import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Subset
from model_defs import ShapeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def run(data_folder="shape_0", epochs=11, samples_per_class=100):
    """
    Train shape classifier for a specific dataset folder

    Args:
        data_folder: "shape_0" or "shape_3mm"
        epochs: Number of training epochs
        samples_per_class: Number of samples per class to use
    """
    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Configuration
    base_dir = ".../pics/shape"
    root_dir = os.path.join(base_dir, data_folder)

    if not os.path.exists(root_dir):
        raise ValueError(f"Data directory not found: {root_dir}")

    batch_size = 32
    lr = 1e-4
    class_names = ['circle', 'ring', 'triangle', 'star']

    print(f"Training on dataset: {data_folder}")
    print(f"Data directory: {root_dir}")

    # Transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load full dataset
    full_dataset = datasets.ImageFolder(root=root_dir, transform=transform)

    # Get indices for each class
    class_indices = {}
    for class_name in class_names:
        class_idx = full_dataset.class_to_idx.get(class_name)
        if class_idx is not None:
            class_indices[class_name] = [i for i, (_, label) in enumerate(full_dataset.samples) if label == class_idx]

    # Select samples_per_class from each class
    selected_indices = []
    for class_name in class_names:
        indices = class_indices.get(class_name, [])
        if indices:
            if len(indices) > samples_per_class:
                selected = random.sample(indices, samples_per_class)
            else:
                selected = indices
                print(f"Warning: Only {len(indices)} samples available for class {class_name}")
            selected_indices.extend(selected)

    # Create subset with balanced classes
    balanced_dataset = Subset(full_dataset, selected_indices)

    # Split the dataset: 70% train, 15% validation, 15% test
    total_samples = len(balanced_dataset)
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    test_size = total_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        balanced_dataset, [train_size, val_size, test_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model setup
    model = ShapeClassifier(num_classes=len(class_names))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # Save model
    os.makedirs("models", exist_ok=True)
    model_filename = f"shape_model_{data_folder}.pt"
    torch.save(model.state_dict(), f"models/{model_filename}")
    print(f"Shape classifier trained and saved as {model_filename}")

    # Evaluate on test set
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)

            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = 100 * test_correct / test_total
    print(f"\nTest Accuracy: {test_accuracy:.2f}%")

    # Classification report
    idx_to_class = {v: k for k, v in full_dataset.class_to_idx.items()}
    class_names_sorted = [idx_to_class[i] for i in range(len(class_names))]

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names_sorted))

    # Confusion matrix with percentages and overall accuracy
    cm = confusion_matrix(all_labels, all_preds)

    # Calculate percentages for each cell
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create labels that show both count and percentage
    labels = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            labels[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues',
                xticklabels=class_names_sorted, yticklabels=class_names_sorted)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Determine surface type from folder name
    surface_type = "Flat" if data_folder == "shape_0" else "Curved (3mm)"
    plt.title(f'Object Detection ({surface_type}) - Confusion Matrix\nAccuracy: {test_accuracy:.2f}%')

    filename = f'train_obj_{data_folder}_confusion_matrix.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved as '{filename}'")
    plt.close()

    return model, test_accuracy


if __name__ == "__main__":
    # Train on both datasets
    for folder in ["shape_0", "shape_3mm"]:
        print(f"\n{'=' * 60}")
        print(f"Training Shape Classifier for {folder}")
        print(f"{'=' * 60}")
        try:
            model, accuracy = run(data_folder=folder)
            print(f"Completed training for {folder} with accuracy: {accuracy:.2f}%")
        except Exception as e:
            print(f"Error training on {folder}: {e}")
        print("\n")