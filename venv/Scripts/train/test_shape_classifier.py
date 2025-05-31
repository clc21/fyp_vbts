import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from model_defs import ShapeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import random


def test_shape_classifier(model_path, test_data_dir, batch_size=32, data_folder="shape_0", samples_per_class=100):
    """
    Test shape classifier on specified dataset with limited samples per class

    Args:
        model_path: Path to trained model
        test_data_dir: Base directory containing shape folders
        batch_size: Batch size for testing
        data_folder: Specific folder to test ("shape_0" or "shape_3mm")
        samples_per_class: Maximum number of samples per class to test (default 100)
    """
    # Set random seed for reproducible sampling
    random.seed(42)

    # Configuration
    class_names = ['circle', 'ring', 'triangle', 'star']
    full_test_dir = os.path.join(test_data_dir, data_folder)

    if not os.path.exists(full_test_dir):
        raise ValueError(f"Test directory not found: {full_test_dir}")

    print(f"Testing on dataset: {data_folder}")
    print(f"Test directory: {full_test_dir}")
    print(f"Maximum samples per class: {samples_per_class}")

    # Transformations (same as training)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load full test dataset
    full_test_dataset = datasets.ImageFolder(root=full_test_dir, transform=transform)

    # Get indices for each class and select samples_per_class from each
    class_indices = {}
    for class_name in class_names:
        class_idx = full_test_dataset.class_to_idx.get(class_name)
        if class_idx is not None:
            class_indices[class_name] = [i for i, (_, label) in enumerate(full_test_dataset.samples) if
                                         label == class_idx]

    # Select samples_per_class from each class
    selected_indices = []
    for class_name in class_names:
        indices = class_indices.get(class_name, [])
        if indices:
            if len(indices) > samples_per_class:
                selected = random.sample(indices, samples_per_class)
                print(f"{class_name}: selected {samples_per_class} from {len(indices)} samples")
            else:
                selected = indices
                print(f"{class_name}: using all {len(indices)} samples (fewer than {samples_per_class})")
            selected_indices.extend(selected)

    # Create subset with balanced classes
    test_dataset = Subset(full_test_dataset, selected_indices)

    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShapeClassifier(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Test the model
    all_preds = []
    all_labels = []
    all_probas = []

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probas.extend(probabilities.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Detailed metrics
    idx_to_class = {v: k for k, v in full_test_dataset.class_to_idx.items()}
    class_names_sorted = [idx_to_class[i] for i in range(len(class_names))]

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names_sorted))

    # Confusion matrix with accuracy
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
    plt.title(f'Object Detection ({surface_type}) - Test Confusion Matrix\nAccuracy: {accuracy:.2f}%')

    filename = f'test_obj_{data_folder}_confusion_matrix.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved as '{filename}'")
    plt.close()

    # Plot some example predictions
    def visualize_predictions(test_loader, model, class_names, num_examples_per_class=3):
        model.eval()
        collected = {cls: [] for cls in range(len(class_names))}

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                for i in range(len(images)):
                    label = labels[i].item()
                    if len(collected[label]) < num_examples_per_class:
                        collected[label].append((images[i].cpu(), preds[i].cpu(), labels[i].cpu()))

                if all(len(collected[cls]) >= num_examples_per_class for cls in collected):
                    break

        num_classes = len(class_names)
        fig = plt.figure(figsize=(4 * num_classes, 3 * num_examples_per_class))

        idx = 1
        for row in range(num_examples_per_class):
            for cls in range(num_classes):
                if row < len(collected[cls]):
                    img, pred, true = collected[cls][row]
                    ax = fig.add_subplot(num_examples_per_class, num_classes, idx, xticks=[], yticks=[])
                    img_np = img[0].numpy()
                    ax.imshow(img_np, cmap='gray')
                    title_color = 'green' if pred == true else 'red'
                    ax.set_title(f"T:{class_names[true]}\nP:{class_names[pred]}", color=title_color)
                    idx += 1

        plt.tight_layout()
        surface_type = "Flat" if data_folder == "shape_0" else "Curved"
        filename = f'obj_{data_folder}_predictions.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Shape predictions saved as '{filename}'")
        plt.close()

    # Run visualization
    visualize_predictions(test_loader, model, class_names_sorted)

    return accuracy, all_preds, all_labels, all_probas


if __name__ == "__main__":
    # Test both datasets with 100 samples each
    base_test_dir = "C:/Users/chenc/OneDrive - Imperial College London/Documents/student stuff/fyp_Y4/pics/shape"

    for folder in ["shape_0", "shape_3mm"]:
        print(f"\n{'=' * 60}")
        print(f"Testing Shape Classifier for {folder}")
        print(f"{'=' * 60}")

        model_path = f"models/shape_model_{folder}.pt"

        if os.path.exists(model_path):
            try:
                accuracy, _, _, _ = test_shape_classifier(
                    model_path=model_path,
                    test_data_dir=base_test_dir,
                    data_folder=folder,
                    samples_per_class=100  # Limit to 100 samples per class
                )
                print(f"Completed testing for {folder} with accuracy: {accuracy:.2f}%")
            except Exception as e:
                print(f"Error testing {folder}: {e}")
        else:
            print(f"Model not found: {model_path}")
        print("\n")