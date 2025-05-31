import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_defs import ShapeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def test_shape_classifier(model_path, test_data_dir, batch_size=32):
    # Configuration
    class_names = ['circle', 'ring', 'triangle', 'star']

    # Transformations (same as training)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load test dataset
    test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform)
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
    idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
    class_names_sorted = [idx_to_class[i] for i in range(len(class_names))]

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names_sorted))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_sorted, yticklabels=class_names_sorted)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('test_confusion_matrix.png')
    print("Confusion matrix saved as 'testCurved_confusion_matrix.png'")

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
        plt.savefig('shape_predictions.png')
        print("Shape predictions saved as 'objCurved_predictions.png'")

    # Run visualization
    visualize_predictions(test_loader, model, class_names_sorted)

    return accuracy, all_preds, all_labels, all_probas


if __name__ == "__main__":
    # Configuration
    model_path = "models/shape_model.pt"
    test_data_dir = "C:/Users/chenc/OneDrive - Imperial College London/Documents/student stuff/fyp_Y4/pics/shape/shape_3mm"

    # Run test
    test_shape_classifier(model_path, test_data_dir)