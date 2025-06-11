import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import random
from collections import defaultdict
import json
import argparse


class DepthTester:
    def __init__(self, model_path='depthModels/depth_classification_model.h5', img_size=(224, 224)):
        self.model_path = model_path
        self.img_size = img_size
        self.depth_mapping = {'0': 0, '05': 1, '1': 2, '15': 3, '2': 4}  # Fixed: '20' -> '2'
        self.depth_labels = ['0mm', '0.5mm', '1mm', '1.5mm', '2mm']
        self.object_types = ['metal', 'black', 'ring', 'triangle']
        self.surface_types = ['curved']
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Please make sure the model file exists and run training first.")

    def load_and_preprocess_image(self, img_path):
        """Load and preprocess a single image"""
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                return None

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize image
            img = cv2.resize(img, self.img_size)

            # Normalize pixel values to [0, 1]
            img = img.astype(np.float32) / 255.0

            return img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

    def check_directory_structure(self, base_path):
        """Check if the directory structure is correct"""
        print("Checking directory structure...")
        print("=" * 50)

        if not os.path.exists(base_path):
            print(f"❌ Base directory '{base_path}' not found!")
            return False

        structure_valid = True
        total_images = 0

        for surface_type in self.surface_types:
            surface_path = os.path.join(base_path, surface_type)

            if not os.path.exists(surface_path):
                print(f"❌ Surface directory '{surface_path}' not found!")
                structure_valid = False
                continue

            print(f"\n📁 {surface_type.upper()} SURFACE:")
            surface_total = 0

            for object_type in self.object_types:
                object_path = os.path.join(surface_path, object_type)

                if not os.path.exists(object_path):
                    print(f"  ❌ Object directory '{object_type}' not found!")
                    structure_valid = False
                    continue

                print(f"  📁 {object_type}:")
                object_total = 0

                for depth_folder in ['0', '05', '1', '15', '2']:
                    depth_path = os.path.join(object_path, depth_folder)

                    if os.path.exists(depth_path):
                        # Count image files
                        image_files = [f for f in os.listdir(depth_path)
                                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

                        count = len(image_files)
                        depth_label = self.depth_labels[self.depth_mapping[depth_folder]]

                        status = "✅" if count > 0 else "⚠️ "
                        print(f"    {status} {depth_label:>6}: {count:>3} images")

                        object_total += count

                        if count == 0:
                            print(f"        Warning: No images found in {depth_path}")
                    else:
                        depth_label = self.depth_labels[self.depth_mapping[depth_folder]]
                        print(f"    ❌ {depth_label:>6}: Directory not found")
                        structure_valid = False

                print(f"    📊 Total {object_type}: {object_total}")
                surface_total += object_total

            print(f"  📊 Total {surface_type} images: {surface_total}")
            total_images += surface_total

        print(f"\n📊 TOTAL IMAGES: {total_images}")

        if structure_valid and total_images > 0:
            print("✅ Directory structure is valid!")
        else:
            print("❌ Directory structure has issues!")

        return structure_valid and total_images > 0

    def load_test_dataset(self, base_path, max_images_per_class=50, surface_filter=None):
        """Load test dataset with the correct structure"""
        images = []
        labels = []
        surface_types = []
        object_types = []
        image_paths = []

        print("Loading test dataset...")
        print(f"Base path: {base_path}")
        print(f"Max images per class: {max_images_per_class}")
        if surface_filter:
            print(f"Surface filter: {surface_filter}")

        # Determine which surfaces to process
        surfaces_to_process = [surface_filter] if surface_filter else self.surface_types

        for surface_type in surfaces_to_process:
            surface_path = os.path.join(base_path, surface_type)
            if not os.path.exists(surface_path):
                print(f"Warning: Path {surface_path} does not exist")
                continue

            for object_type in self.object_types:
                object_path = os.path.join(surface_path, object_type)
                if not os.path.exists(object_path):
                    print(f"Warning: Path {object_path} does not exist")
                    continue

                for depth_folder in ['0', '05', '1', '15', '2']:
                    depth_path = os.path.join(object_path, depth_folder)

                    if not os.path.exists(depth_path):
                        print(f"Warning: Path {depth_path} does not exist")
                        continue

                    # Get all image files
                    image_files = [f for f in os.listdir(depth_path)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

                    # Randomly sample up to max_images_per_class
                    if len(image_files) > max_images_per_class:
                        image_files = random.sample(image_files, max_images_per_class)

                    print(f"Loading {len(image_files)} images from {surface_type}/{object_type}/{depth_folder}")

                    for img_file in image_files:
                        img_path = os.path.join(depth_path, img_file)
                        img = self.load_and_preprocess_image(img_path)

                        if img is not None:
                            images.append(img)
                            labels.append(self.depth_mapping[depth_folder])
                            surface_types.append(0 if surface_type == 'flat' else 1)
                            object_types.append(self.object_types.index(object_type))
                            image_paths.append(img_path)

        print(f"Total test images loaded: {len(images)}")

        if len(images) == 0:
            print("❌ No images found! Please check your directory structure.")
            return None, None, None, None, None

        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        surface_types = np.array(surface_types)
        object_types = np.array(object_types)

        return X, y, surface_types, object_types, image_paths

    def predict_single_image(self, img_path):
        """Predict depth for a single image"""
        if self.model is None:
            print("Model not loaded!")
            return None

        img = self.load_and_preprocess_image(img_path)
        if img is None:
            return None

        # Add batch dimension
        img_batch = np.expand_dims(img, axis=0)

        # Predict
        prediction = self.model.predict(img_batch, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])

        return {
            'predicted_depth': self.depth_labels[predicted_class],
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': prediction[0]
        }

    def evaluate_model(self, X_test, y_test, surface_types_test, object_types_test, image_paths):
        """Comprehensive model evaluation"""
        if self.model is None:
            print("Model not loaded!")
            return

        print("Running model evaluation...")

        # Get predictions
        y_pred_probs = self.model.predict(X_test, verbose=1)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)

        # Overall accuracy
        overall_accuracy = accuracy_score(y_test, y_pred_classes)
        print(f"\nOverall Test Accuracy: {overall_accuracy:.4f}")

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes,
                                    target_names=self.depth_labels,
                                    digits=4))

        # Overall confusion matrix
        cm_overall = confusion_matrix(y_test, y_pred_classes)
        self.plot_confusion_matrix(cm_overall, "Overall", self.depth_labels)

        # Evaluate by surface type
        results = {
            'overall_accuracy': overall_accuracy,
            'surface_accuracies': {},
            'object_accuracies': {}
        }

        for surface_idx, surface_name in enumerate(['flat', 'curved']):
            mask = surface_types_test == surface_idx
            if np.sum(mask) == 0:
                continue

            y_test_surface = y_test[mask]
            y_pred_surface = y_pred_classes[mask]

            surface_accuracy = accuracy_score(y_test_surface, y_pred_surface)
            results['surface_accuracies'][surface_name] = surface_accuracy

            print(f"\n{surface_name.capitalize()} Surface Accuracy: {surface_accuracy:.4f}")

            # Surface-specific classification report
            print(f"\n{surface_name.capitalize()} Surface Classification Report:")
            print(classification_report(y_test_surface, y_pred_surface,
                                        target_names=self.depth_labels,
                                        digits=4))

            # Surface-specific confusion matrix
            cm_surface = confusion_matrix(y_test_surface, y_pred_surface)
            self.plot_confusion_matrix(cm_surface, f"{surface_name.capitalize()} Surface", self.depth_labels)

        # Evaluate by object type
        for object_idx, object_name in enumerate(self.object_types):
            mask = object_types_test == object_idx
            if np.sum(mask) == 0:
                continue

            y_test_object = y_test[mask]
            y_pred_object = y_pred_classes[mask]

            object_accuracy = accuracy_score(y_test_object, y_pred_object)
            results['object_accuracies'][object_name] = object_accuracy

            print(f"\n{object_name.capitalize()} Object Accuracy: {object_accuracy:.4f}")

        # Per-class accuracy analysis
        self.analyze_per_class_accuracy(y_test, y_pred_classes, surface_types_test, object_types_test)

        # Error analysis
        self.analyze_errors(X_test, y_test, y_pred_classes, surface_types_test, object_types_test, image_paths,
                            y_pred_probs)

        # Save results
        self.save_results(results, y_test, y_pred_classes, surface_types_test, object_types_test)

        return results

    def plot_confusion_matrix(self, cm, title, labels):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {title}')
        plt.xlabel('Predicted Depth')
        plt.ylabel('True Depth')
        plt.tight_layout()

        # Save plot
        filename = f'confusion_matrix_{title.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

        # Calculate and display per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        print(f"\nPer-class accuracy for {title}:")
        for i, acc in enumerate(per_class_acc):
            if not np.isnan(acc):
                print(f"  {labels[i]}: {acc:.4f}")

    def analyze_per_class_accuracy(self, y_true, y_pred, surface_types, object_types):
        """Analyze accuracy per depth class"""
        print("\n" + "=" * 50)
        print("PER-CLASS ACCURACY ANALYSIS")
        print("=" * 50)

        for depth_idx, depth_label in enumerate(self.depth_labels):
            mask = y_true == depth_idx
            if np.sum(mask) == 0:
                continue

            y_true_class = y_true[mask]
            y_pred_class = y_pred[mask]
            surface_class = surface_types[mask]
            object_class = object_types[mask]

            class_accuracy = accuracy_score(y_true_class, y_pred_class)

            print(f"\nDepth {depth_label}:")
            print(f"  Overall accuracy: {class_accuracy:.4f} ({np.sum(y_pred_class == depth_idx)}/{len(y_true_class)})")

            # Accuracy by surface type
            for surface_idx, surface_name in enumerate(['flat', 'curved']):
                surface_mask = surface_class == surface_idx
                if np.sum(surface_mask) > 0:
                    y_true_surf_class = y_true_class[surface_mask]
                    y_pred_surf_class = y_pred_class[surface_mask]

                    surf_accuracy = accuracy_score(y_true_surf_class, y_pred_surf_class)
                    correct = np.sum(y_pred_surf_class == depth_idx)
                    total = len(y_true_surf_class)
                    print(f"  {surface_name.capitalize()} surface: {surf_accuracy:.4f} ({correct}/{total})")

            # Accuracy by object type
            for object_idx, object_name in enumerate(self.object_types):
                object_mask = object_class == object_idx
                if np.sum(object_mask) > 0:
                    y_true_obj_class = y_true_class[object_mask]
                    y_pred_obj_class = y_pred_class[object_mask]

                    obj_accuracy = accuracy_score(y_true_obj_class, y_pred_obj_class)
                    correct = np.sum(y_pred_obj_class == depth_idx)
                    total = len(y_true_obj_class)
                    print(f"  {object_name.capitalize()} object: {obj_accuracy:.4f} ({correct}/{total})")

    def analyze_errors(self, X_test, y_true, y_pred, surface_types, object_types, image_paths, y_pred_probs):
        """Analyze prediction errors"""
        print("\n" + "=" * 50)
        print("ERROR ANALYSIS")
        print("=" * 50)

        # Find misclassified samples
        errors = y_true != y_pred
        error_indices = np.where(errors)[0]

        print(
            f"Total errors: {len(error_indices)} out of {len(y_true)} ({len(error_indices) / len(y_true) * 100:.2f}%)")

        if len(error_indices) > 0:
            # Analyze error patterns
            error_matrix = defaultdict(int)
            for idx in error_indices:
                true_label = self.depth_labels[y_true[idx]]
                pred_label = self.depth_labels[y_pred[idx]]
                error_matrix[(true_label, pred_label)] += 1

            print("\nMost common error patterns:")
            sorted_errors = sorted(error_matrix.items(), key=lambda x: x[1], reverse=True)
            for (true_label, pred_label), count in sorted_errors[:10]:
                print(f"  {true_label} → {pred_label}: {count} times")

            # Show some example errors with low confidence
            confidences = np.max(y_pred_probs, axis=1)
            low_conf_errors = error_indices[confidences[error_indices] < 0.7]

            if len(low_conf_errors) > 0:
                print(f"\nLow confidence errors (confidence < 0.7): {len(low_conf_errors)}")
                for i, idx in enumerate(low_conf_errors[:5]):  # Show first 5
                    true_label = self.depth_labels[y_true[idx]]
                    pred_label = self.depth_labels[y_pred[idx]]
                    confidence = confidences[idx]
                    surface = 'flat' if surface_types[idx] == 0 else 'curved'
                    object_type = self.object_types[object_types[idx]]
                    print(f"  {i + 1}. True: {true_label}, Pred: {pred_label}, Conf: {confidence:.3f}, "
                          f"Surface: {surface}, Object: {object_type}")

    def save_results(self, results, y_true, y_pred, surface_types, object_types):
        """Save detailed results to JSON file"""
        detailed_results = {
            'overall_accuracy': float(results['overall_accuracy']),
            'surface_accuracies': {k: float(v) for k, v in results['surface_accuracies'].items()},
            'object_accuracies': {k: float(v) for k, v in results['object_accuracies'].items()},
            'per_class_metrics': {},
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }

        # Calculate per-class metrics
        for depth_idx, depth_label in enumerate(self.depth_labels):
            mask = y_true == depth_idx
            if np.sum(mask) > 0:
                class_accuracy = accuracy_score(y_true[mask], y_pred[mask])
                detailed_results['per_class_metrics'][depth_label] = {
                    'accuracy': float(class_accuracy),
                    'sample_count': int(np.sum(mask))
                }

        # Save to JSON
        with open('test_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)

        print(f"\nDetailed results saved to 'test_results.json'")

    def test_random_samples(self, X_test, y_test, surface_types, object_types, n_samples=10):
        """Test on random samples and show predictions"""
        if self.model is None:
            print("Model not loaded!")
            return

        print(f"\n" + "=" * 50)
        print(f"RANDOM SAMPLE PREDICTIONS")
        print("=" * 50)

        # Select random samples
        indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)

        for i, idx in enumerate(indices):
            img = X_test[idx:idx + 1]  # Keep batch dimension
            true_label = self.depth_labels[y_test[idx]]
            surface = 'flat' if surface_types[idx] == 0 else 'curved'
            object_type = self.object_types[object_types[idx]]

            # Predict
            prediction = self.model.predict(img, verbose=0)
            predicted_class = np.argmax(prediction[0])
            predicted_label = self.depth_labels[predicted_class]
            confidence = np.max(prediction[0])

            status = "✓" if predicted_class == y_test[idx] else "✗"

            print(f"{i + 1:2d}. {status} True: {true_label:>6}, Pred: {predicted_label:>6}, "
                  f"Conf: {confidence:.3f}, Surface: {surface:>6}, Object: {object_type:>8}")


if __name__ == "__main__":
    main()