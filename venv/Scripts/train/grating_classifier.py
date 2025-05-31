import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class GratingClassifier:
    def __init__(self, base_path, data_folder="grating_0"):
        """
        Initialize the grating classifier with KNN models

        Args:
            base_path (str): Base path to the grating board images
            data_folder (str): Specific folder ("grating_0" or "grating_3mm")
        """
        self.base_path = Path(base_path)
        self.data_folder = data_folder
        self.full_path = self.base_path / data_folder
        self.folder_mapping = {
            '05': 0.5,
            '1': 1.0,
            '125': 1.25,
            '150': 1.5,
            '175': 1.75,
            '2': 2.0
        }
        self.reverse_mapping = {v: k for k, v in self.folder_mapping.items()}
        self.images = []
        self.labels = []
        self.feature_vectors = []
        self.scaler = StandardScaler()
        self.models = {}
        self.image_size = (128, 128)

    def load_images(self, max_images_per_folder=100, random_seed=42):
        """Load exactly max_images_per_folder images from each folder"""
        print(f"Loading images from dataset: {self.data_folder}")
        print(f"Full path: {self.full_path}")
        print(f"Loading up to {max_images_per_folder} images from each resolution folder...")

        if not self.full_path.exists():
            raise ValueError(f"Data directory not found: {self.full_path}")

        np.random.seed(random_seed)  # For reproducible selection

        for folder_name, resolution in self.folder_mapping.items():
            folder_path = self.full_path / folder_name

            if not folder_path.exists():
                print(f"Warning: Folder {folder_path} does not exist")
                continue

            image_files = list(folder_path.glob('*.jpg')) + \
                          list(folder_path.glob('*.png')) + \
                          list(folder_path.glob('*.bmp')) + \
                          list(folder_path.glob('*.tif')) + \
                          list(folder_path.glob('*.jpeg'))

            print(f"Found {len(image_files)} images in {folder_name} (resolution: {resolution})")

            # Randomly select up to max_images_per_folder
            if len(image_files) > max_images_per_folder:
                selected_files = np.random.choice(image_files, max_images_per_folder, replace=False)
                print(f"Randomly selected {max_images_per_folder} images from {folder_name}")
            else:
                selected_files = image_files
                print(f"Using all {len(selected_files)} images from {folder_name}")

            loaded_count = 0
            for img_path in selected_files:
                try:
                    # Load image in grayscale
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Resize to standard size
                        img_resized = cv2.resize(img, self.image_size)
                        self.images.append(img_resized)
                        self.labels.append(resolution)
                        loaded_count += 1
                    else:
                        print(f"Warning: Could not load {img_path}")
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

            print(f"Successfully loaded {loaded_count} images from {folder_name}")

        print(f"\nTotal images loaded: {len(self.images)}")
        label_counts = pd.Series(self.labels).value_counts().sort_index()
        print(f"Label distribution:\n{label_counts}")

        return len(self.images)

    def extract_features(self, img):
        """
        Extract comprehensive features from a single image

        Args:
            img: Input grayscale image (should be resized to standard size)

        Returns:
            numpy array: Feature vector
        """
        features = []

        # Basic statistical features
        features.extend([
            np.mean(img),
            np.std(img),
            np.min(img),
            np.max(img),
            np.percentile(img, 25),
            np.percentile(img, 75),
            np.median(img),
            np.var(img)
        ])

        # Texture features using Local Binary Pattern (simplified version)
        def calculate_lbp_histogram(image, radius=1, n_points=8):
            """Calculate LBP histogram"""
            rows, cols = image.shape
            lbp_codes = []

            for i in range(radius, rows - radius):
                for j in range(radius, cols - radius):
                    center = image[i, j]
                    code = 0

                    # Sample points around the center
                    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
                    for k, angle in enumerate(angles):
                        x = int(i + radius * np.cos(angle))
                        y = int(j + radius * np.sin(angle))
                        x = max(0, min(rows - 1, x))
                        y = max(0, min(cols - 1, y))

                        if image[x, y] >= center:
                            code |= (1 << k)

                    lbp_codes.append(code)

            # Create histogram
            hist, _ = np.histogram(lbp_codes, bins=2 ** n_points, range=(0, 2 ** n_points))
            return hist / (np.sum(hist) + 1e-8)  # Normalize

        lbp_hist = calculate_lbp_histogram(img)
        features.extend(lbp_hist[:16])  # Use first 16 bins to keep feature count manageable

        # Gradient features
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        features.extend([
            np.mean(grad_magnitude),
            np.std(grad_magnitude),
            np.max(grad_magnitude),
            np.mean(np.abs(grad_x)),
            np.mean(np.abs(grad_y))
        ])

        # Frequency domain features
        fft = np.fft.fft2(img)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shift)

        # Radial frequency analysis
        center = np.array(magnitude_spectrum.shape) // 2
        y, x = np.ogrid[:magnitude_spectrum.shape[0], :magnitude_spectrum.shape[1]]
        r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)

        # Different frequency bands
        for band_min, band_max in [(0, 10), (10, 30), (30, 50), (50, 64)]:
            mask = (r >= band_min) & (r < band_max)
            if np.any(mask):
                features.append(np.mean(magnitude_spectrum[mask]))
            else:
                features.append(0)

        # Pattern regularity features
        # Detect lines in different orientations
        kernels = {
            'horizontal': np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32),
            'vertical': np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32),
            'diagonal1': np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float32),
            'diagonal2': np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float32)
        }

        for name, kernel in kernels.items():
            response = cv2.filter2D(img.astype(np.float32), -1, kernel)
            features.extend([
                np.mean(np.abs(response)),
                np.std(response)
            ])

        # Grating-specific features
        # Measure periodicity in horizontal and vertical directions
        for axis in [0, 1]:  # 0=horizontal, 1=vertical
            profile = np.mean(img, axis=axis)

            # Autocorrelation to detect periodicity
            autocorr = np.correlate(profile, profile, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]

            # Find peaks (potential periods)
            if len(autocorr) > 10:
                features.append(np.max(autocorr[1:min(len(autocorr), 20)]))  # Max autocorrelation excluding zero lag
                features.append(np.std(autocorr[1:min(len(autocorr), 20)]))
            else:
                features.extend([0, 0])

        return np.array(features)

    def extract_all_features(self):
        """Extract features from all loaded images"""
        print("Extracting features...")

        self.feature_vectors = []
        for i, img in enumerate(self.images):
            if i % 50 == 0:
                print(f"Processing image {i + 1}/{len(self.images)}")

            features = self.extract_features(img)
            self.feature_vectors.append(features)

        self.feature_vectors = np.array(self.feature_vectors)
        print(f"Feature extraction complete. Shape: {self.feature_vectors.shape}")

    def prepare_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        print(f"\nSplitting data: {int((1 - test_size) * 100)}% train, {int(test_size * 100)}% test")

        # Convert labels to integers for classification
        unique_labels = sorted(set(self.labels))
        label_to_int = {res: i for i, res in enumerate(unique_labels)}
        int_to_label = {i: res for res, i in label_to_int.items()}

        y_int = [label_to_int[label] for label in self.labels]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.feature_vectors, y_int,
            test_size=test_size,
            random_state=random_state,
            stratify=y_int
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Print split info
        print(f"Training set: {len(X_train)} images")
        print(f"Test set: {len(X_test)} images")

        # Show distribution in train/test sets
        train_labels = [int_to_label[i] for i in y_train]
        test_labels = [int_to_label[i] for i in y_test]

        print(f"\nTraining set distribution:")
        print(pd.Series(train_labels).value_counts().sort_index())
        print(f"\nTest set distribution:")
        print(pd.Series(test_labels).value_counts().sort_index())

        return X_train_scaled, X_test_scaled, y_train, y_test, int_to_label

    def train_knn_models(self, X_train, y_train):
        """Train KNN models with different k values"""
        print("\nTraining KNN models...")

        # Test different k values
        k_values = [3, 5, 7]

        for k in k_values:
            self.models[f'KNN (k={k})'] = KNeighborsClassifier(
                n_neighbors=k,
                weights='distance',  # Weight by distance for better performance
                metric='euclidean'
            )

            print(f"Training KNN with k={k}...")
            self.models[f'KNN (k={k})'].fit(X_train, y_train)

        print("All KNN models trained successfully!")

    def evaluate_models(self, X_test, y_test, int_to_label):
        """Evaluate all trained KNN models"""
        results = {}

        print(f"\n{'=' * 60}")
        print("MODEL EVALUATION RESULTS")
        print(f"{'=' * 60}")

        for name, model in self.models.items():
            print(f"\n{'-' * 40}")
            print(f"Evaluating {name}")
            print(f"{'-' * 40}")

            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Convert back to resolution labels for reporting
            y_test_labels = [int_to_label[i] for i in y_test]
            y_pred_labels = [int_to_label[i] for i in y_pred]

            print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

            # Detailed classification report
            print(f"\nClassification Report:")
            # Convert float labels (like 0.5, 1.0) to strings for classification report
            y_test_labels_str = [str(label) for label in y_test_labels]
            y_pred_labels_str = [str(label) for label in y_pred_labels]

            print(classification_report(y_test_labels_str, y_pred_labels_str, digits=4))

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)

            # Store results
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred_labels,
                'true_labels': y_test_labels,
                'confusion_matrix': cm
            }

        return results

    def predict_single_image(self, image_path, model_name='KNN (k=5)'):
        """Predict the grating resolution for a single image"""
        if model_name not in self.models:
            available_models = list(self.models.keys())
            raise ValueError(f"Model {model_name} not found. Available models: {available_models}")

        # Load and process image
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Resize to standard size
        img_resized = cv2.resize(img, self.image_size)

        # Extract features
        features = self.extract_features(img_resized)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Predict
        model = self.models[model_name]
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        # Convert prediction back to resolution
        unique_labels = sorted(set(self.labels))
        label_to_int = {res: i for i, res in enumerate(unique_labels)}
        int_to_label = {i: res for res, i in label_to_int.items()}

        predicted_resolution = int_to_label[prediction]

        # Create probability dictionary
        prob_dict = {int_to_label[i]: prob for i, prob in enumerate(probabilities)}

        return predicted_resolution, prob_dict

    def visualize_grating_predictions(self, best_model, X_test, y_test, int_to_label):
        """
        Visualize 1 prediction per resolution class (row=1, cols=6)
        """
        # Convert back to float labels
        unique_classes = sorted(set(y_test))
        plotted_classes = set()

        fig = plt.figure(figsize=(4 * len(unique_classes), 4))

        idx = 1
        test_images_used = []

        # Find original images corresponding to test set
        for i in range(len(X_test)):
            label = y_test[i]
            if label in plotted_classes:
                continue  # only show one image per class

            x = X_test[i].reshape(1, -1)
            prediction = best_model.predict(x)[0]

            # For visualization, we'll use a representative image from the class
            # Since we don't have direct mapping back to original images from test set,
            # we'll find an image from the same class in our loaded images
            true_resolution = int_to_label[label]
            class_images = [img for i, img in enumerate(self.images) if self.labels[i] == true_resolution]

            if class_images:
                representative_img = class_images[0]  # Use first image of this class

                ax = fig.add_subplot(1, len(unique_classes), idx, xticks=[], yticks=[])
                ax.imshow(representative_img, cmap='gray')

                pred_label = int_to_label[prediction]
                color = 'green' if true_resolution == pred_label else 'red'
                ax.set_title(f"T:{true_resolution}\nP:{pred_label}", color=color)

                idx += 1
                plotted_classes.add(label)

            if len(plotted_classes) == len(unique_classes):
                break  # done

        plt.tight_layout()

        # Determine surface type from folder name
        surface_type = "Flat" if self.data_folder == "grating_0" else "Curved (3mm)"
        filename = f'grating_{self.data_folder}_predictions.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved grating predictions as '{filename}'")
        plt.close()

    def run_full_pipeline(self, max_images_per_folder=100, test_size=0.2):
        """Run the complete KNN classification pipeline"""
        print("Starting Grating Classification Pipeline with KNN")
        print("=" * 60)

        # Load images (100 per folder by default)
        total_images = self.load_images(max_images_per_folder)

        if total_images == 0:
            print("No images found. Please check the file paths.")
            return

        # Extract features
        self.extract_all_features()

        # Prepare data (split into train/test)
        X_train, X_test, y_train, y_test, int_to_label = self.prepare_data(test_size=test_size)

        print(f"\nNumber of features extracted: {X_train.shape[1]}")

        # Train KNN models
        self.train_knn_models(X_train, y_train)

        # Evaluate models
        results = self.evaluate_models(X_test, y_test, int_to_label)

        # Summary
        print(f"\n{'=' * 60}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'=' * 60}")

        # Sort results by accuracy
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

        print(f"Best performing model: {sorted_results[0][0]} (Accuracy: {sorted_results[0][1]['accuracy']:.4f})")

        print(f"\nAll KNN model accuracies:")
        for name, result in sorted_results:
            print(f"{name}: {result['accuracy']:.4f} ({result['accuracy'] * 100:.2f}%)")

        # Print some prediction examples
        best_model_name = sorted_results[0][0]
        best_results = sorted_results[0][1]

        print(f"\nExample predictions from best model ({best_model_name}):")
        for i in range(min(10, len(best_results['true_labels']))):
            true_label = best_results['true_labels'][i]
            pred_label = best_results['predictions'][i]
            status = "✓" if true_label == pred_label else "✗"
            print(f"{status} True: {true_label}, Predicted: {pred_label}")

        # Plot and save confusion matrix with accuracy for best model
        print(f"\nSaving confusion matrix for best model ({best_model_name})...")
        best_cm = best_results['confusion_matrix']
        labels_str = sorted(set(best_results['true_labels']))

        # Calculate percentages for each cell
        cm_percent = best_cm.astype('float') / best_cm.sum(axis=1)[:, np.newaxis] * 100

        # Create labels that show both count and percentage
        annotations = np.empty_like(best_cm, dtype=object)
        for i in range(best_cm.shape[0]):
            for j in range(best_cm.shape[1]):
                annotations[i, j] = f'{best_cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

        plt.figure(figsize=(10, 8))
        sns.heatmap(best_cm, annot=annotations, fmt='', cmap='Blues',
                    xticklabels=labels_str,
                    yticklabels=labels_str)

        # Determine surface type from folder name
        surface_type = "Flat" if self.data_folder == "grating_0" else "Curved (3mm)"
        plt.title(
            f'Grating Detection ({surface_type}) - Confusion Matrix - {best_model_name}\nAccuracy: {best_results["accuracy"] * 100:.2f}%')
        plt.ylabel('True Grating Resolution')
        plt.xlabel('Predicted Grating Resolution')
        plt.tight_layout()

        filename = f'grating_{self.data_folder}_confusion_matrix.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved as '{filename}'")

        # Visualize predictions
        best_model = self.models[best_model_name]
        self.visualize_grating_predictions(best_model, X_test, y_test, int_to_label)

        return results, best_results["accuracy"]


# Usage example
if __name__ == "__main__":
    # Test both datasets
    base_path = r"C:\Users\chenc\OneDrive - Imperial College London\Documents\student stuff\fyp_Y4\pics\gratingBoard"

    for folder in ["grating_0", "grating_3mm"]:
        print(f"\n{'=' * 60}")
        print(f"Training Grating Classifier for {folder}")
        print(f"{'=' * 60}")

        try:
            classifier = GratingClassifier(base_path, data_folder=folder)
            results, accuracy = classifier.run_full_pipeline(max_images_per_folder=100, test_size=0.2)
            print(f"Completed training for {folder} with accuracy: {accuracy:.2f}%")
        except Exception as e:
            print(f"Error training on {folder}: {e}")
        print("\n")