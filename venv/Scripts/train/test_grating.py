import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import warnings
import glob

warnings.filterwarnings('ignore')


class GratingTester:
    def __init__(self, model_path, data_folder="grating_0"):
        """
        Initialize the grating classifier tester

        Args:
            model_path (str): Path to saved model file
            data_folder (str): Specific folder ("grating_0" or "grating_3mm")
        """
        self.data_folder = data_folder
        self.model_path = model_path
        self.image_size = (128, 128)

        # Load trained model
        self.load_model()

    def load_model(self):
        """Load pre-trained model and associated components"""
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_to_int = model_data['label_mappings']['label_to_int']
            self.int_to_label = model_data['label_mappings']['int_to_label']

            print(f"Loaded model from: {self.model_path}")
            print(f"Model type: {type(self.model).__name__}")
            if hasattr(self.model, 'n_neighbors'):
                print(f"K neighbors: {self.model.n_neighbors}")
                print(f"Weights: {self.model.weights}")
        except Exception as e:
            raise FileNotFoundError(f"Could not load model from {self.model_path}: {e}")

    def extract_features(self, img):
        """
        Extract features from a single image (same as training)
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
        for axis in [0, 1]:  # 0=horizontal, 1=vertical
            profile = np.mean(img, axis=axis)

            # Autocorrelation to detect periodicity
            autocorr = np.correlate(profile, profile, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]

            if len(autocorr) > 10:
                features.append(np.max(autocorr[1:min(len(autocorr), 20)]))
                features.append(np.std(autocorr[1:min(len(autocorr), 20)]))
            else:
                features.extend([0, 0])

        return np.array(features)

    def predict_single_image(self, image_path):
        """
        Predict the grating resolution for a single image

        Args:
            image_path (str): Path to the image file

        Returns:
            tuple: (predicted_resolution, confidence_scores)
        """
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
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]

        # Convert prediction back to resolution
        predicted_resolution = self.int_to_label[prediction]

        # Create probability dictionary
        prob_dict = {self.int_to_label[i]: prob for i, prob in enumerate(probabilities)}

        return predicted_resolution, prob_dict

    def test_on_saved_features(self):
        """Test the model on saved test features"""
        try:
            # Load the saved test features (assuming they were saved during training)
            features = np.load(f"grating_{self.data_folder}_features.npy")
            labels = np.load(f"grating_{self.data_folder}_labels.npy")

            print(f"Loaded test features: {features.shape}")
            print(f"Loaded test labels: {len(labels)}")

            # Scale features
            features_scaled = self.scaler.transform(features)

            # Convert labels
            y_int = [self.label_to_int[label] for label in labels]

            # Predict
            predictions = self.model.predict(features_scaled)
            probabilities = self.model.predict_proba(features_scaled)

            # Calculate accuracy
            accuracy = accuracy_score(y_int, predictions)

            # Convert back to resolution labels
            pred_labels = [self.int_to_label[pred] for pred in predictions]

            print(f"\nTest Results:")
            print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

            # Classification report
            labels_str = [str(label) for label in labels]
            pred_labels_str = [str(label) for label in pred_labels]
            print(f"\nClassification Report:")
            print(classification_report(labels_str, pred_labels_str, digits=4))

            return accuracy, pred_labels, labels

        except FileNotFoundError:
            print("Saved features not found. Please run feature extraction first.")
            return None, None, None

    def test_on_directory(self, test_dir, max_images_per_class=20):
        """
        Test the model on images from a directory structure

        Args:
            test_dir (str): Path to test directory with subdirectories for each class
            max_images_per_class (int): Maximum images to test per class
        """
        test_path = Path(test_dir)
        if not test_path.exists():
            raise ValueError(f"Test directory not found: {test_dir}")

        folder_mapping = {
            '05': 0.5,
            '1': 1.0,
            '125': 1.25,
            '150': 1.5,
            '175': 1.75,
            '2': 2.0
        }

        all_predictions = []
        all_true_labels = []
        all_probabilities = []

        print(f"Testing on directory: {test_dir}")
        print(f"Testing up to {max_images_per_class} images per class")

        for folder_name, resolution in folder_mapping.items():
            folder_path = test_path / folder_name

            if not folder_path.exists():
                print(f"Warning: Folder {folder_path} does not exist")
                continue

            # Get image files
            image_files = list(folder_path.glob('*.jpg')) + \
                          list(folder_path.glob('*.png')) + \
                          list(folder_path.glob('*.bmp')) + \
                          list(folder_path.glob('*.tif')) + \
                          list(folder_path.glob('*.jpeg'))

            if len(image_files) > max_images_per_class:
                image_files = np.random.choice(image_files, max_images_per_class, replace=False)

            print(f"Testing {len(image_files)} images from class {resolution}")

            for img_path in image_files:
                try:
                    pred_resolution, prob_dict = self.predict_single_image(img_path)
                    all_predictions.append(pred_resolution)
                    all_true_labels.append(resolution)
                    all_probabilities.append(prob_dict)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

        if not all_predictions:
            print("No predictions made. Check test directory structure.")
            return None

        # Calculate overall accuracy
        accuracy = sum(p == t for p, t in zip(all_predictions, all_true_labels)) / len(all_predictions)

        print(f"\nOverall Test Results:")
        print(f"Total images tested: {len(all_predictions)}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        # Per-class accuracy
        print(f"\nPer-class Results:")
        for resolution in sorted(set(all_true_labels)):
            class_indices = [i for i, label in enumerate(all_true_labels) if label == resolution]
            class_predictions = [all_predictions[i] for i in class_indices]
            class_accuracy = sum(p == resolution for p in class_predictions) / len(class_predictions)
            print(
                f"Class {resolution}: {class_accuracy:.4f} ({class_accuracy * 100:.2f}%) - {len(class_indices)} images")

        return accuracy, all_predictions, all_true_labels, all_probabilities

    def visualize_predictions(self, test_dir, num_examples=6):
        """
        Visualize predictions on sample images

        Args:
            test_dir (str): Path to test directory
            num_examples (int): Number of examples to show
        """
        test_path = Path(test_dir)
        if not test_path.exists():
            raise ValueError(f"Test directory not found: {test_dir}")

        folder_mapping = {
            '05': 0.5,
            '1': 1.0,
            '125': 1.25,
            '150': 1.5,
            '175': 1.75,
            '2': 2.0
        }

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        sample_count = 0

        for folder_name, resolution in folder_mapping.items():
            if sample_count >= num_examples:
                break

            folder_path = test_path / folder_name
            if not folder_path.exists():
                continue

            image_files = list(folder_path.glob('*.jpg')) + list(folder_path.glob('*.png'))
            if not image_files:
                continue

            # Pick a random image
            img_path = np.random.choice(image_files)

            try:
                # Load image for display
                img_display = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

                # Make prediction
                pred_resolution, prob_dict = self.predict_single_image(img_path)

                # Display
                ax = axes[sample_count]
                ax.imshow(img_display, cmap='gray')
                ax.set_title(f"True: {resolution}\nPred: {pred_resolution}\nConf: {prob_dict[pred_resolution]:.3f}")
                ax.axis('off')

                # Color the title based on correctness
                color = 'green' if pred_resolution == resolution else 'red'
                ax.title.set_color(color)

                sample_count += 1

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        plt.tight_layout()

        surface_type = "Flat" if self.data_folder == "grating_0" else "Curved (3mm)"
        plt.suptitle(f'Grating Classification Examples - {surface_type}', y=1.02)

        filename = f'grating_{self.data_folder}_test_examples.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Test examples saved as '{filename}'")
        plt.close()

    def run_comprehensive_test(self, test_dir=None):
        """Run comprehensive testing pipeline"""
        print("Starting Comprehensive Grating Classification Testing")
        print("=" * 60)

        # Test on saved features if available
        print("\n1. Testing on saved features...")
        accuracy, pred_labels, true_labels = self.test_on_saved_features()

        # Test on directory if provided
        if test_dir:
            print(f"\n2. Testing on directory: {test_dir}")
            dir_accuracy, dir_preds, dir_true, dir_probs = self.test_on_directory(test_dir)

            # Visualize some predictions
            print("\n3. Creating visualization...")
            self.visualize_predictions(test_dir, num_examples=6)

            return {
                'saved_features_accuracy': accuracy,
                'directory_accuracy': dir_accuracy,
                'directory_predictions': dir_preds,
                'directory_true_labels': dir_true
            }
        else:
            return {
                'saved_features_accuracy': accuracy,
                'predictions': pred_labels,
                'true_labels': true_labels
            }


def run_grating_test(model_path=None, data_folder="grating_0", test_dir=None, test_image=None):
    """
    Standalone function to run grating testing

    Args:
        model_path (str): Path to the trained model, if None will search for models
        data_folder (str): Dataset folder ("grating_0" or "grating_3mm")
        test_dir (str): Directory containing test images
        test_image (str): Path to single image for testing
    """
    print(f"Running grating test for {data_folder}")

    # Find model if not provided
    if model_path is None:
        model_patterns = [
            f"grating_{data_folder}_knn_k*_*_model.pkl",
            f"grating_{data_folder}_*_model.pkl"
        ]

        for pattern in model_patterns:
            matches = glob.glob(pattern)
            if matches:
                model_path = matches[0]
                break

        if model_path is None:
            raise FileNotFoundError(f"No trained model found for {data_folder}")

    # Initialize tester
    tester = GratingTester(model_path, data_folder=data_folder)

    # Test single image if provided
    if test_image:
        if Path(test_image).exists():
            print(f"Testing single image: {test_image}")
            prediction, probabilities = tester.predict_single_image(test_image)
            print(f"Prediction: {prediction}")
            print(f"Probabilities: {probabilities}")
            return prediction, probabilities
        else:
            raise FileNotFoundError(f"Test image not found: {test_image}")

    # Otherwise run comprehensive test
    if test_dir:
        results = tester.run_comprehensive_test(test_dir=test_dir)
        return results
    else:
        # Test on saved features only
        results = tester.run_comprehensive_test()
        return results


if __name__ == "__main__":
    import argparse

    # Test both models when run standalone
    parser = argparse.ArgumentParser(description='Test grating classifier')
    parser.add_argument('--grating_dir', type=str,
                        default=".../pics/gratingBoard",
                        help='Directory containing grating board images')
    parser.add_argument('--data_folder', type=str, default='both',
                        choices=['grating_0', 'grating_3mm', 'both'],
                        help='Which dataset to test')
    parser.add_argument('--test_image', type=str, default=None,
                        help='Path to single image for testing')

    args = parser.parse_args()

    base_path = args.grating_dir

    # Determine which folders to test
    if args.data_folder == 'both':
        folders = ["grating_0", "grating_3mm"]
    else:
        folders = [args.data_folder]

    for folder in folders:
        print(f"\n{'=' * 80}")
        print(f"Testing Grating Classifier for {folder}")
        print(f"{'=' * 80}")

        try:
            test_dir = Path(base_path) / folder if Path(base_path).exists() else None

            results = run_grating_test(
                data_folder=folder,
                test_dir=str(test_dir) if test_dir else None,
                test_image=args.test_image
            )

            print(f"Completed testing for {folder}")

        except Exception as e:
            print(f"Error testing {folder}: {e}")
            import traceback

            traceback.print_exc()
        print("\n")