import os
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class GratingFeatureExtractor:
    def __init__(self, base_path, data_folder="grating_0"):
        """
        Initialize the grating feature extractor

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
        self.image_size = (128, 128)

    def load_images(self, max_images_per_folder=100, random_seed=42):
        """Load exactly max_images_per_folder images from each folder"""
        print(f"Loading images from dataset: {self.data_folder}")
        print(f"Full path: {self.full_path}")
        print(f"Loading up to {max_images_per_folder} images from each resolution folder...")

        if not self.full_path.exists():
            raise ValueError(f"Data directory not found: {self.full_path}")

        np.random.seed(random_seed)

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

            if len(image_files) > max_images_per_folder:
                selected_files = np.random.choice(image_files, max_images_per_folder, replace=False)
                print(f"Randomly selected {max_images_per_folder} images from {folder_name}")
            else:
                selected_files = image_files
                print(f"Using all {len(selected_files)} images from {folder_name}")

            loaded_count = 0
            for img_path in selected_files:
                try:
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
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
        Extract comprehensive features from a single image (without LBP)

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

    def extract_all_features(self):
        """Extract features from all loaded images"""
        print("Extracting features...")

        feature_vectors = []
        for i, img in enumerate(self.images):
            if i % 50 == 0:
                print(f"Processing image {i + 1}/{len(self.images)}")

            features = self.extract_features(img)
            feature_vectors.append(features)

        feature_vectors = np.array(feature_vectors)
        print(f"Feature extraction complete. Shape: {feature_vectors.shape}")

        return feature_vectors, self.labels

    def save_features(self, features, labels, filename_prefix):
        """Save features and labels to files"""
        np.save(f"{filename_prefix}_features.npy", features)
        np.save(f"{filename_prefix}_labels.npy", labels)
        print(f"Saved features to {filename_prefix}_features.npy")
        print(f"Saved labels to {filename_prefix}_labels.npy")

    def run_feature_extraction(self, max_images_per_folder=100):
        """Run the complete feature extraction pipeline"""
        print("Starting Grating Feature Extraction Pipeline")
        print("=" * 60)

        # Load images
        total_images = self.load_images(max_images_per_folder)

        if total_images == 0:
            print("No images found. Please check the file paths.")
            return None, None

        # Extract features
        features, labels = self.extract_all_features()

        # Save features
        self.save_features(features, labels, f"grating_{self.data_folder}")

        return features, labels


if __name__ == "__main__":
    base_path = r"...\pics\gratingBoard"

    for folder in ["grating_0", "grating_3mm"]:
        print(f"\n{'=' * 60}")
        print(f"Extracting features for {folder}")
        print(f"{'=' * 60}")

        try:
            extractor = GratingFeatureExtractor(base_path, data_folder=folder)
            features, labels = extractor.run_feature_extraction(max_images_per_folder=100)
            if features is not None:
                print(f"Completed feature extraction for {folder}")
                print(f"Feature shape: {features.shape}")
        except Exception as e:
            print(f"Error extracting features for {folder}: {e}")
        print("\n")