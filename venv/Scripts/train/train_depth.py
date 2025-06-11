import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import random
from collections import defaultdict
import argparse

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class DepthClassifier:
    def __init__(self, base_path, img_size=(96, 96)):
        self.base_path = base_path
        self.img_size = img_size
        self.depth_mapping = {'0': 0, '05': 1, '1': 2, '15': 3, '2': 4}
        self.depth_labels = ['0mm', '0.5mm', '1.0mm', '1.5mm', '2.0mm']
        self.object_types = ['metal', 'black', 'ring', 'triangle']
        self.surface_types = ['curved']
        self.model = None

    def load_and_preprocess_image(self, img_path):
        """Load and preprocess a single image with robust handling"""
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                return None

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize image
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)

            # Convert to grayscale and back to RGB for consistency
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

            # Normalize pixel values to [0, 1]
            img = img.astype(np.float32) / 255.0

            return img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

    def create_simple_model(self):
        """Create a very simple model that's less prone to overfitting"""
        model = models.Sequential([
            # First conv block
            layers.Conv2D(8, (5, 5), activation='relu', input_shape=(*self.img_size, 3)),
            layers.MaxPooling2D((3, 3)),
            layers.Dropout(0.1),

            # Second conv block
            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),

            # Third conv block
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.GlobalAveragePooling2D(),

            # Simple classifier
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(5, activation='softmax')
        ])

        return model

    def compile_model(self, model):
        """Compile with simpler optimizer settings"""
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),  # Higher LR for faster learning
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def load_dataset(self, max_images_per_class=25, surface_filter=None):
        """Load dataset with immediate validation"""
        images = []
        labels = []
        image_paths = []

        print("Loading dataset with validation...")
        print(f"Base path: {self.base_path}")
        print(f"Max images per class: {max_images_per_class}")

        # Check if base path exists
        if not os.path.exists(self.base_path):
            print(f"❌ Base path does not exist: {self.base_path}")
            return None, None, None

        # Determine which surfaces to process
        surfaces_to_process = [surface_filter] if surface_filter else self.surface_types

        class_counts = defaultdict(int)

        for surface_type in surfaces_to_process:
            surface_path = os.path.join(self.base_path, surface_type)
            print(f"Checking surface path: {surface_path}")

            if not os.path.exists(surface_path):
                print(f"❌ Surface path does not exist: {surface_path}")
                continue

            for object_type in self.object_types:
                object_path = os.path.join(surface_path, object_type)
                print(f"Checking object path: {object_path}")

                if not os.path.exists(object_path):
                    print(f"⚠️ Object path does not exist: {object_path}")
                    continue

                for depth_folder in ['0', '05', '1', '15', '2']:
                    depth_path = os.path.join(object_path, depth_folder)
                    print(f"Checking depth path: {depth_path}")

                    if not os.path.exists(depth_path):
                        print(f"⚠️ Depth path does not exist: {depth_path}")
                        continue

                    # Get all image files
                    try:
                        all_files = os.listdir(depth_path)
                        image_files = [f for f in all_files
                                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

                        print(f"Found {len(image_files)} images in {depth_folder}")

                        if len(image_files) == 0:
                            print(f"⚠️ No images found in {depth_path}")
                            continue

                        # Randomly sample images
                        if len(image_files) > max_images_per_class:
                            image_files = random.sample(image_files, max_images_per_class)

                        loaded_count = 0
                        for img_file in image_files:
                            img_path = os.path.join(depth_path, img_file)
                            img = self.load_and_preprocess_image(img_path)

                            if img is not None:
                                images.append(img)
                                labels.append(self.depth_mapping[depth_folder])
                                image_paths.append(img_path)
                                loaded_count += 1
                                class_counts[self.depth_mapping[depth_folder]] += 1

                        print(
                            f"Successfully loaded {loaded_count} images from {surface_type}/{object_type}/{depth_folder}")

                    except Exception as e:
                        print(f"❌ Error processing {depth_path}: {e}")
                        continue

        print(f"\nTotal images loaded: {len(images)}")

        if len(images) == 0:
            print("❌ No images found! Please check your directory structure.")
            print("Expected structure: base_path/surface_type/object_type/depth_folder/")
            return None, None, None

        # Print detailed class distribution
        print("\nClass distribution:")
        for class_idx in range(5):
            count = class_counts[class_idx]
            print(f"  Class {class_idx} ({self.depth_labels[class_idx]}): {count} images")

        # Check for class imbalance
        min_count = min(class_counts.values()) if class_counts else 0
        max_count = max(class_counts.values()) if class_counts else 0

        if min_count == 0:
            print("❌ Some classes have no images! Cannot proceed with training.")
            return None, None, None

        if max_count / min_count > 3:
            print(f"⚠️ Warning: Large class imbalance detected (ratio: {max_count / min_count:.1f})")

        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)

        return X, y, image_paths

    def train_model_robust(self, X, y, test_size=0.25, validation_size=0.25, epochs=30, batch_size=8):
        """Train with robust settings for small datasets"""
        print("Preparing data for robust training...")

        # Check minimum requirements
        min_samples_needed = 50  # Minimum for meaningful training
        if len(X) < min_samples_needed:
            print(f"❌ Not enough samples for training. Need at least {min_samples_needed}, got {len(X)}")
            return None

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Second split: separate train and validation
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )

        print(f"Training set: {len(X_train)} images")
        print(f"Validation set: {len(X_val)} images")
        print(f"Test set: {len(X_test)} images")

        # Check if we have enough samples per class
        for class_idx in range(5):
            train_count = np.sum(y_train == class_idx)
            val_count = np.sum(y_val == class_idx)
            if train_count < 2 or val_count < 1:
                print(f"❌ Not enough samples for class {class_idx}. Need at least 2 for training, 1 for validation.")
                return None

        # Create simple data augmentation
        datagen = ImageDataGenerator(
            rotation_range=5,
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        datagen.fit(X_train)

        # Create and compile simple model
        print("Creating robust model...")
        self.model = self.create_simple_model()
        self.model = self.compile_model(self.model)

        print("\nModel Architecture:")
        self.model.summary()

        # Conservative callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=5,
                min_lr=1e-5,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_depth_model_robust.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]

        print(f"\nStarting robust training for {epochs} epochs...")

        # Train with augmented data
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=max(1, len(X_train) // batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        # Store test data
        self.X_test = X_test
        self.y_test = y_test

        # Immediate evaluation
        print("\n" + "=" * 50)
        print("TRAINING COMPLETED - IMMEDIATE EVALUATION")
        print("=" * 50)

        # Evaluate on validation set
        val_predictions = self.model.predict(X_val)
        val_predicted_classes = np.argmax(val_predictions, axis=1)
        val_accuracy = np.mean(val_predicted_classes == y_val)
        print(f"Final Validation Accuracy: {val_accuracy:.4f}")

        # Evaluate on test set
        test_predictions = self.model.predict(X_test)
        test_predicted_classes = np.argmax(test_predictions, axis=1)
        test_accuracy = np.mean(test_predicted_classes == y_test)
        print(f"Final Test Accuracy: {test_accuracy:.4f}")

        return history

    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            print("❌ No trained model to save!")
            return False

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save the model
            self.model.save(filepath)
            print(f"✅ Model saved successfully to: {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False

    def load_model(self, filepath):
        """Load a trained model"""
        try:
            self.model = tf.keras.models.load_model(filepath)
            print(f"✅ Model loaded successfully from: {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def evaluate_model(self, X_test=None, y_test=None):
        """Evaluate the model comprehensively"""
        if self.model is None:
            print("❌ No trained model found!")
            return None, None

        # Use stored test data if not provided
        if X_test is None:
            X_test = getattr(self, 'X_test', None)
            y_test = getattr(self, 'y_test', None)

        if X_test is None or y_test is None:
            print("❌ No test data available!")
            return None, None

        print(f"Evaluating on {len(X_test)} test samples...")

        # Make predictions
        predictions = self.model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)

        # Calculate accuracy
        accuracy = np.mean(predicted_classes == y_test)
        print(f"Test Accuracy: {accuracy:.4f}")

        # Detailed per-class analysis
        print("\nPer-class Analysis:")
        for class_idx in range(5):
            class_mask = y_test == class_idx
            if np.sum(class_mask) > 0:
                class_accuracy = np.mean(predicted_classes[class_mask] == y_test[class_mask])
                print(f"  {self.depth_labels[class_idx]}: {class_accuracy:.4f} ({np.sum(class_mask)} samples)")

        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(
            y_test, predicted_classes,
            target_names=self.depth_labels,
            zero_division=0
        ))

        # Confusion matrix
        cm = confusion_matrix(y_test, predicted_classes)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.depth_labels,
                    yticklabels=self.depth_labels)
        plt.title('Confusion Matrix - Robust Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('robust_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        return accuracy, cm


def main():
    """Complete training and evaluation pipeline"""
    print("=" * 60)
    print("ROBUST DEPTH CLASSIFIER TRAINING")
    print("=" * 60)

    # Configuration
    base_path = input("Enter the path to your dataset (or press Enter for default): ").strip()
    if not base_path:
        base_path = ".../pics/depth"

    print(f"Using dataset path: {base_path}")

    # Initialize classifier
    classifier = DepthClassifier(base_path)

    # Load dataset
    print("\nStep 1: Loading dataset...")
    X, y, image_paths = classifier.load_dataset(
        max_images_per_class=20,  # Conservative number
        surface_filter='curved'
    )

    if X is None:
        print("❌ Failed to load dataset. Please check your paths and try again.")
        return

    print(f"✅ Successfully loaded {len(X)} images")

    # Train model
    print("\nStep 2: Training model...")
    history = classifier.train_model_robust(
        X, y,
        epochs=25,  # Conservative number
        batch_size=4  # Small batch size
    )

    if history is None:
        print("❌ Training failed!")
        return

    # Save model
    print("\nStep 3: Saving model...")
    model_path = "depthModels/depth_classification_model_curved.h5"
    classifier.save_model(model_path)

    # Final evaluation
    print("\nStep 4: Final evaluation...")
    accuracy, cm = classifier.evaluate_model()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Final Accuracy: {accuracy:.4f}")
    print(f"Model saved to: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()