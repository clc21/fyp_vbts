import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class GratingTrainer:
    def __init__(self, data_folder="grating_0"):
        """
        Initialize the grating classifier trainer

        Args:
            data_folder (str): Specific folder ("grating_0" or "grating_3mm")
        """
        self.data_folder = data_folder
        self.scaler = StandardScaler()
        self.models = {}
        self.cv_results = {}

        # Load features and labels
        self.features, self.labels = self.load_features()

        # Create label mappings
        unique_labels = sorted(set(self.labels))
        self.label_to_int = {res: i for i, res in enumerate(unique_labels)}
        self.int_to_label = {i: res for res, i in self.label_to_int.items()}

    def load_features(self):
        """Load pre-extracted features and labels"""
        try:
            features = np.load(f"grating_{self.data_folder}_features.npy")
            labels = np.load(f"grating_{self.data_folder}_labels.npy")
            print(f"Loaded features: {features.shape}")
            print(f"Loaded labels: {len(labels)}")
            return features, labels
        except FileNotFoundError:
            raise FileNotFoundError(f"Feature files not found. Please run feature extraction first.")

    def split_data(self, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
        """
        Split data into train, validation, and test sets

        Args:
            train_size (float): Proportion for training
            val_size (float): Proportion for validation
            test_size (float): Proportion for testing
            random_state (int): Random seed for reproducibility
        """
        if abs(train_size + val_size + test_size - 1.0) > 1e-6:
            raise ValueError("train_size + val_size + test_size must equal 1.0")

        print(
            f"\nSplitting data: {train_size * 100:.0f}% train, {val_size * 100:.0f}% val, {test_size * 100:.0f}% test")

        # Convert labels to integers
        y_int = [self.label_to_int[label] for label in self.labels]

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.features, y_int,
            test_size=test_size,
            random_state=random_state,
            stratify=y_int
        )

        # Second split: separate train and validation from remaining data
        val_ratio = val_size / (train_size + val_size)  # Adjust validation ratio
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=random_state,
            stratify=y_temp
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Print split info
        print(f"Training set: {len(X_train)} images")
        print(f"Validation set: {len(X_val)} images")
        print(f"Test set: {len(X_test)} images")

        # Show distribution in each set
        train_labels = [self.int_to_label[i] for i in y_train]
        val_labels = [self.int_to_label[i] for i in y_val]
        test_labels = [self.int_to_label[i] for i in y_test]

        print(f"\nTraining set distribution:")
        print(pd.Series(train_labels).value_counts().sort_index())
        print(f"\nValidation set distribution:")
        print(pd.Series(val_labels).value_counts().sort_index())
        print(f"\nTest set distribution:")
        print(pd.Series(test_labels).value_counts().sort_index())

        # Store the splits for later use
        self.X_train, self.X_val, self.X_test = X_train_scaled, X_val_scaled, X_test_scaled
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

    def perform_cross_validation(self, k_values=[3, 5, 7, 9], cv_folds=5, random_state=42):
        """
        Perform k-fold cross-validation to find best hyperparameters

        Args:
            k_values (list): List of k values to test for KNN
            cv_folds (int): Number of cross-validation folds
            random_state (int): Random seed for reproducibility
        """
        print(f"\nPerforming {cv_folds}-fold cross-validation...")
        print(f"Testing k values: {k_values}")

        # Use stratified k-fold to maintain class distribution
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        # Convert labels to integers for CV
        y_int = [self.label_to_int[label] for label in self.labels]

        # Scale all features for CV
        X_scaled = self.scaler.fit_transform(self.features)

        cv_results = {}

        for k in k_values:
            print(f"\nTesting KNN with k={k}...")

            # Test different distance weighting strategies
            for weights in ['uniform', 'distance']:
                model = KNeighborsClassifier(
                    n_neighbors=k,
                    weights=weights,
                    metric='euclidean'
                )

                # Perform cross-validation
                cv_scores = cross_val_score(
                    model, X_scaled, y_int,
                    cv=skf,
                    scoring='accuracy'
                )

                model_name = f'KNN_k{k}_{weights}'
                cv_results[model_name] = {
                    'scores': cv_scores,
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std(),
                    'k': k,
                    'weights': weights
                }

                print(f"  {model_name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        self.cv_results = cv_results

        # Find best model based on CV results
        best_model_name = max(cv_results.keys(), key=lambda x: cv_results[x]['mean'])
        best_result = cv_results[best_model_name]

        print(f"\nBest model from CV: {best_model_name}")
        print(f"CV Accuracy: {best_result['mean']:.4f} (+/- {best_result['std'] * 2:.4f})")

        return cv_results, best_model_name

    def train_models(self, k_values=[3, 5, 7, 9]):
        """Train KNN models with different hyperparameters"""
        print("\nTraining KNN models...")

        for k in k_values:
            for weights in ['uniform', 'distance']:
                model_name = f'KNN_k{k}_{weights}'

                self.models[model_name] = KNeighborsClassifier(
                    n_neighbors=k,
                    weights=weights,
                    metric='euclidean'
                )

                print(f"Training {model_name}...")
                self.models[model_name].fit(self.X_train, self.y_train)

        print("All models trained successfully!")

    def evaluate_on_validation(self):
        """Evaluate all models on validation set"""
        print(f"\n{'=' * 60}")
        print("VALIDATION SET EVALUATION")
        print(f"{'=' * 60}")

        val_results = {}

        for name, model in self.models.items():
            y_pred = model.predict(self.X_val)
            accuracy = accuracy_score(self.y_val, y_pred)

            val_results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred
            }

            print(f"{name}: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        # Find best model on validation set
        best_model_name = max(val_results.keys(), key=lambda x: val_results[x]['accuracy'])
        print(f"\nBest model on validation: {best_model_name}")
        print(f"Validation accuracy: {val_results[best_model_name]['accuracy']:.4f}")

        return val_results, best_model_name

    def final_evaluation(self, best_model_name):
        """Final evaluation on test set with best model"""
        print(f"\n{'=' * 60}")
        print("FINAL TEST SET EVALUATION")
        print(f"{'=' * 60}")

        best_model = self.models[best_model_name]
        y_pred = best_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)

        # Convert back to resolution labels for reporting
        y_test_labels = [self.int_to_label[i] for i in self.y_test]
        y_pred_labels = [self.int_to_label[i] for i in y_pred]

        print(f"Final Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        # Detailed classification report
        print(f"\nClassification Report:")
        y_test_labels_str = [str(label) for label in y_test_labels]
        y_pred_labels_str = [str(label) for label in y_pred_labels]
        print(classification_report(y_test_labels_str, y_pred_labels_str, digits=4))

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)

        return {
            'accuracy': accuracy,
            'predictions': y_pred_labels,
            'true_labels': y_test_labels,
            'confusion_matrix': cm
        }

    def save_model(self, model_name, filename=None):
        """Save trained model and scaler"""
        if filename is None:
            filename = f"grating_{self.data_folder}_{model_name.lower()}_model.pkl"

        model_data = {
            'model': self.models[model_name],
            'scaler': self.scaler,
            'label_mappings': {
                'label_to_int': self.label_to_int,
                'int_to_label': self.int_to_label
            },
            'data_folder': self.data_folder
        }

        joblib.dump(model_data, filename)
        print(f"Model saved as '{filename}'")

    def plot_cv_results(self):
        """Plot cross-validation results"""
        if not self.cv_results:
            print("No CV results to plot. Run cross-validation first.")
            return

        # Prepare data for plotting
        model_names = list(self.cv_results.keys())
        means = [self.cv_results[name]['mean'] for name in model_names]
        stds = [self.cv_results[name]['std'] for name in model_names]

        # Create plot
        plt.figure(figsize=(12, 6))
        x_pos = np.arange(len(model_names))

        bars = plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        plt.xlabel('Model Configuration')
        plt.ylabel('Cross-Validation Accuracy')
        plt.title(f'Cross-Validation Results - {self.data_folder}')
        plt.xticks(x_pos, model_names, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            plt.text(i, mean + std + 0.005, f'{mean:.3f}',
                     ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        filename = f'grating_{self.data_folder}_cv_results.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"CV results plot saved as '{filename}'")
        plt.close()

    def plot_confusion_matrix(self, test_results):
        """Plot confusion matrix for final results"""
        cm = test_results['confusion_matrix']
        labels_str = sorted(set(test_results['true_labels']))

        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # Create annotations
        annotations = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                    xticklabels=labels_str, yticklabels=labels_str)

        surface_type = "Flat" if self.data_folder == "grating_0" else "Curved (3mm)"
        plt.title(f'Grating Detection ({surface_type}) - Final Test Results\n'
                  f'Accuracy: {test_results["accuracy"] * 100:.2f}%')
        plt.ylabel('True Grating Resolution')
        plt.xlabel('Predicted Grating Resolution')
        plt.tight_layout()

        filename = f'grating_{self.data_folder}_final_confusion_matrix.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Final confusion matrix saved as '{filename}'")
        plt.close()

    def run_full_training_pipeline(self):
        """Run the complete training pipeline"""
        print("Starting Grating Classification Training Pipeline")
        print("=" * 60)

        # Split data
        self.split_data()

        print(f"\nNumber of features: {self.X_train.shape[1]}")

        # Perform cross-validation
        cv_results, best_cv_model = self.perform_cross_validation()

        # Plot CV results
        self.plot_cv_results()

        # Train all models
        self.train_models()

        # Evaluate on validation set
        val_results, best_val_model = self.evaluate_on_validation()

        # Final evaluation on test set
        test_results = self.final_evaluation(best_val_model)

        # Plot final confusion matrix
        self.plot_confusion_matrix(test_results)

        # Save best model
        self.save_model(best_val_model)

        # Summary
        print(f"\n{'=' * 60}")
        print("TRAINING PIPELINE SUMMARY")
        print(f"{'=' * 60}")
        print(f"Best model from CV: {best_cv_model}")
        print(f"Best model from validation: {best_val_model}")
        print(f"Final test accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy'] * 100:.2f}%)")

        return {
            'cv_results': cv_results,
            'val_results': val_results,
            'test_results': test_results,
            'best_model': best_val_model
        }


if __name__ == "__main__":
    for folder in ["grating_0", "grating_3mm"]:
        print(f"\n{'=' * 80}")
        print(f"Training Grating Classifier for {folder}")
        print(f"{'=' * 80}")

        try:
            trainer = GratingTrainer(data_folder=folder)
            results = trainer.run_full_training_pipeline()
            print(f"Completed training for {folder}")
        except Exception as e:
            print(f"Error training on {folder}: {e}")
            import traceback

            traceback.print_exc()
        print("\n")