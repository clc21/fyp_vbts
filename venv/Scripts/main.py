# main.py

import os
import argparse
import pickle
from pathlib import Path
from train.train_shape_classifier import run as run_shape_classifier
from train.train_origin_regressor import run as run_origin_regressor
from train.test_shape_classifier import test_shape_classifier
from train.train_grating import GratingTrainer
from train.test_grating import GratingTester
from train.GratingExtractFeature import GratingFeatureExtractor
from train.train_depth import DepthClassifier
from train.test_depth import DepthTester
from train.depthAnalysis import DepthAnalysisUtils


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and test shape classification, grating, and depth models')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'train_shape', 'train_origin', 'test_shape', 'train_grating', 'test_grating',
                                 'train_depth', 'test_depth', 'check_depth', 'setup_depth'],
                        help='Mode to run: all, train_shape, train_origin, test_shape, train_grating, test_grating, train_depth, test_depth, check_depth, or setup_depth')
    parser.add_argument('--shape_dir', type=str,
                        default=".../pics/shape",
                        help='Directory containing shape data (should contain shape_0 and shape_3mm folders)')
    parser.add_argument('--grating_dir', type=str,
                        default=".../pics/gratingBoard",
                        help='Directory containing grating board images (should contain grating_0 and grating_3mm folders)')
    parser.add_argument('--depth_dir', type=str,
                        default=".../pics/depth",
                        help='Directory containing depth data (should contain flat and curved folders with depth subfolders)')
    parser.add_argument('--max_images_per_folder', type=int, default=100,
                        help='Maximum number of images to load per grating folder')
    parser.add_argument('--max_depth_images_per_class', type=int, default=50,
                        help='Maximum number of images to load per depth class')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing (0.0-1.0)')
    parser.add_argument('--grating_test_image', type=str, default=None,
                        help='Path to a single grating image for testing/prediction')
    parser.add_argument('--depth_test_image', type=str, default=None,
                        help='Path to a single depth image for testing/prediction')
    parser.add_argument('--grating_folder', type=str, default='both',
                        choices=['grating_0', 'grating_3mm', 'both'],
                        help='Which grating dataset to use: grating_0, grating_3mm, or both')
    parser.add_argument('--shape_folder', type=str, default='both',
                        choices=['shape_0', 'shape_3mm', 'both'],
                        help='Which shape dataset to use: shape_0, shape_3mm, or both')
    parser.add_argument('--depth_surface', type=str, default='both',
                        choices=['flat', 'curved', 'both'],
                        help='Which depth surface to use: flat, curved, or both')
    parser.add_argument('--depth_epochs', type=int, default=25,
                        help='Number of epochs for depth model training')
    parser.add_argument('--depth_batch_size', type=int, default=32,
                        help='Batch size for depth model training')
    parser.add_argument('--reference_image', type=str, default=None,
                        help='Path to reference image for depth analysis')

    args = parser.parse_args()

    # Check if models directory exists, create if not
    os.makedirs("models", exist_ok=True)

    if args.mode in ['all', 'setup_depth']:
        print("\n" + "=" * 50)
        print("Setting up depth analysis environment...")
        print("=" * 50)

        utils = DepthAnalysisUtils()
        utils.generate_requirements_file()
        utils.create_run_script()
        print("Depth environment setup completed!")

    if args.mode in ['all', 'check_depth']:
        print("\n" + "=" * 50)
        print("Checking depth directory structure...")
        print("=" * 50)

        utils = DepthAnalysisUtils()
        structure_valid = utils.check_directory_structure(args.depth_dir)

        if structure_valid:
            print("✅ Directory structure is valid!")
            # Analyze image properties
            utils.analyze_image_properties(args.depth_dir)
            # Create sample visualization
            utils.visualize_sample_images(args.depth_dir)
        else:
            print("❌ Directory structure has issues. Please fix before training.")
            return

    if args.mode in ['all', 'train_shape']:
        print("\n" + "=" * 50)
        print("Running shape classifier training...")
        print("=" * 50)

        # Determine which shape folders to train on
        shape_folders = []
        if args.shape_folder == 'both':
            shape_folders = ['shape_0', 'shape_3mm']
        else:
            shape_folders = [args.shape_folder]

        # Train on specified datasets
        for folder in shape_folders:
            print(f"\n{'=' * 60}")
            print(f"Training Shape Classifier for {folder}")
            print(f"{'=' * 60}")
            try:
                model, accuracy = run_shape_classifier(data_folder=folder)
                print(f"Completed training for {folder} with accuracy: {accuracy:.2f}%")
            except Exception as e:
                print(f"Error training on {folder}: {e}")
            print("\n")

        print("Shape classifier training completed!")

    if args.mode in ['all', 'train_origin']:
        print("\n" + "=" * 50)
        print("Running origin regressor training...")
        print("=" * 50)
        run_origin_regressor()
        print("Origin regressor training completed!")

    if args.mode in ['all', 'test_shape']:
        print("\n" + "=" * 50)
        print("Testing shape classifier...")
        print("=" * 50)

        # Determine which shape folders to test on
        shape_folders = []
        if args.shape_folder == 'both':
            shape_folders = ['shape_0', 'shape_3mm']
        else:
            shape_folders = [args.shape_folder]

        # Test on specified datasets
        for folder in shape_folders:
            print(f"\n{'=' * 60}")
            print(f"Testing Shape Classifier for {folder}")
            print(f"{'=' * 60}")

            model_path = f"models/shape_model_{folder}.pt"

            if os.path.exists(model_path):
                try:
                    accuracy, _, _, _ = test_shape_classifier(
                        model_path=model_path,
                        test_data_dir=args.shape_dir,
                        data_folder=folder
                    )
                    print(f"Completed testing for {folder} with accuracy: {accuracy:.2f}%")
                except Exception as e:
                    print(f"Error testing {folder}: {e}")
            else:
                print(f"Model not found: {model_path}")
                print("Please train the model first.")
            print("\n")

    if args.mode in ['all', 'train_grating']:
        print("\n" + "=" * 50)
        print("Running grating classifier training...")
        print("=" * 50)

        if not os.path.exists(args.grating_dir):
            print(f"Error: Grating directory not found at {args.grating_dir}")
            print("Please specify the correct path using --grating_dir")
        else:
            # Determine which grating folders to train on
            grating_folders = []
            if args.grating_folder == 'both':
                grating_folders = ['grating_0', 'grating_3mm']
            else:
                grating_folders = [args.grating_folder]

            # Train on specified datasets
            for folder in grating_folders:
                print(f"\n{'=' * 60}")
                print(f"Training Grating Classifier for {folder}")
                print(f"{'=' * 60}")

                try:
                    # Step 1: Check if features already exist, if not extract them
                    feature_file = f"grating_{folder}_features.npy"
                    label_file = f"grating_{folder}_labels.npy"

                    if not (os.path.exists(feature_file) and os.path.exists(label_file)):
                        print(f"Feature files not found for {folder}. Extracting features...")

                        # Initialize feature extractor
                        extractor = GratingFeatureExtractor(args.grating_dir, data_folder=folder)

                        # Extract features
                        features, labels = extractor.run_feature_extraction(
                            max_images_per_folder=args.max_images_per_folder
                        )

                        if features is None:
                            print(f"Failed to extract features for {folder}")
                            continue

                        print(f"Feature extraction completed for {folder}")
                    else:
                        print(f"Feature files already exist for {folder}. Skipping extraction.")

                    # Step 2: Train the grating classifier
                    print(f"Starting training for {folder}...")
                    trainer = GratingTrainer(data_folder=folder)
                    results = trainer.run_full_training_pipeline()

                    if results:
                        print(f"Grating classifier training for {folder} completed!")
                        print(
                            f"Final test accuracy: {results['test_results']['accuracy']:.4f} ({results['test_results']['accuracy'] * 100:.2f}%)")
                    else:
                        print(f"Failed to train grating classifier for {folder}")

                except Exception as e:
                    print(f"Error in grating pipeline for {folder}: {e}")
                    import traceback
                    traceback.print_exc()
                print("\n")

            print("Grating classifier training completed!")

    if args.mode in ['all', 'test_grating']:
        print("\n" + "=" * 50)
        print("Testing grating classifier...")
        print("=" * 50)

        # Determine which grating folders to test on
        grating_folders = []
        if args.grating_folder == 'both':
            grating_folders = ['grating_0', 'grating_3mm']
        else:
            grating_folders = [args.grating_folder]

        # Test on specified datasets
        for folder in grating_folders:
            print(f"\n{'=' * 60}")
            print(f"Testing Grating Classifier for {folder}")
            print(f"{'=' * 60}")

            # Look for the model files that are created by the trainer
            model_patterns = [
                f"grating_{folder}_knn_k*_*_model.pkl",
                f"grating_{folder}_*_model.pkl"
            ]

            model_path = None
            for pattern in model_patterns:
                import glob
                matches = glob.glob(pattern)
                if matches:
                    model_path = matches[0]  # Use the first match
                    break

            if model_path and os.path.exists(model_path):
                try:
                    print(f"Found model: {model_path}")

                    # Initialize tester
                    tester = GratingTester(model_path, data_folder=folder)

                    # Test directory path
                    test_dir = Path(args.grating_dir) / folder

                    # Run comprehensive testing
                    if args.grating_test_image:
                        if os.path.exists(args.grating_test_image):
                            print(f"Testing single image: {args.grating_test_image}")
                            prediction, probabilities = tester.predict_single_image(args.grating_test_image)
                            print(f"Prediction: {prediction}")
                            print(f"Probabilities: {probabilities}")
                        else:
                            print(f"Test image not found: {args.grating_test_image}")
                    else:
                        # Run full test suite
                        results = tester.run_comprehensive_test(test_dir=str(test_dir))
                        print(f"Comprehensive testing completed for {folder}")

                except Exception as e:
                    print(f"Error testing grating classifier for {folder}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Grating classifier model not found for {folder}")
                print("Expected model files matching pattern: grating_{folder}_*_model.pkl")
                print("Please train the grating model first using --mode train_grating")
            print("\n")

        print("Grating classifier testing completed!")

    if args.mode in ['all', 'train_depth']:
        print("\n" + "=" * 50)
        print("Running depth classifier training...")
        print("=" * 50)

        if not os.path.exists(args.depth_dir):
            print(f"Error: Depth directory not found at {args.depth_dir}")
            print("Please specify the correct path using --depth_dir")
        else:
            # Check directory structure first
            utils = DepthAnalysisUtils()
            structure_valid = utils.check_directory_structure(args.depth_dir)

            if not structure_valid:
                print("❌ Directory structure is invalid. Please fix before training.")
                return

            # Determine which surfaces to train on
            surfaces_to_train = []
            if args.depth_surface == 'both':
                surfaces_to_train = ['flat', 'curved']
            else:
                surfaces_to_train = [args.depth_surface]

            # Train models for each surface type
            for surface_type in surfaces_to_train:
                print(f"\n{'=' * 60}")
                print(f"Training Depth Classifier for {surface_type.upper()} surface")
                print(f"{'=' * 60}")

                try:
                    print(f"Training depth classifier on: {args.depth_dir}")
                    print(f"Surface type: {surface_type}")
                    print(f"Max images per class: {args.max_depth_images_per_class}")
                    print(f"Epochs: {args.depth_epochs}")
                    print(f"Batch size: {args.depth_batch_size}")

                    # Initialize depth classifier
                    classifier = DepthClassifier(base_path=args.depth_dir)

                    # Load dataset for specific surface
                    X, y, image_paths = classifier.load_dataset(
                        max_images_per_class=args.max_depth_images_per_class,
                        surface_filter=surface_type
                    )

                    if X is None or len(X) == 0:
                        print(f"❌ No images found for {surface_type} surface! Please check your directory structure.")
                        continue

                    # Print dataset statistics
                    print(f"\nDataset Statistics for {surface_type} surface:")
                    print(f"Total images: {len(X)}")
                    print(f"Image shape: {X[0].shape}")

                    import numpy as np
                    unique, counts = np.unique(y, return_counts=True)
                    for depth_idx, count in zip(unique, counts):
                        print(f"  Depth {classifier.depth_labels[depth_idx]}: {count} images")

                    # Use the correct method name from your DepthClassifier
                    print(f"\nStarting training for {surface_type} surface...")
                    history = classifier.train_model_robust(
                        X, y,
                        test_size=args.test_size,
                        validation_size=0.2,
                        epochs=args.depth_epochs,
                        batch_size=args.depth_batch_size
                    )

                    if history is None:
                        print(f"❌ Training failed for {surface_type} surface")
                        continue

                    # Evaluate on test set using your class's evaluate_model method
                    if hasattr(classifier, 'X_test') and hasattr(classifier, 'y_test'):
                        print(f"\nEvaluating {surface_type} surface model on test set...")
                        accuracy, cm = classifier.evaluate_model()  # Your method doesn't need parameters
                        if accuracy is not None:
                            print(f"Final Test Accuracy for {surface_type}: {accuracy:.4f}")

                    # Save model using your class's save_model method
                    model_filename = f'models/depth_classification_model_{surface_type}.h5'
                    success = classifier.save_model(model_filename)
                    if success:
                        print(f"Model saved as '{model_filename}'")
                    else:
                        print(f"Failed to save model as '{model_filename}'")

                    print(f"✅ {surface_type.capitalize()} surface training completed!")

                except Exception as e:
                    print(f"❌ Error in depth training for {surface_type}: {e}")
                    import traceback
                    traceback.print_exc()
                print("\n")

        print("Depth classifier training completed!")

    if args.mode in ['all', 'test_depth']:
        print("\n" + "=" * 50)
        print("Testing depth classifier...")
        print("=" * 50)

        if not os.path.exists(args.depth_dir):
            print(f"❌ Depth directory not found at {args.depth_dir}")
            return

        # Determine which surfaces to test
        surfaces_to_test = []
        if args.depth_surface == 'both':
            surfaces_to_test = ['flat', 'curved']
        else:
            surfaces_to_test = [args.depth_surface]

        # Test models for each surface type
        for surface_type in surfaces_to_test:
            print(f"\n{'=' * 60}")
            print(f"Testing Depth Classifier for {surface_type.upper()} surface")
            print(f"{'=' * 60}")

            # Determine model path
            model_path = f'models/depth_classification_model_{surface_type}.h5'

            if not os.path.exists(model_path):
                print(f"❌ Depth classifier model not found: {model_path}")
                print(f"Please train the depth model for {surface_type} surface first using --mode train_depth")
                continue

            try:
                print(f"Testing depth classifier: {model_path}")
                print(f"Data directory: {args.depth_dir}")
                print(f"Surface type: {surface_type}")

                # For testing, we'll create a new classifier instance and load the model
                classifier = DepthClassifier(base_path=args.depth_dir)

                # Load the trained model
                success = classifier.load_model(model_path)
                if not success:
                    print("❌ Cannot proceed without a trained model.")
                    continue

                # Test single image if provided
                if args.depth_test_image:
                    if os.path.exists(args.depth_test_image):
                        print(f"\nTesting single image on {surface_type} model: {args.depth_test_image}")

                        # Load and preprocess the image
                        img = classifier.load_and_preprocess_image(args.depth_test_image)
                        if img is not None:
                            # Add batch dimension and predict
                            img_batch = np.expand_dims(img, axis=0)
                            predictions = classifier.model.predict(img_batch, verbose=0)
                            predicted_class = np.argmax(predictions[0])
                            confidence = predictions[0][predicted_class]

                            print(f"Predicted depth: {classifier.depth_labels[predicted_class]}")
                            print(f"Confidence: {confidence:.4f}")
                            print(f"All probabilities: {predictions[0]}")
                        else:
                            print("❌ Failed to load and preprocess the test image")
                    else:
                        print(f"❌ Test image not found: {args.depth_test_image}")
                    continue

                # Load test dataset for evaluation
                print(f"\nLoading test dataset for {surface_type} surface...")
                X_test, y_test, image_paths = classifier.load_dataset(
                    max_images_per_class=args.max_depth_images_per_class,
                    surface_filter=surface_type
                )

                if X_test is None or len(X_test) == 0:
                    print(f"❌ No test images found for {surface_type} surface! Please check your directory structure.")
                    continue

                # Print dataset statistics
                print(f"\nTest Dataset Statistics for {surface_type} surface:")
                print(f"Total test images: {len(X_test)}")
                print(f"Image shape: {X_test[0].shape}")

                import numpy as np
                unique, counts = np.unique(y_test, return_counts=True)
                for depth_idx, count in zip(unique, counts):
                    print(f"  Depth {classifier.depth_labels[depth_idx]}: {count} images")

                # Store test data in classifier for evaluation
                classifier.X_test = X_test
                classifier.y_test = y_test

                # Run evaluation using your class's method
                print(f"\nEvaluating model on {len(X_test)} test samples...")
                accuracy, cm = classifier.evaluate_model()

                if accuracy is not None:
                    print(f"\n" + "=" * 30)
                    print(f"{surface_type.upper()} SURFACE EVALUATION COMPLETE")
                    print("=" * 30)
                    print(f"Overall Accuracy: {accuracy:.4f}")
                else:
                    print("❌ Evaluation failed")

            except Exception as e:
                print(f"❌ Error in depth testing for {surface_type}: {e}")
                import traceback
                traceback.print_exc()
            print("\n")

        print("Depth classifier testing completed!")
    print("\n" + "=" * 50)
    print("All operations completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()

# Train all models including grating classifier
# python main.py --mode all

# Train only shape classifiers for both datasets
# python main.py --mode train_shape --shape_folder both

# Train only grating classifier for specific dataset
# python main.py --mode train_grating --grating_folder grating_0

# Test shape classifier on specific dataset
# python main.py --mode test_shape --shape_folder shape_3mm

# Test grating classifier with a single image
# python main.py --mode test_grating --grating_test_image "path/to/test/image.jpg"

# Use custom directories
# python main.py --shape_dir "path/to/shape/folder" --grating_dir "path/to/grating/folder"