# main.py

import os
import argparse
import pickle
from train.train_shape_classifier import run as run_shape_classifier
from train.train_origin_regressor import run as run_origin_regressor
from train.test_shape_classifier import test_shape_classifier
from train.grating_classifier import GratingClassifier


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and test shape classification and grating models')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'train_shape', 'train_origin', 'test_shape', 'train_grating', 'test_grating'],
                        help='Mode to run: all, train_shape, train_origin, test_shape, train_grating, or test_grating')
    parser.add_argument('--shape_dir', type=str,
                        default="C:/Users/chenc/OneDrive - Imperial College London/Documents/student stuff/fyp_Y4/pics/shape",
                        help='Directory containing shape data (should contain shape_0 and shape_3mm folders)')
    parser.add_argument('--grating_dir', type=str,
                        default="C:/Users/chenc/OneDrive - Imperial College London/Documents/student stuff/fyp_Y4/pics/gratingBoard",
                        help='Directory containing grating board images (should contain grating_0 and grating_3mm folders)')
    parser.add_argument('--max_images_per_folder', type=int, default=100,
                        help='Maximum number of images to load per grating folder')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing (0.0-1.0)')
    parser.add_argument('--grating_test_image', type=str, default=None,
                        help='Path to a single grating image for testing/prediction')
    parser.add_argument('--grating_folder', type=str, default='both',
                        choices=['grating_0', 'grating_3mm', 'both'],
                        help='Which grating dataset to use: grating_0, grating_3mm, or both')
    parser.add_argument('--shape_folder', type=str, default='both',
                        choices=['shape_0', 'shape_3mm', 'both'],
                        help='Which shape dataset to use: shape_0, shape_3mm, or both')

    args = parser.parse_args()

    # Check if models directory exists, create if not
    os.makedirs("models", exist_ok=True)

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
                    # Initialize and train grating classifier
                    grating_classifier = GratingClassifier(args.grating_dir, data_folder=folder)

                    # Run the full pipeline
                    results, accuracy = grating_classifier.run_full_pipeline(
                        max_images_per_folder=args.max_images_per_folder,
                        test_size=args.test_size
                    )

                    if results:
                        print(f"Grating classifier training for {folder} completed with accuracy: {accuracy:.2f}%!")

                        # Save the trained classifier
                        try:
                            model_filename = f'grating_classifier_{folder}.pkl'
                            model_path = f'models/{model_filename}'

                            with open(model_path, 'wb') as f:
                                pickle.dump(grating_classifier, f)

                            print(f"Grating classifier model saved as: {model_path}")

                        except Exception as e:
                            print(f"Error saving grating classifier model for {folder}: {e}")
                    else:
                        print(f"Failed to train grating classifier for {folder}")

                except Exception as e:
                    print(f"Error training grating classifier for {folder}: {e}")
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

            model_path = f"models/grating_classifier_{folder}.pkl"

            if os.path.exists(model_path):
                try:
                    # Load the trained grating classifier
                    with open(model_path, 'rb') as f:
                        grating_classifier = pickle.load(f)

                    # Test with a single image if provided
                    if args.grating_test_image:
                        if os.path.exists(args.grating_test_image):
                            print(f"Testing single image: {args.grating_test_image}")
                            # Assuming the GratingClassifier has a predict method
                            if hasattr(grating_classifier, 'predict_single_image'):
                                prediction, probabilities = grating_classifier.predict_single_image(args.grating_test_image)
                                print(f"Prediction result: {prediction}")
                                print(f"Probabilities: {probabilities}")
                            else:
                                print("GratingClassifier does not have a predict_single_image method")
                        else:
                            print(f"Test image not found: {args.grating_test_image}")
                    else:
                        # Test on the dataset (if the classifier has a test method)
                        if hasattr(grating_classifier, 'test') or hasattr(grating_classifier, 'evaluate'):
                            print("Running evaluation on test dataset...")
                            # You may need to implement this based on your GratingClassifier structure
                            print("Test evaluation method not implemented in GratingClassifier")
                        else:
                            print("No test method available in GratingClassifier")
                            print("Use --grating_test_image to test a single image")

                    print(f"Completed testing for {folder}")

                except Exception as e:
                    print(f"Error testing grating classifier for {folder}: {e}")
            else:
                print(f"Grating classifier model not found: {model_path}")
                print("Please train the grating model first using --mode train_grating")
            print("\n")

        print("Grating classifier testing completed!")

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