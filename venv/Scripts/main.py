# main.py

import os
import argparse
from train.train_shape_classifier import run as run_shape_classifier
from train.train_origin_regressor import run as run_origin_regressor
from train.test_shape_classifier import test_shape_classifier
from train.grating_classifier import GratingClassifier


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and test shape classification models')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'train_shape', 'train_origin', 'test_shape', 'train_grating', 'test_grating'],
                        help='Mode to run: all, train_shape, train_origin, test_shape, train_grating, or test_grating')
    parser.add_argument('--model_path', type=str, default='models/shape_model.pt',
                        help='Path to the trained shape model for testing')
    parser.add_argument('--test_dir', type=str,
                        default="C:/Users/chenc/OneDrive - Imperial College London/Documents/student stuff/fyp_Y4/pics/shape",
                        help='Directory containing test data')
    parser.add_argument('--grating_dir', type=str,
                        default="C:/Users/chenc/OneDrive - Imperial College London/Documents/student stuff/fyp_Y4/pics/gratingBoard",
                        help='Directory containing grating board images')
    parser.add_argument('--max_images_per_folder', type=int, default=100,
                        help='Maximum number of images to load per grating folder')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing (0.0-1.0)')
    parser.add_argument('--grating_test_image', type=str, default=None,
                        help='Path to a single grating image for testing/prediction')

    args = parser.parse_args()

    # Check if models directory exists, create if not
    os.makedirs("models", exist_ok=True)

    if args.mode in ['all', 'train_shape']:
        print("\n" + "=" * 50)
        print("Running shape classifier training...")
        print("=" * 50)
        shape_model = run_shape_classifier()
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
        if os.path.exists(args.model_path):
            accuracy, _, _, _ = test_shape_classifier(
                model_path=args.model_path,
                test_data_dir=args.test_dir
            )
            print(f"Shape classifier testing completed with accuracy: {accuracy:.2f}%")
        else:
            print(f"Error: Model file not found at {args.model_path}")
            print("Please train the model first or specify the correct path.")

    if args.mode in ['all', 'train_grating']:
        print("\n" + "=" * 50)
        print("Running grating classifier training...")
        print("=" * 50)

        if not os.path.exists(args.grating_dir):
            print(f"Error: Grating directory not found at {args.grating_dir}")
            print("Please specify the correct path using --grating_dir")
        else:
            # Initialize and train grating classifier
            grating_classifier = GratingClassifier(args.grating_dir)

            # Run the full pipeline
            results = grating_classifier.run_full_pipeline(
                max_images_per_folder=args.max_images_per_folder,
                test_size=args.test_size
            )

            if results:
                print("Grating classifier training completed!")

                # Save the trained classifier (optional - you might want to implement this)
                # This would require adding save/load methods to GratingClassifier
                try:
                    import pickle
                    with open('models/grating_classifier.pkl', 'wb') as f:
                        pickle.dump(grating_classifier, f)
                    print("Grating classifier saved to models/grating_classifier.pkl")
                except Exception as e:
                    print(f"Warning: Could not save grating classifier: {e}")
            else:
                print("Grating classifier training failed!")

    if args.mode == 'test_grating':
        print("\n" + "=" * 50)
        print("Testing grating classifier...")
        print("=" * 50)

        # Load trained grating classifier
        classifier_path = 'models/grating_classifier.pkl'
        if os.path.exists(classifier_path):
            try:
                import pickle
                with open(classifier_path, 'rb') as f:
                    grating_classifier = pickle.load(f)
                print("Loaded trained grating classifier")

                if args.grating_test_image:
                    # Test single image
                    if os.path.exists(args.grating_test_image):
                        print(f"Predicting grating resolution for: {args.grating_test_image}")
                        try:
                            predicted_res, probabilities = grating_classifier.predict_single_image(
                                args.grating_test_image
                            )
                            print(f"Predicted resolution: {predicted_res}")
                            print("Prediction probabilities:")
                            for res, prob in sorted(probabilities.items()):
                                print(f"  {res}: {prob:.4f} ({prob * 100:.2f}%)")
                        except Exception as e:
                            print(f"Error predicting image: {e}")
                    else:
                        print(f"Error: Test image not found at {args.grating_test_image}")
                else:
                    print("No test image specified. Use --grating_test_image to test a specific image.")
                    print("Example: python main.py --mode test_grating --grating_test_image path/to/image.jpg")

            except Exception as e:
                print(f"Error loading grating classifier: {e}")
                print("Please train the grating classifier first using --mode train_grating")
        else:
            print(f"Error: Trained grating classifier not found at {classifier_path}")
            print("Please train the grating classifier first using --mode train_grating")


if __name__ == "__main__":
    main()


# Train all models including grating classifier
# python main.py --mode all

# Train only grating classifier with custom settings
# python main.py --mode train_grating --max_images_per_folder 50 --test_size 0.3

# Test grating classifier on a single image
# python main.py --mode test_grating --grating_test_image "path/to/test/image.jpg"

# Train grating classifier with custom directory
# python main.py --mode train_grating --grating_dir "path/to/grating/images"

