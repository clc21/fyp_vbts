# main.py

import os
import argparse
from train.train_shape_classifier import run as run_shape_classifier
from train.train_origin_regressor import run as run_origin_regressor
from train.test_shape_classifier import test_shape_classifier


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and test shape classification models')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'train_shape', 'train_origin', 'test_shape'],
                        help='Mode to run: all, train_shape, train_origin, or test_shape')
    parser.add_argument('--model_path', type=str, default='models/shape_model.pt',
                        help='Path to the trained shape model for testing')
    parser.add_argument('--test_dir', type=str,
                        default="C:/Users/chenc/OneDrive - Imperial College London/Documents/student stuff/fyp_Y4/pics/shape",
                        help='Directory containing test data')

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


if __name__ == "__main__":
    main()