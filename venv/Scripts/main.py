# main.py

from train.train_shape_classifier import run as run_shape_classifier
from train.train_origin_regressor import run as run_origin_regressor


def main():
    print("Running shape classifier training...")
    run_shape_classifier()

    print("\nRunning origin regressor training...")
    run_origin_regressor()


if __name__ == "__main__":
    main()
