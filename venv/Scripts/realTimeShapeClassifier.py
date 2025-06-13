import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights
import time
import sys
import os

# Add model_defs folder to Python path
sys.path.append('model_defs')

try:
    # Try to import from model_defs folder
    from model_defs import ShapeClassifier
except ImportError:
    try:
        # Try alternative import if the file is named differently
        from shape_classifier import ShapeClassifier
    except ImportError:
        # Fallback: define the class here
        print("Could not import ShapeClassifier, using fallback definition")


        class ShapeClassifier(nn.Module):
            def __init__(self, num_classes=4):
                super(ShapeClassifier, self).__init__()
                self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)
                self.model.features.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

            def forward(self, x):
                return self.model(x)


class RealTimeShapeDetector:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the real-time shape detector

        Args:
            model_path: Path to the trained model (.pt file)
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.class_names = ['circle', 'ring', 'triangle', 'star']

        # Load the trained model
        self.model = ShapeClassifier(num_classes=len(self.class_names))
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        # Define transforms (same as training)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            raise ValueError("Could not open camera")

        print(f"Shape detector initialized on {device}")
        print("Classes:", self.class_names)

    def preprocess_frame(self, frame):
        """
        Preprocess frame for model inference

        Args:
            frame: OpenCV frame (BGR format)

        Returns:
            preprocessed tensor ready for model
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply transforms
        tensor = self.transform(frame_rgb)
        tensor = tensor.unsqueeze(0)  # Add batch dimension

        return tensor.to(self.device)

    def predict(self, frame):
        """
        Predict shape from frame

        Args:
            frame: OpenCV frame

        Returns:
            tuple: (predicted_class, confidence, all_probabilities)
        """
        with torch.no_grad():
            # Preprocess frame
            input_tensor = self.preprocess_frame(frame)

            # Get model prediction
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

            # Get predicted class and confidence
            confidence, predicted_idx = torch.max(probabilities, 0)
            predicted_class = self.class_names[predicted_idx.item()]

            return predicted_class, confidence.item(), probabilities.cpu().numpy()

    def draw_prediction(self, frame, predicted_class, confidence, probabilities):
        """
        Draw prediction results on frame

        Args:
            frame: OpenCV frame to draw on
            predicted_class: Predicted class name
            confidence: Confidence score
            probabilities: All class probabilities

        Returns:
            frame with annotations
        """
        height, width = frame.shape[:2]

        # Draw main prediction
        text = f"{predicted_class}: {confidence:.2%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2

        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Draw background rectangle
        cv2.rectangle(frame, (10, 10), (10 + text_width + 10, 10 + text_height + baseline + 10),
                      (0, 0, 0), -1)

        # Draw text
        color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.5 else (0, 0, 255)
        cv2.putText(frame, text, (15, 10 + text_height), font, font_scale, color, thickness)

        # Draw all probabilities
        y_offset = 60
        for i, (class_name, prob) in enumerate(zip(self.class_names, probabilities)):
            prob_text = f"{class_name}: {prob:.2%}"
            color = (0, 255, 0) if i == np.argmax(probabilities) else (255, 255, 255)
            cv2.putText(frame, prob_text, (15, y_offset + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        # Draw FPS counter
        current_time = time.time()
        if hasattr(self, 'last_time'):
            fps = 1.0 / (current_time - self.last_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (width - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self.last_time = current_time

        return frame

    def run(self, show_confidence_threshold=0.3):
        """
        Run real-time shape detection

        Args:
            show_confidence_threshold: Only show predictions above this confidence
        """
        print("Starting real-time shape detection...")
        print("Press 'q' to quit, 's' to save screenshot")

        frame_count = 0

        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                # Make prediction
                predicted_class, confidence, probabilities = self.predict(frame)

                # Draw prediction if confidence is above threshold
                if confidence >= show_confidence_threshold:
                    frame = self.draw_prediction(frame, predicted_class, confidence, probabilities)
                else:
                    # Show low confidence warning
                    cv2.putText(frame, "Low Confidence Detection", (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Display frame
                cv2.imshow('Real-time Shape Detection', frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    filename = f"shape_detection_screenshot_{frame_count}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved as {filename}")

                frame_count += 1

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("Resources cleaned up")


def main():
    """
    Main function to run real-time shape detection
    """
    import os

    # Configuration
    model_paths = {
        'flat': 'models/shape_model_shape_0.pt',
        'curved': 'models/shape_model_shape_3mm.pt'
    }

    # Debug: Check current directory and models folder
    print(f"Current working directory: {os.getcwd()}")
    print(f"Contents of current directory: {os.listdir('.')}")

    if os.path.exists('models'):
        print(f"Contents of models directory: {os.listdir('models')}")
    else:
        print("Models directory does not exist!")

    # Choose which model to use
    print("\nAvailable models:")
    for key, path in model_paths.items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {key}: {path} {exists}")

    model_choice = input("\nChoose model (flat/curved) or enter custom path: ").strip().lower()

    if model_choice in model_paths:
        model_path = model_paths[model_choice]
    else:
        model_path = model_choice if model_choice else model_paths['flat']

    # Additional debugging
    print(f"Trying to load model from: {model_path}")
    print(f"Model file exists: {os.path.exists(model_path)}")

    try:
        # Initialize detector
        detector = RealTimeShapeDetector(model_path)

        # Run detection
        detector.run()

    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        print("Please ensure you have trained the model first using train_shape_classifier.py")
        print("\nTroubleshooting tips:")
        print("1. Check if the file exists with the exact name")
        print("2. Verify you're running from the correct directory")
        print("3. Try using the full absolute path to the model file")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()