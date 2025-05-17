import cv2
import torch
import numpy as np
from torchvision import transforms

class ShapeTracker:
    def __init__(self, img_path, shape_model, origin_model, class_names):
        self.img_path = img_path
        self.img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        self.shape_model = shape_model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.shape_model.eval()
        self.origin_model = origin_model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.origin_model.eval()
        self.class_names = class_names

        self.origin_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        self.shape_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])
        self.origin_x, self.origin_y = self.predict_origin()

    def predict_origin(self):
        input_tensor = self.origin_transform(self.img).unsqueeze(0).to(next(self.origin_model.parameters()).device)
        with torch.no_grad():
            pred = self.origin_model(input_tensor)[0]
        h, w = self.img.shape[:2]
        return int(pred[0].item() * w), int(pred[1].item() * h)

    def classify_shape(self, roi):
        input_tensor = self.shape_transform(roi).unsqueeze(0).to(next(self.shape_model.parameters()).device)
        with torch.no_grad():
            output = self.shape_model(input_tensor)
            pred_class = output.argmax(1).item()
        return self.class_names[pred_class]

    def track_shapes(self):
        blurred = cv2.GaussianBlur(self.img, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            shape_roi = self.img[y:y+h, x:x+w]
            shape_name = self.classify_shape(shape_roi)

            global_x = x + w // 2
            global_y = y + h // 2
            rel_x = global_x - self.origin_x
            rel_y = global_y - self.origin_y

            results.append((shape_name, rel_x, rel_y))

        return results
