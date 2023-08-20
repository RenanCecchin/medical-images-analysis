from ultralytics import YOLO
import cv2 as cv

class yolo_predictor():
    def __init__(self, model):
        self.model = YOLO(model)

    def predict(self, image):
        results = self.model.predict(image)
        return results[0]
    
    def batch_predict(self, images):
        image_results = []
        for image in images:
            image_results.append(self.model.predict(image))
        return image_results