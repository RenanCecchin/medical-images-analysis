from roboflow import Roboflow
import cv2 as cv
import numpy as np
from PIL import Image
import base64
from base64 import decodebytes
import requests
import matplotlib.pyplot as plt

class RoboflowModel:
    def __init__(self, url, access_token, format, headers, class_name, overlap=50, confidence=50, roboflow_size = 720):
        self.url = url
        self.access_token = access_token
        self.format = format
        self.overlap = f"&overlap={overlap}"
        self.confidence = f"&confidence={confidence}"
        self.headers = headers
        self.class_name = class_name
        self.roboflow_size = 720

    def change_overlap(self, overlap):
        self.overlap = f"&overlap={overlap}"

    def change_confidence(self, confidence):
        self.confidence = f"&confidence={confidence}"

    def build_request(self):
        parts = []
        parts.append(self.url)
        parts.append(self.access_token)
        parts.append(self.format)
        parts.append(self.overlap)
        parts.append(self.confidence)
        url_request = "".join(parts)

        return url_request
    
    def show_preds(self, img, preds):
        result_imgs = []
        labels = []
        confs = []
        result_img = np.array(img)
        result_img = cv.cvtColor(result_img, cv.COLOR_RGB2BGR)
        for pred in preds:
            predicted_img = result_img.copy()
            x1 = int(pred["x"])
            y1 = int(pred["y"])
            x2 = int(x1 + pred["width"])
            y2 = int(y1 + pred["height"])
            if self.class_name != None:
                labels.append(pred[self.class_name])
            confs.append(pred["confidence"])
            predicted_img = cv.rectangle(predicted_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            result_imgs.append(predicted_img.astype(np.uint8))
        
        return result_imgs, labels, confs

        
    def predict(self, img):
        img_format = img.format
        img = np.array(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        _ , buffer = cv.imencode(f'.{img_format.lower()}', img)
        img_str = base64.b64encode(buffer.tobytes()).decode("ascii")

        resp = requests.post(self.build_request(), data=img_str, headers=self.headers)
        print(resp.json())
        preds = resp.json()["predictions"]

        return self.show_preds(img, preds)
    



if __name__ == "__main__":
    b = RoboflowModel()
    img = Image.open("brain-example1.png")
    b.predict(img)