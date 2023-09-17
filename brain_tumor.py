from roboflow import Roboflow
import cv2 as cv
import numpy as np
from PIL import Image
import base64
from base64 import decodebytes
import requests
import matplotlib.pyplot as plt

class BrainTumorDetector:
    def __init__(self):
        self.url = "https://detect.roboflow.com/brain-tumor-m2pbp/1"
        self.access_token = "?api_key=pbzrAqfcSaFRN8OIPpKC"
        self.format = "&format=json"
        self.overlap = f"&overlap={30}"
        self.confidence = f"&confidence={50}"
        self.headers = {"Content-Type": "application/x-www-form-urlencoded"}
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
        
    def predict(self, img):
        img_format = img.format
        img = np.array(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        _ , buffer = cv.imencode(f'.{img_format.lower()}', img)
        img_str = base64.b64encode(buffer.tobytes()).decode("ascii")

        resp = requests.post(self.build_request(), data=img_str, headers=self.headers)
        preds = resp.json()["predictions"]

        return preds

if __name__ == "__main__":
    b = BrainTumorDetector()
    img = Image.open("brain-example1.png")
    b.predict(img)