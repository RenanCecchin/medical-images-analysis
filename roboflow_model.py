import cv2 as cv
import numpy as np
import streamlit
from PIL import Image
import base64
from base64 import decodebytes
import requests

from image_selector import ImageSelector
from layout import StreamlitPage
import streamlit as st


class RoboflowModel:
    def __init__(self, url, access_token, format, headers, class_name, overlap=50,
                 roboflow_size=720):
        self.url = url
        self.access_token = access_token
        self.format = format
        self.overlap = f"&overlap={overlap}"
        self.headers = headers
        self.class_name = class_name
        self.roboflow_size = 720

    def change_overlap(self, overlap):
        self.overlap = f"&overlap={overlap}"

    def change_confidence(self, confidence):
        self.confidence = f"&confidence={confidence}"

    def build_request(self, conf_threshold):
        parts = [self.url, self.access_token, self.format, self.overlap, f"&confidence={conf_threshold}"]
        url_request = "".join(parts)

        return url_request

    def show_class_preds(self, img, preds):
        result_imgs = []
        labels = []
        confs = []
        result_img = np.array(img)
        result_img = cv.cvtColor(result_img, cv.COLOR_RGB2BGR)

        if type(preds) == dict:
            preds = [preds]

        for pred in preds:
            predicted_img = result_img.copy()
            result_imgs.append(predicted_img.astype(np.uint8))
            if self.class_name is not None:
                labels.append(pred[self.class_name])
            confs.append(pred["confidence"])

        return result_imgs, labels, confs

    def show_bb_preds(self, img, preds):
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
            if self.class_name is not None:
                labels.append(pred[self.class_name])
            confs.append(pred["confidence"])
            predicted_img = cv.rectangle(predicted_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            result_imgs.append(predicted_img.astype(np.uint8))

        return result_imgs, labels, confs

    def predict(self, img, conf_threshold=50):
        img_format = img.format
        img = np.array(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        _, buffer = cv.imencode(f'.{img_format.lower()}', img)
        img_str = base64.b64encode(buffer.tobytes()).decode("ascii")

        resp = requests.post(self.build_request(conf_threshold), data=img_str, headers=self.headers)
        print(resp.json())
        preds = resp.json()["predictions"]

        return img, preds


class BrainTumorDetector(RoboflowModel, StreamlitPage):

    def __init__(self):
        super().__init__("https://detect.roboflow.com/brain-tumor-m2pbp/1",
                         "?api_key=pbzrAqfcSaFRN8OIPpKC", "&format=json",
                         {"Content-Type": "application/x-www-form-urlencoded"},
                         "class")

    def predict(self, img, conf_threshold=50):
        predicted_img, preds = super().predict(img)
        return self.show_bb_preds(predicted_img, preds)

    def title(self):
        st.title("Detecção de tumores cerebrais")

    def description(self):
        text = ""
        with open("descriptions/braintumor.md", "r", encoding="utf-8") as f:
            text = f.read()
        st.write(text)
        input_img, output_img = st.columns(2)
        input_img.image("sample_data/Brain Tumor/input_sample.png", caption="Imagem de entrada")
        output_img.image("sample_data/Brain Tumor/output_sample.jpg", caption="Imagem de saída com tumor detectado")

    def show_image(self, img, pred_img, label, conf, result_imgs):
        super().show_image(img, pred_img, label, conf, result_imgs)

    def run(self, img, conf_threshold):
        if img is not None:
            img = Image.open(img)
            result_imgs, labels, confs = self.predict(img, conf_threshold=conf_threshold)
            labels = ["Tumor"] * len(result_imgs)
            result_imgs = ImageSelector(result_imgs, labels, confs)
            pred_img, label, conf = result_imgs.get_img(st.session_state.index)
            self.show_image(img, pred_img, label, conf, result_imgs)

        self.title()
        self.description()


class LiverDiseaseDetector(RoboflowModel, StreamlitPage):
    def __init__(self):
        super().__init__("https://detect.roboflow.com/liver-disease/1",
                         "?api_key=pbzrAqfcSaFRN8OIPpKC", "&format=json",
                         {"Content-Type": "application/x-www-form-urlencoded"},
                         "class")

    def predict(self, img, conf_threshold=50):
        predicted_img, preds = super().predict(img)
        return self.show_bb_preds(predicted_img, preds)

    def title(self):
        st.title("Detecção de doenças no fígado")

    def description(self):
        text = ""
        with open("descriptions/liverdisease.md", "r", encoding="utf-8") as f:
            text = f.read()
        st.write(text)
        input_img, output_img = st.columns(2)
        input_img.image("sample_data/Liver Disease/input_sample.png", caption="Imagem de entrada")
        #output_img.image("sample_data/Liver Disease/output_sample.jpg", caption="Imagem de saída com tumor detectado")

    def show_image(self, img, pred_img, label, conf, result_imgs):
        super().show_image(img, pred_img, label, conf, result_imgs)

    def run(self, img, conf_threshold):
        if img is not None:
            img = Image.open(img)
            result_imgs, labels, confs = self.predict(img, conf_threshold=conf_threshold)
            result_imgs = ImageSelector(result_imgs, labels, confs)
            pred_img, label, conf = result_imgs.get_img(st.session_state.index)
            self.show_image(img, pred_img, label, conf, result_imgs)

        self.title()
        self.description()
