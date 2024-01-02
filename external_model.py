import tensorflow as tf
import torch
from tensorflow.python.keras.models import load_model
from layout import StreamlitPage
from PIL import Image
import streamlit as st
from io import StringIO
from image_selector import ImageSelector
import zipfile
import tempfile
import numpy as np
import os


class ExternalModel(StreamlitPage):
    def title(self):
        pass

    def description(self):
        pass

    def show_image(self, img, pred_img, label, conf, result_imgs):
        super().show_image(img, pred_img, label, conf, result_imgs)

    def run(self, img, conf_threshold):
        if img is not None:
            img = Image.open(img)
            if self.framework_choice == "tensorflow":
                input_shape = self.model.layers[0]
                pred_img = img.resize(input_shape.input_shape[1:3])
                pred_img = np.array(pred_img)
                pred_img = np.expand_dims(pred_img, axis=0)
                pred = self.model.predict(pred_img)
                result_imgs = [img] * pred.shape[0]
                result_imgs = ImageSelector(result_imgs, list(range(pred.shape[0])), pred)
                pred_img, label, conf = result_imgs.get_img(st.session_state.index)

                self.show_image(img, pred_img, label, conf_threshold, result_imgs)

    def __init__(self, weights, framework_cboice):
        self.model = None
        self.framework_choice = framework_cboice

        if self.framework_choice == "tensorflow":
            self.model = tf.keras.models.load_model(weights)
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            self.model = torch.load(weights)


class AddModelLayout(StreamlitPage):
    def __init__(self):
        self.external_model = None

    def title(self):
        st.title("Adicionar modelo externo")

    def description(self):
        pass

    def show_image(self, img, pred_img, label, conf, result_imgs):
        pass

    def add_model(self):
        weights = st.file_uploader("Adicione o arquivo do modelo")
        if weights is not None:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(weights.getvalue())
                temp_file_path = temp_file.name
            self.external_model = ExternalModel(temp_file_path, "tensorflow")


    def run(self, img, conf_threshold):
        self.add_model()
        if self.external_model is None:
            self.title()
            self.description()
        else:
            img = st.file_uploader("Selecione uma imagem", type=["jpg", "png", "svg"])
            self.external_model.run(img, conf_threshold)

