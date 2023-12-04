import streamlit as st
from image_selector import ImageSelector
from CheXNet import CheXNet
from roboflow_model import BrainTumorDetector, LiverDiseaseDetector, BreastCancerDetector
import cv2 as cv
import numpy as np
import pandas as pd
import os
import TorchXRayModels
from PIL import Image


@st.cache_resource
def load_model(weight_path):
    return CheXNet(weight_path)


def show_image(img, pred_img, label, conf, result_imgs):
    original, predicted = st.columns(2)
    original.write(f"<p style='text-align: center; font-size: 20px;'>Original</p>", unsafe_allow_html=True)
    original.image(img, channels="RGB")
    if pred_img is None:
        predicted.write("Nada detectado")
        predicted.image(img, channels="RGB")
    else:
        predicted.write(
            f"<p style='text-align: center; font-size: 20px;'>{label} Probabilidade: {round(conf * 100, 2)}</p>",
            unsafe_allow_html=True)
        predicted.image(pred_img, channels="RGB")
        prev, index, nex = predicted.columns(3)
        prev.button("Anterior", on_click=result_imgs.prev)
        index.write(
            f"<p style='text-align: center; font-size: 20px;'>{st.session_state.index + 1}/{len(result_imgs)}</p>",
            unsafe_allow_html=True)
        nex.button("Próxima", on_click=result_imgs.next)


def main():
    # Read the models dataframe
    models = pd.read_csv("models.csv", sep=";")

    selected_model = st.sidebar.selectbox("Selecione o modelo", models["model"].values)
    conf_threshold = st.sidebar.slider("Confidence threshold", 0, 100, 70) / 100.0

    #st.title(models.loc[models["model"] == selected_model, "title"].values[0])
    #descriptions_file_name = models.loc[models["model"] == selected_model, "description_file"].values[0]

    # Write descriptions of the models
    #descriptions_file_path = "descriptions" + "/" + descriptions_file_name
    #if os.path.exists(descriptions_file_path):
    #    with open(descriptions_file_path, "r", encoding="utf-8") as f:
    #        st.markdown(f.read())'''

    # Create the image uploader
    img = st.file_uploader("Upload an image", type=["jpg", "png", "svg"])
    if "index" not in st.session_state:
        st.session_state.index = 0
    print(conf_threshold)
    model = None
    if selected_model == "CheXNet":
        model = CheXNet('model.pth.tar')
    elif selected_model == "Detector de tumores cerebrais":
        model = BrainTumorDetector()
    elif selected_model == "Doença de Fígado":
        model = LiverDiseaseDetector()
    elif selected_model == "Detector de câncer de mama":
        model = BreastCancerDetector()
    elif selected_model == "Detector de 18 doenças de tórax":
        model = TorchXRayModels.DenseNetModel("densenet121-res224-all", conf_threshold=conf_threshold)
    elif selected_model == "Detector de opacidade pulmonar e pneumonia":
        model = TorchXRayModels.DenseNetModel("densenet121-res224-rsna", conf_threshold=conf_threshold)
    elif selected_model == "NIH ChestX-ray14":
        model = TorchXRayModels.DenseNetModel("densenet121-res224-nih", conf_threshold=conf_threshold)
    elif selected_model == "PadChest":
        model = TorchXRayModels.DenseNetModel("densenet121-res224-pc", conf_threshold=conf_threshold)
    elif selected_model == "CheXpert":
        model = TorchXRayModels.DenseNetModel("densenet121-res224-chex", conf_threshold=conf_threshold)
    elif selected_model == "MIMIC CXR CheXPert":
        model = TorchXRayModels.DenseNetModel("densenet121-res224-mimic_nb", conf_threshold=conf_threshold)
    elif selected_model == "MIMIC CXR NegBio":
        model = TorchXRayModels.DenseNetModel("densenet121-res224-mimic_ch", conf_threshold=conf_threshold)
    elif selected_model == "Detector de doenças de tórax ResNet50":
        model = TorchXRayModels.ResNetModel(conf_threshold=conf_threshold)
    elif selected_model == "Detector de doenças de tórax do JF Healthcare":
        model = TorchXRayModels.JFHealthcareModel(conf_threshold=conf_threshold)
    elif selected_model == "Modelo de segmentação anatômica":
        model = TorchXRayModels.SegmentationModel()

    model.run(img, conf_threshold)


if __name__ == "__main__":
    main()
