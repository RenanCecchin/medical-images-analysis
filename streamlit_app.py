import streamlit as st
from image_selector import ImageSelector
from chexnet import CheXNet
from roboflow_model import BrainTumorDetector, LiverDiseaseDetector, BreastCancerDetector
import cv2 as cv
import numpy as np
import pandas as pd
import os
import torchxraymodels
from external_model import ExternalModel, AddModelLayout
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
    conf_threshold = st.sidebar.slider("Limite de Confiança", 0, 100, 70) / 100.0

    st.warning("Quando tiver um momento, por favor, preencha o [formulário de feedback](https://forms.gle/wXiJ9Vomys5SLqAG7)."
               " Sua opinião é extremamente valiosa para este trabalho. Obrigado!", icon = "⚠️")


    if "index" not in st.session_state:
        st.session_state.index = 0

    if selected_model == "Introdução":
        st.title("Introdução")
        st.write("Neste site você pode testar modelos de IA para detecção de doenças em imagens médicas")
        st.write("Para isso, basta selecionar o modelo desejado no menu lateral e fazer o upload de uma imagem")
        st.write("O modelo irá detectar a doença na imagem e mostrar a imagem com a doença destacada")
        st.write(
            "Você pode mudar o limite de confiança do modelo no menu lateral (valores de confiança muito baixos podem gerar falsos positivos)")
        st.write("Você pode mudar a imagem mostrada clicando nos botões de anterior e próximo")
        st.write("Você pode ver a descrição de cada modelo ao clicar no seu respectivo nome no menu lateral")
        st.write("Abaixo você pode ver um vídeo de introdução sobre a plataforma")
        st.video("https://www.youtube.com/watch?v=LG39OWQErSc")

        st.write("Peço que testem a plataforma durante a semana de 04/12/2023 a 10/12/2023 e, se possível, respondam "
                 "a esse [formulário](https://forms.gle/wXiJ9Vomys5SLqAG7) de feedback para que eu possa melhorar a "
                 "plataforma e integrar os resultados ao meu TCC")

    elif selected_model == "Modelo Externo":
        model = AddModelLayout()

        added_model = model.run(None, conf_threshold)

    else:
        # Create the image uploader
        img = st.file_uploader("Selecione uma imagem", type=["jpg", "png", "svg"])
        model = None
        if selected_model == "CheXNet":
            model = CheXNet('model.pth.tar')
        elif selected_model == "Detector de tumores cerebrais":
            model = BrainTumorDetector()
        elif selected_model == "Doença de Fígado":
            model = LiverDiseaseDetector()
        elif selected_model == "Detector de 18 doenças de tórax":
            model = torchxraymodels.DenseNetModel("densenet121-res224-all", conf_threshold=conf_threshold)
        elif selected_model == "Detector de opacidade pulmonar e pneumonia":
            model = torchxraymodels.DenseNetModel("denseness121-res224-rsna", conf_threshold=conf_threshold)
        elif selected_model == "NIH ChestX-ray14":
            model = torchxraymodels.DenseNetModel("densenet121-res224-nih", conf_threshold=conf_threshold)
        elif selected_model == "PadChest":
            model = torchxraymodels.DenseNetModel("densenet121-res224-pc", conf_threshold=conf_threshold)
        elif selected_model == "CheXpert":
            model = torchxraymodels.DenseNetModel("densenet121-res224-chex", conf_threshold=conf_threshold)
        elif selected_model == "MIMIC CXR CheXPert":
            model = torchxraymodels.DenseNetModel("densenet121-res224-mimic_nb", conf_threshold=conf_threshold)
        elif selected_model == "MIMIC CXR NegBio":
            model = torchxraymodels.DenseNetModel("densenet121-res224-mimic_ch", conf_threshold=conf_threshold)
        elif selected_model == "Detector de doenças de tórax ResNet50":
            model = torchxraymodels.ResNetModel(conf_threshold=conf_threshold)
        elif selected_model == "Detector de doenças de tórax do JF Healthcare":
            model = torchxraymodels.JFHealthcareModel(conf_threshold=conf_threshold)
        elif selected_model == "Modelo de segmentação anatômica":
            model = torchxraymodels.SegmentationModel()
        model.run(img, conf_threshold)


if __name__ == "__main__":
    main()
