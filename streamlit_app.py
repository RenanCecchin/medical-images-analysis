import streamlit as st
from image_selector import ImageSelector
from CheXNet import CheXNet
from roboflow_model import RoboflowModel
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image

@st.cache_resource
def load_model(weight_path):
    return CheXNet(weight_path)

def show_image(img, pred_img, label, conf, result_imgs):
    original, predicted = st.columns(2)
    original.write(f"<p style='text-align: center; font-size: 20px;'>Original</p>", unsafe_allow_html=True)
    original.image(img, channels="RGB")
    predicted.write(
        f"<p style='text-align: center; font-size: 20px;'>{label} Probabilidade: {round(conf * 100, 2)}</p>",
        unsafe_allow_html=True)
    predicted.image(pred_img, channels="RGB")
    prev, index, nex = predicted.columns(3)
    prev.button("Anterior", on_click=result_imgs.prev)
    index.write(f"<p style='text-align: center; font-size: 20px;'>{st.session_state.index + 1}/{len(result_imgs)}</p>",
                unsafe_allow_html=True)
    nex.button("Próxima", on_click=result_imgs.next)

def main():
    models = pd.read_csv("models.csv", sep=";")

    selected_model = st.sidebar.selectbox("Selecione o modelo", models["model"].values)

    st.title(models.loc[models["model"] == selected_model,"title"].values[0])
    img = st.file_uploader("Upload an image", type=["jpg", "png", "svg"])

    if "index" not in st.session_state:
        st.session_state.index = 0
    if img is not None:
        img = Image.open(img)

        if selected_model == "CheXNet":
            chexnet = load_model('model.pth.tar')
            pred, confs, cams = chexnet.predict(img)
            result_imgs = []
            result_img = np.array(img)
            result_img = cv.cvtColor(result_img, cv.COLOR_RGB2BGR)
            for cam in cams:
                height, width, _ = result_img.shape
                heatmap = cv.applyColorMap(cv.resize(cam,(width, height)), cv.COLORMAP_JET)
                predicted_img = heatmap * 0.3 + result_img * 0.5
                result_imgs.append(predicted_img.astype(np.uint8))

            result_imgs = ImageSelector(result_imgs, pred, confs)
            pred_img, label, conf = result_imgs.get_img(st.session_state.index)
            show_image(img, pred_img, chexnet.get_class(label), conf, result_imgs)

        elif selected_model == "Detector de tumores cerebrais":
            brain_tumor = RoboflowModel("https://detect.roboflow.com/brain-tumor-m2pbp/1", 
                                        "?api_key=pbzrAqfcSaFRN8OIPpKC", "&format=json", 
                                        {"Content-Type": "application/x-www-form-urlencoded"},
                                        None)

            processed_img, preds = brain_tumor.predict(img)
            result_imgs, labels, confs = brain_tumor.show_bb_preds(processed_img, preds)
            labels = ["Tumor"] * len(result_imgs)
            result_imgs = ImageSelector(result_imgs, labels, confs)

            pred_img, label, conf = result_imgs.get_img(st.session_state.index)
            show_image(img, pred_img, label, conf, result_imgs)
        elif selected_model == "Doença de Fígado":
            liver_disease = RoboflowModel("https://detect.roboflow.com/liver-disease/1",
                                          "?api_key=pbzrAqfcSaFRN8OIPpKC", "&format=json",
                                          {"Content-Type": "application/x-www-form-urlencoded"},
                                          "class")
            
            processed_img, preds = liver_disease.predict(img)
            result_imgs, labels, confs= liver_disease.show_bb_preds(processed_img, preds)
            result_imgs = ImageSelector(result_imgs, labels, confs)

            pred_img, label, conf = result_imgs.get_img(st.session_state.index)
            show_image(img, pred_img, label, conf, result_imgs)

        elif selected_model == "HyperKvasir":
            hyperkvasir = RoboflowModel("https://classify.roboflow.com/hyper-kvasir/1",
                                          "?api_key=pbzrAqfcSaFRN8OIPpKC", "&format=json",
                                          {"Content-Type": "application/x-www-form-urlencoded"},
                                          "class")

            processed_img, preds = hyperkvasir.predict(img)
            result_imgs, labels, confs= hyperkvasir.show_class_preds(processed_img, preds)
            result_imgs = ImageSelector(result_imgs, labels, confs)

            pred_img, label, conf = result_imgs.get_img(st.session_state.index)
            show_image(img, pred_img, label, conf, result_imgs)
        else:
            pass
    else:
        st.session_state.index = 0



if __name__ == "__main__":
    main()