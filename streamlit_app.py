import streamlit as st
from image_selector import ImageSelector
from CheXNet import CheXNet
from roboflow_model import RoboflowModel
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

    st.title(models.loc[models["model"] == selected_model, "title"].values[0])
    descriptions_file_name = models.loc[models["model"] == selected_model, "description_file"].values[0]

    # Write descriptions of the models
    descriptions_file_path = "descriptions" + "/" + descriptions_file_name
    if os.path.exists(descriptions_file_path):
        with open(descriptions_file_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())

    # Create the image uploader
    img = st.file_uploader("Upload an image", type=["jpg", "png", "svg"])
    if "index" not in st.session_state:
        st.session_state.index = 0

    if img is not None:
        img = Image.open(img)
        if selected_model == "CheXNet":
            chexnet = load_model('model.pth.tar')
            pred, confs, cams = chexnet.predict(img, conf_threshold=conf_threshold)
            result_imgs = []
            result_img = np.array(img)
            result_img = cv.cvtColor(result_img, cv.COLOR_RGB2BGR)
            if cams is not None:
                for cam in cams:
                    height, width, _ = result_img.shape
                    heatmap = cv.applyColorMap(cv.resize(cam, (width, height)), cv.COLORMAP_JET)
                    predicted_img = heatmap * 0.3 + result_img * 0.5
                    result_imgs.append(predicted_img.astype(np.uint8))

            result_imgs = ImageSelector(result_imgs, pred, confs)
            pred_img, label, conf = result_imgs.get_img(st.session_state.index)
            cl = chexnet.get_class(label)
            print(cl)
            show_image(img, pred_img, cl, conf, result_imgs)

        elif selected_model == "Detector de tumores cerebrais":
            brain_tumor = RoboflowModel("https://detect.roboflow.com/brain-tumor-m2pbp/1",
                                        "?api_key=pbzrAqfcSaFRN8OIPpKC", "&format=json",
                                        {"Content-Type": "application/x-www-form-urlencoded"},
                                        "class", confidence=conf_threshold)

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
                                          "class", confidence=conf_threshold)

            processed_img, preds = liver_disease.predict(img)
            result_imgs, labels, confs = liver_disease.show_bb_preds(processed_img, preds)
            result_imgs = ImageSelector(result_imgs, labels, confs)

            pred_img, label, conf = result_imgs.get_img(st.session_state.index)
            show_image(img, pred_img, label, conf, result_imgs)
        elif selected_model == "XRV-DenseNet121-res224-all":
            model = TorchXRayModels.DenseNetModel("densenet121-res224-all", threshold=conf_threshold)

            preds = model.predict(np.array(img))["preds"]
            labels = list(preds.keys())
            confs = list(preds.values())
            result_imgs = [np.array(img)] * len(labels)

            result_imgs = ImageSelector(result_imgs, labels, confs)
            pred_img, label, conf = result_imgs.get_img(st.session_state.index)
            show_image(img, pred_img, label, conf, result_imgs)
        elif selected_model == "XRV-DenseNet121-res224-rsna":
            model = TorchXRayModels.DenseNetModel("densenet121-res224-rsna", threshold=conf_threshold)

            preds = model.predict(np.array(img))["preds"]
            labels = list(preds.keys())
            confs = list(preds.values())
            result_imgs = [np.array(img)] * len(labels)

            result_imgs = ImageSelector(result_imgs, labels, confs)
            pred_img, label, conf = result_imgs.get_img(st.session_state.index)
            show_image(img, pred_img, label, conf, result_imgs)
        elif selected_model == "XRV-DenseNet121-res224-nih":
            model = TorchXRayModels.DenseNetModel("densenet121-res224-nih", threshold=conf_threshold)

            preds = model.predict(np.array(img))["preds"]
            labels = list(preds.keys())
            confs = list(preds.values())
            result_imgs = [np.array(img)] * len(labels)

            result_imgs = ImageSelector(result_imgs, labels, confs)
            pred_img, label, conf = result_imgs.get_img(st.session_state.index)
            show_image(img, pred_img, label, conf, result_imgs)
        elif selected_model == "XRV-DenseNet121-res224-pc":
            model = TorchXRayModels.DenseNetModel("densenet121-res224-pc", threshold=conf_threshold)

            preds = model.predict(np.array(img))["preds"]
            labels = list(preds.keys())
            confs = list(preds.values())
            result_imgs = [np.array(img)] * len(labels)

            result_imgs = ImageSelector(result_imgs, labels, confs)
            pred_img, label, conf = result_imgs.get_img(st.session_state.index)
            show_image(img, pred_img, label, conf, result_imgs)
        elif selected_model == "XRV-DenseNet121-res224-chex":
            model = TorchXRayModels.DenseNetModel("densenet121-res224-chex", threshold=conf_threshold)

            preds = model.predict(np.array(img))["preds"]
            labels = list(preds.keys())
            confs = list(preds.values())
            result_imgs = [np.array(img)] * len(labels)

            result_imgs = ImageSelector(result_imgs, labels, confs)
            pred_img, label, conf = result_imgs.get_img(st.session_state.index)
            show_image(img, pred_img, label, conf, result_imgs)
        elif selected_model == "XRV-DenseNet121-res224-mimic_nb":
            model = TorchXRayModels.DenseNetModel("densenet121-res224-mimic_nb", threshold=conf_threshold)

            preds = model.predict(np.array(img))["preds"]
            labels = list(preds.keys())
            confs = list(preds.values())
            result_imgs = [np.array(img)] * len(labels)

            result_imgs = ImageSelector(result_imgs, labels, confs)
            pred_img, label, conf = result_imgs.get_img(st.session_state.index)
            show_image(img, pred_img, label, conf, result_imgs)
        elif selected_model == "XRV-DenseNet121-res224-mimic_ch":
            model = TorchXRayModels.DenseNetModel("densenet121-res224-mimic_ch", threshold=conf_threshold)

            preds = model.predict(np.array(img))["preds"]
            labels = list(preds.keys())
            confs = list(preds.values())
            result_imgs = [np.array(img)] * len(labels)

            result_imgs = ImageSelector(result_imgs, labels, confs)
            pred_img, label, conf = result_imgs.get_img(st.session_state.index)
            show_image(img, pred_img, label, conf, result_imgs)
        elif selected_model == "XRV-ResNet50-res512-all":
            model = TorchXRayModels.ResNetModel(threshold=conf_threshold)

            preds = model.predict(np.array(img))["preds"]
            labels = list(preds.keys())
            confs = list(preds.values())
            result_imgs = [np.array(img)] * len(labels)

            result_imgs = ImageSelector(result_imgs, labels, confs)
            pred_img, label, conf = result_imgs.get_img(st.session_state.index)
            show_image(img, pred_img, label, conf, result_imgs)
        elif selected_model == "XRV-JFHealthcare":
            model = TorchXRayModels.JFHealthcareModel(threshold=conf_threshold)

            preds = model.predict(np.array(img))["preds"]
            labels = list(preds.keys())
            confs = list(preds.values())

            result_imgs = [np.array(img)] * len(labels)
            result_imgs = ImageSelector(result_imgs, labels, confs)

            pred_img, label, conf = result_imgs.get_img(st.session_state.index)
            show_image(img, pred_img, label, conf, result_imgs)
        elif selected_model == "XRV-CheXpert":
            model = TorchXRayModels.CheXpertModel(threshold=conf_threshold)

            preds = model.predict(np.array(img))["preds"]
            labels = list(preds.keys())
            confs = list(preds.values())

            result_imgs = [np.array(img)] * len(labels)
            result_imgs = ImageSelector(result_imgs, labels, confs)

            pred_img, label, conf = result_imgs.get_img(st.session_state.index)
            show_image(img, pred_img, label, conf, result_imgs)
        elif selected_model == "Race Prediction Model":
            model = TorchXRayModels.RacePredictionModel(threshold=conf_threshold)

            preds = model.predict(np.array(img))["preds"]
            labels = list(preds.keys())
            confs = list(preds.values())

            result_imgs = [np.array(img)] * len(labels)
            result_imgs = ImageSelector(result_imgs, labels, confs)

            pred_img, label, conf = result_imgs.get_img(st.session_state.index)
            show_image(img, pred_img, label, conf, result_imgs)
        elif selected_model == "Riken Age Prediction Model":
            model = TorchXRayModels.RikenAgeModel(threshold=conf_threshold)

            preds = model.predict(np.array(img))["preds"]
            labels = list(preds.keys())
            confs = list(preds.values())

            result_imgs = [np.array(img)] * len(labels)
            result_imgs = ImageSelector(result_imgs, labels, confs)

            pred_img, label, conf = result_imgs.get_img(st.session_state.index)
            show_image(img, pred_img, label, conf, result_imgs)
        elif selected_model == "Anatomical Segmentation Model":
            model = TorchXRayModels.SegmentationModel()

            preds = model.predict(np.expand_dims(np.array(img), axis=0))
            preds = preds.numpy()[0]
            #preds = [preds[i, :, :].reshape((512, 512)) for i in range(14)]
            labels = model.get_labels()
            confs = [1] * len(labels)

            result_imgs = ImageSelector(preds, labels, confs)

            pred_img, label, conf = result_imgs.get_img(st.session_state.index)
            # Normalize between 0 and 1
            #pred_img = (pred_img - np.min(pred_img)) / (np.max(pred_img) - np.min(pred_img))
            #pred_img = pred_img.astype(np.float64)
            print("Final")
            print(pred_img)
            print(np.max(pred_img))
            print(np.min(pred_img))
            print(pred_img.shape)
            show_image(img, pred_img, label, conf, result_imgs)
        else:
            pass
    else:
        st.session_state.index = 0


if __name__ == "__main__":
    main()
