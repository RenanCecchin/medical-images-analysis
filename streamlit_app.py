import streamlit as st
from image_selector import ImageSelector
from CheXNet import CheXNet
from brain_tumor import BrainTumorDetector
import  cv2 as cv
import numpy as np
from PIL import Image

@st.cache_resource
def load_model(weight_path):
    return CheXNet(weight_path)

def main(): 
    chexnet = load_model('model.pth.tar')
    selected_model = st.sidebar.selectbox("Selecione o modelo", ("CheXNet", "Brain Tumor Detector"))
    title = {"CheXNet": "Reconhecimento de doenças de Tórax", "Brain Tumor Detector": "Reconhecimento de Tumores Cerebrais"}
    st.title(title[selected_model])
    img = st.file_uploader("Upload an image", type=["jpg", "png", "svg"])

    if "index" not in st.session_state:
        st.session_state.index = 0
    if img is not None:
        img = Image.open(img)

        if selected_model == "CheXNet":
            pred, probs, cams = chexnet.predict(img)
            result_imgs = []
            result_img = np.array(img)
            result_img = cv.cvtColor(result_img, cv.COLOR_RGB2BGR)
            for cam in cams:
                height, width, _ = result_img.shape
                heatmap = cv.applyColorMap(cv.resize(cam,(width, height)), cv.COLORMAP_JET)
                predicted_img = heatmap * 0.3 + result_img * 0.5
                result_imgs.append(predicted_img.astype(np.uint8))

            result_imgs = ImageSelector(result_imgs, pred)
            pred_img, label = result_imgs.get_img(st.session_state.index)
            original, predicted = st.columns(2)
            original.write(f"<p style='text-align: center; font-size: 20px;'>Original</p>", unsafe_allow_html=True)
            original.image(img, channels="RGB")
            
            show_all = predicted.checkbox("Mostrar todas as doenças")
            predicted.write(f"<p style='text-align: center; font-size: 20px;'>{chexnet.get_class(label)} Probabilidade:{round(probs[label]*100, 3)}</p>", unsafe_allow_html=True)
            predicted.image(pred_img, channels="RGB")
            prev, index, nex = predicted.columns(3)
            prev.button("Anterior", on_click=result_imgs.prev)
            index.write(f"<p style='text-align: center; font-size: 20px;'>{st.session_state.index + 1}/{len(result_imgs)}</p>", unsafe_allow_html=True)
            nex.button("Próxima", on_click=result_imgs.next)

        elif selected_model == "Brain Tumor Detector":
            brain_tumor = BrainTumorDetector()
            preds = brain_tumor.predict(img)
            result_imgs = []
            labels = []
            result_img = np.array(img)
            result_img = cv.cvtColor(result_img, cv.COLOR_RGB2BGR)
            for pred in preds:
                predicted_img = result_img.copy()
                x1 = int(pred["x"])
                y1 = int(pred["y"])
                x2 = int(x1 + pred["width"])
                y2 = int(y1 + pred["height"])

                labels.append(pred["confidence"])
                predicted_img = cv.rectangle(predicted_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                result_imgs.append(predicted_img.astype(np.uint8))

            result_imgs = ImageSelector(result_imgs, labels)
            pred_img, label = result_imgs.get_img(st.session_state.index)
            original, predicted = st.columns(2)
            original.write(f"<p style='text-align: center; font-size: 20px;'>Original</p>", unsafe_allow_html=True)
            original.image(img, channels="RGB")
            predicted.write(f"<p style='text-align: center; font-size: 20px;'>Tumor Probabilidade: {round(label*100, 3)}</p>", unsafe_allow_html=True)
            predicted.image(pred_img, channels="RGB")
            prev, index, nex = predicted.columns(3)
            prev.button("Anterior", on_click=result_imgs.prev)
            index.write(f"<p style='text-align: center; font-size: 20px;'>{st.session_state.index + 1}/{len(result_imgs)}</p>", unsafe_allow_html=True)
            nex.button("Próxima", on_click=result_imgs.next)
        else:
            pass
    else:
        st.session_state.index = 0



if __name__ == "__main__":
    main()