import streamlit as st
from image_selector import ImageSelector
from CheXNet import CheXNet
import  cv2 as cv
import numpy as np
from PIL import Image

@st.cache_resource
def load_model(weight_path):
    return CheXNet(weight_path)

def main():
    chexnet = load_model('model.pth.tar')
    st.title("Reconhecimento de doenças de Tórax")
    img = st.file_uploader("Upload an image", type=["jpg", "png", "svg"])

    if "index" not in st.session_state:
        st.session_state.index = 0
    if img is not None:
        img = Image.open(img)
        pred, cams = chexnet.predict(img)
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
        if not show_all:
            predicted.write(f"<p style='text-align: center; font-size: 20px;'>{chexnet.get_class(label)}</p>", unsafe_allow_html=True)
            predicted.image(pred_img, channels="RGB")
            prev, index, nex = predicted.columns(3)
            prev.button("Anterior", on_click=result_imgs.prev)
            index.write(f"<p style='text-align: center; font-size: 20px;'>{st.session_state.index + 1}/{len(result_imgs)}</p>", unsafe_allow_html=True)
            nex.button("Próxima", on_click=result_imgs.next)
        else:
            for image in result_imgs.imgs:
                predicted.image(image, channels="RGB")
    else:
        st.session_state.index = 0



if __name__ == "__main__":
    main()