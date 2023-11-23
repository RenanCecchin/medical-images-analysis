import streamlit as st
from abc import ABC, abstractmethod


class StreamlitPage(ABC):
    @abstractmethod
    def title(self):
        st.title("This is the title")

    @abstractmethod
    def description(self):
        st.write("This is the description")

    @abstractmethod
    def show_image(self, img, pred_img, label, conf, result_imgs):
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

    @abstractmethod
    def run(self, img, conf_threshold):
        self.show_image(img, None, None, None, None)
        self.title()
        self.description()
