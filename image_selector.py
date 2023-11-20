import streamlit as st

class ImageSelector():
    def __init__(self, imgs, labels, confs):
        self.imgs = imgs
        self.labels = labels
        self.confs = confs
    
    def get_img(self, index):
        if index < len(self.imgs):
            return self.imgs[index], self.labels[index], self.confs[index]
        else:
            return None, None, None

    def next(image_selector):
        if st.session_state.index + 1 < len(image_selector.imgs):
            st.session_state.index += 1
        else:
            st.session_state.index = 0

    def prev(image_selector):
        if st.session_state.index - 1 >= 0:
            st.session_state.index -= 1
        else:
            st.session_state.index = len(image_selector.imgs) - 1
    
    def __len__(self):
        return len(self.imgs)