import streamlit as st

class ImageSelector():
    def __init__(self, imgs, labels, confs):
        self.imgs = imgs
        self.labels = labels
        self.confs = confs
    
    def get_img(self, index):
        return self.imgs[index], self.labels[index], self.confs[index]
    
    def next(self):
        if st.session_state.index + 1 < len(self.imgs):
            st.session_state.index += 1
        else:
            st.session_state.index = 0
    
    def prev(self):
        if st.session_state.index - 1 >= 0:
            st.session_state.index -= 1
        else:
            st.session_state.index = len(self.imgs) - 1
    
    def __len__(self):
        return len(self.imgs)