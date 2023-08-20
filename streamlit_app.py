import streamlit as st
from recognition import yolo_predictor
import cv2 as cv
import numpy as np
from PIL import Image

def main():
    predictor = yolo_predictor("yolov8m.pt")
    img = st.file_uploader("Upload an image", type=["jpg", "png"])
    if img is not None:
        img = Image.open(img)
        img = np.array(img)
        st.image(img, channels="RGB")
        result = predictor.predict(img)
        recognized_img = img.copy()
        for box in result.boxes:
            class_id = result.names[box.cls[0].item()]
            coords = box.xyxy[0].tolist()
            coords = [round(x) for x in coords]
            conf = round(box.conf[0].item(), 2)
            scale = 0.5
            #scale = min(recognized_img.shape[0], recognized_img.shape[1]) / (25/scale)
            st.write(scale)
            cv.rectangle(recognized_img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
            cv.putText(recognized_img, f"{class_id} {conf}", (coords[0], coords[1]), cv.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), 2)
        st.image(recognized_img, channels="RGB")


if __name__ == "__main__":
    main()