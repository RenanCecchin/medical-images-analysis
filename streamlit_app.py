import streamlit as st
from ultralytics.yolo.utils.ops import scale_image
from recognition import yolo_predictor
import  cv2 as cv
import numpy as np
from PIL import Image

def main():
    bb_predictor = yolo_predictor("yolov8m.pt")
    segmentation_predictor = yolo_predictor("yolov8m-seg.pt")
    st.title("Object recognition")
    segmentation = st.checkbox("Segmentation")
    img = st.file_uploader("Upload an image", type=["jpg", "png"])
    if img is not None:
        img = Image.open(img)
        img = np.array(img)
        st.image(img, channels="RGB")
        if not segmentation:
            result = bb_predictor.predict(img)
            recognized_img = img.copy()
            for box in result.boxes:
                class_id = result.names[box.cls[0].item()]
                coords = box.xyxy[0].tolist()
                coords = [round(x) for x in coords]
                conf = round(box.conf[0].item(), 2)
                scale = 0.9
                cv.rectangle(recognized_img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
                cv.putText(recognized_img, f"{class_id} {conf}", (coords[0], coords[1]), cv.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), 2)
            st.image(recognized_img, channels="RGB")
        else:
            results = segmentation_predictor.predict(img)
            recognized_img = img.copy()
            for result in results:
                mask_raw = result.masks[0].cpu().data.numpy().transpose(1,2,0)

                mask_3channel = cv.merge((mask_raw, mask_raw, mask_raw))

                h2, w2, c2 = result.orig_img.shape
                mask = cv.resize(mask_3channel, (w2, h2))

                hsv = cv.cvtColor(mask, cv.COLOR_BGR2HSV)

                lower_black = np.array([0, 0, 0])
                upper_black = np.array([0, 0, 1])

                mask = cv.inRange(mask, lower_black, upper_black)
                mask = cv.bitwise_not(mask)

                # Colorize mask
                recognized_img[(mask==255)] = [0, 255, 0]
                cv.addWeighted(recognized_img, 0.5, img, 0.5, 0, recognized_img)

                box = result.boxes
                class_id = results.names[box.cls[0].item()]
                coords = box.xyxy[0].tolist()
                coords = [round(x) for x in coords]

                conf = round(box.conf[0].item(), 2)
                scale = 0.9
                cv.putText(recognized_img, f"{class_id} {conf}", (coords[0], coords[1]), cv.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), 2)

            st.image(recognized_img, channels="RGB")



if __name__ == "__main__":
    main()