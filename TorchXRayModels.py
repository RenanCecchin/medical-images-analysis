import torchxrayvision as xrv
import torch
import torchvision
from abc import ABC, abstractmethod
import cv2 as cv
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from image_selector import ImageSelector
from layout import StreamlitPage


class TorchXRayModel(ABC):
    def __init__(self, threshold):
        self.model = None
        self.threshold = threshold

    def fix_labels(self, labels):
        new_labels = {"preds": {key: value for key, value in labels["preds"].items() if value != 0.5 and value > self.threshold}}
        print(new_labels)
        return new_labels

    def predict(self, img):
        img = xrv.datasets.normalize(img, 255)

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        img = img[None, :, :]

        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                    xrv.datasets.XRayResizer(224)])

        img = transform(img)

        output = {}
        with torch.no_grad():
            img = torch.from_numpy(img).unsqueeze(0)

            preds = self.model(img).cpu()
            output["preds"] = dict(zip(xrv.datasets.default_pathologies, preds[0].detach().numpy()))

        return self.fix_labels(output)


class DenseNetModel(TorchXRayModel, StreamlitPage):
    def __init__(self, model_name, conf_threshold=0.5):
        super().__init__(conf_threshold)
        self.model_name = model_name
        self.model = xrv.models.DenseNet(weights=self.model_name)

    def title(self):
        st.title("Reconhecimento de doenças de tórax com a arquitetura DenseNet121")

    def description(self):
        text = ""
        with open(f"descriptions/{self.model_name}.md", "r", encoding="utf-8") as f:
            text = f.read()

        st.write(text)
        input_img, output_img = st.columns(2)
        input_img.image("sample_data/CheXNet/input_sample.png", caption="Imagem de entrada")
        output_img.image(f"sample_data/CheXNet/output_{self.model_name}_sample.jpg", caption="Imagem de saída com cardiomegalia detectada")

    def show_image(self, img, pred_img, label, conf, result_imgs):
        super().show_image(img, pred_img, label, conf, result_imgs)

    def run(self, img, conf_threshold):
        if img is not None:
            img = Image.open(img)
            pred = self.predict(img)
            labels = list(pred["preds"].keys())
            confs = list(pred["preds"].values())
            result_imgs = [img] * len(labels)

            result_imgs = ImageSelector(result_imgs, labels, confs)
            pred_img, label, conf = result_imgs.get_img(st.session_state.index)
            self.show_image(img, pred_img, label, conf, result_imgs)

        self.title()
        self.description()



class ResNetModel(TorchXRayModel, StreamlitPage):
    def __init__(self, conf_threshold=0.5):
        super().__init__(conf_threshold)
        self.model = xrv.models.ResNet(weights="resnet50-res512-all")

    def title(self):
        st.title("Reconhecimento de doenças de tórax com a arquitetura ResNet50")

    def description(self):
        model_name = "resnet50-res512-all"
        text = ""
        with open(f"descriptions/{model_name}.md", "r", encoding="utf-8") as f:
            text = f.read()

        st.write(text)
        input_img, output_img = st.columns(2)
        input_img.image("sample_data/CheXNet/input_sample.png", caption="Imagem de entrada")
        output_img.image(f"sample_data/CheXNet/output_{model_name}_sample.jpg",
                         caption="Imagem de saída com cardiomegalia detectada")

    def show_image(self, img, pred_img, label, conf, result_imgs):
        super().show_image(img, pred_img, label, conf, result_imgs)

    def run(self, img, conf_threshold):
        if img is not None:
            img = Image.open(img)
            pred = self.predict(img)
            labels = list(pred["preds"].keys())
            confs = list(pred["preds"].values())
            result_imgs = [img] * len(labels)

            result_imgs = ImageSelector(result_imgs, labels, confs)
            pred_img, label, conf = result_imgs.get_img(st.session_state.index)
            self.show_image(img, pred_img, label, conf, result_imgs)

        self.title()
        self.description()

class JFHealthcareModel(TorchXRayModel, StreamlitPage):
    def __init__(self, conf_threshold=0.5):
        super().__init__(conf_threshold)
        self.model = xrv.baseline_models.jfhealthcare.DenseNet()

    def title(self):
        st.title("Reconhecimento de doenças de tórax da JF Healthcare")

    def description(self):
        model_name = "jfhealthcare"
        text = ""
        with open(f"descriptions/{model_name}.md", "r", encoding="utf-8") as f:
            text = f.read()

        st.write(text)
        input_img, output_img = st.columns(2)
        input_img.image("sample_data/CheXNet/input_sample.png", caption="Imagem de entrada")
        output_img.image(f"sample_data/CheXNet/output_{model_name}_sample.jpg",
                         caption="Imagem de saída com cardiomegalia detectada")

    def show_image(self, img, pred_img, label, conf, result_imgs):
        super().show_image(img, pred_img, label, conf, result_imgs)

    def run(self, img, conf_threshold):
        if img is not None:
            img = Image.open(img)
            pred = self.predict(img)
            labels = list(pred["preds"].keys())
            confs = list(pred["preds"].values())
            result_imgs = [img] * len(labels)

            result_imgs = ImageSelector(result_imgs, labels, confs)
            pred_img, label, conf = result_imgs.get_img(st.session_state.index)
            self.show_image(img, pred_img, label, conf, result_imgs)

        self.title()
        self.description()

class SegmentationModel(StreamlitPage):
    def __init__(self):
        self.model = xrv.baseline_models.chestx_det.PSPNet()

    def get_labels(self):
        return self.model.targets

    def predict(self, img):
        img = xrv.datasets.normalize(img, 255)
        img = img.mean(2)[None, ...]

        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(512)])

        img = transform(img)
        img = torch.from_numpy(img)

        with torch.no_grad():
            return self.model(img)

    def title(self):
        st.title("Segmentação de imagens de tórax")

    def description(self):
        text = ""
        with open(f"descriptions/segmentation.md", "r", encoding="utf-8") as f:
            text = f.read()

        st.write(text)
        input_img, output_img = st.columns(2)
        input_img.image("sample_data/CheXNet/input_sample.png", caption="Imagem de entrada")
        output_img.image(f"sample_data/CheXNet/output_segmentation_sample.jpg",
                         caption="Imagem de saída com cardiomegalia detectada")

    def show_image(self, img, pred_img, label, conf, result_imgs):
        super().show_image(img, pred_img, label, conf, result_imgs)

    def run(self, img, conf_threshold):
        if img is not None:
            img = Image.open(img)
            preds = model.predict(np.expand_dims(np.array(img), axis=0))
            preds = preds.numpy()[0]
            preds = self.predict(img)
            preds = preds.numpy()[0]
            labels = self.get_labels()
            confs = [1] * len(labels)

            result_imgs = ImageSelector(preds, labels, confs)

            pred_img, label, conf = result_imgs.get_img(st.session_state.index)
            # Normalize between 0 and 1
            # pred_img = (pred_img - np.min(pred_img)) / (np.max(pred_img) - np.min(pred_img))
            # pred_img = pred_img.astype(np.float64)
            print("Final")
            print(pred_img)
            print(np.max(pred_img))
            print(np.min(pred_img))
            print(pred_img.shape)
            self.show_image(img, pred_img, label, conf, result_imgs)

        self.title()
        self.description()


if __name__ == '__main__':
    model = SegmentationModel()
    img = cv.imread("sample_data/CheXNet/images_001/images/00000001_000.png")
    result = model.predict(img)
    nrows = 2
    ncols = 7
    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 6))
    print(result[0].shape)
    for i in range(nrows):
        for j in range(ncols):
            print(i,j)
            if i*nrows+j < 14:
                axs[i][j].set_title(model.model.targets[i*nrows+j])
                axs[i][j].imshow(result[0][i*nrows+j])

    plt.subplots_adjust(wspace=0.5, hspace=0.2)
    plt.show()
