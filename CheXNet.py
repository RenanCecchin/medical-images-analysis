import os
import re
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import cv2 as cv
from image_selector import ImageSelector
from layout import StreamlitPage
import streamlit as st


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """

    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


class CheXNet(StreamlitPage):

    def __init__(self, ckpt_path):
        self.model = DenseNet121(14)
        self.CLASS_NAMES = ["Atelectasia", "Cardiomegalia", "Derrame pleural (Efusão)", "Infiltração", "Massa",
                            "Nódulo", "Pneumonia", "Pneumotórax", "Consolidação", "Edema",
                            "Enfisema", "Fibrose", "Espessamento pleural", "Hérnia"]
        # hook the feature extractor
        self.features_blobs = []

        def hook_feature(module, input, output):
            self.features_blobs.append(output.data.cpu().numpy())

        self.model.densenet121._modules.get('features').register_forward_hook(hook_feature)

        # get the softmax weight
        params = list(self.model.parameters())
        self.weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

        self.model = torch.nn.DataParallel(self.model)

        if os.path.isfile(ckpt_path):
            print("=> loading checkpoint '{}'".format(ckpt_path))
            checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

            # Fix for original code trained model loading
            state_dict = checkpoint['state_dict']
            remove_data_parallel = False
            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            for key in list(state_dict.keys()):
                match = pattern.match(key)
                new_key = match.group(1) + match.group(2) if match else key
                new_key = new_key[7:] if remove_data_parallel else new_key
                state_dict[new_key] = state_dict[key]
                # Delete old key only if modified.
                if match or remove_data_parallel:
                    del state_dict[key]

            self.model.load_state_dict(state_dict)
            print("=> loaded checkpoint '{}' (epoch {})".format(ckpt_path, checkpoint['epoch']))
        else:
            raise Exception("=> no checkpoint found at '{}'".format(ckpt_path))

        self.model.eval()

        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda
            (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda
            (lambda crops: torch.stack([normalize(crop) for crop in crops]))
        ])

    def get_class(self, pred):
        if pred is not None:
            return self.CLASS_NAMES[pred]
        else:
            return None

    def returnCAM(self, feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 256x256
        size_upsample = (256, 256)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:
            feature_conv_flat = feature_conv.reshape((nc, bz * h * w))
            cam = weight_softmax[idx].dot(feature_conv_flat)
            # Slice a 7x7 matrix
            cam = cam[0:49]
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv.resize(cam_img, size_upsample))
        return output_cam

    def predict(self, image, conf_threshold=0.5):
        """
        Args:
            image: a PIL image
        """
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        with torch.no_grad():
            pred = self.model(image)

        pred_mean = pred.mean(0)

        over_threshold = pred_mean > conf_threshold
        print(pred_mean)
        print(over_threshold)
        over_threshold_indices = over_threshold.nonzero(as_tuple=True)

        if len(over_threshold_indices[0]) > 0:
            pred_classes = over_threshold_indices[0].tolist()
            pred_probs = pred_mean[over_threshold].tolist()

            cams = self.returnCAM(self.features_blobs[0], self.weight_softmax, pred_classes)

            return pred_classes, pred_probs, cams
        else:
            return None, None, None

    def show_image(self, img, pred_img, label, conf, result_imgs):
        super().show_image(img, pred_img, label, conf, result_imgs)

    def title(self):
        st.title("Reconhecimento de 14 doenças de tórax")

    def description(self):
        text = ""
        with open("descriptions/chexnet.md", "r", encoding="utf-8") as f:
            text = f.read()
        st.write(text)
        input_img, output_img = st.columns(2)
        input_img.image("sample_data/CheXNet/input_sample.png", caption="Imagem de entrada")
        output_img.image("sample_data/CheXNet/output_sample.jpg", caption="Imagem de saída com cardiomegalia detectada")

    def run(self, img, conf_threshold):
        if img is not None:
            img = Image.open(img)
            pred, confs, cams = self.predict(img, conf_threshold=conf_threshold)
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
            cl = self.get_class(label)

            self.show_image(img, pred_img, cl, conf, result_imgs)

        self.title()
        self.description()
