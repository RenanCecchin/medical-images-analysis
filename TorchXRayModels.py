import torchxrayvision as xrv
import torch
import torchvision
from abc import ABC, abstractmethod
import cv2 as cv
import matplotlib.pyplot as plt


class TorchXRayModel(ABC):
    def __init__(self, threshold):
        self.model = None
        self.threshold = threshold

    def fix_labels(self, labels):
        new_labels = {"preds": {key: value for key, value in labels["preds"].items() if value != 0.5 and value > self.threshold}}
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


class DenseNetModel(TorchXRayModel):
    def __init__(self, model_name, threshold=0.5):
        super().__init__(threshold)
        self.model_name = model_name
        self.model = xrv.models.DenseNet(weights=self.model_name)


class ResNetModel(TorchXRayModel):
    def __init__(self, threshold=0.5):
        super().__init__(threshold)
        self.model = xrv.models.ResNet(weights="resnet50-res512-all")


class JFHealthcareModel(TorchXRayModel):
    def __init__(self, threshold=0.5):
        super().__init__(threshold)
        self.model = xrv.baseline_models.jfhealthcare.DenseNet()


class CheXpertModel(TorchXRayModel):
    def __init__(self, threshold=0.5):
        super().__init__(threshold)
        self.model = xrv.baseline_models.chexpert.DenseNet(weights_zip="chexpert_weights.zip")


class RacePredictionModel(TorchXRayModel):
    def __init__(self, threshold=0.5):
        super().__init__(threshold)
        self.model = xrv.baseline_models.emory_hiti.RaceModel()


class RikenAgeModel(TorchXRayModel):
    def __init__(self, threshold=0.5):
        super().__init__(threshold)
        self.model = xrv.baseline_models.riken.AgeModel()


class SegmentationModel:
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


if __name__ == '__main__':
    model = SegmentationModel()
    img = cv.imread("sample_data/CheXNet/images_001/images/00000001_000.png")
    result = model.predict(img)
    print("SAI")
    print(result.shape)
    for i in range(14):
        plt.title(model.model.targets[i])
        plt.imshow(result[0][i])
        plt.show()
