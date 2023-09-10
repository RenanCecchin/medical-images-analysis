import os
import re
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.feature_extraction import get_graph_node_names
from torch.utils.data import DataLoader
from PIL import Image
import cv2 as cv

CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

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

class CheXNet():
    def __init__(self, ckpt_path):
        cudnn.benchmark = True
        self.model = DenseNet121(14).cuda() 
        # hook the feature extractor
        self.features_blobs = []
        def hook_feature(module, input, output):
            self.features_blobs.append(output.data.cpu().numpy())

        self.model.densenet121._modules.get('features').register_forward_hook(hook_feature)

        # get the softmax weight
        params = list(self.model.parameters())
        self.weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

        self.model = torch.nn.DataParallel(self.model).cuda()

        if os.path.isfile(ckpt_path):
            print("=> loading checkpoint '{}'".format(ckpt_path))
            checkpoint = torch.load(ckpt_path)

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
        self.transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.TenCrop(224),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                        ])
    

    def get_class(self, pred):
        return CLASS_NAMES[pred] 

    def returnCAM(self, feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 256x256
        size_upsample = (256, 256)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:
            feature_conv_flat = feature_conv.reshape((nc, bz*h*w))
            cam = weight_softmax[idx].dot(feature_conv_flat)
            # Slice a 7x7 matrix
            cam = cam[0:49]
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv.resize(cam_img, size_upsample))
        return output_cam   
    
    def predict(self, image):
        """
        Args:
            image: a PIL image
        """
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        with torch.no_grad():
            pred = self.model(image)
        
        h_x = F.softmax(pred, dim = 1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.cpu().numpy()
        idx = idx.cpu().numpy()
        cam = self.returnCAM(self.features_blobs[0], self.weight_softmax, idx[0])
        
        return idx[0], cam


if __name__ == '__main__':
    ckpt_path = 'model.pth.tar'
    model = CheXNet(ckpt_path)

    image = Image.open('chest-example1.png')

    pred, cams = model.predict(image)
    
    #print(type(cam.data.cpu().numpy()[0][0]))
    #print(type(pred.data.cpu().numpy()[0][0]))
    #print(np.shape(pred.data.cpu().numpy()))
    #print(np.shape(cam.data.cpu().numpy()))
    #print(np.shape(cam.data.cpu().numpy()[:, :, 0, 0]))

    image = np.array(image)
    img = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    cv.imshow('original', image)
    cv.imshow('image', img)
    i = 0
    for cam in cams:
        print("A")
        #cv.imshow('pred', pred.data.cpu().numpy())
        height, width, _ = img.shape
        heatmap = cv.applyColorMap(cv.resize(cam,(width, height)), cv.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        result = result.astype(np.uint8)
        cv.imshow('CAM' + str(i), result)
        i += 1


    print(pred.data.cpu().numpy().argmax())
    cv.waitKey(0)
    cv.destroyAllWindows()