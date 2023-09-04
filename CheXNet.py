import os
import re
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
        #self.activation = {}
        #def get_activation(name):
        #    def hook(model, input, output):
        #        self.activation[name] = output.detach()
        #    return hook
        #self.model.densenet121.features.denseblock4.denselayer16.conv2.register_forward_hook(get_activation('denseblock4.denselayer16.conv2'))

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

        train_nodes, eval_nodes = get_graph_node_names(torchvision.models.densenet121(pretrained=True))
        print(train_nodes)
        print(eval_nodes)

    def returnCAM(self, feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 256x256
        size_upsample = (256, 256)
        bz, nc, h, w = feature_conv.shape
        print(weight_softmax.shape)
        print(nc, h*w)
        output_cam = []
        for idx in class_idx:
            cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample))
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
        #return pred, self.activation['denseblock4.denselayer16.conv2']
        print(np.shape(self.features_blobs[0]))
        return pred, self.returnCAM(self.features_blobs[0], self.weight_softmax, [idx[0]])


if __name__ == '__main__':
    ckpt_path = 'model.pth.tar'
    model = CheXNet(ckpt_path)

    image = Image.open('chest-example1.png')

    pred, cam = model.predict(image)
    
    print(type(cam.data.cpu().numpy()[0][0]))
    print(type(pred.data.cpu().numpy()[0][0]))
    print(np.shape(pred.data.cpu().numpy()))
    print(np.shape(cam.data.cpu().numpy()))
    print(np.shape(cam.data.cpu().numpy()[:, :, 0, 0]))

    
    cv.imshow('image', np.array(image))
    cv.imshow('pred', pred.data.cpu().numpy())
    cv.imshow('cam', cam.data.cpu().numpy()[:, :, 0, 0])

    print(pred.data.cpu().numpy().argmax())
    cv.waitKey(0)
    cv.destroyAllWindows()