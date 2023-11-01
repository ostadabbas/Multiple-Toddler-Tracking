import timm
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging
from PIL import Image
model_name = 'vit_base_patch16_224'
vit_model = timm.create_model(model_name, pretrained=True)
# detr_model = timm.create_model('detr_res50', pretrained=True)


class ViTExtractor(object):
    def __init__(self, use_cuda=True):
        self.model_name = 'vit_base_patch16_224'
        self.net = timm.create_model(self.model_name, pretrained=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.net.to(self.device)
        self.norm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        def _to_pil(im):
            return Image.fromarray(im)

        im_batch = torch.cat([self.norm(_to_pil(im)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()

class DETRExtractor(object):
    def __init__(self, use_cuda=True):
        self.model_name = 'detr_res50'
        self.net = timm.create_model(self.model_name, pretrained=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.net.to(self.device)
        self.norm = transforms.Compose([
            transforms.Resize((800, 800)),  # DETR expects 800x800 input size
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        def _to_pil(im):
            return Image.fromarray(im)

        im_batch = torch.cat([self.norm(_to_pil(im)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            # Extract features from one of the intermediate layers
            features = self.net.backbone(im_batch)['3']  # This gives features after the third block
        return features.cpu().numpy()
