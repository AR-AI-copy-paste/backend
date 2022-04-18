import errno
import os
import sys
import urllib.request

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from hsh.library.hash import Hasher
from PIL import Image
from skimage import transform
from torchvision import transforms
from tqdm import tqdm
import gdown
import shutil

from . import u2DataLoader, u2Net



def load_model(model_name: str = "u2net"):
    hasher = Hasher()

    net = u2Net.U2NET(3, 1)
    path = r"./models/u2net/u2Paths/u2net.pth"

    if not (os.path.isfile(path)):
        print("Trying to download the path data")
        gdown.download("https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ&export=download", "u2net.pth", quiet=False)
        shutil.move("u2net.pth", path)

    try:
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(path))
            net.to(torch.device("cuda"))
        else:
            net.load_state_dict(
                torch.load(
                    path,
                    map_location="cpu",
                )
            )
    except FileNotFoundError:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), model_name + ".pth"
        )

    net.eval()

    return net


def norm_pred(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)

    return dn


def preprocess(image):
    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[0:2])

    if 3 == len(label_3.shape):
        label = label_3[:, :, 0]
    elif 2 == len(label_3.shape):
        label = label_3

    if 3 == len(image.shape) and 2 == len(label.shape):
        label = label[:, :, np.newaxis]
    elif 2 == len(image.shape) and 2 == len(label.shape):
        image = image[:, :, np.newaxis]
        label = label[:, :, np.newaxis]

    transform = transforms.Compose(
        [u2DataLoader.RescaleT(320), u2DataLoader.ToTensorLab(flag=0)]
    )
    sample = transform({"imidx": np.array([0]), "image": image, "label": label})

    return sample


def predict(net, item):

    sample = preprocess(item)

    with torch.no_grad():

        if torch.cuda.is_available():
            inputs_test = torch.cuda.FloatTensor(
                sample["image"].unsqueeze(0).cuda().float()
            )
        else:
            inputs_test = torch.FloatTensor(sample["image"].unsqueeze(0).float())

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        pred = d1[:, 0, :, :]
        predict = norm_pred(pred)

        predict = predict.squeeze()
        predict_np = predict.cpu().detach().numpy()
        img = Image.fromarray(predict_np * 255).convert("RGB")
        del d1, d2, d3, d4, d5, d6, d7, pred, predict, predict_np, inputs_test, sample

        return img