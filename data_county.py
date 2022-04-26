import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
import PIL.Image


class Dataset_County(Dataset):
    def __init__(self, filename, img_dir, transform=None, target_transform=None, multi=False, multi_label=None):
        self.img_labels = filename
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.multi = multi
        self.multi_label = multi_label

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = PIL.Image.open(img_path)
        if self.multi:
            label = self.multi_label[idx]
            label = label.type(torch.LongTensor)
            if label > 1:
                bin_label = 1
            else:
                bin_label = 0
        else:
            bin_label = -1
            label = img_name.split("_")[0]
            if "notwindmill" in label:
                label = 1
            else:
                label = 0
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, bin_label
