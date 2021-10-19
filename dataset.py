"""
Tiny-ImageNet 200 classes
train and test dataset

MicroImageNet classification challenge is similar to the classification challenge in the full ImageNet ILSVRC.
MicroImageNet contains 200 classes for training. Each class has 500 images. The test set contains 10,000 images.
All images are 64x64 colored ones. Objective is to classify the 10,000 test set as accurately as possible.
https://www.kaggle.com/c/tiny-imagenet/overview
"""
import os
import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pathlib import Path
import urllib.request
import shutil
import zipfile


class InverseNormalize:
    def __init__(self, mean, std):
        self.mean = torch.Tensor(mean)[None, :, None, None]
        self.std = torch.Tensor(std)[None, :, None, None]

    def __call__(self, sample):
        return (sample * self.std) + self.mean

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


class TinyImagenet200(Dataset):
    """Tiny imagenet 200 dataloader"""

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    dataset = None

    def __init__(self, root="./data", *args, train=True, download=False, **kwargs):
        super().__init__()

        if download:
            self.download(root=root)
        dataset = _TinyImagenet200Train if train else _TinyImagenet200Val
        self.root = root
        self.dataset = dataset(root, *args, **kwargs)
        self.classes = self.dataset.classes
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    @staticmethod
    def transform_val_inverse():
        return InverseNormalize(
            [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
        )

    def download(self, root="./"):
        """Download and unzip Imagenet200 files in the `root` directory."""
        dir = os.path.join(root, "tiny-imagenet-200")
        dir_train = os.path.join(dir, "train")
        if os.path.exists(dir) and os.path.exists(dir_train):
            print('Files already downloaded and verified')
            return

        path = Path(os.path.join(root, "tiny-imagenet-200.zip"))
        if not os.path.exists(path):
            os.makedirs(path.parent, exist_ok=True)

            print("==> Downloading TinyImagenet200...")
            with urllib.request.urlopen(self.url) as response, open(
                str(path), "wb"
            ) as out_file:
                shutil.copyfileobj(response, out_file)

        print("==> Extracting TinyImagenet200...")
        with zipfile.ZipFile(str(path)) as zf:
            zf.extractall(root)

    def __getitem__(self, i):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)


class _TinyImagenet200Train(datasets.ImageFolder):
    def __init__(self, root="./data", *args, **kwargs):
        super().__init__(os.path.join(root, "tiny-imagenet-200/train"), *args, **kwargs)


class _TinyImagenet200Val(datasets.ImageFolder):
    def __init__(self, root="./data", *args, **kwargs):
        super().__init__(os.path.join(root, "tiny-imagenet-200/val"), *args, **kwargs)

        self.path_to_class = {}
        with open(os.path.join(self.root, "val_annotations.txt")) as f:
            for line in f.readlines():
                parts = line.split()
                path = os.path.join(self.root, "images", parts[0])
                self.path_to_class[path] = parts[1]

        self.classes = list(sorted(set(self.path_to_class.values())))
        self.class_to_idx = {label: self.classes.index(label) for label in self.classes}

    def __getitem__(self, i):
        sample, _ = super().__getitem__(i)
        path, _ = self.samples[i]
        label = self.path_to_class[path]
        target = self.class_to_idx[label]
        return sample, target

    def __len__(self):
        return super().__len__()
