from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from utils.conf import data_path
from PIL import Image
from datasets.utils.federated_dataset import FederatedDataset,partition_label_skew_loaders
from datasets.utils.public_dataset import PublicDataset,random_loaders
from typing import Tuple
from backbone.ResNet import resnet50
from datasets.transforms.denormalization import DeNormalize
from backbone.ResNet import resnet10,resnet12,resnet18
from torch.utils.data import Dataset
import numpy as np
import os

class TinyImagenet(Dataset):
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data = []
        for num in range(20):
            self.data.append(np.load(os.path.join(
                root, 'processed/x_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.data = np.concatenate(np.array(self.data))

        self.targets = []
        for num in range(20):
            self.targets.append(np.load(os.path.join(
                root, 'processed/y_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.targets = np.concatenate(np.array(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(np.uint8(255 * img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)


        return img, target


class MyTinyImagenet(TinyImagenet):

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(MyTinyImagenet, self).__init__(
            root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]


        img = Image.fromarray(np.uint8(255 * img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class FedLeaTinyImagenet(FederatedDataset):
    NAME = 'fl_tiny_imagenet'
    SETTING = 'label_skew'
    N_SAMPLES_PER_Class = None
    N_CLASS = 200
    Nor_TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5),
                              (0.5, 0.5, 0.5))])

    def get_data_loaders(self, train_transform=None):
        if not train_transform:
            train_transform = self.Nor_TRANSFORM

        train_dataset = MyTinyImagenet(data_path()+'TINYIMG', train=True,
                                  download=False, transform=train_transform)

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        test_dataset = MyTinyImagenet(data_path()+'TINYIMG', train=False,
                                  download=False, transform=test_transform)

        traindls, testdl, net_cls_counts = partition_label_skew_loaders(train_dataset, test_dataset, self)

        return traindls, testdl, net_cls_counts


    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), FedLeaTinyImagenet.Nor_TRANSFORM])
        return transform

    @staticmethod
    def get_backbone(parti_num, names_list, model_name=''):
        nets_list = []
        for j in range(parti_num):
            nets_list.append(resnet50(num_classes=FedLeaTinyImagenet.N_CLASS, name='resnet50'))
        return nets_list

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        return transform