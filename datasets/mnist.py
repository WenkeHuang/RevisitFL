import torch

import torchvision.transforms as transforms

from backbone.SimpleCNN import SimpleCNN

from utils.conf import data_path
from PIL import Image
from datasets.utils.federated_dataset import FederatedDataset, partition_label_skew_loaders
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
from torchvision.datasets import MNIST


class MyMNIST(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False, data_name=None) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.dataset = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        return dataobj

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        img, target = self.dataset[index]

        return img, target


class FedMNIST(FederatedDataset):
    NAME = 'fl_mnist'
    SETTING = 'label_skew'
    N_SAMPLES_PER_Class = None
    N_CLASS = 10
    Singel_Channel_Nor_TRANSFORM = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.1307, 0.1307, 0.1307),
                                 (0.3081, 0.3081, 0.3081))])

    def get_data_loaders(self, train_transform=None):

        pri_aug = self.args.pri_aug
        if pri_aug == 'weak':
            train_transform = self.Singel_Channel_Nor_TRANSFORM

        train_dataset = MyMNIST(root=data_path(), train=True,
                                download=False, transform=train_transform)
        test_transform = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor(),
             transforms.Lambda(lambda x: x.repeat(3, 1, 1)), self.get_normalization_transform()])
        test_dataset = MyMNIST(data_path(), train=False,
                               download=False, transform=test_transform)
        traindls, testdl, net_cls_counts = partition_label_skew_loaders(train_dataset, test_dataset, self)
        return traindls, testdl, net_cls_counts

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), FedMNIST.Singel_Channel_Nor_TRANSFORM])
        return transform

    @staticmethod
    def get_backbone(parti_num, names_list, model_name=''):
        nets_list = []

        for j in range(parti_num):
            nets_list.append(SimpleCNN(FedMNIST.N_CLASS))

        return nets_list

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(mean=(0.1307, 0.1307, 0.1307),
                                         std=(0.3081, 0.3081, 0.3081))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(mean=(0.1307, 0.1307, 0.1307),
                                std=(0.3081, 0.3081, 0.3081))
        return transform
