from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from backbone.SimpleCNNAlign import SimpleCNNAilgn
from backbone.resnet_fedalign import resnet56_fedalign
from utils.conf import data_path
from PIL import Image
from datasets.utils.federated_dataset import FederatedDataset,partition_label_skew_loaders
from typing import Tuple
from backbone.SimpleCNN import SimpleCNN
import torchvision.transforms as T
from datasets.transforms.denormalization import DeNormalize
from torch.autograd import Variable
import torch.nn.functional as F


class MyCIFAR10(CIFAR10):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(MyCIFAR10, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class FedLeaCIFAR10(FederatedDataset):
    NAME = 'fl_cifar10'
    SETTING = 'label_skew'
    N_SAMPLES_PER_Class = None
    N_CLASS = 10
    torchvision_normalization = T.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2615))
    torchvision_denormalization = DeNormalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2615))

    Nor_TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         torchvision_normalization])

    CON_TRANSFORMS = transforms.Compose([
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                torchvision_normalization])

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(
            Variable(x.unsqueeze(0), requires_grad=False),
            (4, 4, 4, 4), mode='reflect').data.squeeze()),
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        torchvision_normalization
    ])

    def get_data_loaders(self,train_transform=None):
        pri_aug = self.args.pri_aug
        if pri_aug =='weak':
            train_transform = self.Nor_TRANSFORM
        elif pri_aug =='strong':
            train_transform = self.CON_TRANSFORM
        else:
            train_transform = self.transform_train

        train_dataset = MyCIFAR10(root=data_path(), train=True,
                        download=False, transform=train_transform)
        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])
        test_dataset = CIFAR10(data_path(), train=False,
                               download=False, transform=test_transform)
        traindls, testdl,net_cls_counts = partition_label_skew_loaders(train_dataset, test_dataset, self)
        return traindls, testdl,net_cls_counts

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), FedLeaCIFAR10.Nor_TRANSFORM])
        return transform

    @staticmethod
    def get_backbone(parti_num,names_list,model_name=''):
        nets_list = []

        if model_name == 'fedalign':
            for j in range(parti_num):
                nets_list.append(SimpleCNNAilgn(FedLeaCIFAR10.N_CLASS))
        else:
            for j in range(parti_num):
                nets_list.append(SimpleCNN(FedLeaCIFAR10.N_CLASS))

        return nets_list

    @staticmethod
    def get_normalization_transform():
        transform = FedLeaCIFAR10.torchvision_normalization
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = FedLeaCIFAR10.torchvision_denormalization
        return transform
