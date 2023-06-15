from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms

from backbone.ResNet import resnet50
from backbone.resnet_fedalign import resnet50_fedalign
from utils.conf import data_path
from PIL import Image
from datasets.utils.federated_dataset import FederatedDataset, partition_label_skew_loaders
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize

class MyCifar100(CIFAR100):

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(MyCifar100, self).__init__(
            root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

class FedLeaCIFAR100(FederatedDataset):
    NAME = 'fl_cifar100'
    SETTING = 'label_skew'
    N_SAMPLES_PER_Class = None
    N_CLASS = 100
    Nor_TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                              (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])


    def get_data_loaders(self, train_transform=None):
        if not train_transform:
            train_transform = self.Nor_TRANSFORM
        train_dataset = CIFAR100(root=data_path(), train=True,
                                 download=False, transform=train_transform)

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        test_dataset = CIFAR100(data_path(), train=False,
                                download=False, transform=test_transform)

        traindls, testdl, net_cls_counts = partition_label_skew_loaders(train_dataset, test_dataset, self)
        return traindls, testdl, net_cls_counts

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), FedLeaCIFAR100.Nor_TRANSFORM])
        return transform

    @staticmethod
    def get_backbone(parti_num, names_list, model_name=''):
        nets_list = []
        if model_name == 'moon':
            for j in range(parti_num):
                nets_list.append(resnet50(num_classes=FedLeaCIFAR100.N_CLASS, name='resnet50'))
        elif model_name == 'fedalign':
            for j in range(parti_num):
                nets_list.append(resnet50_fedalign(class_num=FedLeaCIFAR100.N_CLASS, name='resnet50'))
        else:
            for j in range(parti_num):
                nets_list.append(resnet50(num_classes=FedLeaCIFAR100.N_CLASS, name='resnet50'))
        return nets_list

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                              (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                              (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        return transform
