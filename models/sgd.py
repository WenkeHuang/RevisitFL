from utils.args import *
from models.utils.federated_model import FederatedModel
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated Learning via SGD.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class Sgd(FederatedModel):
    NAME = 'sgd'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(Sgd, self).__init__(nets_list, args, transform)
        self.local_epoch = args.local_epoch
        self.local_lr = args.local_lr

    def loc_update(self, priloader_list):
        for i in range(self.args.parti_num):
            self._train_net(i, self.nets_list[i], priloader_list[i])
        self.global_net = self.nets_list[0]

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        if self.args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=self.local_lr, weight_decay=self.args.reg)
        elif self.args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=self.local_lr, momentum=0.9,
                                  weight_decay=self.args.reg)
        # optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                optimizer.step()
