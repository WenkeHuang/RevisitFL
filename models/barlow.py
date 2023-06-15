import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel
import torch


# https://github.com/QinbinLi/MOON


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via Barlow.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser



class Barlow(FederatedModel):
    NAME = 'barlow'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(Barlow, self).__init__(nets_list, args, transform)
        self.lambd = args.lambd

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])
        self.aggregate_nets(None)
        return None

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        self.global_net = self.global_net.to(self.device)
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                f = net.features(images)
                z = net.classifier(f)

                _,feat_dim = f.shape
                with torch.no_grad():
                    g_f = self.global_net.features(images)
                    g_z = self.global_net.classifier(g_f)

                loss_Feat_Barlow = self.barlow_loss(f,g_f,feat_dim)
                loss_Feat_Barlow = loss_Feat_Barlow/feat_dim

                _,logtis_dim = z.shape
                loss_Logits_Barlow = self.barlow_loss(z,g_z,logtis_dim)
                loss_Logits_Barlow = loss_Logits_Barlow/logtis_dim

                lossCE = criterion(z, labels)

                loss = lossCE + loss_Feat_Barlow + loss_Logits_Barlow
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d CE = %0.3f,FeBa = %0.3f,LoBa = %0.3f" % (index, lossCE, loss_Feat_Barlow,loss_Logits_Barlow)
                optimizer.step()

    def off_diagonal(self,x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def barlow_loss(self,z1, z2,dim):
        bn = nn.BatchNorm1d(dim, affine=False).to(self.device)
        # empirical cross-correlation matrix
        c = bn(z1).T @ bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.local_batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss
