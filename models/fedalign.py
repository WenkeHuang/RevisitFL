import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import copy
from models.utils.federated_model import FederatedModel


# Local Learning Matters: Rethinking Data Heterogeneity in Federated Learning
# CVPR 2022

class FedAlign(FederatedModel):
    NAME = 'fedalign'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedAlign, self).__init__(nets_list, args, transform)
        self.width_range = [0.25, 1.0]

    def transmitting_matrix(self, fm1, fm2):
        if fm1.size(2) > fm2.size(2):
            fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))

        fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
        fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1, 2)

        fsp = torch.bmm(fm1, fm2) / fm1.size(2)
        return fsp

    def top_eigenvalue(self, K, n_power_iterations=10, dim=1):
        v = torch.ones(K.shape[0], K.shape[1], 1).to(self.device)
        for _ in range(n_power_iterations):
            m = torch.bmm(K, v)
            n = torch.norm(m, dim=1).unsqueeze(1)
            v = m / n

        top_eigenvalue = torch.sqrt(n / torch.norm(v, dim=1).unsqueeze(1))
        return top_eigenvalue

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
        net.train()
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

                # outputs = net(images)
                # features = net.features(images)

                net.apply(lambda m: setattr(m, 'width_mult', self.width_range[-1]))
                t_feats, t_out = net.extract_feature(images)
                loss = criterion(t_out, labels)
                loss.backward()
                loss_CE = loss.item()
                net.apply(lambda m: setattr(m, 'width_mult', self.width_range[0]))
                s_feats = net.reuse_feature(t_feats[-2].detach())

                TM_s = torch.bmm(self.transmitting_matrix(s_feats[-2], s_feats[-1]), self.transmitting_matrix(s_feats[-2], s_feats[-1]).transpose(2, 1))
                TM_t = torch.bmm(self.transmitting_matrix(t_feats[-2].detach(), t_feats[-1].detach()),
                                 self.transmitting_matrix(t_feats[-2].detach(), t_feats[-1].detach()).transpose(2, 1))
                loss = F.mse_loss(self.top_eigenvalue(K=TM_s), self.top_eigenvalue(K=TM_t))
                loss = self.args.mu * (loss_CE / loss.item()) * loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                optimizer.step()
