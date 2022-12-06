import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel
import numpy as np

class LogitNormLoss(nn.Module):

    def __init__(self, t=1.0):
        super(LogitNormLoss, self).__init__()
        self.t = t

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return F.cross_entropy(logit_norm, target)

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via FedOursNormLogExp.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class FedOurNormLogExp(FederatedModel):
    NAME = 'fedournormlogexp'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list,args, transform):
        super(FedOurNormLogExp, self).__init__(nets_list, args, transform)
        self.t = args.t
        self.w = args.w
        self.norm_dict = {}

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for net_idex,net in enumerate(self.nets_list):
            net.load_state_dict(global_w)
            self.norm_dict[net_idex]=0

    def loc_update(self,priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients,self.online_num,replace=False).tolist()
        self.online_clients = online_clients

        for i in online_clients:
            self._train_net(i,self.nets_list[i], priloader_list[i])
        self.norm_aggregate_nets()

        return  None

    def norm_aggregate_nets(self):
        global_net = self.global_net
        nets_list = self.nets_list

        online_clients = self.online_clients
        global_w = self.global_net.state_dict()

        online_clients_dl = [self.trainloaders[online_clients_index] for online_clients_index in online_clients]
        online_clients_len = [len(dl.sampler.indices) for dl in online_clients_dl]
        online_clients_all = np.sum(online_clients_len)
        data_freq_weight = online_clients_len / online_clients_all

        online_clients_norm = [self.norm_dict[online_clients_index] for online_clients_index in online_clients]
        # online_clients_norm = [torch.log(self.norm_dict[online_clients_index] for online_clients_index in online_clients)]
        online_clients_mean_norm = np.mean(online_clients_norm)

        # online_clients_norm_weight = [np.log(online_clients_mean_norm /(item * self.w)) for item in online_clients_norm]

        # online_clients_norm_weight = [online_clients_mean_norm / np.log((item * self.w)) for item in online_clients_norm]
        online_clients_norm_weight = np.array([online_clients_mean_norm / (item * self.w) for item in online_clients_norm])

        # online_clients_norm_weight = np.exp(online_clients_norm_weight) / np.sum(np.exp(online_clients_norm_weight),axis=0)

        online_clients_norm_weight = np.log(online_clients_norm_weight+1) / np.sum(np.log(online_clients_norm_weight+1),axis=0)

        # online_clients_norm_weight = (online_clients_norm_weight) / np.sum(online_clients_norm_weight,axis=0)
        online_client_weight = np.multiply(data_freq_weight, online_clients_norm_weight)

        online_client_weight = online_client_weight / np.sum(online_client_weight)

        first = True
        for index,net_id in enumerate(online_clients):
            net = nets_list[net_id]
            net_para = net.state_dict()
            # if net_id == 0:
            if first:
                first = False
                for key in net_para:
                    global_w[key] = net_para[key] * online_client_weight[index]
            else:
                for key in net_para:
                    global_w[key] += net_para[key] * online_client_weight[index]

        global_net.load_state_dict(global_w)

        for _, net in enumerate(nets_list):
            net.load_state_dict(global_net.state_dict())


    def _train_net(self,index,net,train_loader):
        net = net.to(self.device)

        if self.args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=self.local_lr, weight_decay=self.args.reg)
        elif self.args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=self.local_lr, momentum=0.9,
                                  weight_decay=self.args.reg)

        net.train()
        criterion = LogitNormLoss(t=self.t)
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
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index,loss)
                optimizer.step()

        net.eval()
        g_norm = 0
        # g_t = 0
        private_len = len(train_loader.sampler.indices)
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            outpus = net(images)
            batch_norm = torch.norm(outpus, p=2, dim=-1, keepdim=True) # batch x 1
            g_norm += torch.sum(batch_norm.detach())
        g_norm = torch.div(g_norm,private_len)
        self.norm_dict[index] = round(np.float(g_norm.cpu()),4)