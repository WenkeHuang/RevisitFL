import torch.optim as optim
import torch.nn as nn
from torch.distributions import Bernoulli
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel

# https://github.com/katsura-jp/fedavg.pytorch
# https://github.com/vaseline555/Federated-Averaging-PyTorch

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via FedReg.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class FedReg(FederatedModel):
    NAME = 'fedreg'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list,args, transform):
        super(FedReg, self).__init__(nets_list,args,transform)
        self.reserve_p = args.reserve_p

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _,net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self,priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients,self.online_num,replace=False).tolist()
        self.online_clients = online_clients

        for i in online_clients:
            self._train_net(i,self.nets_list[i], priloader_list[i])

        self.aggregate_nets(None)

        return  None

    def _train_net(self,index,net,train_loader):
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
                outputs = net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()

                for p in net.parameters():
                    if p.grad is None:
                        continue
                    grad = p.grad.data

                    reserve_p = self.reserve_p  # the ratio of gradients that are reserved.

                    grad_mask = Bernoulli(grad.new_full(size=grad.size(), fill_value=reserve_p))
                    grad *= grad_mask.sample() / reserve_p

                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index,loss)
                optimizer.step()

