import torch.optim as optim
import torch.nn as nn
from torch.distributions import Bernoulli
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel
import torch
import numpy as np
import torch.nn.functional as F


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via OURS.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


def store_grad(params, grads, grad_dims):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
    """
    # store the gradients
    grads.fill_(0.0)
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[:count + 1])
            grads[begin: end].copy_(param.grad.data.view(-1))
        count += 1


def overwrite_grad(params, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = sum(grad_dims[:count + 1])
            this_grad = newgrad[begin: end].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        count += 1


def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger


class Ours(FederatedModel):
    NAME = 'ours'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(Ours, self).__init__(nets_list, args, transform)
        self.prev_nets_list = []

        self.m = args.m
        self.reserve_p = args.reserve_p
        # self.log_weight = args.log_weight
        self.grad_dims = []
        self.grads_global = None
        self.grads_local = None

    def ini(self):
        for j in range(self.args.parti_num):
            self.prev_nets_list.append(copy.deepcopy(self.nets_list[j]))
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

        # Allocate temporary synaptic memory
        for pp in self.global_net.parameters():
            self.grad_dims.append(pp.data.numel())

        self.grads_global = torch.zeros(np.sum(self.grad_dims)).to(self.device)
        self.grads_local = torch.zeros(np.sum(self.grad_dims)).to(self.device)

    def momentum_update_key_encoder(self, local_model, global_model):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(local_model.parameters(), global_model.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

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

        global_net = copy.deepcopy(self.global_net.to(self.device))  # 复制出一个新的

        global_optimizer = optim.SGD(global_net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)

        criterionCE = nn.CrossEntropyLoss()
        criterionCE.to(self.device)
        criterionKL = nn.KLDivLoss(reduction='batchmean')
        criterionKL.to(self.device)
        criterionL2 = nn.MSELoss()
        criterionL2.to(self.device)

        iterator = tqdm(range(self.local_epoch))

        cos = torch.nn.CosineSimilarity(dim=-1)

        nominal_epoch = self.args.communication_epoch - 1
        residual_epoch = nominal_epoch - self.epoch_index
        # target_cos = round(self.epoch_index / self.args.communication_epoch, 2)
        target_cos = round(residual_epoch / nominal_epoch, 2)
        target_cos = torch.tensor(target_cos).to(self.device)

        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Calculate the Global Net Gradient on Private Data
                global_optimizer.zero_grad()
                global_outputs = global_net(images)
                penatly = criterionCE(global_outputs, labels)
                penatly.backward()
                store_grad(global_net.parameters, self.grads_global, self.grad_dims)

                # Calculate the Local Net Gradient on Private Data
                optimizer.zero_grad()
                local_feature = net.features(images)
                local_outputs = net.classifier(local_feature)
                loss = criterionCE(local_outputs, labels)
                loss.backward(retain_graph=True)
                store_grad(net.parameters, self.grads_local, self.grad_dims)

                # 从整体参数的梯度层面进行优化
                self.grads_local = self.grads_local.requires_grad_()
                grad_mask = Bernoulli(self.grads_local.new_full(size=self.grads_local.size(),
                                                                fill_value=1-self.reserve_p)).sample()
                grads_local_old=copy.deepcopy(self.grads_local)
                gradient_cos = cos(self.grads_local, self.grads_global)
                # if True:
                if gradient_cos <= target_cos:

                    optimizer_grad = optim.SGD([self.grads_local], lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
                    loss_disssim = torch.nn.functional.l1_loss(gradient_cos, target_cos, reduction='sum')
                    loss_disssim = loss_disssim
                    loss_disssim.backward()

                    optimizer_grad.step()
                    grads_local_new=grad_mask*self.grads_local+grads_local_old*(1-grad_mask)
                    overwrite_grad(net.parameters, grads_local_new, self.grad_dims)
                    optimizer_grad.zero_grad()

                # for p in net.parameters():
                #     if p.grad is None:
                #         continue
                #     grad = p.grad.data
                #
                #     reserve_p = self.reserve_p
                #     grad_mask = Bernoulli(grad.new_full(size=grad.size(), fill_value=reserve_p))
                #     grad *= grad_mask.sample() / reserve_p

                # iterator.desc = "Local Pariticipant %d lossCE = %0.3f lossFea = %0.3f lossLog = %0.3f" % (index,loss)
                iterator.desc = "Local Pariticipant %d lossCE = %0.3f" % (index, loss)
                optimizer.step()
                self.grads_local = torch.zeros(np.sum(self.grad_dims)).to(self.device)

            # with torch.no_grad():  # no gradient to keys
            #     self.momentum_update_key_encoder(net, global_net)  # update the key encoder
