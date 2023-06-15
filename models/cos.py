import torch.optim as optim
import torch.nn as nn
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


class COS(FederatedModel):
    NAME = 'cos'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(COS, self).__init__(nets_list, args, transform)
        self.prev_nets_list = []


        self.grad_dims = []
        self.grad_scale = args.grad_scale
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
            param_k.data = param_k.data * 0.999 + param_q.data * (1. - 0.999)

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        for i in online_clients:
            self._train_net(i,self.nets_list[i], priloader_list[i])
        self.aggregate_nets(None)
        return None

    def _train_net(self,index, net, train_loader):
        net = net.to(self.device)
        global_net = copy.deepcopy(self.global_net.to(self.device))
        if self.args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=self.local_lr, weight_decay=self.args.reg)
            global_optimizer = optim.Adam(filter(lambda p: p.requires_grad, global_net.parameters()), lr=self.local_lr, weight_decay=self.args.reg)
        elif self.args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=self.local_lr, momentum=0.9,
                                  weight_decay=self.args.reg)
            global_optimizer = optim.SGD(filter(lambda p: p.requires_grad, global_net.parameters()), lr=self.local_lr, momentum=0.9,
                                  weight_decay=self.args.reg)
        # optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        # global_optimizer = optim.SGD(global_net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)

        criterionCE = nn.CrossEntropyLoss()
        criterionCE.to(self.device)
        criterionKL = nn.KLDivLoss(reduction='batchmean')
        criterionKL.to(self.device)

        iterator = tqdm(range(self.local_epoch))

        cos = torch.nn.CosineSimilarity(dim=-1)
        # target_cos = round(self.epoch_index/self.args.communication_epoch,2)
        # target_cos = torch.tensor(target_cos).to(self.device)

        nominal_epoch = self.args.communication_epoch-1
        residual_epoch = nominal_epoch-self.epoch_index
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
                penatly = criterionCE(global_outputs,labels)
                penatly.backward()
                store_grad(global_net.parameters, self.grads_global, self.grad_dims)

                # Calculate the Local Net Gradient on Private Data
                optimizer.zero_grad()
                local_outputs = net(images)
                loss = criterionCE(local_outputs,labels)
                loss.backward(retain_graph=True)
                store_grad(net.parameters, self.grads_local, self.grad_dims)

                # gradient_cos = cos(self.grads_local, self.grads_global)
                # if gradient_cos <=target_cos:
                #     # g_tilde = project(gxy=self.grads_local, ger=self.grads_global)
                #     g_tilde = project(gxy=self.grads_global, ger=self.grads_local)
                #     overwrite_grad(net.parameters, g_tilde, self.grad_dims)
                # else:
                #     overwrite_grad(net.parameters, self.grads_local, self.grad_dims)
                #
                # optimizer.step()
                # for name, parms in net.named_parameters():
                #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
                #           ' -->grad_value:', parms.grad)
                #     break
                # 从整体参数的梯度层面进行优化
                # loss_disssim = 0
                # self.grads_local = self.grads_local.requires_grad_()
                # grad_lr = self.local_lr * self.grad_scale
                # optimizer_grad = optim.SGD([self.grads_local], lr=grad_lr, momentum=0.9, weight_decay=1e-5)
                # optimizer_grad.zero_grad()
                # gradient_cos = cos(self.grads_local, self.grads_global)
                # loss_disssim = torch.nn.functional.l1_loss(gradient_cos, target_cos, reduction='sum')
                # loss_disssim.backward()
                # optimizer_grad.step()
                # overwrite_grad(net.parameters, self.grads_local, self.grad_dims)
                # for name, parms in net.named_parameters():
                #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
                #           ' -->grad_value:', parms.grad)
                #     break
                #
                # self.grads_local = torch.zeros(np.sum(self.grad_dims)).to(self.device)

                # 从整体参数的梯度层面进行优化
                loss_disssim = 0
                self.grads_local = self.grads_local.requires_grad_()
                gradient_cos = cos(self.grads_local, self.grads_global)
                if gradient_cos <= target_cos:
                    grad_lr = self.local_lr * self.grad_scale
                    if self.args.optimizer == 'adam':
                        optimizer_grad = optim.Adam([self.grads_local], lr=grad_lr,
                                               weight_decay=self.args.reg)
                    elif self.args.optimizer == 'sgd':
                        optimizer_grad = optim.SGD([self.grads_local], lr=grad_lr,
                                              momentum=0.9,
                                              weight_decay=self.args.reg)
                    # optimizer_grad = optim.SGD([self.grads_local], lr=grad_lr, momentum=0.9, weight_decay=1e-5)
                    loss_disssim = torch.nn.functional.l1_loss(gradient_cos, target_cos, reduction='sum')
                    loss_disssim = loss_disssim
                    loss_disssim.backward()
                    optimizer_grad.step()
                    overwrite_grad(net.parameters, self.grads_local, self.grad_dims)
                    optimizer_grad.zero_grad()

                iterator.desc = "Local Pariticipant %d CE = %0.3f,Dissim = %0.3f" % (index, loss, loss_disssim)
                optimizer.step()

                self.grads_local = torch.zeros(np.sum(self.grad_dims)).to(self.device)

            # with torch.no_grad():  # no gradient to keys
            #     self.momentum_update_key_encoder(net, global_net)  # update the key encoder