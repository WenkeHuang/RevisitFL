import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel
import torch.nn.functional as F
import torch

# https://github.com/katsura-jp/fedavg.pytorch
# https://github.com/vaseline555/Federated-Averaging-PyTorch
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via Fedavg.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class FedLocal(FederatedModel):
    NAME = 'fedlocal'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list,args, transform):
        super(FedLocal, self).__init__(nets_list, args, transform)
        self.m = args.m
        self.log_weight = args.log_weight

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _,net in enumerate(self.nets_list):
            net.load_state_dict(global_w)
        self.prev_epoch_net = None

    def loc_update(self,priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients,self.online_num,replace=False).tolist()
        self.online_clients = online_clients

        for i in online_clients:
            self._train_net(i,self.nets_list[i], priloader_list[i])
        self.aggregate_nets(None)

        return  None

    def momentum_update_key_encoder(self, local_model, global_model):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(local_model.parameters(), global_model.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    def _train_net(self,index,net,train_loader):
        net_cls_counts = self.net_cls_counts[index]
        class_num = len(train_loader.dataset.classes)
        rich_class_num = int(class_num / 2)
        net_cls_counts = (sorted(net_cls_counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
        rich_class_index_list = []
        for i, (k, _) in enumerate(net_cls_counts):
            if i <= rich_class_num - 1:
                rich_class_index_list.append(k)
        poor_class_index_list = list(set([i for i in range(class_num)]).difference(set(rich_class_index_list)))

        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterionCE = nn.CrossEntropyLoss()
        criterionCE.to(self.device)
        criterionKL = nn.KLDivLoss(reduction='batchmean')
        criterionL2 = nn.MSELoss()
        criterionL1 = nn.L1Loss()
        iterator = tqdm(range(self.local_epoch))
        self.prev_epoch_net = copy.deepcopy(self.global_net.to(self.device))
        for _ in iterator:
            # self.prev_epoch_net = copy.deepcopy(net)
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                feature = net.features(images)
                outputs = net.classifier(feature)
                loss_CE = criterionCE(outputs, labels)

                rich_outputs = outputs[:, rich_class_index_list]
                poor_outputs = outputs[:, poor_class_index_list]

                # soft_outputs = F.softmax(outputs,dim=1)
                # bs,class_num = outputs.shape
                # non_targets_mask = torch.ones(bs,class_num).to(self.device).scatter_(1, labels.view(-1,1), 0)
                # nt_soft_outputs = soft_outputs[non_targets_mask.bool()].view(bs, class_num-1)
                # nt_logsoft_outputs = torch.log(nt_soft_outputs)

                with torch.no_grad():
                    prev_feature = self.prev_epoch_net.features(images)
                    prev_outputs = self.prev_epoch_net.classifier(prev_feature)
                    # soft_prev_outpus = F.softmax(prev_outputs,dim=1)
                    # nt_soft_prev_outputs = soft_prev_outpus[non_targets_mask.bool()].view(bs, class_num-1)
                    rich_prev_outputs = prev_outputs[:, rich_class_index_list]
                    poor_prev_outputs = prev_outputs[:, poor_class_index_list]
                # Feature KL
                # loss_distill = criterionKL(F.log_softmax(feature),F.softmax(prev_feature))

                # Feature l2
                loss_fea_dis = criterionL2((feature),(prev_feature))
                # loss_distill1 = criterionL1(feature,prev_feature)

                # Logits KL
                loss_log_dis = criterionKL(F.log_softmax(rich_outputs),F.softmax(rich_prev_outputs))+\
                               criterionKL(F.log_softmax(poor_outputs),F.softmax(poor_prev_outputs))
                loss_log_dis = 0.5*loss_log_dis
                loss_log_dis = self.log_weight * loss_log_dis
                # b = F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1)
                # b = -1.0 * b.sum(dim=1)
                # loss_cpr = b.mean()
                # loss_cpr = -loss_cpr


                # loss_distill = criterionKL(F.log_softmax(outputs),F.softmax(prev_outputs))
                # loss_distill = criterionKL(nt_logsoft_outputs,nt_soft_prev_outputs)
                # loss = loss_CE + loss_distill + loss_cpr
                loss = loss_CE + loss_fea_dis + loss_log_dis
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d lossCE = %0.3f lossFea = %0.3f lossLog = %0.3f" % (index,loss_CE,loss_fea_dis,loss_log_dis)
                optimizer.step()

            with torch.no_grad():  # no gradient to keys
                self.momentum_update_key_encoder(net,self.prev_epoch_net)  # update the key encoder