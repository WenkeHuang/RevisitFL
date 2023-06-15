import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel
import torch
import numpy as np


# https://github.com/katsura-jp/fedavg.pytorch
# https://github.com/vaseline555/Federated-Averaging-PyTorch
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via Fedavg.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class FedAvG(FederatedModel):
    NAME = 'fedavg'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list,args, transform):
        super(FedAvG, self).__init__(nets_list,args,transform)

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
        # self._classifier_weight()
        self.aggregate_nets(None)
        return  None

    def _train_net(self,index,net,train_loader):
        net = net.to(self.device)
        net.train()
        # if self.args.optimizer == 'adam':
        #     optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=self.local_lr, weight_decay=self.args.reg)
        # elif self.args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9,
                              weight_decay=self.args.reg)
        # optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                # if len(images)!=1:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index,loss)
                optimizer.step()
        # all_outputs = None
        # all_labels = None
        # with torch.no_grad():
        #     for batch_idx, (images, labels) in enumerate(train_loader):
        #         if len(images)!=1:
        #             images = images.to(self.device)
        #             labels = labels.to(self.device)
        #             outputs = net(images)
        #             if all_outputs == None:
        #                 all_outputs = outputs
        #                 all_labels = labels
        #             else:
        #                 all_outputs= torch.cat((all_outputs,outputs),dim=0)
        #                 all_labels= torch.cat((all_labels,labels),dim=0)
        #     # print(criterion(all_outputs, all_labels))
        #     self._norm_evaluate(net,all_outputs, all_labels)
        # print('Wenke')

    def _norm_evaluate(self,net,outputs,labels):
        norms = torch.norm(outputs, p=2, dim=-1, keepdim=True) + 1e-7
        labels_set = torch.unique(labels)
        index_dict = {}
        for _, unique_label in enumerate(labels_set):
            # print(unique_label)
            index_list = [index for (index, value) in enumerate(labels) if value == unique_label.item()]
            # print(index_list)
            index_dict[unique_label.item()] = index_list

        weight_dict = {}
        # cls_fc_weight = net.l3.weight.detach().cpu().numpy()
        cls_fc_weight = net.fc.weight.detach().cpu().numpy()
        # cls_weight = torch.norm(net.fc.weight.detach(), 2, 1)
        cls_fc_weight_norm = np.linalg.norm(cls_fc_weight, axis=1)

        # for _, unique_label in enumerate(labels_set):
        #     class_weight = net.l3.weight[unique_label, :]
        #     class_mean_weight = torch.mean(class_weight)
        #     weight_dict[unique_label.item()] = round(class_mean_weight.item(),3)
        # sum_len = 0
        # sum_scale = 0
        # for _, unique_label in enumerate(labels_set):
        #     norms_list = norms[index_dict[unique_label.item()]]
        #     norms_mean = torch.mean(norms_list)
        #     scale = len(norms_list)
        #     print('Label:'+str(unique_label.item())+' '+'Length'+str(round(norms_mean.item(),2))+' '+'Number'+str(scale))
        #     sum_len +=norms_mean.item() *scale
        #     sum_scale +=scale
        # print(sum_len/sum_scale)

        for clas_index in range(len(cls_fc_weight_norm)):
            weight = cls_fc_weight_norm[clas_index]
            scale = len(index_dict[clas_index])
            print('Label:'+str(clas_index)+' '+'Weight'+str(round(weight,2))+'Number'+str(scale))

    def _classifier_weight(self):
        weight_dict = {}
        nets_list = self.nets_list
        for net_index, net in enumerate(nets_list):
            # cls_fc_weight = net.l3.weight.detach().cpu().numpy()
            cls_fc_weight = net.fc.weight.detach().cpu().numpy()
            cls_fc_weight_norm = np.linalg.norm(cls_fc_weight, axis=1)
            each_weight = []
            for clas_index in range(len(cls_fc_weight_norm)):
                weight = cls_fc_weight_norm[clas_index]
                each_weight.append(weight)
                # print('Label:' + str(clas_index) + ' ' + 'Weight' + str(round(weight, 2)))
            weight_dict[net_index]=each_weight

        path = str(self.args.beta)+'classifier.csv'
        with open(path, 'w') as f:
            for k1 in weight_dict:
                data = weight_dict[k1]
                out_str = ''
                for k2 in range(len(data)):
                    out_str += str(data[k2]) + ','
                out_str += '\n'
                f.write(out_str)


        # for _, unique_label in enumerate(labels_set):
        #     class_weight = net.l3.weight[unique_label.item(), :]
        #     class_mean_weight = torch.mean(class_weight)
        #     weight_dict[unique_label.item()] = round(class_mean_weight.item(),3)

