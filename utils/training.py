import torch
from argparse import Namespace
from models.utils.federated_model import FederatedModel
from datasets.utils.federated_dataset import FederatedDataset
from typing import Tuple
from torch.utils.data import DataLoader
from utils.logger import CsvWriter

def global_evaluate(model: FederatedModel, test_dl: DataLoader, setting: str,name: str) -> Tuple[list, list]:
    accs = 0
    dl = test_dl
    net = model.global_net
    status = net.training
    net.eval()
    correct, total,top1,top5 = 0.0, 0.0,0.0,0.0
    for batch_idx, (images, labels) in enumerate(dl):
        with torch.no_grad():
            images, labels = images.to(model.device), labels.to(model.device)
            outputs = net(images)
            _, max5 = torch.topk(outputs, 5, dim=-1)
            labels = labels.view(-1, 1)
            top1 += (labels == max5[:, 0:1]).sum().item()
            top5 += (labels == max5).sum().item()
            total += labels.size(0)
    top1acc= round(100 * top1 / total,2)
    top5acc= round(100 * top5 / total,2)
    if name in ['fl_cifar10']:
        accs = top1acc
    elif name in ['fl_cifar100']:
        accs = top1acc
    net.train(status)
    return accs

def train(model: FederatedModel, private_dataset: FederatedDataset,
          args: Namespace) -> None:
    if args.csv_log:
        csv_writer = CsvWriter(args, private_dataset)


    pri_train_loaders, test_loaders,net_cls_counts = private_dataset.get_data_loaders()
    model.trainloaders = pri_train_loaders
    model.testlodaers = test_loaders
    model.net_cls_counts =net_cls_counts

    if hasattr(model, 'ini'):
        model.ini()

    accs_list = []

    Epoch  = args.communication_epoch
    option_learning_decay = args.learning_decay
    for epoch_index in range(Epoch):
        model.epoch_index = epoch_index
        if hasattr(model, 'loc_update'):
            epoch_loc_loss_dict = model.loc_update(pri_train_loaders)
            # if epoch_loc_loss_dict !=None:
            #     loc_loss_dict[epoch_index] = epoch_loc_loss_dict
            # 这里先去掉降低lr
            if option_learning_decay ==True:
                model.local_lr = args.local_lr * (1 - epoch_index/Epoch * 0.9)


        accs = global_evaluate(model, test_loaders, private_dataset.SETTING,private_dataset.NAME)

        # if epoch_index %10 == 0:
        #     save_networks(model,epoch_index)
        # if epoch_index ==args.communication_epoch-1:
        #     save_networks(model,epoch_index)
        # save_networks(model,epoch_index)

        accs = round(accs,3)
        accs_list.append(accs)


        print('The '+str(epoch_index)+' Communcation Accuracy:'+str(accs) + 'Method:'+model.args.model)



    if args.csv_log:
        csv_writer.write_acc(accs_list)
