from unicodedata import decimal
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from datasets.cifar10 import MyCIFAR10
from utils.args import *
from models.utils.federated_model import FederatedModel
from utils.util import *
# from datasets.ImbalanceCIFAR import *
# from datasets.testagnosticdataloader import *
import numpy as np
# https://github.com/katsura-jp/fedavg.pytorch
# https://github.com/vaseline555/Federated-Averaging-PyTorch
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via Fedavg.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class FedCL(FederatedModel):
    NAME = 'fedcl'
    COMPATIBILITY = ['homogeneity']
    def __init__(self, nets_list,args, transform):
        super(FedCL, self).__init__(nets_list,args,transform)

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _,net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self,priloader_list):
        # self.aggregate_nets(None)
        train_epoch = self._train_net
        if self.args.feat_regular == 1:
            train_epoch = self._train_with_regular
        for i in range(self.args.parti_num):
            train_epoch(i,self.nets_list[i], priloader_list[i])
        # 是否使用cl学习聚合权重
        if self.args.cl_aggregate == 1:
            ws = self.test_train_validate()
            self.aggregate_nets(ws)
        else:
            self.aggregate_nets(None)

        return  None

    def _train_net(self,index,net,train_loader):
        net = net.to(self.device)
        net.train()
        optimizer = optim.Adam(net.parameters(), lr=self.local_lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        for epoch in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index,loss)
                optimizer.step()

    def _train_with_regular(self,index,net,train_loader):
        net = net.to(self.device)
        global_net = self.global_net.to(self.device)
        global_net.train()
        net.train()
        optimizer = optim.Adam(net.parameters(), lr=self.local_lr, weight_decay=1e-5)
        ce = nn.CrossEntropyLoss()
        cos = nn.CosineEmbeddingLoss()
        ce.to(self.device)
        cos.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        for epoch in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                client_features = net.features(images)
                client_pred = net.classifier(client_features)
                client_features = F.normalize(client_features, dim=1)
                global_features = global_net.features(images)
                celoss = ce(client_pred,labels)
                cosloss = cos(client_features,global_features,torch.ones(labels.shape[0]).to(self.device))
                loss = celoss + cosloss
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d celoss = %0.3f ,cosloss = %0.3f" % (index,celoss,cosloss)
                optimizer.step()


    # 这里是为了在不使用长尾的时候搞的，用普通的feddataloader，所以我们把普通的fedset传入
    # 这里我们不需要labelskew之后的train，只需要正常的train和testloader就行，
    # 其实是只要test，因为本质上这是对global上利用test除标签外的信息
    # 所以用test里的dataset重新构造一个不skew的train

    # 这里不需要了，我们直接在imbalancecifar10中进行了对lt——ratio是否为-1的判断，以此返回普通或是长尾
    def get_feddataset(self,private_dataset):
        self.normal_dataset  = private_dataset
    def get_feddataloader(self,train_tran):
        train_dataset,test_dataset = self.private_dataset.get_normal_loaders(TwoCropsTransform(train_tran))
        train_loader = DataLoader(train_dataset,batch_size=self.args.local_batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset,batch_size=self.args.local_batch_size, shuffle=False, num_workers=4)
        return train_loader,test_loader


    def test_train_validate(self):
        parti_num = self.args.parti_num
        epochs = self.args.local_epoch 
        num_classes = FederatedModel.N_CLASS
        data_loader = TestAgnosticImbalanceCIFAR10DataLoader(
            data_dir=data_path(),
            batch_size=128,
            shuffle=False,
            training=False,
            imb_factor=self.args.lt_ratio,
            num_workers=2
        )


        train_cls_num_list = data_loader.cls_num_list
        #b = np.load("../data/shot_list.npy")
        train_cls_num_list=torch.tensor(train_cls_num_list)
        many_shot = train_cls_num_list > 100
        few_shot =train_cls_num_list <20
        medium_shot =~many_shot & ~few_shot

        train_data_loader= data_loader.train_set()
        valid_data_loader = data_loader.test_set()
            
        # aggregation_weight = torch.nn.Parameter(torch.FloatTensor(parti_num)).to(self.device)
        aggregation_weight = torch.Tensor(parti_num).float()
        aggregation_weight =  aggregation_weight.to(self.device)
        aggregation_weight.requires_grad = True
        optimizer = optim.SGD([aggregation_weight], lr=self.local_lr, weight_decay=5e-4,momentum=0.9,nesterov=True)
        # aggregation_weight.requires_grad = True
        aggregation_weight.data.fill_(1/parti_num) 
        
        # optimizer = config.init_obj('optimizer', torch.optim, [aggregation_weight])

        
            # self.cl_aggregate_nets(aggregation_weight)
        weight_record = self.test_training(train_data_loader, aggregation_weight, optimizer)
        print("Aggregation weight:")
        for i,w in enumerate(weight_record):
            print(f"Expert:{i} is {w:.2f}")
        # record = self.test_validation(valid_data_loader, num_classes, aggregation_weight, many_shot, medium_shot, few_shot)    
        
        print('\n')        
        print('='*25, ' Final results ', '='*25)
        print('\n')
        # print('Top-1 accuracy on many-shot, medium-shot, few-shot and all classes:')
        # print(record)            
        print('\n')
        print('Aggregation weights of three experts:')    
        print(weight_record)
        return weight_record

                
    def test_training(self,train_data_loader,  aggregation_weight, optimizer,epochs=1):
        model = self.global_net
        model.eval()
        weight_record = [] #保存权重
        device = self.device
        losses = AverageMeter('Loss', ':.4e') 
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        for k in range(epochs):
            for i, (data, _) in enumerate(tqdm(train_data_loader)):
                data[0] = data[0].to(device)
                data[1] = data[1].to(device) 
                aggregation_softmax = torch.nn.functional.softmax(aggregation_weight) # softmax for normalization
                aggregation_softmax.to(device)
                cl_outputs = []
                for d in data:
                    output = []
                    for m in self.nets_list:
                        m.to(device)
                        m.eval()
                        out = m(d)
                        output.append(out)
                    output = torch.stack(output,dim=0)
                    for o,w in zip(output,aggregation_softmax):
                        temp = o
                        o = temp*w
                    output = torch.sum(output,dim=0)
                    cl_outputs.append(F.softmax(output,dim=1))
                
                # SSL loss: similarity maxmization
                cos_similarity = cos(cl_outputs[0],cl_outputs[1]).mean()
                ssl_loss =  cos_similarity
                loss =  - ssl_loss 
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.update(ssl_loss, data[0].shape[0])
            flag = False #如果有w小于0.05退出训练
            for w in aggregation_weight:
                if w < 0.05:
                    flag = True
            if flag: break
        aggregation_softmax = torch.nn.functional.softmax(aggregation_weight, dim=0).detach().cpu().numpy()
        print('aggregation_softmax',aggregation_softmax)
        for w in aggregation_softmax:
            w = np.round(w,decimals=2)
            weight_record.append(w)
        return  weight_record

    def test_validation(self,data_loader, model, num_classes, aggregation_weight, many_shot, medium_shot, few_shot):
        device = self.device
        model.eval()  
        aggregation_weight.requires_grad = False 
        confusion_matrix = torch.zeros(num_classes, num_classes).cuda()
        total_logits = torch.empty((0, num_classes)).cuda()
        total_labels = torch.empty(0, dtype=torch.long).cuda()
        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(data_loader)):
                data, target = data.to(device), target.to(device)
                output = model(data)
                expert1_logits_output = output['logits'][:,0,:] 
                expert2_logits_output = output['logits'][:,1,:]
                expert3_logits_output = output['logits'][:,2,:]
                aggregation_softmax = torch.nn.functional.softmax(aggregation_weight) # softmax for normalization
                aggregation_output = aggregation_softmax[0] * expert1_logits_output + aggregation_softmax[1] * expert2_logits_output + aggregation_softmax[2] * expert3_logits_output
                for t, p in zip(target.view(-1), aggregation_output.argmax(dim=1).view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                total_logits = torch.cat((total_logits, aggregation_output))
                total_labels = torch.cat((total_labels, target))  
                
    
        probs, preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)

        # Calculate the overall accuracy and F measurement
        eval_acc_mic_top1= mic_acc_cal(preds[total_labels != -1],
                                            total_labels[total_labels != -1])
            
        acc_per_class = confusion_matrix.diag()/confusion_matrix.sum(1)
        acc = acc_per_class.cpu().numpy() 
        many_shot_acc = acc[many_shot].mean()
        medium_shot_acc = acc[medium_shot].mean()
        few_shot_acc = acc[few_shot].mean()
        print("Many-shot {0:.2f}, Medium-shot {1:.2f}, Few-shot {2:.2f}, All {3:.2f}".format(many_shot_acc * 100, medium_shot_acc * 100, 
                            few_shot_acc * 100, eval_acc_mic_top1* 100))     
        return np.round(many_shot_acc * 100, decimals=2), np.round(medium_shot_acc * 100, decimals=2), np.round(few_shot_acc * 100, decimals=2), np.round(eval_acc_mic_top1 * 100, decimals=2)
