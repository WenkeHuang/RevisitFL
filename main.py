import torch.multiprocessing
import setproctitle
import datetime
import socket
import torch
import uuid
import sys
import os

torch.multiprocessing.set_sharing_strategy('file_system')
conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/datasets')
sys.path.append(conf_path + '/backbone')
sys.path.append(conf_path + '/models')
from datasets import Priv_NAMES as DATASET_NAMES
from utils.args import add_management_args
from utils.conf import set_random_seed
from datasets import get_prive_dataset
from utils.best_args import best_args
from argparse import ArgumentParser
from models import get_all_models
from utils.training import train
from models import get_model


def parse_args():
    parser = ArgumentParser(description='Federated Learning with Label Skew', allow_abbrev=False)
    parser.add_argument('--device_id', type=int, default=0, help='The Device Id for Experiment')
    parser.add_argument('--communication_epoch', type=int, default=100, help='The Communication Epoch in Federated Learning')
    # 10
    parser.add_argument('--local_epoch', type=int, default=10, help='The Local Epoch for each Participant')
    parser.add_argument('--local_batch_size', type=int, default=64)
    # 10
    parser.add_argument('--parti_num', type=int, default=10, help='The Number for Participants')

    parser.add_argument('--seed', type=int, default=0, help='The random seed.')
    parser.add_argument('--model', type=str, default='fedopt',  # moon fedavg fedreg fedavgnorm fedalign fedours
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--structure', type=str, default='homogeneity')
    parser.add_argument('--dataset', type=str, default='fl_cifar10',choices=DATASET_NAMES) # fl_tiny_imagenet,fl_mnist fl_cifar10, fl_cifar100
    parser.add_argument('--pri_aug', type=str, default='weak', help='Private data augmentation')
    # 0.3 0.5
    parser.add_argument('--beta', type=float, default=0.01, help='The beta for label skew')
    # 1 0.5
    parser.add_argument('--online_ratio', type=float, default=1, help='The ratio for online clients')

    parser.add_argument('--optimizer', type=str, default='sgd', help='adam or sgd')
    parser.add_argument('--local_lr', type=float, default=0.1, help='The learning rate for local updating')
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")

    parser.add_argument('--learning_decay', type=bool, default=False, help='Learning rate decay')
    parser.add_argument('--averaing', type=str, default='weight', help='Averaging strategy')

    parser.add_argument('--test_time', action='store_true',)

    ########
    parser.add_argument('--t', type=float, default=0.35)

    torch.set_num_threads(4)
    add_management_args(parser)
    args = parser.parse_args()

    best = best_args[args.dataset][args.model]
    if args.beta in best:
        best = best[args.beta]
    else:
        best = best[0.5]

    for key, value in best.items():
        setattr(args, key, value)

    if args.seed is not None:
        set_random_seed(args.seed)

    if args.dataset == 'fl_cifar10':
        args.local_lr = 0.01
    elif args.dataset == 'fl_cifar100':
        args.local_lr = 0.1
    else:
        args.local_lr = 0.01

    if args.dataset in ['fl_cifar10','fl_cifar100']:
        args.communication_epoch = 100
    else:
        args.communication_epoch = 50
    return args

def main(args=None):
    if args is None:
        args = parse_args()

    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    priv_dataset = get_prive_dataset(args)
    backbones_list = priv_dataset.get_backbone(args.parti_num, None, model_name=args.model)

    model = get_model(backbones_list, args, priv_dataset.get_transform())
    args.arch = model.nets_list[0].name
    print('{}_{}_{}_{}_{}_{}_{}'.format(args.model, args.parti_num, args.dataset, args.beta, args.online_ratio, args.communication_epoch, args.local_epoch))
    if args.test_time:
        setproctitle.setproctitle('test speed')
    else:
        setproctitle.setproctitle('{}_{}_{}_{}_{}_{}'.format(args.model, args.dataset, args.beta, args.online_ratio, args.communication_epoch, args.local_epoch))
    train(model, priv_dataset, args)


if __name__ == '__main__':
    main()
