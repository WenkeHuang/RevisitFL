import os
import sys
import socket
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/datasets')
sys.path.append(conf_path + '/backbone')
sys.path.append(conf_path + '/models')
from datasets import Priv_NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import get_prive_dataset
from models import get_model
from utils.training import train
from utils.best_args import best_args
from utils.conf import set_random_seed
import setproctitle

import torch
import uuid
import datetime


def parse_args():
    parser = ArgumentParser(description='You Only Need Me', allow_abbrev=False)
    parser.add_argument('--device_id', type=int, default=5, help='The Device Id for Experiment')
    parser.add_argument('--communication_epoch', type=int, default=100, help='The Communication Epoch in Federated Learning')

    # 10
    parser.add_argument('--local_epoch', type=int, default=10, help='The Local Epoch for each Participant')
    parser.add_argument('--local_batch_size', type=int, default=64)

    # 10
    parser.add_argument('--parti_num', type=int, default=10, help='The Number for Participants')

    parser.add_argument('--seed', type=int, default=0, help='The random seed.')
    parser.add_argument('--model', type=str, default='fedavgnorm',  # moon fedavg fedreg fedavgnorm fedalign fedours
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--structure', type=str, default='homogeneity')
    parser.add_argument('--dataset', type=str, default='fl_cifar10',  #
                        choices=DATASET_NAMES, help='Which scenario to perform experiments on.')
    parser.add_argument('--pri_aug', type=str, default='weak',  # weak strong
                        help='Augmentation for Private Data')
    # 0.5
    parser.add_argument('--beta', type=float, default=0.2, help='The Beta for Label Skew')
    parser.add_argument('--online_ratio', type=float, default=1, help='The Ratio for Online Clients')

    parser.add_argument('--optimizer', type=str, default='sgd', help='adam or sgd')
    parser.add_argument('--local_lr', type=float, default=0.01, help='The Learning Rate for local updating')
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")

    parser.add_argument('--learning_decay', type=bool, default=False, help='The Option for Learning Rate Decay')
    parser.add_argument('--averaing', type=str, default='weight', help='The Option for averaging strategy')

    torch.set_num_threads(4)
    add_management_args(parser)
    args = parser.parse_args()

    best = best_args[args.dataset][args.model]
    if args.beta in best:
        best = best[args.beta]
    else:
        best = best[0.5]  # Reasonable
    for key, value in best.items():
        setattr(args, key, value)

    if args.seed is not None:
        set_random_seed(args.seed)
    if args.parti_num == 0:
        if args.dataset in ['fl_cifar10', 'fl_cifar100']:
            args.parti_num = 10
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

    print('{}_{}_{}_{}_{}_{}'.format(args.model, args.parti_num, args.dataset, args.beta, args.communication_epoch, args.local_epoch))
    setproctitle.setproctitle('{}_{}_{}_{}_{}'.format(args.model, args.dataset, args.beta, args.communication_epoch, args.local_epoch))

    train(model, priv_dataset, args)


if __name__ == '__main__':
    main()
