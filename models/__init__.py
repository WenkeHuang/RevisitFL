# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import importlib


def get_all_models():
    return [model.split('.')[0] for model in os.listdir('models')
            if not model.find('__') > -1 and 'py' in model]


names = {}
for model in get_all_models():
    mod = importlib.import_module('models.' + model)
    class_name = {x.lower(): x for x in mod.__dir__()}[model.replace('_', '')]
    names[model] = getattr(mod, class_name)


# 上面获得了fedavg上面的模型的名字，下面相当于调用他们的初始化了，fedavg(a,b,c)
def get_model(nets_list, args, transform):
    return names[args.model](nets_list, args, transform)
