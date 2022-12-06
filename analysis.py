import pandas as pd
import  os
import numpy as np
import csv

path = './data/'
# scenario = 'fl_cifar10' # fl_cifar100,
scenarios_list = ['fl_cifar100']
beta = 0.5
comm_epoch = 100
local_epoch = 10
local_lr = 0.1
online_ratio = 1.0
column_mean_acc_list = ['method', 'paragroup'] + ['epoch'+str(i) for i in range(comm_epoch)]+['MEAN','MAX']
not_include_method = ['fedours','fedoursnoexp','fedournormexp','fedournorm']
# not_include_method = ['']

def load_mean_acc_list(structure_path):
    acc_dict = {}
    experiment_index = 0
    for model in os.listdir(structure_path):
        if model != '' and model not in not_include_method:
            model_path = os.path.join(structure_path, model)
            if os.path.isdir(model_path):
                for para in os.listdir(model_path):
                    para_path = os.path.join(model_path, para)
                    args_path = para_path+'/args.csv'
                    args_pd = pd.read_table(args_path,sep=",")
                    args_pd = args_pd.loc[:, args_pd.columns]
                    args_comm_epoch = args_pd['communication_epoch'][0]
                    args_loca_epoch = args_pd['local_epoch'][0]
                    args_lr = args_pd['local_lr'][0]
                    args_online_ratio = args_pd['online_ratio'][0]
                    if args_comm_epoch == comm_epoch and args_loca_epoch ==local_epoch and args_lr==local_lr and args_online_ratio ==online_ratio:
                        if len(os.listdir(para_path)) != 1:
                            data = pd.read_table(para_path + '/acc.csv', sep=",")
                            data = data.loc[:, data.columns]
                            acc_value = data.values
                            mean_acc_value = np.mean(acc_value, axis=0)
                            mean_acc_value = mean_acc_value.tolist()
                            mean_acc_value = [round(item, 2) for item in mean_acc_value]
                            max_acc_value = max(mean_acc_value)
                            last_acc_vale = mean_acc_value[-3:]
                            last_acc_vale= np.mean(last_acc_vale)
                            mean_acc_value.append(round(last_acc_vale,3))
                            mean_acc_value.append(max_acc_value)
                            acc_dict[experiment_index] = [model, para] + mean_acc_value
                            experiment_index += 1
    return acc_dict


if __name__=='__main__':
    for _,scenario in enumerate (scenarios_list):
        print('**************************************************************')
        scenario_path = os.path.join(path, scenario)
        print('Scenario: ' + scenario+' Beta: '+ str(beta))
        scenario_beta_path = os.path.join(scenario_path, str(beta))
        mean_acc_dict = load_mean_acc_list(scenario_beta_path)
        mean_df = pd.DataFrame(mean_acc_dict)
        mean_df = mean_df.T
        mean_df.columns = column_mean_acc_list
        print(mean_df)
        mean_df.to_excel(os.path.join(scenario_beta_path,'output.xls'), na_rep=True)
        print('**************************************************************')