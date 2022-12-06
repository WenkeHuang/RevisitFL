import os

def create_if_not_exists(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def save_networks(model,communication_idx):
    nets_list = model.nets_list
    model_name = model.NAME
    save_option = True

    if save_option:
        checkpoint_path = model.checkpoint_path
        model_path = os.path.join(checkpoint_path, model_name)
        model_para_path = os.path.join(model_path, 'para')
        create_if_not_exists(model_para_path)
        for net_idx,network in enumerate(nets_list):
            each_network_path = os.path.join(model_para_path, str(communication_idx) + '_' + str(net_idx) + '.ckpt')
            torch.save(network.state_dict(),each_network_path)

import torch


# def get_centroids(feats_, labels_,num_classes):
#     centroids = []
#     # for i in np.unique(labels_):
#     for i in range(num_classes):
#         temp = feats_[labels_ == i]
#         # 部分client可能不包含所有类，但为了方便，我们将不存在的类的中心设置为0
#         centroids.append(np.mean(temp, axis=0) if temp.size > 0 else np.zeros(feat_dim))
#     return np.stack(centroids)
#
# def get_knncentroids(feats_all,labels_all,num_classes):
#     print('===> Calculating KNN centroids.')
#     feat_dim = feats_all[0].shape
#     feats_all = feats_all.cpu().numpy()
#     labels_all = labels_all.cpu().numpy()
#     feats = np.concatenate(feats_all)
#     labels = np.concatenate(labels_all)
#
#     featmean = feats.mean(axis=0)
#
#     # Get unnormalized centorids 未经过normalized 的原始特征
#     un_centers = get_centroids(feats, labels,num_classes)
#     un_centers = torch.from_numpy(un_centers)
#
#     # Get l2n centorids l2正则特征
#     l2n_feats = torch.Tensor(feats.copy())
#     norm_l2n = torch.norm(l2n_feats, 2, 1, keepdim=True)
#     l2n_feats = l2n_feats / norm_l2n
#     l2n_centers = get_centroids(l2n_feats.numpy(), labels,num_classes)
#     l2n_centers = torch.from_numpy(l2n_centers)
#
#     # Get cl2n centorids l2 中心正则特征
#     cl2n_feats = torch.Tensor(feats.copy())
#     cl2n_feats = cl2n_feats - torch.Tensor(featmean)
#     norm_cl2n = torch.norm(cl2n_feats, 2, 1, keepdim=True)
#     cl2n_feats = cl2n_feats / norm_cl2n
#     cl2n_centers = get_centroids(cl2n_feats.numpy(), labels,num_classes)
#     cl2n_centers = torch.from_numpy(cl2n_centers)
#
#     featmean = torch.from_numpy(featmean)
#
#     return {'mean': featmean,
#             'uncs': un_centers,
#             'l2ncs': l2n_centers,
#             'cl2ncs': cl2n_centers}

