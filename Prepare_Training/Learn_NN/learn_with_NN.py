import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelWithLMHead
)
import logging
import time
import argparse
import os
import json
from datetime import datetime, timedelta
from tqdm import tqdm, trange
import torch.multiprocessing as mp
import warnings
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import random
from sklearn.linear_model import LinearRegression, Lasso
from torch.optim.lr_scheduler import LambdaLR

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "start loading")

time_list_const = ['q_2019', 'q_2020_1', 'q_2020_2', 'q_2020_3', 'q_2020_4', 'q_2021_1', 'q_2021_2', 'q_2021_3', 'q_2021_4', 'q_2022_1', 'q_2022_2', 'q_2022_3', 'q_2022_4']
time_feature = "q_2022_4"
print(f"the time feature is {time_feature}")

time_index = time_list_const.index(time_feature)
if time_index > 0:
    time_previous = time_list_const[time_index - 1]
rank = 1000

with open(f"../Calculate_P/control_probing_matrix_{time_feature}/LM_step_N.json", "r") as f:
    X_P_N = json.load(f)

with open(f"../Calculate_P/control_probing_matrix_{time_feature}/LM_learn.json", "r") as f:
    P_matrix = json.load(f)

print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "start loading")

matrix_after = np.load('../Cooccurance_Matrix/co_occurance_matrix_q_2022_1.npy')
matrix_before = np.load('../Cooccurance_Matrix/co_occurance_matrix_q_2021_4.npy')

print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "start loading")

data = np.load(f"svd_results_vertical_rank_{rank}_{time_feature}.npz")
U, sigma, V = data["U"], data["sigma"], data["V"]
Sigma = np.diag(sigma)
U_D = np.dot(U, np.sqrt(Sigma))
D_V = np.dot(np.sqrt(Sigma), V)

print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "start loading")

with open('../Cooccurance_Matrix/subject_map.json', 'r') as file:
    subject_map = json.load(file)
with open('../Cooccurance_Matrix/subject_inverse_map.json', 'r') as file:
    subject_inverse_map = json.load(file)
with open('../Cooccurance_Matrix/object_map.json', 'r') as file:
    object_map = json.load(file)
with open('../Cooccurance_Matrix/object_inverse_map.json', 'r') as file:
    object_inverse_map = json.load(file)
with open('/m-ent1/ent1/xihc20/ALL_DATA_PREPROCESSING/Extract_Relation/Words_PMID/Exact_intersection/ALL_Object_dict.json', 'r') as file:
    object_atall = json.load(file)
with open('/m-ent1/ent1/xihc20/ALL_DATA_PREPROCESSING/Extract_Relation/Words_PMID/Exact_intersection/ALL_Subject_dict.json', 'r') as file:
    subject_atall = json.load(file)

print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "finish loading")

# PC_prob_dict = {}
# for i, (sub, obj_dict) in tqdm(enumerate(P_matrix.items()), total=len(P_matrix)):
#     for j, (obj, prob) in enumerate(obj_dict.items()):
#         PC_prob_dict[sub, obj] = [prob, matrix_after[int(sub), int(obj)],
#                                   matrix_after[int(sub), :].sum(), matrix_after[:, int(obj)].sum(),
#                                   U_D[int(sub) + U_D.shape[0] // 2, :], D_V[:, int(obj)]]
#
# P_N_prob_dict = {}
# for i, (sub, obj_dict) in tqdm(enumerate(X_P_N.items()), total=len(X_P_N)):
#     for j, (obj, prob) in enumerate(obj_dict.items()):
#         P_N_prob_dict[sub, obj] = [prob, matrix_before[int(sub), int(obj)],
#                                   matrix_before[int(sub), :].sum(), matrix_before[:, int(obj)].sum(),
#                                   U_D[int(sub), :], D_V[:, int(obj)]]

# np.save(f'PC_prob_dict_{rank}_{time_feature}.npy', PC_prob_dict)
PC_prob_dict = np.load(f'PC_prob_dict_{rank}_{time_feature}.npy', allow_pickle=True).item()

# np.save(f'PN_prob_dict_{rank}_{time_feature}.npy', P_N_prob_dict)
P_N_prob_dict = np.load(f'PN_prob_dict_{rank}_{time_feature}.npy', allow_pickle=True).item()

def split_train_test(data_dict, train_ratio):
    # 随机打乱字典的键
    random_keys = list(data_dict.keys())
    random.shuffle(random_keys)

    num_samples = len(random_keys)
    num_train = int(num_samples * train_ratio)
    num_test = num_samples - num_train

    train_data = {key: data_dict[key] for key in random_keys[:num_train]}
    test_data = {key: data_dict[key] for key in random_keys[num_train:num_train + num_test]}

    return train_data, test_data

def normalize_data(Data):
    means = np.mean(Data, axis=0)
    stds = np.std(Data, axis=0)
    normalized_Data = (Data - means) / stds
    return normalized_Data, means, stds

train_ratio = 0.7
nonzero_train_data, nonzero_test_data = split_train_test(PC_prob_dict, train_ratio)
print(f"train set size = {len(nonzero_train_data)}, test set size = {len(nonzero_test_data)}, "
      f"at all {len(nonzero_train_data) + len(nonzero_test_data)}")
print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "start training")

# # , u, v
# X_train = [np.hstack((x, y, z)) for w, x, y, z, u, v in nonzero_train_data.values()]
# y_train = [np.log10(w) for w, x, y, z, u, v in nonzero_train_data.values()]
# X_test = [np.hstack((x, y, z)) for w, x, y, z, u, v in nonzero_test_data.values()]
# y_test = [np.log10(w) for w, x, y, z, u, v in nonzero_test_data.values()]
#
# X_train, means, stds = normalize_data(X_train)
# # y_train, means, stds = normalize_data(y_train)
# X_test, means, stds = normalize_data(X_test)
# # y_test, means, stds = normalize_data(y_test)
#
# X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
# def spearman_rho(list1, list2):
#     rho, p_value = spearmanr(list1, list2)
#     print("Spearman's秩相关系数: ", rho)
#     print("p值: ", p_value)
#
# # model = LinearRegression()
# model = Lasso(alpha=1e-5)
# model.fit(X_train, y_train)
# xyz_model_coef = model.coef_
#
# print('@' * 100)
# spearman_rho([w for w, x, y, z, u, v in nonzero_train_data.values()], [x for w, x, y, z, u, v in nonzero_train_data.values()])
# spearman_rho([w for w, x, y, z, u, v in nonzero_train_data.values()], [y for w, x, y, z, u, v in nonzero_train_data.values()])
# spearman_rho([w for w, x, y, z, u, v in nonzero_train_data.values()], [z for w, x, y, z, u, v in nonzero_train_data.values()])
# spearman_rho([w for w, x, y, z, u, v in nonzero_train_data.values()], [model.intercept_ + (model.coef_ * np.hstack((x, y, z))).sum() for w, x, y, z, u, v in tqdm(nonzero_train_data.values())])
# spearman_rho([w for w, x, y, z, u, v in nonzero_train_data.values()], [model.intercept_ + (model.coef_ * X_train[idx]).sum() for idx in range(X_train.shape[0])])
# print('-' * 50)
# spearman_rho([w for w, x, y, z, u, v in nonzero_test_data.values()], [x for w, x, y, z, u, v in nonzero_test_data.values()])
# spearman_rho([w for w, x, y, z, u, v in nonzero_test_data.values()], [y for w, x, y, z, u, v in nonzero_test_data.values()])
# spearman_rho([w for w, x, y, z, u, v in nonzero_test_data.values()], [z for w, x, y, z, u, v in nonzero_test_data.values()])
# spearman_rho([w for w, x, y, z, u, v in nonzero_test_data.values()], [model.intercept_ + (model.coef_ * np.hstack((x, y, z))).sum() for w, x, y, z, u, v in tqdm(nonzero_test_data.values())])
# spearman_rho([w for w, x, y, z, u, v in nonzero_test_data.values()], [model.intercept_ + (model.coef_ * X_test[idx]).sum() for idx in range(X_test.shape[0])])
#
# import IPython
# IPython.embed()
#
# ####################################################################
#
# X_train = [np.hstack((u, v)) for w, x, y, z, u, v in nonzero_train_data.values()]
# y_train = [w for w, x, y, z, u, v in nonzero_train_data.values()]
# X_test = [np.hstack((u, v)) for w, x, y, z, u, v in nonzero_test_data.values()]
# y_test = [w for w, x, y, z, u, v in nonzero_test_data.values()]
#
# X_train, means, stds = normalize_data(X_train)
# # y_train, means, stds = normalize_data(y_train)
# X_test, means, stds = normalize_data(X_test)
# # y_test, means, stds = normalize_data(y_test)
#
# X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
# def spearman_rho(list1, list2):
#     rho, p_value = spearmanr(list1, list2)
#     print("Spearman's秩相关系数: ", rho)
#     print("p值: ", p_value)
#
# # model = LinearRegression()
# model = Lasso(alpha=1e-5)
# model.fit(X_train, y_train)
# uv_model_coef = model.coef_
# # w1, w2, w3, w4, w5 = model.coef_
# # intercept = model.intercept_
# # print("回归系数：w1 =", w1, ", w2 =", w2, ", w3 =", w3)
#
# print('@' * 100)
# spearman_rho([w for w, x, y, z, u, v in nonzero_train_data.values()], [x for w, x, y, z, u, v in nonzero_train_data.values()])
# spearman_rho([w for w, x, y, z, u, v in nonzero_train_data.values()], [y for w, x, y, z, u, v in nonzero_train_data.values()])
# spearman_rho([w for w, x, y, z, u, v in nonzero_train_data.values()], [z for w, x, y, z, u, v in nonzero_train_data.values()])
# spearman_rho([w for w, x, y, z, u, v in nonzero_train_data.values()], [model.intercept_ + (model.coef_ * np.hstack((u, v))).sum() for w, x, y, z, u, v in tqdm(nonzero_train_data.values())])
# spearman_rho([w for w, x, y, z, u, v in nonzero_train_data.values()], [model.intercept_ + (model.coef_ * X_train[idx]).sum() for idx in range(X_train.shape[0])])
# print('-' * 50)
# spearman_rho([w for w, x, y, z, u, v in nonzero_test_data.values()], [x for w, x, y, z, u, v in nonzero_test_data.values()])
# spearman_rho([w for w, x, y, z, u, v in nonzero_test_data.values()], [y for w, x, y, z, u, v in nonzero_test_data.values()])
# spearman_rho([w for w, x, y, z, u, v in nonzero_test_data.values()], [z for w, x, y, z, u, v in nonzero_test_data.values()])
# spearman_rho([w for w, x, y, z, u, v in nonzero_test_data.values()], [model.intercept_ + (model.coef_ * np.hstack((u, v))).sum() for w, x, y, z, u, v in tqdm(nonzero_test_data.values())])
# spearman_rho([w for w, x, y, z, u, v in nonzero_test_data.values()], [model.intercept_ + (model.coef_ * X_test[idx]).sum() for idx in range(X_test.shape[0])])
#
# # import IPython
# # IPython.embed()
#
# #####################################################################################
# # , u, v
# X_train = [np.hstack((x, y, z, u, v)) for w, x, y, z, u, v in nonzero_train_data.values()]
# y_train = [np.log10(w) for w, x, y, z, u, v in nonzero_train_data.values()]
# X_test = [np.hstack((x, y, z, u, v)) for w, x, y, z, u, v in nonzero_test_data.values()]
# y_test = [np.log10(w) for w, x, y, z, u, v in nonzero_test_data.values()]
#
# X_train, means, stds = normalize_data(X_train)
# # y_train, means, stds = normalize_data(y_train)
# X_test, means, stds = normalize_data(X_test)
# # y_test, means, stds = normalize_data(y_test)
#
# X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
# def spearman_rho(list1, list2):
#     rho, p_value = spearmanr(list1, list2)
#     print("Spearman's秩相关系数: ", rho)
#     print("p值: ", p_value)
#
# # model = LinearRegression()
# model = Lasso(alpha=1e-5)
# model.fit(X_train, y_train)
# xyzuv_model_coef = model.coef_
# # model.coef_[:3] = xyz_model_coef
#
# print('@' * 100)
# spearman_rho([w for w, x, y, z, u, v in nonzero_train_data.values()], [x for w, x, y, z, u, v in nonzero_train_data.values()])
# spearman_rho([w for w, x, y, z, u, v in nonzero_train_data.values()], [y for w, x, y, z, u, v in nonzero_train_data.values()])
# spearman_rho([w for w, x, y, z, u, v in nonzero_train_data.values()], [z for w, x, y, z, u, v in nonzero_train_data.values()])
# spearman_rho([w for w, x, y, z, u, v in nonzero_train_data.values()], [model.intercept_ + (model.coef_ * np.hstack((x, y, z, u, v))).sum() for w, x, y, z, u, v in tqdm(nonzero_train_data.values())])
# spearman_rho([w for w, x, y, z, u, v in nonzero_train_data.values()], [model.intercept_ + (model.coef_ * X_train[idx]).sum() for idx in range(X_train.shape[0])])
# print('-' * 50)
# spearman_rho([w for w, x, y, z, u, v in nonzero_test_data.values()], [x for w, x, y, z, u, v in nonzero_test_data.values()])
# spearman_rho([w for w, x, y, z, u, v in nonzero_test_data.values()], [y for w, x, y, z, u, v in nonzero_test_data.values()])
# spearman_rho([w for w, x, y, z, u, v in nonzero_test_data.values()], [z for w, x, y, z, u, v in nonzero_test_data.values()])
# spearman_rho([w for w, x, y, z, u, v in nonzero_test_data.values()], [model.intercept_ + (model.coef_ * np.hstack((x, y, z, u, v))).sum() for w, x, y, z, u, v in tqdm(nonzero_test_data.values())])
# spearman_rho([w for w, x, y, z, u, v in nonzero_test_data.values()], [model.intercept_ + (model.coef_ * X_test[idx]).sum() for idx in range(X_test.shape[0])])
#
# model.coef_[:3] = xyz_model_coef
# model.coef_[3:] = uv_model_coef
# print('@' * 100)
# spearman_rho([w for w, x, y, z, u, v in nonzero_train_data.values()], [x for w, x, y, z, u, v in nonzero_train_data.values()])
# spearman_rho([w for w, x, y, z, u, v in nonzero_train_data.values()], [y for w, x, y, z, u, v in nonzero_train_data.values()])
# spearman_rho([w for w, x, y, z, u, v in nonzero_train_data.values()], [z for w, x, y, z, u, v in nonzero_train_data.values()])
# spearman_rho([w for w, x, y, z, u, v in nonzero_train_data.values()], [model.intercept_ + (model.coef_ * np.hstack((x, y, z, u, v))).sum() for w, x, y, z, u, v in tqdm(nonzero_train_data.values())])
# spearman_rho([w for w, x, y, z, u, v in nonzero_train_data.values()], [model.intercept_ + (model.coef_ * X_train[idx]).sum() for idx in range(X_train.shape[0])])
# print('-' * 50)
# spearman_rho([w for w, x, y, z, u, v in nonzero_test_data.values()], [x for w, x, y, z, u, v in nonzero_test_data.values()])
# spearman_rho([w for w, x, y, z, u, v in nonzero_test_data.values()], [y for w, x, y, z, u, v in nonzero_test_data.values()])
# spearman_rho([w for w, x, y, z, u, v in nonzero_test_data.values()], [z for w, x, y, z, u, v in nonzero_test_data.values()])
# spearman_rho([w for w, x, y, z, u, v in nonzero_test_data.values()], [model.intercept_ + (model.coef_ * np.hstack((x, y, z, u, v))).sum() for w, x, y, z, u, v in tqdm(nonzero_test_data.values())])
# spearman_rho([w for w, x, y, z, u, v in nonzero_test_data.values()], [model.intercept_ + (model.coef_ * X_test[idx]).sum() for idx in range(X_test.shape[0])])
#
# import IPython
# IPython.embed()

torch.set_printoptions(precision=6)
unnormalized_X_train = [np.hstack((x, y, z, u, v)) for w, x, y, z, u, v in nonzero_train_data.values()]
y_train = [np.log10(w) for w, x, y, z, u, v in nonzero_train_data.values()]
unnormalized_X_test = [np.hstack((x, y, z, u, v)) for w, x, y, z, u, v in nonzero_test_data.values()]
y_test = [np.log10(w) for w, x, y, z, u, v in nonzero_test_data.values()]

unnormalized_PN_X = [np.hstack((x, y, z, u, v)) for w, x, y, z, u, v in P_N_prob_dict.values()]
PN_y = [np.log10(w) for w, x, y, z, u, v in P_N_prob_dict.values()]

X_train, means, stds = normalize_data(unnormalized_X_train)
# y_train, means, stds = normalize_data(y_train)
X_test, means, stds = normalize_data(unnormalized_X_test)
# y_test, means, stds = normalize_data(y_test)
PN_X, means, stds = normalize_data(unnormalized_PN_X)

# torch.tensor(np.array( )).float().cuda()
X_train, y_train, X_test, y_test = torch.tensor(np.array(X_train)).float().cuda(), torch.tensor(np.array(y_train)).float().cuda(),\
                                   torch.tensor(np.array(X_test)).float().cuda(), torch.tensor(np.array(y_test)).float().cuda()
PN_X, PN_y = torch.tensor(np.array(PN_X)).float().cuda(), torch.tensor(np.array(PN_y)).float().cuda()
# 定义自定义的二层MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size * 4)

        self.fc3 = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.fc4 = nn.Linear(hidden_size * 2, hidden_size * 2)

        self.fc5 = nn.Linear(hidden_size * 2, hidden_size * 1)
        self.output = nn.Linear(hidden_size, output_size)
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        #
        # self.fc1.weight.data[:, :3] = 1.0
        # self.fc1.weight.data[:, :3] =
        # self.fc1.weight.data[:, 3:] = 0.0


        # self.fc2.weight.data.fill_(1.0)

    def forward(self, x):
        # x = self.fc1(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x) + x)
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x) + x)
        x = self.relu(self.fc5(x))

        x = self.output(x)
        return x


# 定义模型的输入维度、隐藏层维度和输出维度
input_size = X_train.shape[1]
hidden_size = 32
output_size = 1

print(f"input size {input_size}")
time.sleep(1)
# 创建MLP模型实例
model = MLP(input_size, hidden_size, output_size).cuda()

def lr_scheduler(step):
    if step < 6000:
        return 1.0  # 初始学习率
    else:
        factor = 0.1 ** ((step - 6000) // 3000)  # 每4000步学习率减小十倍
        return factor

# 设置训练参数
num_epochs = 30000
batch_size = 1000000
l1_lambda = 1e-4
l2_lambda = 1e-4
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=5e-3)
scheduler = LambdaLR(optimizer, lr_lambda=lr_scheduler)

def spearman_rho(list1, list2):
    rho, p_value = spearmanr(list1, list2)
    print("Spearman's秩相关系数: ", rho)
    print("p值: ", p_value)
    return rho

# import IPython
# IPython.embed()

# 执行训练
for epoch in trange(num_epochs):

    # 模型训练完成后，你可以使用它进行预测
    if epoch % 500 == 0:
        print('-' * 50, f"epoch {epoch}", '-' * 50)
        print(model.fc1.weight)
    if (epoch) % 500 == 0 and epoch > 24000:
        examine_train_list, examine_test_list = [], []
        for idx in trange(len(unnormalized_X_train)):
            with torch.no_grad():
                batch_X = unnormalized_X_train[idx]
                batch_X = torch.tensor(np.array(batch_X)).float().cuda()
                outputs = model(batch_X)
                examine_train_list.append(outputs.cpu().numpy())

        for idx in trange(len(unnormalized_X_test)):
            with torch.no_grad():
                batch_X = unnormalized_X_test[idx]
                batch_X = torch.tensor(np.array(batch_X)).float().cuda()
                outputs = model(batch_X)
                examine_test_list.append(outputs.cpu().numpy())

        spearman_rho([w for w, x, y, z, u, v in nonzero_train_data.values()], [x for w, x, y, z, u, v in nonzero_train_data.values()])
        spearman_rho([w for w, x, y, z, u, v in nonzero_train_data.values()], [y for w, x, y, z, u, v in nonzero_train_data.values()])
        spearman_rho([w for w, x, y, z, u, v in nonzero_train_data.values()], [z for w, x, y, z, u, v in nonzero_train_data.values()])
        spearman_rho([w for w, x, y, z, u, v in nonzero_train_data.values()], examine_train_list)
        print('-' * 50)
        spearman_rho([w for w, x, y, z, u, v in nonzero_test_data.values()], [x for w, x, y, z, u, v in nonzero_test_data.values()])
        spearman_rho([w for w, x, y, z, u, v in nonzero_test_data.values()], [y for w, x, y, z, u, v in nonzero_test_data.values()])
        spearman_rho([w for w, x, y, z, u, v in nonzero_test_data.values()], [z for w, x, y, z, u, v in nonzero_test_data.values()])
        spearman_rho([w for w, x, y, z, u, v in nonzero_test_data.values()], examine_test_list)

        examine_train_list, examine_test_list = [], []
        for idx in trange(X_train.shape[0]):
            with torch.no_grad():
                batch_X = X_train[idx]
                outputs = model(batch_X)
                examine_train_list.append(outputs.cpu().numpy())

        for idx in trange(X_test.shape[0]):
            with torch.no_grad():
                batch_X = X_test[idx]
                outputs = model(batch_X)
                examine_test_list.append(outputs.cpu().numpy())
        spearman_rho([w for w, x, y, z, u, v in nonzero_train_data.values()], examine_train_list)
        spearman_rho([w for w, x, y, z, u, v in nonzero_test_data.values()], examine_test_list)

        if epoch > 24000:
            import IPython
            IPython.embed()

    # 将数据划分为小批量进行训练
    indices = torch.randperm(X_train.size(0))
    for batch_start in range(0, X_train.size(0), batch_size):
        batch_indices = indices[batch_start:batch_start + batch_size]
        batch_X = X_train[batch_indices]
        batch_y = y_train[batch_indices]

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y.unsqueeze(1))

        if epoch % 500 == 0 or epoch > 24000:
            print("origin: ", loss)

        # 计算 L1 正则化项
        l1_reg = torch.tensor(0.).to(loss)
        for param in model.parameters():
            l1_reg += torch.norm(param, p=1)
        loss += l1_lambda * l1_reg

        # 计算 L1 正则化项
        l2_reg = torch.tensor(0.).to(loss)
        for param in model.parameters():
            l1_reg += torch.norm(param, p=2)
        loss += l2_lambda * l2_reg

        if epoch % 500 == 0 or epoch > 24000:
            print("elastic: ", loss)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if epoch % 500 == 0 or epoch > 24000:
            with torch.no_grad():
                batch_X = X_test
                batch_y = y_test

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                print("test loss: ", loss)

                if loss < 8.5:
                    with torch.no_grad():
                        batch_X = PN_X
                        batch_y = PN_y

                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y.unsqueeze(1))
                        print("PN loss: ", loss)

                    aaa = outputs.squeeze().tolist()
                    bbb = PN_y.tolist()
                    rho = spearman_rho(aaa, bbb)

                    if rho > 0.75:
                        estimate_P = {}
                        cnt = 0
                        for kk in X_P_N.keys():
                            estimate_P[kk] = {}
                            for k in X_P_N[kk].keys():
                                estimate_P[kk][k] = np.power(10, aaa[cnt])
                                cnt += 1
                        assert cnt == len(aaa)

                        with open(f"../Calculate_P/control_probing_matrix_{time_feature}/Estimate_step_N.json", "w") as f:
                            json.dump(estimate_P, f, indent=4)

                        import IPython
                        IPython.embed()
    # 打印训练损失
    if (epoch + 1) % 500 == 0 or epoch > 24000:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
