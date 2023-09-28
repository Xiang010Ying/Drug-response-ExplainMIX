# coding: utf-8
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_curve
from collections import  Counter
# from import_path import *
import os
from GDSC.MOFGCN.model import GModel
from GDSC.MOFGCN.optimizer import Optimizer
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from GDSC.MOFGCN.Entire_Drug_Cell.sampler import Sampler
from GDSC.MOFGCN.myutils import roc_auc, translate_result
# from MOFGCN.Entire_Drug_Cell.Grid_algorithm import grid_main
from sklearn.model_selection import train_test_split

data_dir ='../../'+'processed_data/'

# 加载细胞系-药物矩阵
# cell_drug = pd.read_csv(data_dir + "cell_drug_common_binary_we.csv", index_col=0, header=0)
"""aaa"""
cell_drug = pd.read_csv(
            r"D:\xiang_needread\项目代码（重要）\MOFGCN-main\GDSC\processed_data\cell_drug_common_binary_we.csv",index_col=0, header=0)
null_mask = pd.read_csv(r"D:\xiang_needread\项目代码（重要）\MOFGCN-main\GDSC\processed_data\null_mask_we.csv",
                        index_col=0, header=0)

null_mask.fillna(0, inplace=True)
null_mask = np.array(null_mask, dtype=np.float32)

cell_drug.fillna(0,inplace=True)

print('cell_drug.shape',cell_drug.shape)
print('null_mask.shape',null_mask.shape)
cell_lines = [73, 100, 121, 151, 12, 33, 43, 81, 94, 105, 113, 183, 44, 74, 101, 185, 153, 190, 76, 1, 67, 80, 108, 136, 170, 0, 53, 98, 26, 27, 39, 68, 90, 109, 120, 123, 128, 147, 148, 187, 59, 175, 69, 88, 65, 71]
drugs = [72, 67, 121, 335, 186, 218, 172, 17, 309, 152, 264, 297, 28, 234, 21, 187, 185, 228, 171, 287, 75, 183, 137, 280, 164, 277, 151, 130, 324, 315, 29, 329, 184, 42, 194, 20, 105, 54, 224, 58, 276, 239, 257, 208, 306, 312, 106, 113, 265, 84, 225, 99, 327, 271, 73, 161, 350, 233, 199, 57, 213, 86, 96, 33, 302, 40, 299, 120, 15, 25, 80, 177, 244, 258, 32, 179, 55, 283, 198, 201, 285, 88, 216, 255, 7, 62, 220, 139, 338, 197, 81, 135, 178, 118, 209, 12, 107, 166, 27]

k=-1
test_index=[]
train_index=[]
for i in range(cell_drug.shape[0]):
    for j in range(cell_drug.shape[1]):
        if cell_drug.iloc[i,j] == 1:
            k += 1
            if (i in cell_lines) and (j in drugs):
                test_index.append(k)
            elif j not in drugs and i not in cell_lines:
                train_index.append(k)

cell_drug = np.array(cell_drug, dtype=np.float32)
adj_mat_coo_data = sp.coo_matrix(cell_drug).data

# 加载药物-指纹特征矩阵
drug_feature = pd.read_csv(data_dir + "drug_feature_we.csv", index_col=0, header=0)
feature_drug = np.array(drug_feature, dtype=np.float32)
# 加载细胞系-基因特征矩阵
gene = pd.read_csv(data_dir + "cell_gene_feature_we.csv", index_col=0, header=0)
gene = np.array(gene, dtype=np.float32)

# 加载细胞系-cna特征矩阵
cna = pd.read_csv(data_dir + "cell_gene_cna_we.csv", index_col=0, header=0)
cna = cna.fillna(0)
cna = np.array(cna, dtype=np.float32)

# 加载细胞系-mutaion特征矩阵
mutation = pd.read_csv(data_dir + "cell_gene_mutation_we.csv", index_col=0, header=0)
mutation = np.array(mutation, dtype=np.float32)

# 加载null_mask
null_mask = pd.read_csv(data_dir + "null_mask_we.csv", index_col=0, header=0)
null_mask = np.array(null_mask, dtype=np.float32)


epochs = []
true_datas = pd.DataFrame()
predict_datas = pd.DataFrame()
k = 10
#kfold = KFold(n_splits=k, shuffle=True, random_state=11)
kfold = StratifiedKFold(n_splits=10,shuffle=False)

K = 0

train_index=np.array(train_index)
test_index=np.array(test_index)
sampler = Sampler(cell_drug, train_index, test_index, null_mask)

model = GModel(adj_mat=sampler.train_data, gene=gene, cna=cna, mutation=mutation, sigma=2, k=11, iterates=3,
               feature_drug=feature_drug, n_hid1=128, n_hid2=38, alpha=5.74)
opt = Optimizer(model, sampler.train_data, sampler.test_data, sampler.test_mask, sampler.train_mask,
                roc_auc, lr=1e-7, epochs=1000)

predict_data_masked, true_data = opt()
# K += 1
# pred_data = predict_data_masked.detach().numpy()
#
# print('pred_data',pred_data)
# # 大于零为1小于零为0
# pred_data = np.where(pred_data >= 0.5, 1, 0)
# true_data = true_data.detach().cpu().numpy()
# # print('true_data',true_data)
# true_data = np.where(true_data >= 0.5, 1, 0)
#
# tn, fp, fn, tp = confusion_matrix(true_data,pred_data).ravel()
# prec, reca, _ = precision_recall_curve(true_data, pred_data)
# aupr = auc(reca, prec)
# auc_ = roc_auc_score(true_data, pred_data)
# accuracy = (tp + tn) / (tn + fp + fn + tp)
# recall = tp / (tp + fn)
# precision = tp / (tp + fp)
# f1 = 2 * precision * recall / (precision + recall)
#
# print(
#     'acc={:.4f}|auc={:.4f}|aupr={:.4f}|precision={:.4f}|recall={:.4f}|f1={:.4f}'.format(accuracy, auc_, aupr,
#                                                                                         precision, recall, f1))
# print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
# print('y_pred: ', Counter(pred_data))
# print('y_true: ', Counter(true_data))

# for train_index, test_index in kfold.split(np.arange(adj_mat_coo_data.shape[0]),y):
#     print('train_index',train_index)
#     sampler = Sampler(cell_drug, train_index, test_index, null_mask)
#     model = GModel(adj_mat=sampler.train_data, gene=gene, cna=cna, mutation=mutation, sigma=2, k=11, iterates=3,
#                    feature_drug=feature_drug, n_hid1=38, n_hid2=10, alpha=3.74)
#     opt = Optimizer(model, sampler.train_data, sampler.test_data, sampler.test_mask, sampler.train_mask,
#                     roc_auc, lr=1e-4, epochs=200)
#
#     predict_data_masked, true_data = opt()
#     K += 1
#
#     pred_data = predict_data_masked.detach().numpy()
#     # 大于零为1小于零为0
#     pred_data = np.where(pred_data >= 0.5, 1, 0)
#     true_data = true_data.detach().cpu().numpy()
#     true_data = np.where(true_data >= 0.5, 1, 0)
#
#     tn, fp, fn, tp = confusion_matrix(true_data,pred_data).ravel()
#     prec, reca, _ = precision_recall_curve(true_data, pred_data)
#     aupr = auc(reca, prec)
#     auc_ = roc_auc_score(true_data, pred_data)
#     accuracy = (tp + tn) / (tn + fp + fn + tp)
#     recall = tp / (tp + fn)
#     precision = tp / (tp + fp)
#     f1 = 2 * precision * recall / (precision + recall)
#
#     all_accuracy += accuracy
#     all_roc_auc += auc_
#     all_aupr += aupr
#     all_precision += precision
#     all_recall += recall
#     all_f1 += f1
#     print(
#         'acc={:.4f}|auc={:.4f}|aupr={:.4f}|precision={:.4f}|recall={:.4f}|f1={:.4f}'.format(accuracy, auc_, aupr,
#                                                                                             precision, recall, f1))
#     print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
#     print('y_pred: ', Counter(pred_data))
#     print('y_true: ', Counter(true_data))
#
# print('次数：',K)
# print('acc={:.4f}|auc={:.4f}|aupr={:.4f}|precision={:.4f}|recall={:.4f}|f1={:.4f}'.format(all_accuracy/K, all_roc_auc/K, all_aupr/K,
#                                                                                           all_precision/K, all_recall/K, all_f1/K))
# from sklearn.model_selection import train_test_split
#
# train_index, test_index = train_test_split(np.arange(adj_mat_coo_data.shape[0]), test_size = 1200, random_state = 12348, shuffle =True)
# print('train_index',train_index)
# print('shape',adj_mat_coo_data.shape)
# sampler = Sampler(cell_drug, train_index, test_index, null_mask)
# model = GModel(adj_mat=sampler.train_data, gene=gene, cna=cna, mutation=mutation, sigma=2, k=11, iterates=3,
#                feature_drug=feature_drug, n_hid1=192, n_hid2=36, alpha=5.74)
# opt = Optimizer(model, sampler.train_data, sampler.test_data, sampler.test_mask, sampler.train_mask,
#                 roc_auc, lr=1e-5, epochs=500)
# epoch, true_data, predict_data = opt()
# epochs.append(epoch)
# true_datas = true_datas.append(translate_result(true_data))
# predict_datas = predict_datas.append(translate_result(predict_data))

# file = open("./result_data/epochs.txt", "w")
# file.write(str(epochs))
# file.close()
pd.DataFrame(true_datas).to_csv("./result_data/true_data.csv")
pd.DataFrame(predict_datas).to_csv("./result_data/predict_data.csv")

"""
alphas = np.linspace(start=4.5, stop=6.5, num=101)
save_format = "{:^10.5f}{:^10.4f}"
save_file = open("./alpha_grid_result.txt", "w")
for alpha in alphas:
    alpha = float(alpha)
    grid_main(fold_k=5, random_state=11, original_adj_mat=cell_drug, null_mask=null_mask, gene=gene,
              cna=cna, mutation=mutation, drug_feature=feature_drug, sigma=2, knn=11,
              iterates=3, n_hid1=192, n_hid2=36, alpha=alpha, evaluate_fun=roc_auc, lr=5e-4,
              epochs=1000, neg_sample_times=9, device="cuda", str_format=save_format,
              file=save_file)
save_file.close()
"""
'''
for train_index, test_index in kfold.split(np.arange(adj_mat_coo_data.shape[0]),y):
    sampler = Sampler(cell_drug, train_index, test_index, null_mask)
    model = GModel(adj_mat=sampler.train_data, gene=gene, cna=cna, mutation=mutation, sigma=2, k=11, iterates=3,
                   feature_drug=feature_drug, n_hid1=38, n_hid2=16, alpha=3.74)
    opt = Optimizer(model, sampler.train_data, sampler.test_data, sampler.test_mask, sampler.train_mask,
                    roc_auc, lr=1e-3, epochs=1000)

    predict_data_masked, true_data = opt()
    K += 1

    pred_data = predict_data_masked.detach().numpy()
    # 大于零为1小于零为0
    pred_data = np.where(pred_data >= 0.5, 1, 0)
    true_data = true_data.detach().cpu().numpy()
    true_data = np.where(true_data >= 0.5, 1, 0)

    tn, fp, fn, tp = confusion_matrix(true_data,pred_data).ravel()
    prec, reca, _ = precision_recall_curve(true_data, pred_data)
    aupr = auc(reca, prec)
    auc_ = roc_auc_score(true_data, pred_data)
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * precision * recall / (precision + recall)

    all_accuracy += accuracy
    all_roc_auc += auc_
    all_aupr += aupr
    all_precision += precision
    all_recall += recall
    all_f1 += f1
    print(
        'acc={:.4f}|auc={:.4f}|aupr={:.4f}|precision={:.4f}|recall={:.4f}|f1={:.4f}'.format(accuracy, auc_, aupr,
                                                                                            precision, recall, f1))
    print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
    print('y_pred: ', Counter(pred_data))
    print('y_true: ', Counter(true_data))
'''
