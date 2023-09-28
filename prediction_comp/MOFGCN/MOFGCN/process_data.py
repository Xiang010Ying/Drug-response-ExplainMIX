import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist

"""
count 65276
sum 123904
percent 0.5268272210743802
"""

'''
药物相似性计算
'''
# drug = pd.read_csv(r'D:\XIAOXIANG\可解释性实验\data\CCLE\Drug_Characteristic\drug_fingerprints.csv',index_col=0)
#
# count=0
# sum=0
#
# #相似性
# drug_sim = pd.DataFrame(columns=range(len(drug)),index=range(len(drug)))
# for i in range(len(drug)):
#     for j in range(len(drug)):
#         X = np.vstack([drug.iloc[i], drug.iloc[j]])
#         d2 = pdist(X, 'jaccard')
#         drug_sim.iloc[i,j]=d2[0]
#
# for i in range(len(drug_sim)):
#     for j in range(len(drug_sim)):
#         sum+=1
#         if drug_sim.iloc[i,j] >= 0.5:
#             count+=1
# print('count',count)
# print('sum',sum)
# print('percent',count/sum)
# drug_sim.to_csv(r'D:\xiang_needread\项目代码（重要）\MOFGCN-main\GDSC\processed_data\drug_feature_we.csv')

'''
建立药物细胞系关联矩阵和无关联矩阵
细胞192 药物352
'''
cell_lines = [73, 100, 121, 151, 12, 33, 43, 81, 94, 105, 113, 183, 44, 74, 101, 185, 153, 190, 76, 1, 67, 80,
              108, 136, 170, 0, 53, 98, 26, 27, 39, 68, 90, 109, 120, 123, 128, 147, 148, 187, 59, 175, 69, 88,
              65, 71]
drugs = [72, 67, 121, 335, 186, 218, 172, 17, 309, 152, 264, 297, 28, 234, 21, 187, 185, 228, 171, 287, 75, 183,
         137, 280, 164, 277, 151, 130, 324, 315, 29, 329, 184, 42, 194, 20, 105, 54, 224, 58, 276, 239, 257,
         208, 306, 312, 106, 113, 265, 84, 225, 99, 327, 271, 73, 161, 350, 233, 199, 57, 213, 86, 96, 33, 302,
         40, 299, 120, 15, 25, 80, 177, 244, 258, 32, 179, 55, 283, 198, 201, 285, 88, 216, 255, 7, 62, 220,
         139, 338, 197, 81, 135, 178, 118, 209, 12, 107, 166, 27]
# sensitive = pd.read_csv(r'D:\XIAOXIANG\可解释性实验\data\CCLE\CCLE_Characteristic\6_multiple_omic_data\sensitive_index_data.csv',header=None)
# nosensitive = pd.read_csv(r'D:\XIAOXIANG\可解释性实验\data\CCLE\CCLE_Characteristic\6_multiple_omic_data\nosensitive_index_data.csv',header=None)

# matrix = pd.DataFrame(index=range(192),columns=range(192,544))
#
# for i in range(len(nosensitive)):
#     matrix.iloc[nosensitive.iloc[i,0],nosensitive.iloc[i,1]] = 0
#
# for i in range(len(sensitive)):
#     matrix.iloc[sensitive.iloc[i,0],sensitive.iloc[i,1]] = 1
#
# print(matrix)
#
# null_matrix = matrix.isna()
# for i in range(matrix.shape[0]):
#     for j in range(matrix.shape[1]):
#         if null_matrix.iloc[i,j]==True:
#             null_matrix.iloc[i, j] = 1
#         else:
#             null_matrix.iloc[i, j] = 0
# print(null_matrix)
# null_matrix.to_csv(r'D:\xiang_needread\项目代码（重要）\MOFGCN-main\GDSC\processed_data\null_mask_we.csv')
# # matrix.fillna(0,inplace=True)
# matrix.to_csv(r'D:\xiang_needread\项目代码（重要）\MOFGCN-main\GDSC\processed_data\cell_drug_common_binary_we.csv')
#对输入进行修改
cell_drug=pd.read_csv(r'D:\xiang_needread\项目代码（重要）\MOFGCN-main\GDSC\processed_data\cell_drug_common_binary_we.csv')
matrix = pd.read_csv(r'D:\xiang_needread\项目代码（重要）\MOFGCN-main\GDSC\processed_data\null_mask_we.csv')

for i in range(cell_drug.shape[0]):
    for j in range(cell_drug.shape[1]):
        if matrix.iloc[i,j]==1 and (j in drugs and i in cell_lines):
            matrix.iloc[i, j]=0
matrix.to_csv(r'D:\xiang_needread\项目代码（重要）\MOFGCN-main\GDSC\processed_data\null_mask_we.csv')
'''
组学数据标准化
'''
# cna = pd.read_csv(r'D:\XIAOXIANG\可解释性实验\data\CCLE\CCLE_Characteristic\6_multiple_omic_data\cell_line_copynumber.csv',index_col=0)
# data_norm = (cna - cna.min()) / (cna.max() - cna.min())#z-score标准化
# data_norm.fillna(0,inplace=True)
# # print('data_norm',data_norm)
# data_norm.to_csv(r'D:\xiang_needread\项目代码（重要）\MOFGCN-main\GDSC\processed_data\cell_gene_cna_we.csv')
#
#
# expression = pd.read_csv(r'D:\XIAOXIANG\可解释性实验\data\CCLE\CCLE_Characteristic\6_multiple_omic_data\cell_line_expression.csv',index_col=0)
# data_norm = (expression - expression.min()) / (expression.max() - expression.min())#z-score标准化
# data_norm.fillna(0,inplace=True)
# data_norm.to_csv(r'D:\xiang_needread\项目代码（重要）\MOFGCN-main\GDSC\processed_data\cell_gene_feature_we.csv')
#
# mutation = pd.read_csv(r'D:\XIAOXIANG\可解释性实验\data\CCLE\CCLE_Characteristic\6_multiple_omic_data\cell_line_mutation.csv',index_col=0)
# data_norm = (mutation - mutation.min()) / (mutation.max() - mutation.min())#z-score标准化
# data_norm.fillna(0,inplace=True)
# data_norm.to_csv(r'D:\xiang_needread\项目代码（重要）\MOFGCN-main\GDSC\processed_data\cell_gene_mutation_we.csv')
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_curve
from collections import Counter
import random

'''计算SRMF指标'''

true = []
pred = []
#true_data = pd.read_csv(r'D:\xiang_needread\项目代码（重要）\SRMF-master\SRMF\cell_drug_common_binary_we.csv',header=None)
true_data = pd.read_csv(r'D:\xiang_needread\项目代码（重要）\MOFGCN-main\GDSC\processed_data\cell_drug_common_binary_we.csv',index_col=0)
pred_data = pd.read_csv(r'D:\xiang_needread\项目代码（重要）\SRMF-master\SRMF\pred.csv',header=None)
pred_data = pred_data.T

test_index = []
cell_lines = [73, 100, 121, 151, 12, 33, 43, 81, 94, 105, 113, 183, 44, 74, 101, 185, 153, 190, 76, 1, 67, 80,
              108, 136, 170, 0, 53, 98, 26, 27, 39, 68, 90, 109, 120, 123, 128, 147, 148, 187, 59, 175, 69, 88,
              65, 71]
drugs = [72, 67, 121, 335, 186, 218, 172, 17, 309, 152, 264, 297, 28, 234, 21, 187, 185, 228, 171, 287, 75, 183,
         137, 280, 164, 277, 151, 130, 324, 315, 29, 329, 184, 42, 194, 20, 105, 54, 224, 58, 276, 239, 257,
         208, 306, 312, 106, 113, 265, 84, 225, 99, 327, 271, 73, 161, 350, 233, 199, 57, 213, 86, 96, 33, 302,
         40, 299, 120, 15, 25, 80, 177, 244, 258, 32, 179, 55, 283, 198, 201, 285, 88, 216, 255, 7, 62, 220,
         139, 338, 197, 81, 135, 178, 118, 209, 12, 107, 166, 27]
# print(true_data.shape)
# for i in range(true_data.shape[0]):
#     for j in range(true_data.shape[1]):
#         if i in cell_lines:
#             true_data.iloc[i,j] = np.nan
# true_data.to_csv(r'D:\xiang_needread\项目代码（重要）\SRMF-master\SRMF\cell_drug_common_binary_we.csv')
#
'''计算SNF指标'''
# k=-1
# for i in range(true_data.shape[0]):
#     for j in range(true_data.shape[1]):
#         if not np.isnan(true_data.iloc[i,j]):
#             true.append(true_data.iloc[i,j])
#             pred.append(pred_data.iloc[i,j])
#             k += 1
#             print(k)
#             if (i in cell_lines) and (j in drugs):
#                 test_index.append(k)
#
# y_p = []
# for i in pred:
#     if (i < 0.5):
#         y_p.append(0)
#     else:
#         y_p.append(1)
#
# y_t = true
#
# index=np.array([i for i in range(len(y_p))])
# random.shuffle(index)
#
# y_pred = y_p
# y_true = y_t
#
# y_pred = []
# y_true = []
#
# for i in index[test_index]:
#     y_true.append(y_t[i])
#     y_pred.append(y_p[i])
#
# tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
# accuracy = (tp + tn) / (tn + fp + fn + tp)
# roc_auc = roc_auc_score(y_true, y_pred)
# prec, reca, _ = precision_recall_curve(np.array(y_true), np.array(y_pred))
# aupr = auc(reca, prec)
#
# recall = tp / (tp + fn)
# precision = tp / (tp + fp)
# f1 = 2 * precision * recall / (precision + recall)
#
# print(
#     'acc={:.4f}|auc={:.4f}|aupr={:.4f}|precision={:.4f}|recall={:.4f}|f1={:.4f}'.format(accuracy, roc_auc, aupr,
#                                                                                         precision, recall, f1))
# print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
# print('y_pred: ', Counter(y_pred))
# print('y_true: ', Counter(y_true))

'''生成cell_drug_common_binary_we.csv'''
# sensitive = pd.read_csv(r'D:\XIAOXIANG\可解释性实验\data\CCLE\CCLE_Characteristic\6_multiple_omic_data\sensitive_index_data.csv',header=None)
# nosensitive = pd.read_csv(r'D:\XIAOXIANG\可解释性实验\data\CCLE\CCLE_Characteristic\6_multiple_omic_data\nosensitive_index_data.csv',header=None)
#
# sensitive_ic50 = pd.read_csv(r'D:\XIAOXIANG\可解释性实验\data\CCLE\CCLE_Characteristic\6_multiple_omic_data\sensitive_drug_celline.csv')
# nosensitive_ic50 = pd.read_csv(r'D:\XIAOXIANG\可解释性实验\data\CCLE\CCLE_Characteristic\6_multiple_omic_data\nosensitive_drug_celline.csv')
#
# matrix = pd.DataFrame(index=range(192),columns=range(192,544))
#
# for i in range(len(nosensitive)):
#     # matrix.iloc[nosensitive.iloc[i,0],nosensitive.iloc[i,1]] = nosensitive_ic50.iloc[i,2]
#     matrix.iloc[nosensitive.iloc[i, 0], nosensitive.iloc[i, 1]] = 0
# for i in range(len(sensitive)):
#     # matrix.iloc[sensitive.iloc[i,0],sensitive.iloc[i,1]] = sensitive_ic50.iloc[i,2]
#     matrix.iloc[sensitive.iloc[i, 0], sensitive.iloc[i, 1]] = 1
# matrix.to_csv(r'D:\xiang_needread\项目代码（重要）\SRMF-master\SRMF\cell_drug_common_binary_we.csv')

# for i in range(len(nosensitive)):
#     matrix.iloc[nosensitive.iloc[i,0],nosensitive.iloc[i,1]] = 0
# for i in range(len(sensitive)):
#     matrix.iloc[sensitive.iloc[i,0],sensitive.iloc[i,1]] = 1
# matrix.to_csv(r'D:\xiang_needread\项目代码（重要）\MOFGCN-main\GDSC\processed_data\cell_drug_common_binary_we.csv')

#新测试集
#cell_drug = pd.read_csv('../cell_drug_response.csv')
# cell_lines = ["RCHACV","NCIH929","MOLM13","PATU8988T","G402"]
cell_lines = [57,143,61,81,2]
# drugs = ["Fulvestrant","Docetaxel","Dactinomycin","Vinblastine","Crizotinib","SB505124","Temsirolimus","Irinotecan","Alisertib","PD173074"]
drugs = [120,23,50,22,114,228,25,115,109,249]

train = []
test = []

# for i in range(cell_drug.shape[0]):
#     if (cell_drug.iloc[i,1] in cell_lines) or (cell_drug.iloc[i,2] in drugs):
#         test.append(i)
#     else:
#         train.append(i)

import scipy.sparse as sp

# k=-1
#cell_drug = pd.read_csv(r"D:\xiang_needread\项目代码（重要）\MOFGCN-main\GDSC\processed_data\cell_drug_common_binary_we.csv", index_col=0, header=0)
# print('cell_drug.shape',cell_drug.shape)
# for i in range(cell_drug.shape[0]):
#     for j in range(cell_drug.shape[1]):
#         if cell_drug.iloc[i,j] == 1:
#             k += 1
#             if (i in cell_lines) or (j in drugs):
#                 train.append(k)
# print('k',k)
# print('train',len(train))
# cell_drug.fillna(0,inplace=True)
# cell_drug = np.array(cell_drug, dtype=np.float32)
# adj_mat_coo_data = sp.coo_matrix(cell_drug).data
# print(adj_mat_coo_data.shape)
from GDSC.MOFGCN.myutils import to_coo_matrix

# null_mask = pd.read_csv(r"D:\xiang_needread\项目代码（重要）\MOFGCN-main\GDSC\processed_data\null_mask_we.csv", index_col=0, header=0)
# null_mask.fillna(0,inplace=True)
# null_mask = np.array(null_mask, dtype=np.float32)
# pos_adj_mat = null_mask + cell_drug-np.array(1)
#
# z=0
# for i in range(pos_adj_mat.shape[0]):
#     for j in range(pos_adj_mat.shape[1]):
#         if pos_adj_mat.iloc[i,j] == -1:
#             z+=1
#             if (i in cell_lines) or (j in drugs):
#                 test.append(z)
# print(len(test))
