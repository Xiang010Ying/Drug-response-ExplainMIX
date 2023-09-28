import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
#min_max_scaler = preprocessing.MinMaxScaler()
from scipy.spatial.distance import pdist

# sensitive= pd.read_csv(r'D:\xiang_needread\项目代码（重要）\TGSA-master\data\sensitive.csv',index_col=0)
# no_sensitive= pd.read_csv(r'D:\xiang_needread\项目代码（重要）\TGSA-master\data\nosensitive.csv',index_col=0)
#
# data =pd.concat([sensitive,no_sensitive])
# print(data)
# data.to_csv(r'D:\xiang_needread\项目代码（重要）\TGSA-master\data\PANCANCER_IC_we.csv')
np.set_printoptions(threshold=np.inf)
# test=np.load('./data/CellLines_DepMap/CCLE_580_18281/census_706/edge_index_PPI_0.95.npy',encoding = "latin1")  #加载文件
#
# print(test.shape)
# print(test)
# copy_number = pd.read_csv(r'D:\XIAOXIANG\可解释性实验\data\CCLE\CCLE_Characteristic\6_multiple_omic_data\cell_line_copynumber.csv',index_col=0)
# expression = pd.read_csv(r'D:\XIAOXIANG\可解释性实验\data\CCLE\CCLE_Characteristic\6_multiple_omic_data\cell_line_expression.csv',index_col=0)
#mutation = pd.read_csv(r'D:\XIAOXIANG\可解释性实验\data\CCLE\CCLE_Characteristic\6_multiple_omic_data\cell_line_mutation.csv',index_col=0)
# methylation = pd.read_csv(r'D:\XIAOXIANG\可解释性实验\data\CCLE\CCLE_Characteristic\6_multiple_omic_data\cell_line_methylation.csv',index_col=0)
# metabolomic = pd.read_csv(r'D:\XIAOXIANG\可解释性实验\data\CCLE\CCLE_Characteristic\6_multiple_omic_data\cell_line_metabolomic.csv',index_col=0)
# protein = pd.read_csv(r'D:\XIAOXIANG\可解释性实验\data\CCLE\CCLE_Characteristic\6_multiple_omic_data\cell_line_protein.csv',index_col=0)
# protein.fillna(value=0,inplace=True)

# min_max_scaler = preprocessing.MinMaxScaler()
# copy_number = min_max_scaler.fit_transform(copy_number)
# expression = min_max_scaler.fit_transform(expression)
# mutation = min_max_scaler.fit_transform(mutation)
# methylation = min_max_scaler.fit_transform(methylation)
# metabolomic = min_max_scaler.fit_transform(metabolomic)
# protein = min_max_scaler.fit_transform(protein)
# mutation_similary = pd.DataFrame(index=range(0,192),columns=range(0,192))
# for i in range(len(mutation)):
#     for j in range(len(mutation)):
#         if mutation_similary.iloc[i,j] >0:
#             mutation_similary.iloc[i, j]=1
#
# for i in range(len(mutation)):
#     for j in range(len(mutation)):
#         X = np.vstack([mutation.iloc[i,:], mutation.iloc[j,:]])
#         mutation_similary.iloc[i,j] = 1-pdist(X,'jaccard')[0]
#
# print('mutation_similary',mutation_similary)

# copy_number_similary = pd.DataFrame(cosine_similarity(copy_number))
# expression_similary = pd.DataFrame(cosine_similarity(expression))
# mutation_similary = pd.DataFrame(cosine_similarity(mutation))
# methylation_similary = pd.DataFrame(cosine_similarity(methylation))
# metabolomic_similary = pd.DataFrame(cosine_similarity(metabolomic))
# protein_similary = pd.DataFrame(cosine_similarity(protein))

# copy_number_similary.to_csv(r'D:\XIAOXIANG\可解释性实验\data\CCLE\CCLE_Characteristic\6_multiple_omic_data\copy_number_similary.csv')
# expression_similary.to_csv(r'D:\XIAOXIANG\可解释性实验\data\CCLE\CCLE_Characteristic\6_multiple_omic_data\expression_similary.csv')
#mutation_similary.to_csv(r'D:\XIAOXIANG\可解释性实验\data\CCLE\CCLE_Characteristic\6_multiple_omic_data\mutation_similary.csv')
# methylation_similary.to_csv(r'D:\XIAOXIANG\可解释性实验\data\CCLE\CCLE_Characteristic\6_multiple_omic_data\methylation_similary.csv')
# metabolomic_similary.to_csv(r'D:\XIAOXIANG\可解释性实验\data\CCLE\CCLE_Characteristic\6_multiple_omic_data\metabolomic_similary.csv')
# protein_similary.to_csv(r'D:\XIAOXIANG\可解释性实验\data\CCLE\CCLE_Characteristic\6_multiple_omic_data\protein_similary.csv')

# data = pd.read_csv(r'D:/XIAOXIANG/可解释性实验/data/CCLE/CCLE_Characteristic/6_multiple_omic_data/aaaaaaaaaaaaa.csv',index_col=0)
# count=0
# sum=0
# for i in range(len(data)):
#     for j in range(len(data)):
#         sum+=1
#         if data.iloc[i,j] > 0.003:
#             count+=1
# print('count',count-192)
# print('sum',sum)
# print('percent',(count-192)/sum)

# drug_dict = np.load('./data/Drugs/drug_feature_graph.npy', allow_pickle=True).item()
# with open('./data/Drugs/drug_feature_graph.npy','w') as f:
#     f.write(drug_dict)
# print('drug_dict',len(drug_dict))
# print('drug_dict',drug_dict)
# cell_dict = np.load('./data/CellLines_DepMap/CCLE_580_18281/census_706/cell_feature_all.npy',
#                     allow_pickle=True).item()
# print('cell_dict', len(cell_dict))
# print('cell_dict',cell_dict)
# edge_index = np.load('./data/CellLines_DepMap/CCLE_580_18281/census_706/edge_index_PPI_{}.npy'.format(0.95))
# print(edge_index.shape)
# print(edge_index)

# save_path = np.load('./data/CellLines_DepMap/CCLE_580_18281/census_706/cell_feature_cn_std_we.npy',allow_pickle=True)
# print(save_path.shape)

# cell = np.load('./data/CellLines_DepMap/CCLE_580_18281/census_706/cell_feature_cn_std_we.npy',
#                         allow_pickle=True).item()
# print(cell)

# sample = pd.read_csv(r'D:\xiang_needread\项目代码（重要）\TGSA-master\data\sample_info.csv',index_col=0)
# data = pd.read_csv(r'D:\xiang_needread\项目代码（重要）\TGSA-master\data\PANCANCER_IC_we - 副本.csv',index_col=0)
# print(data)
# print(data.index)
# for i in data.index:
#     print(i)
#     if i in sample.index:
#         data.loc[i,'DepMap_ID'] = sample.loc[i,'DepMap_ID']
#
# data.to_csv(r'D:\xiang_needread\项目代码（重要）\TGSA-master\data\PANCANCER_IC_we - 副本.csv')

# edges = pd.read_csv('./data/9606.protein.links.detailed.v11.0.txt', sep=' ')
# print(edges)

# sample = pd.read_csv(r'D:\xiang_needread\项目代码（重要）\TGSA-master\data\PANCANCER_IC_we.csv',index_col=0)
# data = pd.read_csv(r'D:\xiang_needread\项目代码（重要）\TGSA-master\data\PANCANCER_IC_we - 副本.csv',index_col=0)
#
# for i in sample.index:
#     print(i)
#     if i in sample.index:
#         sample.loc[i, 'DepMap_ID'] = data.loc[i,'DepMap_ID']
#
# sample.to_csv(r'D:\xiang_needread\项目代码（重要）\TGSA-master\data\PANCANCER_IC_we.csv')

# cell_dict = np.load('./data/CellLines_DepMap/CCLE_580_18281/census_706/cell_feature_cn_std_we.npy',
#                         allow_pickle=True).item()
# print(cell_dict['ACH-000750'])
import pickle
# dict_dir = 'data/similarity_augment/dict/'
# data=pd.read_csv(r'D:\xiang_needread\项目代码（重要）\TGSA-master\data\PANCANCER_IC_we - 副本.csv',index_col=0)
# cell_dict = {}
# index=0
# for i in data.index:
#     cell_dict[i] =index
#     index+=1
# f= open(dict_dir + "cell_id2idx_dict_we", 'wb')
# pickle.dump(cell_dict,f)
#
# with open(dict_dir + "cell_id2idx_dict_we", 'rb') as f:
#     cell_id2idx_dict = pickle.load(f)
# print(cell_id2idx_dict)

# dict_dir = 'data/similarity_augment/dict/'
# data=pd.read_csv(r'D:\xiang_needread\项目代码（重要）\TGSA-master\data\PANCANCER_IC_we - 副本.csv',index_col=0)
# cell_dict = {}
# index=0
# for i in data.index:
#     cell_dict[index] =i
#     index+=1
# f= open(dict_dir + "cell_idx2id_dict_we", 'wb')
# pickle.dump(cell_dict,f)
#
# with open(dict_dir + "cell_idx2id_dict_we", 'rb') as f:
#     cell_id2idx_dict = pickle.load(f)
# print(cell_id2idx_dict)

# dict_dir = 'data/similarity_augment/dict/'
# data=pd.read_csv(r'D:\xiang_needread\项目代码（重要）\TGSA-master\data\Drugs\drug_smiles_we.csv',index_col=0)
# cell_dict = {}
# index=0
# for i in data.index:
#     cell_dict[i] =index+192
#     index+=1
# f= open(dict_dir + "drug_name2idx_dict_we", 'wb')
# pickle.dump(cell_dict,f)
#
# with open(dict_dir + "drug_name2idx_dict_we", 'rb') as f:
#     cell_id2idx_dict = pickle.load(f)
# print(cell_id2idx_dict)

# dict_dir = 'data/similarity_augment/dict/'
# data=pd.read_csv(r'D:\xiang_needread\项目代码（重要）\TGSA-master\data\Drugs\drug_smiles_we.csv',index_col=0)
# cell_dict = {}
# index=192
# for i in data.index:
#     cell_dict[index] =i
#     index+=1
# f= open(dict_dir + "drug_idx2name_dict_we", 'wb')
# pickle.dump(cell_dict,f)
#
# with open(dict_dir + "drug_idx2name_dict_we", 'rb') as f:
#     cell_id2idx_dict = pickle.load(f)
# print(cell_id2idx_dict)

# data=pd.read_csv(r'D:\xiang_needread\项目代码（重要）\TGSA-master\data\CellLines_DepMap\CCLE_580_18281\census_706\cn_we.csv',index_col=0)
# imp_gene=pd.read_csv(r'D:\xiang_needread\项目代码（重要）\TGSA-master\data\gene.csv',index_col=0,header=0)
# new_data=pd.DataFrame(index=data.index,columns=imp_gene.index)
# print(data)
# print(imp_gene)
# print(new_data)
# for i in data.columns.values.tolist():
#     if i in imp_gene.index:
#         new_data.loc[:,i] =data.loc[:,i]
# new_data.to_csv(r'D:\xiang_needread\项目代码（重要）\TGSA-master\data\CellLines_DepMap\CCLE_580_18281\census_706\cn_we_706.csv')

data=pd.read_csv(r'D:\xiang_needread\项目代码（重要）\TGSA-master\data\CellLines_DepMap\CCLE_580_18281\census_706\exp_we.csv',index_col=0)
imp_gene=pd.read_csv(r'D:\xiang_needread\项目代码（重要）\TGSA-master\data\gene.csv',index_col=0,header=0)
new_data=pd.DataFrame(index=data.index,columns=imp_gene.index)

for i in data.columns.values.tolist():
    if i.split(' ')[0] in imp_gene.index:
        new_data.loc[:,i.split(' ')[0]] =data.loc[:,i]
new_data.to_csv(r'D:\xiang_needread\项目代码（重要）\TGSA-master\data\CellLines_DepMap\CCLE_580_18281\census_706\exp_we_706.csv')

# data=pd.read_csv(r'D:\xiang_needread\项目代码（重要）\TGSA-master\data\CellLines_DepMap\CCLE_580_18281\census_706\mu_we.csv',index_col=0)
# imp_gene=pd.read_csv(r'D:\xiang_needread\项目代码（重要）\TGSA-master\data\gene.csv',index_col=0,header=0)
# new_data=pd.DataFrame(index=data.index,columns=imp_gene.index)
# print(data)
# print(imp_gene)
# print(new_data)
# for i in data.columns.values.tolist():
#     if i in imp_gene.index:
#         new_data.loc[:,i] =data.loc[:,i]
# new_data.to_csv(r'D:\xiang_needread\项目代码（重要）\TGSA-master\data\CellLines_DepMap\CCLE_580_18281\census_706\mu_we_706.csv')