import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

"""

This part, as the input preparation of the code, 
includes four kinds of relationships, 
constructing the sensitivity and insensitivity relationship between cell lines and drugs, 
cell line similarity relationship, and drug similarity relationship

"""


"""
cell-lines ---- drugs
"""
triples = []
triples0 = []
triples1 = []

# sensitive data
sensitive_data = pd.read_csv(r'..\data\sensitive_index_data.csv',header=None)

# no_sensitive data
nosensitive_data = pd.read_csv(r'..\data\nosensitive_index_data.csv',header=None)

# 6 192 cell lines
cellnum = 192
for i in range(len(nosensitive_data)):
    a=nosensitive_data.iloc[i,0]
    b = nosensitive_data.iloc[i, 1]+cellnum
    temp = [b, 0, a]
    triples0.append(temp)

for i in range(len(sensitive_data)):
    a=sensitive_data.iloc[i,0]
    b = sensitive_data.iloc[i, 1]+cellnum
    temp = [b, 1, a]
    triples1.append(temp)

np.set_printoptions(threshold=np.inf)
triples = triples0+triples1
train_triples, test_triples = train_test_split(triples, test_size=2968, random_state = 42, shuffle =True)#12348  11  42 22 123
print('train_triples',len(train_triples))


"""
cell-lines ---- cell lines

To generate the cell line combination fusion file, 
we use the SNF method, 
refer to https://www.nature.com/articles/nmeth.2810 for details, 
you can also choose other methods, such as average value

'../data/6_omic_data.csv'  The file contains data for six omics fusion
'../data/5_omic_data_rmm.csv' The file contains data for five omics fusions and no metabolome data
'../data/5_omic_data_rmp.csv' The file contains data for five omics fusions and no proteomic data
'../data/4_omic_data_rmpm.csv' The file contains data for five omics fusion, excluding proteomic and metabolomic data

"""

threshold1 = 0.003
cell_line=pd.read_csv(r'..\data\6_omic_data.csv',index_col=0)
cell_line=cell_line.fillna(0)

for i in range(cell_line.shape[0]):
    for j in range(cell_line.shape[1]):
        if(cell_line.iloc[i,j]>threshold1 and i!=j):  #   >0
            train_triples.append([i,2,j])


'''
drugs ---- drugs

To generate the drug profile data, we use PaDEL software,
https://www.researchgate.net/publication/50598672_PaDEL-Descriptor_An_Open_Source_Software_to_Calculate_Molecular_Descriptors_and_Fingerprints
to calculate the drug similarity we use cosine similarity, 
you can also choose other methods, 
such as Euclidean distance and so on

'''
threshold2 = 0.5
drug_sim=pd.read_csv(r'..\data\drug_sim.csv',index_col=0)

for i in range(drug_sim.shape[0]):
    for j in range(drug_sim.shape[1]):
        if(drug_sim.iloc[i,j]>=threshold2 and i!=j):
            train_triples.append([i+192,3,j+192])

train_triples = np.array(train_triples)
test_triples = np.array(test_triples)

'''
Scrambled data to improve the generalization ability of the model
'''

index=[i for i in range(len(train_triples))]
random.shuffle(index)
train_triples = train_triples[index]

index=[i for i in range(len(test_triples))]
random.shuffle(index)
test_triples = test_triples[index]

#entity  6 544
entities=[]
for i in range(544):
    entities.append(i)
entities=np.array(entities)

np.save('../data/ccle_6_train_triples.npy',train_triples)
np.save('../data/ccle_6_test_triples.npy',test_triples)
np.save('../data/ccle_6_entities.npy',entities)

print('----train_triples----',len(train_triples))
print('----test_triples----',len(test_triples))
print('----entities----',entities.shape)
