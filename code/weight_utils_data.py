import pandas as pd
import numpy as np
import os

cell_drug = []
for i in range(544):
    cell_drug.append(i)

data = pd.DataFrame(
    columns=cell_drug,
    index=cell_drug
)
sensitive_data = pd.read_csv(r'..\data\sensitive_index_data.csv',header=None)

for i in range(len(sensitive_data)):
    a = sensitive_data.iloc[i, 0]
    b = sensitive_data.iloc[i, 1] + 192
    data.iloc[a, b] = 1
    data.iloc[b, a] = 1

nosensitive_data = pd.read_csv(r'..\data\nosensitive_index_data.csv',header=None)
for i in range(len(nosensitive_data)):
    a = nosensitive_data.iloc[i, 0]
    b = nosensitive_data.iloc[i, 1] + 192
    data.iloc[a, b] = -1
    data.iloc[b, a] = -1

np.set_printoptions(threshold=np.inf)
cell_line = pd.read_csv(r'D:\XIAOXIANG\可解释性实验\data\CCLE\CCLE_Characteristic\6_multiple_omic_data\aaaaaaaaaaaaa.csv', index_col=0)
for i in range(cell_line.shape[0]):
    for j in range(cell_line.shape[1]):
        if (cell_line.iloc[i, j] > 0.004): #>0
            data.iloc[i, j] = 0

drug_line = pd.read_csv(r'D:\XIAOXIANG\可解释性实验\data\CCLE\Drug_Characteristic\drug_sim.csv', index_col=0)
for i in range(drug_line.shape[0]):
    for j in range(drug_line.shape[1]):
        if (drug_line.iloc[i, j] > 0.5):
            data.iloc[i+192, j+192] = 0
data.to_csv(r'../data/weight_utils_data_6.csv')
print(data)