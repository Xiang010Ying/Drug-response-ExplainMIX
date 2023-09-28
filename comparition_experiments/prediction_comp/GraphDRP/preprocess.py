import os
import csv
from pubchempy import *
import numpy as np
import numbers
import h5py
import math
import pandas as pd
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *
import random
import pickle
import sys
import matplotlib.pyplot as plt
import argparse

def is_not_float(string_list):
    try:
        for string in string_list:
            float(string)
        return False
    except:
        return True

"""
The following 4 function is used to preprocess the drug data. We download the drug list manually, and download the SMILES format using pubchempy. Since this part is time consuming, I write the cids and SMILES into a csv file. 
"""

folder = "data/"
#folder = ""

def load_drug_list():
    filename = folder + "Druglist.csv"
    csvfile = open(filename, "rb")
    reader = csv.reader(csvfile)
    next(reader, None)
    drugs = []
    for line in reader:
        drugs.append(line[0])
    drugs = list(set(drugs))
    return drugs

def write_drug_cid():
    drugs = load_drug_list()
    drug_id = []
    datas = []
    outputfile = open(folder + 'pychem_cid.csv', 'wb')
    wr = csv.writer(outputfile)
    unknow_drug = []
    for drug in drugs:
        c = get_compounds(drug, 'name')
        if drug.isdigit():
            cid = int(drug)
        elif len(c) == 0:
            unknow_drug.append(drug)
            continue
        else:
            cid = c[0].cid
        print(drug, cid)
        drug_id.append(cid)
        row = [drug, str(cid)]
        wr.writerow(row)
    outputfile.close()
    outputfile = open(folder + "unknow_drug_by_pychem.csv", 'wb')
    wr = csv.writer(outputfile)
    wr.writerow(unknow_drug)

def cid_from_other_source():
    """
    some drug can not be found in pychem, so I try to find some cid manually.
    the small_molecule.csv is downloaded from http://lincs.hms.harvard.edu/db/sm/
    """
    f = open(folder + "small_molecule.csv", 'r')
    reader = csv.reader(f)
    reader.next()
    cid_dict = {}
    for item in reader:
        name = item[1]
        cid = item[4]
        if not name in cid_dict: 
            cid_dict[name] = str(cid)

    unknow_drug = open(folder + "unknow_drug_by_pychem.csv").readline().split(",")
    drug_cid_dict = {k:v for k,v in cid_dict.iteritems() if k in unknow_drug and not is_not_float([v])}
    return drug_cid_dict

def load_cid_dict():
    reader = csv.reader(open(folder + "pychem_cid.csv"))
    pychem_dict = {}
    for item in reader:
        pychem_dict[item[0]] = item[1]
    pychem_dict.update(cid_from_other_source())
    return pychem_dict


def download_smiles():
    cids_dict = load_cid_dict()
    cids = [v for k,v in cids_dict.iteritems()]
    inv_cids_dict = {v:k for k,v in cids_dict.iteritems()}
    download('CSV', folder + 'drug_smiles.csv', cids, operation='property/CanonicalSMILES,IsomericSMILES', overwrite=True)
    f = open(folder + 'drug_smiles.csv')
    reader = csv.reader(f)
    header = ['name'] + reader.next()
    content = []
    for line in reader:
        content.append([inv_cids_dict[line[0]]] + line)
    f.close()
    f = open(folder + "drug_smiles.csv", "w")
    writer = csv.writer(f)
    writer.writerow(header)
    for item in content:
        writer.writerow(item)
    f.close()

"""
The following code will convert the SMILES format into onehot format
"""

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index

def load_drug_smile():
    # reader = csv.reader(open(folder + "drug_smiles.csv"))
    reader = csv.reader(open(folder + "drug_smiles_we.csv"))
    next(reader, None)

    drug_dict = {}
    drug_smile = []

    for item in reader:
        name = item[0]
        smile = item[2]

        if name in drug_dict:
            pos = drug_dict[name]
        else:
            pos = len(drug_dict)
            drug_dict[name] = pos
        drug_smile.append(smile)
    
    smile_graph = {}
    for smile in drug_smile:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    
    return drug_dict, drug_smile, smile_graph

def save_cell_mut_matrix():
    # f = open(folder + "PANCANCER_Genetic_feature.csv")
    f = open(folder + "PANCANCER_Genetic_feature_we.csv")
    reader = csv.reader(f)
    next(reader)
    features = {}
    cell_dict = {}
    mut_dict = {}
    matrix_list = []

    for item in reader:
        cell_id = item[1]
        mut = item[5]
        is_mutated = int(item[6])

        if mut in mut_dict:
            col = mut_dict[mut]
        else:
            col = len(mut_dict)
            mut_dict[mut] = col

        if cell_id in cell_dict:
            row = cell_dict[cell_id]
        else:
            row = len(cell_dict)
            cell_dict[cell_id] = row
        if is_mutated == 1:
            matrix_list.append((row, col))
    
    cell_feature = np.zeros((len(cell_dict), len(mut_dict)))

    for item in matrix_list:
        cell_feature[item[0], item[1]] = 1

    with open('mut_dict', 'wb') as fp:
        pickle.dump(mut_dict, fp)
    
    return cell_dict, cell_feature


"""
This part is used to extract the drug - cell interaction strength. it contains IC50, AUC, Max conc, RMSE, Z_score
"""
from sklearn.model_selection import train_test_split
def save_mix_drug_cell_matrix():
    # f = open(folder + "PANCANCER_IC.csv")
    f = open(folder + "PANCANCER_IC_we.csv")
    reader = csv.reader(f)
    next(reader)
    IC = pd.read_csv(folder + "PANCANCER_IC_we.csv")

    cell_dict, cell_feature = save_cell_mut_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()

    temp_data = []
    bExist = np.zeros((len(drug_dict), len(cell_dict)))

    for item in reader:
        drug = item[0]
        cell = item[3]
        ic50 = item[8]
        # ic50 = float(ic50)
        #ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        ic50 = 0 if float(ic50)>0 else 1
        temp_data.append((drug, cell, ic50))

    cell_lines = ['WM115', 'KMRC1', 'MDAMB231', 'DANG', 'LCLC103H', 'LXF289', 'SKCO1', 'PATU8988T', 'CAL851', 'RT112',
                  'HCC1806', 'BT20', 'SNU61', 'SNU449', 'NCIH196', 'OVCAR4', 'KYSE510', 'RVH421', 'CORL88', 'KMS12BM',
                  'MDAMB453', 'CAL120', 'MONOMAC6', 'SKES1', 'SBC5', 'UACC62', 'SNU1', 'CAL33', 'NCIN87', 'NCIH747',
                  'HCC1937', 'SNUC5', 'NCIH2052', 'NCIH661', 'HCC1500', 'IGR37', 'SUIT2', 'OSRC2', 'OV90', 'TE4',
                  'NCIH2030', 'SKMEL5', 'KMS11', 'NCIH146', 'NCIH2291', 'NALM6']
    drugs = ['JNJ38877605', 'CP724714', 'Talazoparib', 'YM201636', 'S-Trityl-L-cysteine', 'Tenovin-6', 'Brivanib',
             'Omipalisib', 'WYE-125132', 'AZD3759', 'CZC24832', 'Mitomycin-C', 'Motesanib', 'SL0101', 'ARRY-520',
             'TGX221', 'Cyclopamine', 'SB505124', 'TPCA-1', 'LJI308', 'OSI-930', 'NSC319726', 'VE-822', 'P22077',
             'Sepantronium bromide', 'I-BRD9', 'Afuresertib', 'Lapatinib', 'PHA-665752', 'VX-702', 'Elesclomol', 'CMK',
             'IOX2', 'Dactolisib', 'Embelin', 'Panobinostat', 'Olaparib', 'Foretinib', 'Ara-G', 'Fluorouracil',
             'GSK2578215A', 'EHT-1864', 'AZ960', 'FR-180204', 'AR-42', 'Refametinib', 'Axitinib', 'Obatoclax Mesylate',
             'GSK2606414', 'PF-00299804', 'IMD-0354', 'Capivasertib', 'CGP-082996', 'Elephantin', 'Masitinib',
             'MK-8776', 'GNE-317', 'BX795', 'GW-2580', 'Etoposide', 'SU11274', 'GSK1059615', 'Lenalidomide', 'AZD6094',
             'BMS-754807', 'Afatinib', 'JNK-9L', 'Fulvestrant', 'CUDC-101', 'Temsirolimus', 'GSK690693', 'A-770041',
             'BPTES', 'XAV939', 'Trametinib', 'GW843682X', 'Pyrimethamine', 'WEHI-539', 'LFM-A13', 'BX-912',
             'Pyridostatin', 'BIBF-1120', 'Y-39983', 'YK-4-279', 'Vinorelbine', 'Avagacestat', 'TWS119', 'OTX015',
             'Bosutinib', 'BAY-61-3606', 'T0901317', 'Entinostat', 'WH-4-023', 'Niraparib', 'Tubastatin A',
             'Ispinesib Mesylate', 'PLX-4720', 'TW 37', 'Tanespimycin']
    test_index = []
    for i in range(IC.shape[0]):
        if (IC.loc[i, 'Cell line name'] in cell_lines) and (IC.loc[i, 'Drug name'] in drugs):
            test_index.append(i)
    test_set = test_index

    train_index = []
    for i in range(IC.shape[0]):
        if (i not in test_index) and (IC.loc[i,'Cell line name'] not in cell_lines) and  (IC.loc[i, 'Drug name'] not in drugs):
            train_index.append(i)
    print('all_learn',IC.shape[0])
    print('test_learn',len(test_set))
    print('train_learn',len(train_index))
    train_set = train_index
    train_set, val_set = train_test_split(train_set, test_size=0.2, random_state=22)

    xd = []
    xc = []
    y = []
    lst_drug = []
    lst_cell = []
    random.shuffle(temp_data)
    for data in temp_data:
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict:
            xd.append(drug_smile[drug_dict[drug]])
            xc.append(cell_feature[cell_dict[cell]])
            y.append(ic50)
            bExist[drug_dict[drug], cell_dict[cell]] = 1
            lst_drug.append(drug)
            lst_cell.append(cell)

    with open('drug_dict_we', 'wb') as fp:
        pickle.dump(drug_dict, fp)

    xd, xc, y = np.asarray(xd), np.asarray(xc), np.asarray(y)

    size = int(xd.shape[0] * 0.8)
    size1 = int(xd.shape[0] * 0.9)

    with open('list_drug_mix_test_we', 'wb') as fp:
        pickle.dump(np.array(lst_drug)[test_set], fp)
        
    with open('list_cell_mix_test_we', 'wb') as fp:
        pickle.dump(np.array(lst_cell)[test_set], fp)

    # xd_train = xd[:size]
    # xd_val = xd[size:size1]
    # xd_test = xd[size1:]
    #
    # xc_train = xc[:size]
    # xc_val = xc[size:size1]
    # xc_test = xc[size1:]
    #
    # y_train = y[:size]
    # y_val = y[size:size1]
    # y_test = y[size1:]

    xd_train = xd[train_set]
    xd_val = xd[val_set]
    xd_test = xd[test_set]

    xc_train = xc[train_set]
    xc_val = xc[val_set]
    xc_test = xc[test_set]

    y_train = y[train_set]
    y_val = y[val_set]
    y_test = y[test_set]

    dataset = 'GDSC'
    print('preparing ', dataset + '_train.pt in pytorch format!')

    train_data = TestbedDataset(root='data', dataset=dataset+'_train_mix', xd=xd_train, xt=xc_train, y=y_train, smile_graph=smile_graph)
    val_data = TestbedDataset(root='data', dataset=dataset+'_val_mix', xd=xd_val, xt=xc_val, y=y_val, smile_graph=smile_graph)
    test_data = TestbedDataset(root='data', dataset=dataset+'_test_mix', xd=xd_test, xt=xc_test, y=y_test, smile_graph=smile_graph)


def save_blind_drug_matrix():
    # f = open(folder + "PANCANCER_IC.csv")
    f = open(folder + "PANCANCER_IC_we.csv")
    reader = csv.reader(f)
    next(reader)

    cell_dict, cell_feature = save_cell_mut_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()

    matrix_list = []

    temp_data = []

    xd_train = []
    xc_train = []
    y_train = []

    xd_val = []
    xc_val = []
    y_val = []

    xd_test = []
    xc_test = []
    y_test = []

    xd_unknown = []
    xc_unknown = []
    y_unknown = []

    dict_drug_cell = {}

    bExist = np.zeros((len(drug_dict), len(cell_dict)))

    for item in reader:
        drug = item[0]
        cell = item[3]
        ic50 = item[8]
        ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        
        temp_data.append((drug, cell, ic50))

    random.shuffle(temp_data)
    
    for data in temp_data:
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict:
            if drug in dict_drug_cell:
                dict_drug_cell[drug].append((cell, ic50))
            else:
                dict_drug_cell[drug] = [(cell, ic50)]
            
            bExist[drug_dict[drug], cell_dict[cell]] = 1

    lstDrugTest = []

    size = int(len(dict_drug_cell) * 0.8)
    size1 = int(len(dict_drug_cell) * 0.9)
    pos = 0
    for drug,values in dict_drug_cell.items():
        pos += 1
        for v in values:
            cell, ic50 = v
            if pos < size:
                xd_train.append(drug_smile[drug_dict[drug]])
                xc_train.append(cell_feature[cell_dict[cell]])
                y_train.append(ic50)
            elif pos < size1:
                xd_val.append(drug_smile[drug_dict[drug]])
                xc_val.append(cell_feature[cell_dict[cell]])
                y_val.append(ic50)
            else:
                xd_test.append(drug_smile[drug_dict[drug]])
                xc_test.append(cell_feature[cell_dict[cell]])
                y_test.append(ic50)
                lstDrugTest.append(drug)

    with open('drug_bind_test_we', 'wb') as fp:
        pickle.dump(lstDrugTest, fp)
    
    print(len(y_train), len(y_val), len(y_test))

    xd_train, xc_train, y_train = np.asarray(xd_train), np.asarray(xc_train), np.asarray(y_train)
    xd_val, xc_val, y_val = np.asarray(xd_val), np.asarray(xc_val), np.asarray(y_val)
    xd_test, xc_test, y_test = np.asarray(xd_test), np.asarray(xc_test), np.asarray(y_test)

    dataset = 'GDSC'
    print('preparing ', dataset + '_train.pt in pytorch format!')
    train_data = TestbedDataset(root='data', dataset=dataset+'_train_blind', xd=xd_train, xt=xc_train, y=y_train, smile_graph=smile_graph)
    val_data = TestbedDataset(root='data', dataset=dataset+'_val_blind', xd=xd_val, xt=xc_val, y=y_val, smile_graph=smile_graph)
    test_data = TestbedDataset(root='data', dataset=dataset+'_test_blind', xd=xd_test, xt=xc_test, y=y_test, smile_graph=smile_graph)


def save_blind_cell_matrix():
    # f = open(folder + "PANCANCER_IC.csv")
    f = open(folder + "PANCANCER_IC_we.csv")
    reader = csv.reader(f)
    next(reader)

    cell_dict, cell_feature = save_cell_mut_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()

    matrix_list = []

    temp_data = []

    xd_train = []
    xc_train = []
    y_train = []

    xd_val = []
    xc_val = []
    y_val = []

    xd_test = []
    xc_test = []
    y_test = []

    xd_unknown = []
    xc_unknown = []
    y_unknown = []

    dict_drug_cell = {}

    bExist = np.zeros((len(drug_dict), len(cell_dict)))

    for item in reader:
        drug = item[0]
        cell = item[3]
        ic50 = item[8]
        ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        
        temp_data.append((drug, cell, ic50))

    random.shuffle(temp_data)
    
    for data in temp_data:
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict:
            if cell in dict_drug_cell:
                dict_drug_cell[cell].append((drug, ic50))
            else:
                dict_drug_cell[cell] = [(drug, ic50)]
            
            bExist[drug_dict[drug], cell_dict[cell]] = 1

    lstCellTest = []

    size = int(len(dict_drug_cell) * 0.8)
    size1 = int(len(dict_drug_cell) * 0.9)

    pos = 0
    for cell,values in dict_drug_cell.items():
        pos += 1
        for v in values:
            drug, ic50 = v
            if pos < size:
                xd_train.append(drug_smile[drug_dict[drug]])
                xc_train.append(cell_feature[cell_dict[cell]])
                y_train.append(ic50)
            elif pos < size1:
                xd_val.append(drug_smile[drug_dict[drug]])
                xc_val.append(cell_feature[cell_dict[cell]])
                y_val.append(ic50)
            else:
                xd_test.append(drug_smile[drug_dict[drug]])
                xc_test.append(cell_feature[cell_dict[cell]])
                y_test.append(ic50)
                lstCellTest.append(cell)

    with open('cell_bind_test_we', 'wb') as fp:
        pickle.dump(lstCellTest, fp)
    
    print(len(y_train), len(y_val), len(y_test))

    xd_train, xc_train, y_train = np.asarray(xd_train), np.asarray(xc_train), np.asarray(y_train)
    xd_val, xc_val, y_val = np.asarray(xd_val), np.asarray(xc_val), np.asarray(y_val)
    xd_test, xc_test, y_test = np.asarray(xd_test), np.asarray(xc_test), np.asarray(y_test)

    dataset = 'GDSC'
    print('preparing ', dataset + '_train.pt in pytorch format!')
    train_data = TestbedDataset(root='data', dataset=dataset+'_train_cell_blind', xd=xd_train, xt=xc_train, y=y_train, smile_graph=smile_graph)
    val_data = TestbedDataset(root='data', dataset=dataset+'_val_cell_blind', xd=xd_val, xt=xc_val, y=y_val, smile_graph=smile_graph)
    test_data = TestbedDataset(root='data', dataset=dataset+'_test_cell_blind', xd=xd_test, xt=xc_test, y=y_test, smile_graph=smile_graph)

def save_best_individual_drug_cell_matrix():
    f = open(folder + "PANCANCER_IC.csv")
    reader = csv.reader(f)
    next(reader)

    cell_dict, cell_feature = save_cell_mut_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()

    matrix_list = []

    temp_data = []

    xd_train = []
    xc_train = []
    y_train = []

    dict_drug_cell = {}

    bExist = np.zeros((len(drug_dict), len(cell_dict)))
    i=0
    for item in reader:
        drug = item[0]
        cell = item[3]
        ic50 = item[8]
        ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        
        if drug == "Bortezomib":
            temp_data.append((drug, cell, ic50))
    random.shuffle(temp_data)
    
    for data in temp_data:
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict:
            if drug in dict_drug_cell:
                dict_drug_cell[drug].append((cell, ic50))
            else:
                dict_drug_cell[drug] = [(cell, ic50)]
            
            bExist[drug_dict[drug], cell_dict[cell]] = 1
    cells = []
    for drug,values in dict_drug_cell.items():
        for v in values:
            cell, ic50 = v
            xd_train.append(drug_smile[drug_dict[drug]])
            xc_train.append(cell_feature[cell_dict[cell]])
            y_train.append(ic50)
            cells.append(cell)

    xd_train, xc_train, y_train = np.asarray(xd_train), np.asarray(xc_train), np.asarray(y_train)
    with open('cell_blind_sal', 'wb') as fp:
        pickle.dump(cells, fp)
    dataset = 'GDSC'
    print('preparing ', dataset + '_train.pt in pytorch format!')
    train_data = TestbedDataset(root='data', dataset=dataset+'_bortezomib', xd=xd_train, xt=xc_train, y=y_train, smile_graph=smile_graph, saliency_map=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='prepare dataset to train model')
    parser.add_argument('--choice', type=int, required=False, default=0, help='0.mix test, 1.saliency value, 2.drug blind, 3.cell blind')
    args = parser.parse_args()
    choice = args.choice
    if choice == 0:
        # save mix test dataset
        save_mix_drug_cell_matrix()
    elif choice == 1:
        # save saliency map dataset
        save_best_individual_drug_cell_matrix()
    elif choice == 2:
        # save blind drug dataset
        save_blind_drug_matrix()
    elif choice == 3:
        # save blind cell dataset
        save_blind_cell_matrix()
    else:
        print("Invalide option, choose 0 -> 4")