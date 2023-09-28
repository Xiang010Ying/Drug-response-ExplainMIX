import numpy as np
import pandas as pd

data = pd.read_csv('../data/weight_utils_data_6.csv', index_col=0)

def link_num(node1, node2):
    ''' Find the number of nodes that are connected to both nodes, and the number of nodes that are connected to both nodes '''

    if len(np.argwhere(np.array(data.iloc[node1, :].notnull()) == True)) == 0:
        return 0
    gather1 = np.hstack(np.argwhere(np.array(data.iloc[node1, :].notnull()) == True))
    if len(np.argwhere(np.array(data.iloc[node2, :].notnull()) == True)) == 0:
        return 0
    gather2 = np.hstack(np.argwhere(np.array(data.iloc[node2, :].notnull()) == True))

    mid_nodes = []
    for i in gather1:
        if i in gather2:
            mid_nodes.append(i)
    num12 = len(mid_nodes)

    return num12

if __name__ == '__main__':
    cell_drug = []
    for i in range(544):
        cell_drug.append(i)

    num_data = pd.DataFrame(
        columns=cell_drug,
        index=cell_drug
    )
    print('num_data.shape', num_data.shape)
    print('data.shape',data.shape)
    print(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            num = link_num(i,j)
            num_data.iloc[i, j] = num
            num_data.iloc[j, i] = num
    num_data.to_csv('../data/relevent_num.csv')
