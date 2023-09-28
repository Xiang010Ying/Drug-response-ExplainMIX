import pandas as pd
import numpy as np

class Weight_Cal():
    def __init__(self,  **kwargs):
        super(Weight_Cal, self).__init__(**kwargs)
        metric_data = pd.read_csv('../data/weight_utils_data_6.csv', index_col=0)
        weight_num = pd.read_csv('../data/relevent_num.csv', index_col=0)
        logic_weight_sens = pd.read_csv('../data/logic_data_sens.csv', index_col=0)
        logic_weight_nosens = pd.read_csv('../data/logic_data_nosens.csv', index_col=0)

        self.data = metric_data
        self.weightnum = weight_num.fillna(0)
        self.logic_weight_sens = logic_weight_sens.fillna(0)
        self.logic_weight_nosens = logic_weight_nosens.fillna(0)


    def logic_weight(self, head, tail, pred):
        '''Find a logical probability'''

        i = head.numpy()[0]
        j = tail.numpy()[0]
        pred = pred.numpy()[0]
        pred_ij = 1 if pred>=0.5 else -1

        if pred_ij == 1:
            P = self.logic_weight_sens.iloc[i,j]
        else:
            P = self.logic_weight_nosens.iloc[i,j]
        return P

    def link_num(self, node1, node2):
        '''Find the number of nodes that are connected to both nodes, and the number of nodes that are connected to both nodes'''

        gather1 = np.hstack(np.argwhere(np.array(self.data.iloc[node1, :].notnull()) == True))
        gather2 = np.hstack(np.argwhere(np.array(self.data.iloc[node2, :].notnull()) == True))

        num1 = len(gather1)
        num2 = len(gather2)
        mid_nodes = []
        for i in gather1:
            if i in gather2:
                mid_nodes.append(i)
        num12 = len(mid_nodes)

        return num1, num2, num12

    def relevent_weight(self, head, tail):
        '''Calculate how important two nodes are to each other'''

        i = head.numpy()[0]
        j = tail.numpy()

        numi, numj, numij = self.link_num(i, j)

        snum_i = 0
        n = len(self.weightnum.iloc[i, :])
        snum_i = snum_i + n

        snum_j = 0
        n = len(self.weightnum.iloc[j, :])
        snum_j = snum_j + n

        # 计算P权重
        P = numij*(1/numi+1/numj)/(1/numi*snum_i+1/numj*snum_j)

        return P

if __name__ == '__main__':
    import weight_utils
    weight_calulate=weight_utils.Weight_Cal()
    weight_calulate.logic_weight(2,1000,0.6)
    a=weight_calulate.relevent_weight(1, 10)
    print(a)
