import numpy as np
import scipy.sparse as sp
from GDSC.MOFGCN.myutils import to_coo_matrix, to_tensor, mask
import pandas as pd


class Sampler(object):
    # 对原始边进行采样
    # 采样后生成测试集、训练集
    # 处理完后的训练集转换为torch.tensor格式

    def __init__(self, adj_mat_original, train_index, test_index, null_mask):
        super(Sampler, self).__init__()
        self.adj_mat = to_coo_matrix(adj_mat_original)
        self.train_index = train_index
        self.test_index = test_index
        self.null_mask = null_mask
        self.train_pos = self.sample(train_index)
        self.test_pos = self.sample(test_index)
        self.train_neg, self.test_neg = self.sample_negative()
        self.train_mask = mask(self.train_pos, self.train_neg, dtype=int)
        self.test_mask = mask(self.test_pos, self.test_neg, dtype=bool)
        # print(self.test_pos)
        # print(self.test_neg)
        # print(self.test_mask)
        # self.test_mask = (mask[self.test_pos]==True)
        self.train_data = to_tensor(self.train_pos)
        self.test_data = to_tensor(self.test_pos)

    def sample(self, index):
        row = self.adj_mat.row
        col = self.adj_mat.col
        data = self.adj_mat.data
        sample_row = row[index]
        sample_col = col[index]
        sample_data = data[index]

        sample = sp.coo_matrix((sample_data, (sample_row, sample_col)), shape=self.adj_mat.shape)
        return sample

    def sample_negative(self):
        # identity 表示邻接矩阵是否为二部图
        # 二部图：边的两个节点，是否属于同类结点集
        pos_adj_mat = self.null_mask + self.adj_mat.toarray()
        neg_adj_mat = sp.coo_matrix(np.abs(pos_adj_mat - np.array(2)))
        all_row = neg_adj_mat.row
        all_col = neg_adj_mat.col
        all_data = neg_adj_mat.data
        #index = np.arange(all_data.shape[0])
        index=[]


        # 采样负测试集
        test_n = self.test_index.shape[0]
        #test_neg_index = np.random.choice(index, test_n, replace=False)
        test_neg_index = []
        train_neg_index = []
        cell_lines = [73, 100, 121, 151, 12, 33, 43, 81, 94, 105, 113, 183, 44, 74, 101, 185, 153, 190, 76, 1, 67, 80,
                      108, 136, 170, 0, 53, 98, 26, 27, 39, 68, 90, 109, 120, 123, 128, 147, 148, 187, 59, 175, 69, 88,
                      65, 71]
        drugs = [72, 67, 121, 335, 186, 218, 172, 17, 309, 152, 264, 297, 28, 234, 21, 187, 185, 228, 171, 287, 75, 183,
                 137, 280, 164, 277, 151, 130, 324, 315, 29, 329, 184, 42, 194, 20, 105, 54, 224, 58, 276, 239, 257,
                 208, 306, 312, 106, 113, 265, 84, 225, 99, 327, 271, 73, 161, 350, 233, 199, 57, 213, 86, 96, 33, 302,
                 40, 299, 120, 15, 25, 80, 177, 244, 258, 32, 179, 55, 283, 198, 201, 285, 88, 216, 255, 7, 62, 220,
                 139, 338, 197, 81, 135, 178, 118, 209, 12, 107, 166, 27]

        cell_drug = pd.read_csv(
            r"D:\xiang_needread\项目代码（重要）\MOFGCN-main\GDSC\processed_data\cell_drug_common_binary_we.csv", index_col=0,
            header=0)
        null_mask = pd.read_csv(r"D:\xiang_needread\项目代码（重要）\MOFGCN-main\GDSC\processed_data\null_mask_we.csv",
                                index_col=0, header=0)
        null_mask.fillna(0, inplace=True)
        null_mask = np.array(cell_drug, dtype=np.float32)
        pos_adj_mat = null_mask + cell_drug - np.array(1)

        z = -1
        for i in range(pos_adj_mat.shape[0]):
            for j in range(pos_adj_mat.shape[1]):
                if pos_adj_mat.iloc[i, j] == -1:
                    z += 1
                    index.append(z)
                    if (i in cell_lines) and (j in drugs):
                        test_neg_index.append(z)

        test_row = all_row[test_neg_index]
        test_col = all_col[test_neg_index]
        test_data = all_data[test_neg_index]
        test = sp.coo_matrix((test_data, (test_row, test_col)), shape=self.adj_mat.shape)

        # 采样训练集

        train_neg_index = np.delete(index, test_neg_index)
        train_row = all_row[train_neg_index]
        train_col = all_col[train_neg_index]
        train_data = all_data[train_neg_index]
        train = sp.coo_matrix((train_data, (train_row, train_col)), shape=self.adj_mat.shape)

        return train, test
