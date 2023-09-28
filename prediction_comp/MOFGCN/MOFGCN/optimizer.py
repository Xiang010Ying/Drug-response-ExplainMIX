import torch
from abc import ABC
import torch.nn as nn
import torch.optim as optim
from GDSC.MOFGCN.model import EarlyStop
from GDSC.MOFGCN.myutils import cross_entropy_loss
import scipy.sparse as sp
from GDSC.MOFGCN.myutils import accuracy_binary,precision_binary,recall_binary,f1_score_binary
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_curve
from collections import  Counter

class Optimizer(nn.Module, ABC):
    def __init__(self, model, train_data, test_data, test_mask, train_mask, evaluate_fun,
                 lr=0.01, epochs=100, test_freq=20, device="cpu"):
        super(Optimizer, self).__init__()
        self.model = model.to(device)
        self.train_data = train_data.to(device)
        self.test_data = test_data.to(device)
        self.test_mask = test_mask.to(device)
        self.train_mask = train_mask.to(device)
        self.evaluate_fun = evaluate_fun
        self.lr = lr
        self.epochs = epochs
        self.test_freq = test_freq
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.05)

    def forward(self):
        true_data = torch.masked_select(self.test_data, self.test_mask)
        # true_data = torch.tensor(sp.coo_matrix(self.test_data).data)
        #print('true_data', true_data.shape)
        early_stop = EarlyStop(tolerance=8, data_len=true_data.size()[0])
        # print('self.train_data',self.train_data)
        for epoch in torch.arange(self.epochs):
            predict_data = self.model()

            loss = cross_entropy_loss(self.train_data, predict_data, self.train_mask)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print('epoch',epoch)
            print('loss',loss)
            predict_data_masked = torch.masked_select(predict_data, self.test_mask)
            # if epoch % self.test_freq == 0:
            #     predict_data_masked = torch.masked_select(predict_data, self.test_mask)
                # print('self.test_mask',self.test_mask.shape)
                # print('predict_data', predict_data.shape)
                # print('predict_data', type(predict_data))
                #predict_data_masked = torch.tensor(sp.coo_matrix(predict_data).data)
                #auc_1 = self.evaluate_fun(true_data, predict_data_masked)
               # print("epoch:%4d" % epoch.item(), "loss:%.6f" % loss.item(), "auc:%.4f" % auc_1)
               #  flag = early_stop.stop(auc=auc_1, epoch=epoch.item(), predict_data=predict_data_masked)
               #  if flag:
               #      break
            pred_data = predict_data_masked.detach().numpy()

            print('pred_data', pred_data)
            # 大于零为1小于零为0
            pred_data = np.where(pred_data >= 0.5, 1, 0)
            # print('true_data',true_data)
            true_data = np.where(true_data >= 0.5, 1, 0)

            tn, fp, fn, tp = confusion_matrix(true_data, pred_data).ravel()
            prec, reca, _ = precision_recall_curve(true_data, pred_data)
            aupr = auc(reca, prec)
            auc_ = roc_auc_score(true_data, pred_data)
            accuracy = (tp + tn) / (tn + fp + fn + tp)
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            f1 = 2 * precision * recall / (precision + recall)

            print(
                'acc={:.4f}|auc={:.4f}|aupr={:.4f}|precision={:.4f}|recall={:.4f}|f1={:.4f}'.format(accuracy, auc_,
                                                                                                    aupr,
                                                                                                    precision,
                                                                                                    recall, f1))
            print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
            print('y_pred: ', Counter(pred_data))
            print('y_true: ', Counter(true_data))
        print("Fit finished.")
        #max_index = early_stop.get_best_index()
        # best_epoch = early_stop.epoch_pre[max_index]
        #best_predict = early_stop.predict_data_pre[max_index, :]
        #print('best_predict',best_predict)

        # accuracy = accuracy_binary(true_data, predict_data_masked,0.5)
        # precision = precision_binary(true_data, predict_data_masked,0.5)
        # recall = recall_binary(true_data, predict_data_masked,0.5)

        return predict_data_masked,true_data
