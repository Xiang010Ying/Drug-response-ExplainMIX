#!/usr/bin/env python3

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
import weight_utils
import math

weight_calulate = weight_utils.Weight_Cal()

def get_pred(adj_mats,num_relations,tape,pred,head,tail,top_k):

    scores = []
    relat = []

    for i in range(num_relations):

        adj_mat_i = adj_mats[i]

        partitions_i = tf.cast(tf.reduce_any(tf.equal(tf.reshape(adj_mat_i.indices[:, 1], [-1, 1]), head.numpy()[0]), 1), tf.int32)
        partitions_j = tf.cast(tf.reduce_any(tf.equal(tf.reshape(adj_mat_i.indices[:, 1], [-1, 1]), tail.numpy()[0]), 1), tf.int32)
        gradient_score = tape.gradient(pred, adj_mat_i.values)
        # if gradient_score is None:
        #     gradient_score = np.zeros(51895)
        # else:
        #     gradient_score = gradient_score.numpy()
        gradient_score = gradient_score.numpy()


        k_i = tf.where(tf.equal(partitions_i, 1)).numpy()

        p_logic = weight_calulate.logic_weight(head, tail, pred)

        for a in k_i:
            score = gradient_score[a][0]
            p_rele = weight_calulate.relevent_weight(head, adj_mat_i.indices[a[0], 2])
            weight = math.exp(np.sign(pred.numpy())*(p_logic+p_rele))
            score = weight*score
            relat.append((head.numpy()[0], i, adj_mat_i.indices[a[0], 2].numpy()))
            scores.append(score)

        k_j = tf.where(tf.equal(partitions_j, 1)).numpy()
        for b in k_j:
            score = gradient_score[b][0]
            p_rele = weight_calulate.relevent_weight(tail, adj_mat_i.indices[b[0], 2])
            weight = math.exp(np.sign(pred.numpy()) * (p_logic + p_rele))
            score = weight * score
            relat.append((tail.numpy()[0], i, adj_mat_i.indices[b[0],2].numpy()))
            scores.append(score)

    score_dic = dict(zip(relat, scores))
    top_k_scores = sorted(score_dic.items(), key=lambda x : x[1],reverse=True)[:top_k]
    print('top_k_scores',top_k_scores)

    return top_k_scores


if __name__ == '__main__':

    import os
    import utils
    import random as rn
    import RGCN
    from collections import Counter

    SEED = 123
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)

    DATASET = 'cellline_drug'
    RULE = 'link'
    TOP_K = 5
    EMBEDDING_DIM = 32

    MAX_PADDING = 3

    train_triples = np.load('../data/ccle_6_train_triples.npy')
    test_triples = np.load('../data/ccle_6_test_triples.npy')

    entities = np.load(r'../data/ccle_6_entities.npy')
    np.set_printoptions(threshold=np.inf)
    relations=[0,1,2,3]
    NUM_ENTITIES = len(entities)
    NUM_RELATIONS = len(relations)
    OUTPUT_DIM = 15

    ALL_INDICES = tf.reshape(tf.range(0,NUM_ENTITIES,1,dtype=tf.int64), (1,-1))

    ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
    rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

    idx2ent = dict(zip(range(NUM_ENTITIES),entities))
    idx2rel = dict(zip(range(NUM_RELATIONS),relations))

    train_triples = np.array(train_triples)
    test_triples = np.array(test_triples)
    X_train_triples = train_triples
    X_test_triples = test_triples

    ADJACENCY_DATA = tf.concat([
        X_train_triples,
    ], axis=0
    )

    model = RGCN.get_RGCN_Model(
        num_entities=NUM_ENTITIES, #1168
        num_relations=NUM_RELATIONS, #4
        embedding_dim=EMBEDDING_DIM, #10
        output_dim=OUTPUT_DIM, #10
        seed=SEED
    )

    # Import the weights after model training
    model.load_weights(os.path.join('..','data','weights','drug_celline','drug_celline'+'_'+RULE+'.h5'))
    ADJ_MATS = utils.get_adj_mats(ADJACENCY_DATA, NUM_ENTITIES, NUM_RELATIONS)  # 每一种关系为一个大项
    tf_data = tf.data.Dataset.from_tensor_slices(
            (X_test_triples[:,0],X_test_triples[:,1],X_test_triples[:,2],X_test_triples)).batch(1)

    pred_exps = []
    pred_list = []
    y_pred = []
    y_true = []
    y_prob = []

    for head, rel, tail , true_exp in tf_data:
        with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape:
            tape.watch(ADJ_MATS)

            pred = model([
                ALL_INDICES,
                tf.reshape(head,(1,-1)),
                tf.reshape(rel,(1,-1)),
                tf.reshape(tail,(1,-1)),
                ADJ_MATS
                ]
            )
        print('pred',pred)

        pred_list.append([head.numpy()[0], pred.numpy()[0][0], rel.numpy()[0], tail.numpy()[0]])
        if(pred.numpy()[0][0]<0.5):
            y_pred.append(0)
            y_prob.append(pred.numpy()[0][0])
        else:
            y_pred.append(1)
            y_prob.append(pred.numpy()[0][0])
        y_true.append(rel.numpy()[0])

        pred_exp = get_pred(ADJ_MATS,NUM_RELATIONS,tape,pred,head,tail,TOP_K)

        pred_exps.append(pred_exp)

    preds = pred_exps
    np.set_printoptions(threshold = np.inf)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    roc_auc = roc_auc_score(y_true,y_prob)
    prec, reca, _ = precision_recall_curve(np.array(y_true), np.array(y_prob))
    aupr = auc(reca,prec)

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * precision * recall / (precision + recall)
    print(
        'acc={:.4f}|auc={:.4f}|aupr={:.4f}|precision={:.4f}|recall={:.4f}|f1={:.4f}'.format(accuracy, roc_auc, aupr, precision, recall, f1))
    print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
    print('y_pred: ', Counter(y_pred))
    print('y_true: ', Counter(y_true))

    with open('../result/rgcn_results.txt','w') as f:
            f.write('acc={:.4f}|auc={:.4f}|aupr={:.4f}|precision={:.4f}|recall={:.4f}|f1={:.4f}'.format(accuracy, roc_auc, aupr, precision, recall, f1)+'\n')

    with open('../result/rgcn_preds.txt','w') as f:
        for pred in pred_list:
            f.write(str(pred)+'\n')

    print(f'Embedding dim: {EMBEDDING_DIM}')
    print(f"{DATASET} {RULE}")

    with open('../result/weight_best_explain.txt','w') as f:
        for pred in preds:
            f.write(str(pred)+'\n')

    print('Done.')
