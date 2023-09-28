#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_curve



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
    TOP_K = 10
    EMBEDDING_DIM = 32  # 125
    MAX_PADDING = 3

    np.set_printoptions(threshold=np.inf)
    train_triples = np.load('../data/ccle_6_train_triples.npy')
    test_triples = np.load('../data/ccle_6_test_triples.npy')

    entities = np.load(r'../data/ccle_6_entities.npy')

    relations = [0, 1, 2, 3]  # sensitive nosensitive
    NUM_ENTITIES = len(entities)
    NUM_RELATIONS = len(relations)
    OUTPUT_DIM = 15  # 15

    ALL_INDICES = tf.reshape(tf.range(0, NUM_ENTITIES, 1, dtype=tf.int64), (1, -1))

    ent2idx = dict(zip(entities, range(NUM_ENTITIES)))

    rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

    idx2ent = dict(zip(range(NUM_ENTITIES), entities))
    idx2rel = dict(zip(range(NUM_RELATIONS), relations))

    train_triples = np.array(train_triples)
    test_triples = np.array(test_triples)
    X_train_triples = train_triples
    X_test_triples = test_triples

    ADJACENCY_DATA = tf.concat([
        X_train_triples,
    ], axis=0
    )

    model = RGCN.get_RGCN_Model(
        num_entities=NUM_ENTITIES,  # 1168
        num_relations=NUM_RELATIONS,  # 4
        embedding_dim=EMBEDDING_DIM,  # 10
        output_dim=OUTPUT_DIM,  # 10
        seed=SEED
    )
    model.load_weights(os.path.join('..','data','weights','drug_celline','drug_celline'+'_'+RULE+'.h5'))

    X_train_triples = np.expand_dims(X_train_triples, axis=0)

    ADJ_MATS = utils.get_adj_mats(ADJACENCY_DATA, NUM_ENTITIES, NUM_RELATIONS)  # 每一种关系为一个大项
    tf_data = tf.data.Dataset.from_tensor_slices(
        (X_test_triples[:, 0], X_test_triples[:, 1], X_test_triples[:, 2], X_test_triples)).batch(1)

    pred_exps = []
    pred_list = []
    y_pred = []
    y_true = []
    y_prob = []

    for head, rel, tail, true_exp in tf_data:

        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(ADJ_MATS)

            pred = model([
                ALL_INDICES,
                tf.reshape(head, (1, -1)),
                tf.reshape(rel, (1, -1)),
                tf.reshape(tail, (1, -1)),
                ADJ_MATS
            ]
            )
        print('pred', pred)
        pred_list.append([head.numpy()[0], pred.numpy()[0][0], rel.numpy()[0], tail.numpy()[0]])
        if (pred.numpy()[0][0] < 0.5):
            y_pred.append(0)
            y_prob.append(pred.numpy()[0][0])
        else:
            y_pred.append(1)
            y_prob.append(pred.numpy()[0][0])
        y_true.append(rel.numpy()[0])

    np.set_printoptions(threshold=np.inf)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    roc_auc = roc_auc_score(y_true, y_prob)
    prec, reca, _ = precision_recall_curve(np.array(y_true), np.array(y_prob))
    aupr = auc(reca, prec)

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * precision * recall / (precision + recall)


    print(
        'acc={:.4f}|auc={:.4f}|aupr={:.4f}|precision={:.4f}|recall={:.4f}|f1={:.4f}'.format(accuracy, roc_auc, aupr,
                                                                                            precision, recall, f1))
    print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
    print('y_pred: ', Counter(y_pred))
    print('y_true: ', Counter(y_true))

    with open('./rgcn_preds.txt', 'w') as f:
        for pred in pred_list:
            f.write(str(pred) + '\n')

    print(f'Embedding dim: {EMBEDDING_DIM}')
    print(f"{DATASET} {RULE}")

    print('Done.')
