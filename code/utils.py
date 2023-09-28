#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

def pad_weight(trace,longest_trace,unk_weight):

    while trace.shape[0] != longest_trace:
        trace = np.concatenate([trace,unk_weight],axis=0)

    return trace

def f1(precision,recall):
    return 2 * (precision*recall) / (precision + recall)

def jaccard_score_np(true_exp,pred_exp):
        
    num_true_traces = true_exp.shape[0]
    num_pred_traces = pred_exp.shape[0]

    count = 0
    for pred_row in pred_exp:
        for true_row in true_exp:
            if (pred_row == true_row).all():
                count +=1

    score = count / (num_true_traces + num_pred_traces-count)
    
    return score

def jaccard_score_tf(true_exp,pred_exp):

    num_true_traces = tf.shape(true_exp)[0]
    num_pred_traces = tf.shape(pred_exp)[0]

    count = 0
    for i in range(num_pred_traces):

        pred_row = pred_exp[i]

        for j in range(num_true_traces):

            true_row = true_exp[j]

            count += tf.cond(tf.reduce_all(pred_row == true_row), lambda :1, lambda:0)

    score = count / (num_true_traces + num_pred_traces-count)
    
    return score

def remove_padding_np(exp,unk_ent_id, unk_rel_id,axis=1):

    unk = np.array([unk_ent_id, unk_rel_id, unk_ent_id],dtype=object)

    exp_mask = (exp != unk).all(axis=axis)

    masked_exp = exp[exp_mask]

    return masked_exp

def remove_padding_tf(exp,unk_ent_id, unk_rel_id,axis=-1):

    unk = tf.cast(
        tf.convert_to_tensor([unk_ent_id, unk_rel_id, unk_ent_id]),
        dtype=exp.dtype)

    exp_mask = tf.reduce_all(tf.math.not_equal(exp, unk),axis=axis)

    masked_exp = tf.boolean_mask(exp,exp_mask)

    return masked_exp

def max_jaccard_np(current_traces,pred_exp,true_weight,
    unk_ent_id,unk_rel_id,unk_weight_id,return_idx=False):

    ''''
    pred_exp must have shape[0] >= 1

    pred_exp: 2 dimensional (num_triples,3)

    '''
    
    jaccards = []
    sum_weights = []
    
    for i in range(len(current_traces)):
        
        true_exp = remove_padding_np(current_traces[i],unk_ent_id,unk_rel_id)

        weight = true_weight[i][true_weight[i] != unk_weight_id]

        sum_weight = sum([float(num) for num in weight])

        sum_weights.append(sum_weight)

        jaccard = jaccard_score_np(true_exp, pred_exp)

        jaccards.append(jaccard)

    max_indices = np.array(jaccards) == max(jaccards)

    if max_indices.sum() > 1:
        max_idx = np.argmax(max_indices * sum_weights)
        max_jaccard = jaccards[max_idx]
    else:
        max_jaccard = max(jaccards)
        max_idx = np.argmax(jaccards)
    
    if return_idx:
        return max_jaccard, max_idx
    return max_jaccard

def max_jaccard_tf(current_traces,pred_exp,unk_ent_id,unk_rel_id):
    
    jaccards = []
    
    for i in range(len(current_traces)):
        
        trace = remove_padding_tf(current_traces[i],unk_ent_id,unk_rel_id)

        jaccard = jaccard_score_tf(trace, pred_exp)

        jaccards.append(jaccard)

    return max(jaccards)


def array2idx(dataset,ent2idx,rel2idx):
    
    if dataset.ndim == 2:
        
        data = []
        
        for head, rel, tail in dataset:
            
            head_idx = ent2idx[head]
            tail_idx = ent2idx[tail]
            rel_idx = rel2idx[rel]
            
            data.append((head_idx, rel_idx, tail_idx))

        data = np.array(data,dtype=np.int64)

    elif dataset.ndim == 3:
        
        data = []

        for i in range(len(dataset)):
            
            temp_array = []
        
            for head,rel,tail in dataset[i,:,:]:

                head_idx = ent2idx[head]
                tail_idx = ent2idx[tail]
                rel_idx = rel2idx[rel]

                temp_array.append((head_idx,rel_idx,tail_idx))

            data.append(temp_array)
            
        data = np.array(data,dtype=np.int64).reshape(-1,dataset.shape[1],3)

    elif dataset.ndim == 4:

        data = []

        for i in range(len(dataset)):

            temp_array = []

            for j in range(len(dataset[i])):

                temp_array_1 = []

                for head,rel,tail in dataset[i,j]:

                    head_idx = ent2idx[head]
                    tail_idx = ent2idx[tail]
                    rel_idx = rel2idx[rel]

                    temp_array_1.append((head_idx,rel_idx,tail_idx))

                temp_array.append(temp_array_1)

            data.append(temp_array)

        data = np.array(data)

    return data

def idx2array(dataset,idx2ent,idx2rel):
    
    if dataset.ndim == 2:
        
        data = []
        
        for head_idx, rel_idx, tail_idx in dataset:
            
            head = idx2ent[head_idx]
            tail = idx2ent[tail_idx]
            rel = idx2rel[rel_idx]
            
            data.append((head, rel, tail))

        data = np.array(data)

    elif dataset.ndim == 3:
        
        data = []

        for i in range(len(dataset)):
            
            temp_array = []
        
            for head_idx, rel_idx, tail_idx in dataset[i,:,:]:

                head = idx2ent[head_idx]
                tail = idx2ent[tail_idx]
                rel = idx2rel[rel_idx]

                temp_array.append((head,rel,tail))

            data.append(temp_array)
            
        data = np.array(data).reshape(-1,dataset.shape[1],3)

    elif dataset.ndim == 4:

        data = []

        for i in range(len(dataset)):

            temp_array = []

            for j in range(len(dataset[i])):

                temp_array_1 = []

                for head_idx, rel_idx, tail_idx in dataset[i,j]:

                    head = idx2ent[head_idx]
                    tail = idx2ent[tail_idx]
                    rel = idx2rel[rel_idx]

                    temp_array_1.append((head,rel,tail))

                temp_array.append(temp_array_1)

            data.append(temp_array)

        data = np.array(data)

    return data

def distinct(a):
    _a = np.unique(a,axis=0)
    return _a

def get_adj_mats(data,num_entities,num_relations):

    adj_mats = []

    for i in range(num_relations):

        data_i = data[data[:,1] == i]


        if not data_i.shape[0]:   #data_i.shape[0]
            indices = tf.zeros((1,2),dtype=tf.int64)
            values = tf.zeros((1), dtype=tf.int64)

        else:
            indices = tf.gather(data_i,[0,2],axis=1)
            indices = tf.py_function(distinct,[indices],indices.dtype)
            values = tf.ones((indices.shape[0]))


        sparse_mat = tf.sparse.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=(num_entities,num_entities),
            )

        sparse_mat = tf.sparse.reorder(sparse_mat)

        sparse_mat = tf.sparse.reshape(sparse_mat, shape=(1,num_entities,num_entities))

        adj_mats.append(sparse_mat)

    return adj_mats

def get_negative_triples(head, rel, tail, num_entities, random_state=123):

    cond = tf.random.uniform(tf.shape(head), 0, 2, dtype=tf.int64, seed=random_state)
    rnd = tf.random.uniform(tf.shape(head), 0, num_entities-1, dtype=tf.int64, seed=random_state)
    
    neg_head = tf.where(cond == 1, head, rnd)
    neg_tail = tf.where(cond == 1, rnd, tail)
    print('neg_head',neg_head)

    return neg_head, neg_tail

