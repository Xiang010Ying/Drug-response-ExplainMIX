#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import utils
import random as rn
import RGCN


def get_neighbors(data_subset, node_idx):
    head_neighbors = tf.boolean_mask(data_subset, data_subset[:, 0] == node_idx)
    tail_neighbors = tf.boolean_mask(data_subset, data_subset[:, 2] == node_idx)

    neighbors = tf.concat([head_neighbors, tail_neighbors], axis=0)

    return neighbors


def get_computation_graph(head, rel, tail, data, num_relations):
    '''Get k hop neighbors (may include duplicates)'''

    neighbors_head = get_neighbors(data, head)
    neighbors_tail = get_neighbors(data, tail)

    all_neighbors = tf.concat([neighbors_head, neighbors_tail], axis=0)

    return all_neighbors


def replica_step(head, rel, tail, explanation, num_entities, num_relations):
    comp_graph = get_computation_graph(head, rel, tail, ADJACENCY_DATA, num_relations)

    adj_mats = utils.get_adj_mats(comp_graph, num_entities, num_relations)

    total_loss = 0.0

    for epoch in range(NUM_EPOCHS):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(masks)
            masked_adjs = [adj_mats[i] * tf.sigmoid(masks[i]) for i in range(num_relations)]

            before_pred = model([
                ALL_INDICES,
                tf.reshape(head, (1, -1)),
                tf.reshape(rel, (1, -1)),
                tf.reshape(tail, (1, -1)),
                adj_mats
            ]
            )

            pred = model([
                ALL_INDICES,
                tf.reshape(head, (1, -1)),
                tf.reshape(rel, (1, -1)),
                tf.reshape(tail, (1, -1)),
                masked_adjs
            ]
            )

            loss = - before_pred * tf.math.log(pred + 0.00001)

            tf.print(f"current loss {loss}")

            total_loss += loss

        grads = tape.gradient(loss, masks)
        optimizer.apply_gradients(zip(grads, masks))

    current_pred = []
    current_scores = []

    for i in range(num_relations):

        mask_i = adj_mats[i] * tf.nn.sigmoid(masks[i])

        mask_idx = mask_i.values > THRESHOLD

        non_masked_indices = tf.gather(mask_i.indices[mask_idx], [1, 2], axis=1)

        if tf.reduce_sum(non_masked_indices) != 0:
            rel_indices = tf.cast(tf.ones((non_masked_indices.shape[0], 1)) * i, tf.int64)

            triple = tf.concat([non_masked_indices, rel_indices], axis=1)

            triple = tf.gather(triple, [0, 2, 1], axis=1)

            score_array = mask_i.values[mask_idx]

            current_pred.append(triple)
            current_scores.append(score_array)

    current_scores = tf.concat([array for array in current_scores], axis=0)
    top_k_scores = tf.argsort(current_scores, direction='DESCENDING')[0:10]

    pred_exp = tf.reshape(tf.concat([array for array in current_pred], axis=0), (-1, 3))
    pred_exp = tf.gather(pred_exp, top_k_scores, axis=0)

    for mask in masks:
        mask.assign(value=init_value)

    return total_loss, pred_exp


def distributed_replica_step(head, rel, tail, explanation, num_entities, num_relations):
    per_replica_losses, current_preds = replica_step(head, rel, tail, explanation, num_entities, num_relations)

    reduce_loss = per_replica_losses / NUM_EPOCHS

    return reduce_loss, current_preds


if __name__ == '__main__':

    import argparse
    import tensorflow as tf

    SEED = 123
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)

    parser = argparse.ArgumentParser()

    DATASET = 'cellline_drug'
    RULE = 'link'
    TOP_K = 10
    EMBEDDING_DIM = 32  # 125
    LEARNING_RATE = 0.1
    NUM_EPOCHS = 100

    MAX_PADDING = 3
    train_triples = np.load('../data/ccle_6_train_triples.npy')
    test_triples = np.load('../data/ccle_6_test_triples.npy')

    entities = np.load(r'../data/ccle_6_entities.npy')
    relations = [0, 1, 2, 3]
    NUM_ENTITIES = len(entities)
    NUM_RELATIONS = len(relations)

    MAX_PADDING = 2

    train_triples = np.array(train_triples)
    test_triples = np.array(test_triples)
    X_train_triples = train_triples
    X_test_triples = test_triples

    NUM_ENTITIES = len(entities)
    NUM_RELATIONS = len(relations)
    OUTPUT_DIM = 15
    THRESHOLD = .01

    ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
    rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

    idx2ent = dict(zip(range(NUM_ENTITIES), entities))
    idx2rel = dict(zip(range(NUM_RELATIONS), relations))

    UNK_ENT_ID = 0
    UNK_REL_ID = 0

    ALL_INDICES = tf.reshape(tf.range(0, NUM_ENTITIES, 1, dtype=tf.int64), (1, -1))

    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')

    model = RGCN.get_RGCN_Model(
        num_entities=NUM_ENTITIES,
        num_relations=NUM_RELATIONS,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        seed=SEED
    )

    model.load_weights(os.path.join('..', 'data', 'weights', 'drug_celline', 'drug_celline' + '_' + RULE + '.h5'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    init_value = tf.random.normal(
        (1, NUM_ENTITIES, NUM_ENTITIES),
        mean=0,
        stddev=1,
        dtype=tf.float32,
        seed=SEED
    )

    masks = [tf.Variable(
        initial_value=init_value,
        name='mask_' + str(i),
        trainable=True) for i in range(NUM_RELATIONS)
    ]

    ADJACENCY_DATA = tf.concat([
        X_train_triples,
    ], axis=0)

    best_preds = []
    explanations = []
    tf_data = tf.data.Dataset.from_tensor_slices(
        (X_test_triples[:, 0], X_test_triples[:, 1], X_test_triples[:, 2], X_test_triples)).batch(1)

    dist_dataset = strategy.experimental_distribute_dataset(tf_data)

    for head, rel, tail, explanation in dist_dataset:
        loss, current_preds = distributed_replica_step(head, rel,
                                                       tail, explanation, NUM_ENTITIES, NUM_RELATIONS)
        explanations.append(explanation)
        best_preds.append(current_preds)

    explanations = [array.numpy() for array in explanations]
    best_preds = [array.numpy() for array in best_preds]

    out_preds = []

    for i in range(len(best_preds)):
        preds_i = utils.idx2array(best_preds[i], idx2ent, idx2rel)

        out_preds.append(preds_i)

    out_preds = np.array(out_preds, dtype=object)

    print(f'Rule: {RULE}')
    print(f'Num epochs: {NUM_EPOCHS}')
    print(f'Embedding dim: {EMBEDDING_DIM}')
    print(f'learning_rate: {LEARNING_RATE}')
    print(f'threshold {THRESHOLD}')

    np.savez(os.path.join('..', 'data', 'preds',
                          DATASET, 'gnn_explainer_' + DATASET + '_' + RULE + '_' + '_preds.npz'),
             preds=out_preds
             )

    print('Done.')

