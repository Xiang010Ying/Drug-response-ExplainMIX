#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Lambda
import utils

class RGCN_Layer(tf.keras.layers.Layer):
    def __init__(self,num_entities,num_relations,output_dim,seed,**kwargs):
        super(RGCN_Layer,self).__init__(**kwargs)
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.output_dim = output_dim
        self.seed = seed
        
    def build(self,input_shape):

        input_dim = int(input_shape[-2][-1])
        
        self.relation_kernel = self.add_weight(
            shape=(self.num_relations,input_dim, self.output_dim),
            name="relation_kernels",
            trainable=True,
            initializer=tf.keras.initializers.RandomNormal(
                mean=0.0,
                stddev=1,
                seed=self.seed
            )
        )

        self.self_kernel = self.add_weight(
            shape=(input_dim, self.output_dim),
            name="self_kernel",
            trainable=True,
            initializer=tf.keras.initializers.RandomNormal(
                mean=0.0,
                stddev=1,
                seed=self.seed
            )
        )

    def call(self,inputs):

        embeddings,head_idx,head_e,tail_idx,tail_e,*adj_mats = inputs

        head_output = tf.matmul(head_e,self.self_kernel)
        tail_output = tf.matmul(tail_e,self.self_kernel)
                
        for i in range(self.num_relations):
            
            adj_i = tf.sparse.reshape(adj_mats[0][i],shape=(self.num_entities,self.num_entities))

            rows = tf.sparse.to_dense(adj_i)
            sess =tf.compat.v1.Session()

            with open('./adj_metrics.txt', 'w') as f:
                f.write(str(rows))

            sum_embeddings = tf.sparse.sparse_dense_matmul(adj_i, embeddings)
            
            head_update = tf.nn.embedding_lookup(sum_embeddings,head_idx) #在sum_embeddings中查找head_idx相关张量
            tail_update = tf.nn.embedding_lookup(sum_embeddings,tail_idx)
            
            head_output += tf.matmul(head_update,self.relation_kernel[i]) #矩阵相乘
            tail_output += tf.matmul(tail_update,self.relation_kernel[i])
       
        return tf.sigmoid(head_output), tf.sigmoid(tail_output)

class DistMult(tf.keras.layers.Layer):
    def __init__(self, num_relations,seed,**kwargs):
        super(DistMult,self).__init__(**kwargs)
        self.num_relations = num_relations
        self.seed = seed
        
    def build(self,input_shape):
        
        embedding_dim = input_shape[0][-1]
        
        self.kernel = self.add_weight(
            shape=(self.num_relations,embedding_dim),
            trainable=True,
            initializer=tf.keras.initializers.RandomNormal(
                mean=0.0,
                stddev=1,
                seed=self.seed
            ),
            name='rel_embedding'
        )
        
    def call(self,inputs):

        head_e,rel_idx,tail_e = inputs
        rel_e = tf.nn.embedding_lookup(self.kernel,rel_idx) #cong
        score = tf.sigmoid(tf.reduce_sum(head_e*rel_e*tail_e,axis=-1)) # 对同一行的元素进行累加     这里计算返回的分数
        return tf.expand_dims(score,axis=0)

class RGCN_Model(tf.keras.Model):

    def __init__(self,num_entities,*args,**kwargs):
        super(RGCN_Model,self).__init__(*args, **kwargs)
        self.num_entities = num_entities

    def train_step(self,data):
        all_indices,pos_head,rel,pos_tail,*adj_mats = data[0]
        y_pos_true = data[1]

        neg_head, neg_tail = utils.get_negative_triples(
                head=pos_head, 
                rel=rel, 
                tail=pos_tail,
                num_entities=self.num_entities
            )

        with tf.GradientTape() as tape:

            y_pos_pred = self([
                    all_indices,
                    pos_head,
                    rel,
                    pos_tail,
                    adj_mats
                    ],
                    training=True
                )

            y_neg_pred = self([
                    all_indices,
                    neg_head,
                    rel,
                    neg_tail,
                    adj_mats
                    ],
                    training=True
                )

            y_pred = tf.concat([y_pos_pred,y_neg_pred],axis=1)
            y_true = tf.concat([y_pos_true,tf.zeros_like(y_pos_true)],axis=1)

            loss = self.compiled_loss(y_true,y_pred)

            loss *= (1/ self.num_entities)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.compiled_metrics.update_state(y_pos_true, y_pos_pred)

        return {m.name: m.result() for m in self.metrics}

def get_RGCN_Model(num_entities,num_relations,embedding_dim,output_dim,seed):

    head_input = tf.keras.Input(shape=(None,), name='head_input',dtype=tf.int64)
    rel_input = tf.keras.Input(shape=(None,), name='rel_input',dtype=tf.int64)
    tail_input = tf.keras.Input(shape=(None,), name='tail_input',dtype=tf.int64)
    all_entities = tf.keras.Input(shape=(None,), name='all_entities',dtype=tf.int64)

    adj_inputs = [tf.keras.Input(
        shape=(num_entities,num_entities),   #均质网络
        dtype=tf.float32,
        name='adj_inputs_'+str(i),
        sparse=True,
        ) for i in range(num_relations)]

    entity_embeddings = Embedding(
        input_dim=num_entities,
        output_dim=embedding_dim,
        name='entity_embeddings',
        embeddings_initializer=tf.keras.initializers.RandomUniform(
            minval=0,
            maxval=1,
            seed=seed
        )
    )

    head_e = entity_embeddings(head_input)
    tail_e = entity_embeddings(tail_input)
    all_e = entity_embeddings(all_entities)

    head_e = Lambda(lambda x:x[0,:,:])(head_e)
    tail_e = Lambda(lambda x:x[0,:,:])(tail_e)
    all_e = Lambda(lambda x:x[0,:,:])(all_e)
    #print(head_e)

    head_index = Lambda(lambda x:x[0,:])(head_input)
    rel_index = Lambda(lambda x:x[0,:])(rel_input)
    tail_index = Lambda(lambda x:x[0,:])(tail_input)
    #print('head_e',head_e)
    # print('head_index:',head_index)

    new_head,new_tail = RGCN_Layer(
        num_relations=num_relations,
        num_entities=num_entities,
        output_dim=output_dim,
        seed=seed)([
            all_e,
            head_index,
            head_e,
            tail_index,
            tail_e,
            adj_inputs
            ]
        )
    output = DistMult(num_relations=num_relations,seed=seed,name='DistMult')([
        new_head,rel_index,new_tail
        ]
    )

    model = RGCN_Model(
        inputs=[all_entities,head_input,rel_input,tail_input] + adj_inputs,
        outputs=[output],
        num_entities=num_entities
    )

    return model

if __name__ == '__main__':

    import numpy as np
    import argparse
    import os
    import random as rn

    SEED = 123
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)

    DATASET = 'cellline_drug'
    RULE = 'link'
    TOP_K = 10
    EMBEDDING_DIM = 32#125

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

    #划分训练集和测试集
    train_triples = np.array(train_triples)
    test_triples = np.array(test_triples)
    X_train_triples = train_triples
    X_test_triples = test_triples

    ADJACENCY_DATA = tf.concat([
        X_train_triples,
        ],axis=0
    )

    model = get_RGCN_Model(
        num_entities=NUM_ENTITIES,
        num_relations=NUM_RELATIONS,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        seed=SEED
    )

    LEARNING_RATE=0.001
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    )
    ADJ_MATS = utils.get_adj_mats(X_train_triples, NUM_ENTITIES, NUM_RELATIONS)

    X_train_triples = np.expand_dims(X_train_triples, axis=0)
    NUM_EPOCHS = 400

    model.fit(
        x=[
            ALL_INDICES,
            X_train_triples[:,:,0],
            X_train_triples[:,:,1],
            X_train_triples[:,:,2],
            ADJ_MATS
        ],
        y=X_train_triples[:,:, 1].reshape(1, -1),
        epochs=NUM_EPOCHS,
        batch_size=1,
        verbose=1
    )

    model.save_weights(os.path.join('..', 'data', 'weights', 'drug_celline',  'drug_celline' + '_' + RULE + '.h5'))