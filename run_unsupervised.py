import os
import time
import tensorflow as tf
import numpy as np
import easydict
import pickle as pkl
import networkx as nx
import matplotlib.pyplot as plt
import sys
from tensorflow.python.util import deprecation

from src.data_loader import DataLoader
from src.minibatch import EdgeBatch, NeighborSampler
from src.model import LayerInfo, UnsupervisedSAGE, UnsupervisedGAT

args = easydict.EasyDict({
    "infolder": "../dataset/yelp/sample-167888",
    "outfolder": "../dataset/yelp/sample-167888/embeddings",
    "gpu": 0,
    "model": "GAT",
    "epoch": 1,
    "batch_size": 512,
    "dropout": 0.0,
    "ffd_dropout": 0.0,
    "attn_dropout": 0.0,
    "weight_decay": 0.0,
    "learning_rate": 0.0001,
    "max_degree": 100,
    "sample1": 25,
    "sample2": 10,
    "dim1": 128,
    "dim2": 128,
    "neg_sample": 20,
    "head1": 8,
    "head2": 1
})
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
deprecation._PRINT_DEPRECATION_WARNINGS = False

def train(data_trn):
    # data: graph, node features, random walks
    (G, features, walks) = data_trn
    print ('===== start training on graph(node={}, edge={}, walks={})====='.format(len(G.nodes()), len(G.edges()), len(walks)))
    print ('batch_size: ', '{}\n'.format(args.dropout),
           'max_degree', '{}\n'.format(args.max_degree),
           'sample1: ', '{}\n'.format(args.sample1),
           'sample2: ', '{}\n'.format(args.sample2),
           'neg_sample: ', '{}\n'.format(args.neg_sample),
           'dropout: ', '{}\n'.format(args.dropout))
    
    # placeholders
    placeholders = {
        'batch1': tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'batch2': tf.placeholder(tf.int32, shape=(None), name='batch2'),
        'neg_sample': tf.placeholder(tf.int32, shape=(None,), name='neg_sample_size'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'ffd_dropout': tf.placeholder_with_default(0., shape=(), name='ffd_dropout'),
        'attn_dropout': tf.placeholder_with_default(0., shape=(), name='attn_dropout'),
        'batch_size': tf.placeholder(tf.int32, name='batch_size'),
    }
    
    # batch of edges (positive samples)
    minibatch = EdgeBatch(G, placeholders, walks, batch_size=args.batch_size, max_degree=args.max_degree)
    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")
    # sample of neighbor for convolution
    sampler = NeighborSampler(adj_info)
    # two layers
    layer_infos = [LayerInfo('layer1', sampler, args.sample1, args.dim1, args.head1),
                   LayerInfo('layer2', sampler, args.sample2, args.dim2, args.head2)]
    # GCN model
    if args.model == 'SAGE':
        model = UnsupervisedSAGE(placeholders, features, adj_info, minibatch.deg, layer_infos, args)
    else:
        model = UnsupervisedGAT(placeholders, features, adj_info, minibatch.deg, layer_infos, args)
    
    # initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})
    
    # begin training
    t = time.time()
    for epoch in range(args.epoch):
        minibatch.shuffle()
        
        iter = 0
        print ('Epoch: {} (batch={})'.format(epoch + 1, minibatch.left_edge()))
        while not minibatch.end_edge():
            # construct feed dictionary
            feed_dict, _ = minibatch.next_edgebatch_feed_dict()
            feed_dict.update({placeholders['dropout']: args.dropout})
            feed_dict.update({placeholders['ffd_dropout']: args.ffd_dropout})
            feed_dict.update({placeholders['attn_dropout']: args.attn_dropout})
            
            # train
            outs = sess.run([model.loss, model.mrr, model.outputs1], 
                            feed_dict=feed_dict)
            train_loss = outs[0]
            train_mrr = outs[1]
            
            # print log
            if iter % 50 == 0:
                print ('-- iter: ', '{:4d}'.format(iter),
                       'train_loss=', '{:.5f}'.format(train_loss),
                       'train_mrr=', '{:.5f}'.format(train_mrr),
                       'time so far=', '{:.5f}'.format((time.time() - t)/60))
            iter += 1
            
    print ('Training finished!')
    
    # save embeddings
    embeddings = []
    nodes = []
    seen = set()
    minibatch.shuffle()
    while not minibatch.end_node():
        feed_dict, edges = minibatch.next_nodebatch_feed_dict()
        outs = sess.run([model.loss, model.mrr, model.outputs1], 
                        feed_dict=feed_dict)
        # only save embeds1 because of planetoid
        for i, edge in enumerate(edges):
            node = edge[0]
            if not node in seen:
                embeddings.append(outs[-1][i, :])
                nodes.append(node)
                seen.add(node)
    if not os.path.exists(args.outfolder):
        os.makedirs(args.outfolder)
    with open('{}/{}.bin'.format(args.outfolder, args.model), 'wb') as f:
        pkl.dump((embeddings, nodes), f)

def main():
    # load data
    loader = DataLoader(args.infolder)
    # train
    train((loader.G_trn, loader.features, loader.walks))

if __name__=='__main__':
    main()







