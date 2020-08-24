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
from src.model import LayerInfo, UnsupervisedSAGE, UnsupervisedGAT, UnsupervisedCGAT, UnsupervisedCGAT_2

args = easydict.EasyDict({
    "infolder": "../dataset/stackoverflow/sample-218016", # yelp/sample-641938, stackoverflow/sample-218016
    "outfolder": "../dataset/stackoverflow/sample-218016/embeddings", # yelp/sample-641938/embeddings, stackoverflow/sample-218016
    "gpu": 1,
    "model": "SAGE",
    "epoch": 1,
    "batch_size": 64, # 64 for GAT; 128 for SAGE
    "dropout": 0.,
    "ffd_dropout": 0.,
    "attn_dropout": 0.,
    "vae_dropout": 0.,
    "weight_decay": 0.0,
    "learning_rate": 0.0005,
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
    (G, features, walks, edgetexts, vocab_dim) = data_trn
    print ('===== start {} training on graph(node={}, edge={}, walks={})====='.format(
            args.model, len(G.nodes()), len(G.edges()), len(walks)))
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
        'vae_dropout': tf.placeholder_with_default(0., shape=(), name='vae_dropout'),
        'batch_size': tf.placeholder(tf.int32, name='batch_size'),
    }
    
    # batch of edges
    minibatch = EdgeBatch(G, edgetexts, placeholders, walks, 
                          batch_size=args.batch_size, max_degree=args.max_degree, vocab_dim=vocab_dim)
    # adj_info
    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")
    # (node1, node2) -> edge_idx
    edge_idx_ph = tf.placeholder(dtype=tf.int32, shape=minibatch.edge_idx.shape)
    edge_idx = tf.Variable(edge_idx_ph, trainable=False, name='edge_idx')
    # edge_vecs
    edge_vec_ph = tf.placeholder(dtype=tf.float32, shape=minibatch.edge_vec.shape)
    edge_vec = tf.Variable(edge_vec_ph, trainable=False, name='edge_vec')

    # sample of neighbor for convolution
    sampler = NeighborSampler(adj_info)
    # two layers
    layer_infos = [LayerInfo('layer1', sampler, args.sample1, args.dim1, args.head1),
                   LayerInfo('layer2', sampler, args.sample2, args.dim2, args.head2)]

    # initialize session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        
    # GCN model
    if args.model == 'SAGE':
        model = UnsupervisedSAGE(placeholders, features, minibatch.deg, layer_infos, args)
    elif args.model == 'GAT':
        model = UnsupervisedGAT(placeholders, features, minibatch.deg, layer_infos, args)
    elif args.model == 'CGAT':
        model = UnsupervisedCGAT(placeholders, features, vocab_dim, edge_idx, edge_vec, 
                                 minibatch.deg, layer_infos, args)
    else:
        model = UnsupervisedCGAT_2(placeholders, features, vocab_dim, edge_idx, edge_vec, 
                                 minibatch.deg, layer_infos, args)

    sess.run(tf.global_variables_initializer(), 
             feed_dict={adj_info_ph: minibatch.adj, 
                        edge_idx_ph: minibatch.edge_idx, 
                        edge_vec_ph: minibatch.edge_vec})

    # print out model size
    para_size = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print ("Model size: {}".format(para_size))
    
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
            feed_dict.update({placeholders['vae_dropout']: args.vae_dropout})
            
            # train  
            if args.model.startswith('CGAT'):
                outs = sess.run([model.graph_loss, model.reconstr_loss, model.kl_loss, model.loss, model.mrr], 
                                feed_dict=feed_dict)
                graph_loss = outs[0]
                reconstr_loss = outs[1]
                kl_loss = outs[2]
                train_loss = outs[3]
                train_mrr = outs[4]
            
                # print log
                if iter % 100 == 0:
                    print ('-- iter: ', '{:4d}'.format(iter),
                           'graph_loss=', '{:.5f}'.format(graph_loss),
                           'reconstr_loss=', '{:.5f}'.format(reconstr_loss),
                           'kl_loss=', '{:.5f}'.format(kl_loss),
                           'train_loss=', '{:.5f}'.format(train_loss),
                           'train_mrr=', '{:.5f}'.format(train_mrr),
                           'time so far=', '{:.5f}'.format((time.time() - t)/60))
            else:
                outs = sess.run([model.loss, model.mrr, model.inputs1, model.batch_size], feed_dict=feed_dict)
                if iter % 100 == 0:
                    print ('-- iter: ', '{:4d}'.format(iter),
                           'train_loss=', '{:.5f}'.format(outs[0]),
                           'train_mrr=', '{:.5f}'.format(outs[1]),
                           'time so far=', '{:.5f}'.format((time.time() - t)/60))
            iter += 1
            
    print ('Training {} finished!'.format(args.model))
    
    # save embeddings
    embeddings = []
    nodes = []
    seen = set()
    minibatch.shuffle()
    iter = 0
    while not minibatch.end_node():
        feed_dict, edges = minibatch.next_nodebatch_feed_dict()
        print ('-- iter: ', '{:4d}'.format(iter), edges)
        for p in edges:
            (n, _) = p
            if n >= len(G.nodes()):
                print ('Gotcha!{}'.format(n))
        if args.model.startswith('CGAT'):
            outs = sess.run([model.outputs1, model.beta, model.phi], 
                        feed_dict=feed_dict)
        else:
            outs = sess.run([model.outputs1], 
                        feed_dict=feed_dict)
        # only save embeds1 because of planetoid
        for i, edge in enumerate(edges):
            node = edge[0]
            if not node in seen:
                embeddings.append(outs[0][i, :])
                nodes.append(node)
                seen.add(node)
        if iter % 100 == 0:
            print ('-- iter: ', '{:4d}'.format(iter), 
                   'node_embeded=', '{}'.format(len(seen)))
        iter += 1
    if not os.path.exists(args.outfolder):
        os.makedirs(args.outfolder)
    with open('{}/{}.bin'.format(args.outfolder, args.model), 'wb') as f:
        pkl.dump((embeddings, nodes), f)
    
    if args.model.startswith('CGAT'):
        with open('{}/{}softmax_topic.bin'.format(args.outfolder, args.model), 'wb') as f:
            pkl.dump((outs[1], outs[2]), f)
        
def main():
    # load data
    loader = DataLoader(args.infolder)
    # train
    train((loader.G_trn, loader.features, loader.walks, loader.edge_text, len(loader.vocab)))

if __name__=='__main__':
    main()

