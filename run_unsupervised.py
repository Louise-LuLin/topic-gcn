import os
import time
import tensorflow as tf
import numpy as np
import argparse
import pickle as pkl
import networkx as nx
from tensorflow.python.util import deprecation
import logging

from src.data_loader import DataLoader
from src.minibatch import EdgeBatch, NeighborSampler
from src.model import LayerInfo, CGAT


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training-data-dir', type=str, required=True,
                        help='path of training data')  # ../dataset/stackoverflow/sample-51130/
    parser.add_argument('--embed-dir', type=str, required=True,
                        help='HDFS path or local directory')  # ../dataset/stackoverflow/sample-51130/embeddings
    parser.add_argument('--gpu', type=int, default=1,
                        help='index of gpu card')

    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epoch')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Number of batch_size')
    parser.add_argument('--dim1', type=int, default=128, 
                        help='Size of hidden dim for layer 1')
    parser.add_argument('--dim2', type=int, default=128, 
                        help='Size of hidden dim for layer 2')
    parser.add_argument('--attn-head1', type=int, default=8, 
                        help='Number of attention head for layer 1')
    parser.add_argument('--attn-head2', type=int, default=1, 
                        help='Number of attention head for layer 2')
    parser.add_argument('--sample1', type=int, default=25,
                        help="Number of neighbor for layer 1")
    parser.add_argument('--sample2', type=int, default=10,
                        help="Number of neighbor for layer 2")
    parser.add_argument('--neg-sample', type=int, default=20,
                        help="Number of negative sample")
    parser.add_argument('--max-degree', type=int, default=100,
                        help='Maximum degree per node')

    parser.add_argument('--learning-rate', type=float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0, 
                        help="L2 weight factor")
    parser.add_argument('--dropout', type=float, default=0.0, 
                        help="Fraction for dropout  (1 - keep probability)")
    parser.add_argument('--ffd-dropout', type=float, default=0.0, 
                        help="Fraction for dropout  (1 - keep probability)")
    parser.add_argument('--attn-dropout', type=float, default=0.0, 
                        help="Fraction for dropout  (1 - keep probability)")
    parser.add_argument('--vae-dropout', type=float, default=0.0, 
                        help="Fraction for dropout  (1 - keep probability)")

    parser.add_argument('--max-steps', type=int, default=1000000, 
                        help="Maximum number of steps to batches to train for")
    parser.add_argument('--eval-steps', type=int, default=1000, 
                        help="Number of steps to run for validation")
    parser.add_argument('--checkpoint-steps', type=int, default=1000, 
                        help="Number of steps between checkpoints")

    return parser.parse_args()
 
def train(data_trn, args):
    # data: graph, node features, random walks
    (G, features, walks, edgetexts, vocab_dim) = data_trn
    print ('===== start training on graph(node={}, edge={}, walks={})====='.format(
            len(G.nodes()), len(G.edges()), len(walks)))
    print ('batch_size: ', '{}\n'.format(args.batch_size),
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
    layer_infos = [LayerInfo('layer1', sampler, args.sample1, args.dim1, args.attn_head1),
                   LayerInfo('layer2', sampler, args.sample2, args.dim2, args.attn_head2)]

    # initialize session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        
    # GCN model
    model = CGAT(placeholders, features, vocab_dim, edge_idx, edge_vec, 
                             minibatch.deg, layer_infos, 
                             args.neg_sample, args.learning_rate, args.weight_decay)
    
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
            
            iter += 1
            
    print ('Training finished!')
    
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

        outs = sess.run([model.outputs1, model.beta, model.phi], 
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
    if not os.path.exists(args.embed_dir):
        os.makedirs(args.embed_dir)
    with open('{}/CGAT.bin'.format(args.embed_dir), 'wb') as f:
        pkl.dump((embeddings, nodes), f)
    
    with open('{}/CGAT_topic.bin'.format(args.embed_dir), 'wb') as f:
        pkl.dump((outs[1], outs[2]), f)
        
def main():
    print(tf.__version__)

    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    deprecation._PRINT_DEPRECATION_WARNINGS = False

    # tf.logging.set_verbosity(tf.logging.INFO)

    # load data
    loader = DataLoader(args.training_data_dir)

    # train
    train((loader.G_trn, loader.features, loader.walks, loader.edge_text, len(loader.vocab)), args)

if __name__=='__main__':
    main()

