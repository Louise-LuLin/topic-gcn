import tensorflow as tf
import math
from collections import namedtuple

import src.loss as loss
from src.layer import MeanAggregator, AttentionAggregator

# LayerInfo is a namedtuple that specifies the parameters 
# of the recursive layers
LayerInfo = namedtuple("LayerInfo",
    ['layer_name', # name of the layer (to get feature embedding etc.)
     'neighbor_sampler', # callable neigh_sampler constructor
     'num_samples', # num of sampled neighbor
     'output_dim', # the output (i.e., hidden) dimension
     'num_head' # num of head for GAT
])

class UnsupervisedSAGE(object):
    """
    Unsupervised GraphSAGE
    """
    def __init__(self, placeholders, features, adj, degrees, layer_infos, args):
        self.inputs1 = placeholders['batch1']
        self.inputs2 = placeholders['batch2']
        self.batch_size = placeholders['batch_size']
        self.placeholders = placeholders
        
        self.adj_info = adj
        self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
        self.degrees = degrees
        self.neg_sample_size = args.neg_sample
        
        self.dims = [features.shape[1]]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.layer_infos = layer_infos
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        self.weight_decay = args.weight_decay
        
        self.build()
        
    def build(self):
        self._build()
        self._loss()
        self._accuracy()
        
        self.loss = self.loss / tf.cast(self.batch_size, tf.float32)
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                 for grad, var in grads_and_vars]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        
    def _build(self):
        # negative sampling
        labels = tf.reshape(tf.cast(self.placeholders['batch2'], dtype=tf.int64), [self.batch_size, 1])
        self.neg_samples, _, _ = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels,
            num_true=1,
            num_sampled=self.neg_sample_size,
            unique=False,
            range_max=len(self.degrees),
            distortion=0.75,
            unigrams=self.degrees.tolist()
        )
        
        # convolution for three set of nodes
        # sample layers of nodes
        samples1, support_sizes1 = self.sample(self.inputs1, self.batch_size)
        samples2, support_sizes2 = self.sample(self.inputs2, self.batch_size)
        neg_samples, neg_support_sizes = self.sample(self.neg_samples, self.neg_sample_size)
        
        # initialize the aggregators
        self.init_aggregator()
        
        # aggregate
        self.outputs1 = self.aggregate(samples1, support_sizes1, self.batch_size)
        self.outputs2 = self.aggregate(samples2, support_sizes2, self.batch_size)
        self.neg_outputs = self.aggregate(neg_samples, neg_support_sizes, self.neg_sample_size)
        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)
        self.outputs2 = tf.nn.l2_normalize(self.outputs2, 1)
        self.neg_outputs = tf.nn.l2_normalize(self.neg_outputs, 1)
            
    def _loss(self):
        self.loss = loss.xent_loss(self.outputs1, self.outputs2, self.neg_outputs)
        for layer in range(len(self.layer_infos) + 1):
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='aggregate/layer_' + str(layer)):
                self.loss += self.weight_decay * tf.nn.l2_loss(var)
        
    def _accuracy(self):
        aff = loss.affinity(self.outputs1, self.outputs2)
        neg_aff = loss.neg_cost(self.outputs1, self.neg_outputs)
        neg_aff = tf.reshape(neg_aff, [self.batch_size, self.neg_sample_size])
        _aff = tf.expand_dims(aff, axis=1)
        aff_all = tf.concat(axis=1, values=[neg_aff, _aff])
        size = tf.shape(aff_all)[1]
        _, indices_of_ranks = tf.nn.top_k(aff_all, k=size)
        _, ranks = tf.nn.top_k(-indices_of_ranks, k=size)
        self.mrr = tf.reduce_mean(tf.div(1.0, tf.cast(ranks[:, -1] + 1, tf.float32)))
        
    def sample(self, inputs, batch_size):
        """
        Sample neighbors to be the supportive set for convolution
        """
        samples = [inputs]
        support_size = 1
        support_sizes = [support_size]
        for k in range(len(self.layer_infos)):
            # expanding neighbors of input nodes layer by layer backward
            # layer_info: forward, samples: backward
            t = len(self.layer_infos) - k - 1
            support_size *= self.layer_infos[t].num_samples
            sampler = self.layer_infos[t].neighbor_sampler
            node = sampler((samples[k], self.layer_infos[t].num_samples))
            samples.append(tf.reshape(node, [support_size * batch_size,]))
            support_sizes.append(support_size)
        return samples, support_sizes

    def init_aggregator(self):
        """ Initialize aggregator layers with creating reuseble convolution variables
        """
        self.aggregators = []
        for layer in range(len(self.dims) - 1):
            name = 'layer_' + str(layer)
            dim_mult = 2 if layer != 0 else 1
            if layer == len(self.dims) - 2:
                aggregator = MeanAggregator(name, dim_mult*self.dims[layer], self.dims[layer+1], 
                                            dropout=self.placeholders['dropout'], act=lambda x:x)
            else:
                aggregator = MeanAggregator(name, dim_mult*self.dims[layer], self.dims[layer+1],
                                            dropout=self.placeholders['dropout'])
            self.aggregators.append(aggregator)
    
    def aggregate(self, samples, support_sizes, batch_size):
        """ Aggregate embeddings of neighbors to compute the embeddings at next layer
        Args:
            samples: a list of node samples hops away at each layer. size=K+1
            support_sizes: a list of node numbers at each layer. size=K+1
            batch_size: input size
        Returns:
            The final embedding for input nodes
        """
        hidden = [tf.nn.embedding_lookup([self.features], node_sample) for node_sample in samples]
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos] # neighbor sample size for each node
        for layer in range(len(num_samples)):
            # embedding at current layer for all support nodes hops away
            next_hidden = []
            for hop in range(len(num_samples) - layer):
                dim_mult = 2 if layer != 0 else 1
                neighbor_dims = [batch_size * support_sizes[hop], 
                                 num_samples[len(num_samples) - hop - 1],
                                 dim_mult * self.dims[layer]]
                inputs = (hidden[hop], tf.reshape(hidden[hop + 1], neighbor_dims))
                h = self.aggregators[layer](inputs)
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]
                
class UnsupervisedGAT(UnsupervisedSAGE):
    """
    Unsupervised GAT
    """
    def __init__(self, placeholders, features, adj, degrees, layer_infos, args):
        self.heads = [layer_infos[i].num_head for i in range(len(layer_infos))]
        # define heads first, otherwise cannot _build
        UnsupervisedSAGE.__init__(self, placeholders, features, adj, degrees, layer_infos, args)
    
    def init_aggregator(self):
        """ Initialize aggregator layers with creating reuseble convolution variables
        """
        self.aggregators = []
        for layer in range(len(self.dims) - 1):
            dim_mult = 1 if layer==0 else self.heads[layer-1]
            multihead_attns = []
            for head in range(self.heads[layer]):
                name = 'layer_' + str(layer) + '_' + str(head)
                if layer == len(self.dims) - 2:
                    aggregator = AttentionAggregator(name, dim_mult*self.dims[layer], self.dims[layer+1],
                                                     ffd_drop=self.placeholders['ffd_dropout'],
                                                     attn_drop=self.placeholders['attn_dropout'], act=lambda x:x)
                else:
                    aggregator = AttentionAggregator(name, dim_mult*self.dims[layer], self.dims[layer+1],
                                                     ffd_drop=self.placeholders['ffd_dropout'], 
                                                     attn_drop=self.placeholders['attn_dropout'])
                multihead_attns.append(aggregator)
            self.aggregators.append(multihead_attns)
    
    def aggregate(self, samples, support_sizes, batch_size):
        """ Aggregate embeddings of neighbors to compute the embeddings at next layer
        Args:
            samples: a list of node samples hops away at each layer. size=K+1
            support_sizes: a list of node numbers at each layer. size=K+1
            batch_size: input size
        Returns:
            The final embedding for input nodes
        """
        hidden = [tf.nn.embedding_lookup([self.features], node_sample) for node_sample in samples]
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos] # neighbor sample size for each node
        for layer in range(len(num_samples)):
            # embedding at current layer for all support nodes hops away
            next_hidden = []
            for hop in range(len(num_samples) - layer):
                dim_mult = 1 if layer==0 else self.heads[layer-1]
                neighbor_dims = [batch_size * support_sizes[hop], 
                                     num_samples[len(num_samples) - hop - 1],
                                     dim_mult * self.dims[layer]]
                inputs = (hidden[hop], tf.reshape(hidden[hop + 1], neighbor_dims))
                attns = []
                for head in range(self.heads[layer]):
                    h = self.aggregators[layer][head](inputs)
                    attns.append(h)
                if layer == len(num_samples) - 1: # last layer
                    next_hidden.append(tf.add_n(attns) / self.heads[layer])
                else:
                    next_hidden.append(tf.concat(attns, axis=-1))
            hidden = next_hidden
        
        return hidden[0]
    
    
    
    
        