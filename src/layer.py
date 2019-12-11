import tensorflow as tf
import numpy as np

def uniform_init(shape, scale=0.05):
    return tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)

def glorot_init(shape):
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    return tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)

def zeros_init(shape):
    return tf.zeros(shape, dtype=tf.float32)

def ones_init(shape):
    return tf.ones(shape, dtype=tf.float32)

class MeanAggregator(object):
    """ Aggregate via mean followed by MLP
    """
    def __init__(self, name, input_dim, output_dim, dropout=0., usebias=False, act=tf.nn.relu):
        self.name = name
        self.dropout = dropout
        self.usebias = usebias
        self.act = act
        
        self.vars = {}
        with tf.variable_scope(name) as scope:
            self.vars['self_weights'] = tf.get_variable('self_weights',
                                                       initializer=glorot_init((input_dim, output_dim)))
            self.vars['neighbor_weights'] = tf.get_variable('neighbor_weights',
                                                           initializer=glorot_init((input_dim, output_dim)))
            if usebias:
                self.vars['bias'] = tf.get_variable('bias',
                                                   initializer=zeros_init((2 * output_dim)))

    def __call__(self, inputs):
        """
        Args:
            input: (self_vecs, neighbor_vecs)
            self_vecs.shape = [batch_size, dim]
            neighbor_vecs.shape = [batch_size, num_samples, dim]
        """
        self_vecs, neighbor_vecs = inputs
        # dropout
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)
        neighbor_vecs = tf.nn.dropout(neighbor_vecs, 1-self.dropout)
        neighbor_mean = tf.reduce_mean(neighbor_vecs, axis=1)
        # aggregate
        with tf.variable_scope(self.name) as scope:
            from_self = tf.matmul(self_vecs, self.vars['self_weights'])
            from_neighbor = tf.matmul(neighbor_mean, self.vars['neighbor_weights'])
            # concat(self, neighbor)
            output = tf.concat([from_self, from_neighbor], axis=1) # shape: [batch_size, 2*output_dim]
            if self.usebias:
                output += self.vars['bias']
        return self.act(output)
       
class AttentionAggregator(object):
    """ one head attention aggregator for GAT
    """
    def __init__(self, name, input_dim, output_dim, ffd_drop=0., attn_drop=0., usebias=False, act=tf.nn.elu):
        self.name = name
        self.ffd_drop = ffd_drop
        self.attn_drop = attn_drop
        self.usebias = usebias
        self.act = act
        
        with tf.variable_scope(name) as scope:
            self.conv1 = tf.layers.Conv1D(filters=output_dim, kernel_size=1, name='conv1')
            self.conv2 = tf.layers.Conv1D(filters=1, kernel_size=1, name='conv2')
            if usebias:
                self.bias = tf.get_variable('bias',
                                            initializer=zero_init((output_dim)))
                
    def __call__(self, inputs):
        self_vecs, neighbor_vecs = inputs
        # reshape: [batch_size, num_samples+1, dim]
        self_vecs = tf.expand_dims(self_vecs, axis=1)
        vecs = tf.concat([self_vecs, neighbor_vecs], axis=1)
        # dropout
        vecs = tf.nn.dropout(vecs, 1-self.ffd_drop)
        # transform and self attention
        with tf.variable_scope(self.name) as scope:
            vecs_trans = self.conv1(vecs) # shape: [batch_size, 1+num_samples, output_dim]
            f_1 = self.conv2(vecs_trans)  # shape: [batch_size, 1+num_samples, 1]
            f_2 = self.conv2(vecs_trans)
            logits = f_1 + tf.transpose(f_2, [0, 2, 1]) # shape: [batch_size, 1+num_samples, 1+num_samples]
            coefs = tf.nn.softmax(tf.nn.leaky_relu(logits))
            # dropout
            coefs = tf.nn.dropout(coefs, 1-self.attn_drop)
            vecs_trans = tf.nn.dropout(vecs_trans, 1-self.ffd_drop)
            # aggregate
            output = tf.matmul(coefs, vecs_trans) # shape: [batch_size, 1+num_samples, output_dim]
            # only maintain the target node for each batch
            output = tf.slice(output, [0, 0, 0], [-1, 1, -1])
            output = tf.squeeze(output) # shape: [batch_size, output_dim]
            if self.usebias:
                output += self.bias
        return self.act(output)
    
    
            
        
        









