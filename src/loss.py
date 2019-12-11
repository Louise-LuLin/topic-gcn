import tensorflow as tf

def affinity(inputs1, inputs2):
    """
    Affinity between batch of inputs1 and inputs2
    inputs1: [batch_size * feature_size]
    return: [batch_size]
    """
    return tf.reduce_sum(inputs1 * inputs2, axis=1)

def neg_cost(inputs, neg_samples):
    """
    For each input in batch, compute its affinity to negative samples
    return: [batch_size * num_neg_samples]
    """
    return tf.matmul(inputs, tf.transpose(neg_samples))

def skipgram_loss(inputs1, inputs2, neg_samples):
    aff = affinity(inputs1, inputs2)
    neg_aff = neg_cost(inputs1, neg_samples)
    neg_cost = tf.log(tf.reduce_sum(tf.exp(neg_aff), axis=1))
    return tf.reduce_sum(aff - neg_cost)

def xent_loss(inputs1, inputs2, neg_samples, neg_sample_weight=1.0):
    aff = affinity(inputs1, inputs2)
    neg_aff = neg_cost(inputs1, neg_samples)
    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(aff), logits=aff)
    neg_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_aff), logits=neg_aff)
    return tf.reduce_sum(true_xent) + neg_sample_weight * (tf.reduce_mean(neg_xent))