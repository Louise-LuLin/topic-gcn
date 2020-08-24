import numpy as np
import random
from scipy.sparse import *
import tensorflow as tf

np.random.seed(123)

class NeighborSampler(object):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info):
        self.adj_info = adj_info

    def __call__(self, inputs):
        ids, num_samples = inputs
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids) 
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        return adj_lists
    
class EdgeBatch(object):
    """
    sample edge batch
    """
    def __init__(self, G, edgetexts, placeholders, walks, batch_size=100, max_degree=25, vocab_dim=5000):
        self.G = G
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0
        
        self.nodes = np.random.permutation(G.nodes())
        self.edges = np.random.permutation(walks)
        self.adj, self.deg = self.construct_adj()
        
        pairs = np.array(sorted([list(k) for k in edgetexts.keys()], key=lambda x:(x[0], x[1])), dtype=np.int64)
        rows = np.array([p[0] for p in pairs])
        cols = np.array([p[1] for p in pairs])
        indexs = np.array([i for i in range(len(edgetexts))], dtype=np.int32)
        self.edge_idx = csr_matrix((indexs, (rows,cols)), shape=(self.adj.shape[0], self.adj.shape[0])).todense()
    
        self.edge_vec = np.array([self.onehot(edgetexts[k], vocab_dim) for k in edgetexts.keys()])
        
    def construct_adj(self):
        adj = len(self.nodes) * np.ones((len(self.nodes), self.max_degree))
        deg = np.zeros((len(self.nodes), ))
        
        for nid in self.G.nodes():
            neighbors = np.array([n for n in self.G.neighbors(nid)])
            degree = len(neighbors)
            deg[nid] = degree
            if degree == 0:
                continue
            if degree > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif degree < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[nid, :] = neighbors
        return adj, deg
    
    def onehot(self, doc, min_len):
        vec = []
        for w_idx, w_cnt in doc.items():
            for i in range(w_cnt):
                vec.append(w_idx)
        return np.bincount(np.array(vec).astype('int'), minlength=min_len)

    def end_edge(self):
        return self.batch_num * self.batch_size >= len(self.edges)
    
    def end_node(self):
        return self.batch_num * self.batch_size >= len(self.nodes)
    
    def left_edge(self):
        return len(self.edges) // self.batch_size
    
    def left_node(self):
        return len(self.nodes) // self.batch_size
    
    def next_edgebatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.edges))
        edge_batch = self.edges[start_idx : end_idx]
        return (self.batch_feed_dict(edge_batch), edge_batch)

    def next_nodebatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.nodes))
        node_batch = self.nodes[start_idx : end_idx]
        edge_batch = [(n, n) for n in node_batch]
        return (self.batch_feed_dict(edge_batch), edge_batch)
        
    def batch_feed_dict(self, edge_batch):
        batch1 = [n1 for n1, n2 in edge_batch]
        batch2 = [n2 for n1, n2 in edge_batch]
        feed_dict = {self.placeholders['batch_size']: len(edge_batch), 
                     self.placeholders['batch1']: batch1,
                     self.placeholders['batch2']: batch2}
        return feed_dict
        
    def batch_num(self):
        return len(self.edges) // self.batch_size + 1
    
    def shuffle(self):
        self.edges = np.random.permutation(self.edges)
        self.nodes = np.random.permutation(self.nodes)
        self.batch_num = 0
        
        
        