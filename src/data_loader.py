import numpy as np
import random
import sys
import os
import os.path
from os import path
import networkx as nx
import pickle as pkl

WALK_LEN=5
WALK_N=50

class DataLoader(object):
    def __init__(self, folder, uni_flag=True, seed=448, split="Edge"):
        uni_str = "_uni" if uni_flag else ""
        # id to idx
        with open('{}/user_map.bin'.format(folder), 'rb') as f:
            self.user_dict = pkl.load(f)   # dict: k=userID, v=idx
        with open('{}/item_map.bin'.format(folder), 'rb') as f:
            self.item_dict = pkl.load(f)   # dict: k=userID, v=idx
        with open("{}/vocab_map.bin".format(folder), 'rb') as f: #dict: k=word, v=idx
            self.vocab = pkl.load(f)
        # label
#         with open("{}/label_map.bin".format(folder), 'rb') as f:  #dict: k=label, v=idx
#             self.label_dict = pkl.load(f)
#         with open("{}/node_label{}.bin".format(folder, uni_str), 'rb') as f:  #np.array: node * classN
#             self.node_label = pkl.load(f)
        # ajacent dict
        with open('{}/adj_all.bin'.format(folder), 'rb') as f: #dict: k=node_idx, v=[neighbors_idx]
            self.adj = pkl.load(f)

        # load edge info
#         with open("{}/edge_rate.bin".format(folder), 'rb') as f: #dict: k=(u_idx, i_idx)/(i_idx, u_idx), v=int(ave(rating))
#             self.edge_rate = pkl.load(f)
        with open("{}/edge_text.bin".format(folder), 'rb') as f: #dict: k=(u_idx, i_idx)/(i_idx, u_idx), v={word_idx: count}
            self.edge_text = pkl.load(f)
            
        print ('===== load data =====')
        print ('{} nodes: {} users, {} items'.format(len(self.adj), len(self.user_dict), len(self.item_dict)))
#         print ("{} unique edges".format(len(self.edge_rate)))
        print ("{} features".format(len(self.vocab)))
#         print ("{} labels".format(len(self.label_dict)))
        
        self.G = nx.from_dict_of_lists(self.adj)
        
        if split == "Edge":
            if path.exists("{}/graph_{}.bin".format(folder, seed)):
                (self.G_trn, self.G_tst) = pkl.load(open("{}/graph_{}.bin".format(folder, seed), 'rb'))
                self.features = pkl.load(open("{}/feature_{}.bin".format(folder, seed), 'rb'))
                self.walks = pkl.load(open("{}/walk_{}.bin".format(folder, seed), 'rb'))
            else:
                (self.G_trn, self.G_tst, self.features, self.walks) = self.split_by_edge(seed, folder)
                
    
    # split into train/test set
    def split_by_edge(self, seed, folder):
        print ('===== split trn/tst/ set=====')        
        rand = random.Random(seed)
        # randomly sample 0.1 of edges for test for each user
        adj_trn = {}
        adj_tst = {idx:[] for idx in self.adj.keys()}
        for _, uidx in self.user_dict.items():
            nei_items = self.adj[uidx]
            tst_num = 0.1 * len(nei_items)
            if len(adj_tst[uidx]) >= tst_num:
                continue
            tst_nei_items = rand.sample(nei_items, int(tst_num))
            adj_tst[uidx] = tst_nei_items
            for n in tst_nei_items:
                adj_tst[n].append(uidx)
            
            
        for k, v in self.adj.items():
            adj_trn[k] = list(set(v) - set(adj_tst[k]))
        # statistics
        lens = []
        for k, v in adj_trn.items():
            lens.append(len(v))
        print ("trn edge stats: ave={}, max={}, min={}".format(sum(lens)/len(lens), max(lens), min(lens)))
        # test
        lens = []
        for k, v in adj_tst.items():
            lens.append(len(v))
        print ("tst edge stats: ave={}, max={}, min={}".format(sum(lens)/len(lens), max(lens), min(lens)))
        # user
        lens_all = []
        lens_trn = []
        lens_tst = []
        ratio = []
        for uid, idx in self.user_dict.items():
            lens_all.append(len(self.adj[idx]))
            lens_trn.append(len(adj_trn[idx]))
            lens_tst.append(len(adj_tst[idx]))
            ratio.append(len(adj_tst[idx])/len(self.adj[idx]))
        print ("=== user node ===")
        print ("all stats: ave={}, max={}, min={}".format(sum(lens_all)/len(lens_all), max(lens_all), min(lens_all)))
        print ("trs stats: ave={}, max={}, min={}".format(sum(lens_trn)/len(lens_trn), max(lens_trn), min(lens_trn)))
        print ("tst stats: ave={}, max={}, min={}".format(sum(lens_tst)/len(lens_tst), max(lens_tst), min(lens_tst)))
        print ("ratio for tst: {}".format(sum(ratio)/len(ratio)))
        # item
        lens_all = []
        lens_trn = []
        lens_tst = []
        ratio = []
        for iid, idx in self.item_dict.items():
            lens_all.append(len(self.adj[idx]))
            lens_trn.append(len(adj_trn[idx]))
            lens_tst.append(len(adj_tst[idx]))
            ratio.append(len(adj_tst[idx])/len(self.adj[idx]))
        print ("=== item node ===")
        print ("all stats: ave={}, max={}, min={}".format(sum(lens_all)/len(lens_all), max(lens_all), min(lens_all)))
        print ("trs stats: ave={}, max={}, min={}".format(sum(lens_trn)/len(lens_trn), max(lens_trn), min(lens_trn)))
        print ("tst stats: ave={}, max={}, min={}".format(sum(lens_tst)/len(lens_tst), max(lens_tst), min(lens_tst)))
        print ("ratio for tst: {}".format(sum(ratio)/len(ratio)))
        
        # aggregate node feature
        feat = self.get_feature(adj_trn)
        
        # generate random walks
        G_trn = nx.from_dict_of_lists(adj_trn)
        G_tst = nx.from_dict_of_lists(adj_tst)
        walks = self.gen_random_walk(G_trn, G_trn.nodes(), seed)
        
        with open("{}/graph_{}.bin".format(folder, seed), 'wb') as f:
            pkl.dump((G_trn, G_tst), f)
        with open("{}/feature_{}.bin".format(folder, seed), 'wb') as f:
            pkl.dump(feat, f)
        with open("{}/walk_{}.bin".format(folder, seed), 'wb') as f:
            pkl.dump(walks, f)
            
        return (G_trn, G_tst, feat, walks)
       
    # aggregate edge feature to node, and normalize
    def get_feature(self, adj):
        x = np.zeros((len(self.adj), len(self.vocab)))
        for k, v in adj.items():
            for n in v:
                pair = (k, n)
                if pair in self.edge_text:
                    for feat, freq in self.edge_text[pair].items():
                        x[k][feat] += freq
                        x[n][feat] += freq
        # normalize
        row_sum = x.sum(axis=1)
        x = x / row_sum[:, np.newaxis]
#         from sklearn.preprocessing import StandardScaler
#         scaler = StandardScaler()
#         scaler.fit(x)
#         x = scaler.transform(x)
        # stats
        lens = []
        for i in range(x.shape[0]):
            lens.append(np.count_nonzero(x[i]))
        print ("node feature stats: ave={}, max={}, min={}, zeros={}".format(sum(lens)/len(lens), max(lens), min(lens), 
                                                                           len(lens) - np.count_nonzero(lens)))
        return x

    def gen_random_walk(self, G, nodes, seed):
        print ("===== generate random walk =====")
        rand = random.Random(seed)
        pairs = []
        for cnt, node in enumerate(nodes):
            if G.degree(node) == 0:
                continue
            for i in range(WALK_N):
                cur_node = node
                for j in range(WALK_LEN):
                    next_node = rand.choice(G.neighbors(cur_node))
                    # self co-occurrences are useless
                    if cur_node != node:
                        pairs.append((node,cur_node))
                    cur_node = next_node
            if cnt % 1000 == 0:
                print("--- Done walks for", cnt, "nodes")
        return pairs
    