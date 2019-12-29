import os
import numpy as np
from tqdm import tqdm
import scipy
from scipy import sparse
import pickle as pkl
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import json
import random
import math
import nltk
import operator
nltk.download('punkt')
nltk.download('stopwords')

class yelpProcessor(object):
    def __init__(self, folder):
        # read business.json
        self.item_objs = self.load_json('{}/business.json'.format(folder))
        # read user.json
        self.user_objs = self.load_json('{}/user.json'.format(folder))
        # read review.json
        self.review_objs = self.load_json('{}/review.json'.format(folder))

        print ('------ load data -----')
        print ("item: {}".format(len(self.item_objs)))
        print ("user: {}".format(len(self.user_objs)))
        print ("review: {}".format(len(self.review_objs)))
        
        # read vocab
        self.vocab = {}
        with open('{}/vocab.txt'.format(folder), 'r') as f:
            for line in f:
                if line[0] != '#':
                    self.vocab[line.strip()] = len(self.vocab)
        print ("vocab size: {}".format(len(self.vocab)))
        
    def load_json(self, filename):
        objs = []
        with open(filename, 'r') as f:
            for line in f:
                objs.append(json.loads(line.strip()))
        return objs

    def count_review(self, review_objs, user_dict, item_dict):
        for obj in review_objs:
            u_id = obj['user_id']
            i_id = obj['business_id']
            if u_id in user_dict and i_id in item_dict:
                user_dict[u_id][0] += 1
                item_dict[i_id][0] += 1
        return (user_dict, item_dict)

    # filter the raw data to obtain a dense subgraph
    def filter_dense(self, user_lim=30, item_lim=40, iter_lim=10):
        print ('----- filtering graph -----')
        print ('set: user_lim={}, item_lim={}, iter_lim={}'.format(user_lim, item_lim, iter_lim))
        
        user_dict = {}
        item_dict = {}
        for i, obj in enumerate(self.user_objs):
            u_id = obj['user_id']
            user_dict[u_id] = [0, i]
        for i, obj in enumerate(self.item_objs):
            i_id = obj['business_id']
            item_dict[i_id] = [0, i]

        # filter sparse nodes
        for i in range(iter_lim):
            user_dict, item_dict = self.count_review(self.review_objs, user_dict, item_dict)

            for k in list(user_dict):
                if user_dict[k][0] < user_lim:
                    del user_dict[k]
            for k in list(item_dict):
                if item_dict[k][0] < item_lim:
                    del item_dict[k]
            print ('-- iter {} --'.format(i))
            print ("user: {}".format(len(user_dict)))
            print ("item: {}".format(len(item_dict)))

            # reset and recount
            for k, v in user_dict.items():
                user_dict[k][0] = 0
            for k, v in item_dict.items():
                item_dict[k][0] = 0

        filter_user_objs = []
        filter_item_objs = []
        filter_review_objs = []
        for k, v in user_dict.items():
            filter_user_objs.append(self.user_objs[v[1]])
        for k, v in item_dict.items():
            filter_item_objs.append(self.item_objs[v[1]])
        for obj in self.review_objs:
            u_id = obj['user_id']
            i_id = obj['business_id']
            if u_id in user_dict and i_id in item_dict:
                filter_review_objs.append(obj)
        print ('----- filtering result -----')
        print ('user: {}'.format(len(filter_user_objs)))
        print ('item: {}'.format(len(filter_item_objs)))
        print ('review: {}'.format(len(filter_review_objs)))
        return (filter_user_objs, filter_item_objs, filter_review_objs)
    
    def obj2dict(self, json_objs, strname, shift):
        id_dict = {}
        for obj in json_objs:
            obj_id = obj[strname]
            if obj_id not in id_dict:
                id_dict[obj_id] = shift + len(id_dict)
        return id_dict
    
    # construct adjacent list
    # mind duplicate edges between (u, i) pair: concatenate all reviews
    def construct_graph(self, user_objs, item_objs, review_objs):
        print ('----- constructing graph -----')
        
        user_dict = self.obj2dict(user_objs, 'user_id', 0)            
        item_dict = self.obj2dict(item_objs, 'business_id', len(user_dict))
        print ("user size: {}".format(len(user_dict)))
        print ("item size: {}".format(len(item_dict)))
        
        adj_dict = {}
        edge_dict = {}
        for obj in review_objs:
            u_id = obj['user_id']
            i_id = obj['business_id']
            if u_id in user_dict and i_id in item_dict:
                u_idx = user_dict[u_id]
                i_idx = item_dict[i_id]
                pair = [(u_id, i_id), (i_id, u_id)]
                for p in pairs:
                    if p not in edge_dict:
                        edge_dict[p] = 1
                    else:
                        edge_dict[p] += 1

                if u_idx not in adj_dict:
                    adj_dict[u_idx] = [i_idx]
                else:
                    adj_dict[u_idx] = list(set(adj_dict[u_idx] + [i_idx]))

                if i_idx not in adj_dict:
                    adj_dict[i_idx] = [u_idx]
                else:
                    adj_dict[i_idx] = list(set(adj_dict[i_idx] + [u_idx]))
        # statistics
        lens = []
        for k, v in edge_dict.items():
            lens.append(v)
        print ("all reviews: {}".format(sum(lens)))
        print ("unique edges: {}".format(len(edge_dict)))
        lens = []
        for k, v in adj_dict.items():
            lens.append(len(v))
        print ("edge stats: all={}, ave={}, max={}, min={}".format(sum(lens), sum(lens)/len(lens), max(lens), min(lens)))
        return (user_dict, item_dict, adj_dict)
    
    def string2gram(self, line, N):
        # split into words
        tokens = nltk.tokenize.word_tokenize(line)
        # convert to lower case
        tokens = [w.lower() for w in tokens]
        # remove punctuation
        words = [w for w in tokens if w.isalpha() or w.replace('.','',1).isdigit()]
        # stem words
        porter = nltk.stem.porter.PorterStemmer()
        stemmed = [porter.stem(w) for w in words]
        for i in range(len(stemmed)):
            if(stemmed[i].replace('.','',1).isdigit()):
                stemmed[i] = 'NUM'
        grams = []
        for n in range(N):
            for i in range(len(stemmed)-n):
                gram = stemmed[i]
                for j in range(n):
                    gram += "-" + stemmed[i+j+1]
                grams.append(gram)
        return grams

    # process edge rating and text
    # mind duplicate reviews: average rating, concatenate reviews
    def process_edge(self, review_objs, user_dict, item_dict):
        print ('----- processing edges -----')
        # process rating
        def rate_edge(review_objs, user_dict, item_dict):
            edge_rate = {}
            for obj in review_objs:
                u_idx = user_dict[obj['user_id']]
                i_idx = item_dict[obj['business_id']]
                rate = int(obj['stars'])
                pairs = [(u_idx, i_idx), (i_idx, u_idx)]
                for p in pairs:
                    if p not in edge_rate:
                        edge_rate[p] = [rate]
                    else:
                        edge_rate[p].append(rate)
            for k, v in edge_rate.items():
                edge_rate[k] = int(sum(v)/len(v))
            return edge_rate

        edge_rate = rate_edge(review_objs, user_dict, item_dict)
        # statistics
        rates = {}
        for k,v in edge_rate.items():
            if v not in rates:
                rates[v] = 1
            else:
                rates[v] += 1
        print ("rating stats: {}".format(rates))
        
        # process text
        edge_text = {}
        try:
            with tqdm(review_objs) as objs:
                for obj in objs:
                    u_idx = user_dict[obj['user_id']]
                    i_idx = item_dict[obj['business_id']]
                    feat = {}
                    tokens = self.string2gram(obj['text'], 2)
                    for t in tokens:
                        if t in self.vocab:
                            if t not in feat:
                                feat[self.vocab[t]] = 1
                            else:
                                feat[self.vocab[t]] += 1
                    pairs = [(u_idx, i_idx), (i_idx, u_idx)]
                    for p in pairs:
                        if p not in edge_text:
                            edge_text[p] = feat
                        else:
                            for k, v in feat.items():
                                if k not in edge_text[p]:
                                    edge_text[p][k] = v
                                else:
                                    edge_text[p][k] += v
        except KeyboardInterrupt:
            objs.close()
            raise
        objs.close()
        # statistics
        lens = []
        for k, v in edge_text.items():
            lens.append(len(v))
        print ("text stats: ave={}, max={}, min={}, zeros={}".format(sum(lens)/len(lens), max(lens), min(lens), 
                                                                   len(lens) - np.count_nonzero(lens)))
        return (edge_rate, edge_text)
    
    def process_label(self, item_objs, adj_dict, user_dict, item_dict):
        print ('----- processing lable -----')
        noises = ['restaur', 'new', 'food']
        class_dict = {}
        item_class = {}
        item_uni_class = {}
        # only pick the 50 most common labels
        class_count = {}
        for obj in item_objs:
            labels = self.string2gram(obj['categories'], 1)
            trim_labels = []
            for lbl in labels:
                if lbl in noises:
                    continue
                trim_labels.append(lbl)
            for lbl in trim_labels:
                if lbl not in class_count:
                    class_count[lbl] = 1
                else:
                    class_count[lbl] += 1
        sorted_class = sorted(class_count.items(), key=lambda kv: kv[1])[::-1]
        for i in range(50):
            class_dict[sorted_class[i][0]]=len(class_dict)

        for obj in item_objs:
            labels = self.string2gram(obj['categories'], 1)
            trim_labels = []
            for lbl in labels:
                if lbl in class_dict:
                    trim_labels.append(lbl)
            if len(trim_labels) > 0:
                trim_uni_labels = trim_labels[:1]
            item_class[item_dict[obj['business_id']]] = [class_dict[lbl] for lbl in trim_labels]
            item_uni_class[item_dict[obj['business_id']]] = [class_dict[lbl] for lbl in trim_uni_labels]
        
        # aggregate item_node label to user_node label
        y = np.zeros((len(adj_dict), len(class_dict)))
        y_uni = np.zeros((len(adj_dict), len(class_dict)))
        for k, v in adj_dict.items():
            if k in item_class: # item node
                for lbl in item_class[k]:
                    y[k][lbl] = 1
                for lbl in item_uni_class[k]:
                    y_uni[k][lbl] = 1
            else: # user node
                user_lbls = np.zeros(len(class_dict))
                for n in v:
                    for lbl in item_class[n]:
                        user_lbls[lbl] += 1
                y_uni[k][np.argmax(user_lbls)] = 1
                y[k] = [1 if i > 0 else 0 for i in user_lbls]
        # statistics
        print ("classe number: {}".format(len(class_dict)))
        print (class_dict)
        lens = []
        for k, v in item_dict.items():
            lens.append(np.count_nonzero(y[v]))
        print ("item class stats: ave={}, max={}, min={}, zeros={}".format(sum(lens)/len(lens), max(lens), min(lens), 
                                                                           len(lens) - np.count_nonzero(lens)))
        lens = []
        for k, v in item_dict.items():
            lens.append(np.count_nonzero(y_uni[v]))
        print ("item unique class stats: ave={}, max={}, min={}, zeros={}".format(sum(lens)/len(lens), max(lens), min(lens), 
                                                                           len(lens) - np.count_nonzero(lens)))
        lens = []
        for k, v in user_dict.items():
            lens.append(np.count_nonzero(y[v]))
        print ("user class stats: ave={}, max={}, min={}, zeros={}".format(sum(lens)/len(lens), max(lens), min(lens), 
                                                                           len(lens) - np.count_nonzero(lens)))
        lens = []
        for k, v in user_dict.items():
            lens.append(np.count_nonzero(y_uni[v]))
        print ("user unique class stats: ave={}, max={}, min={}, zeros={}".format(sum(lens)/len(lens), max(lens), min(lens), 
                                                                           len(lens) - np.count_nonzero(lens)))
        return (class_dict, y, y_uni)
        
if __name__ == "__main__":
    """ processing yelp data"""
    # input folder
    folder = "../../dataset/yelp"
    processor = yelpProcessor(folder)
    # filter dense graph
    (user_objs, item_objs, review_objs) = processor.filter_dense(50, 55, 13)
    # output folder
    path = os.path.join(folder, "sample-"+str(len(review_objs)))
    if not os.path.exists(path):
        os.makedirs(path)
    
    with open('{}/business.json'.format(path), 'w') as f:
        json.dump(item_objs, f)
    with open('{}/user.json'.format(path), 'w') as f:
        json.dump(user_objs, f)
    with open('{}/review.json'.format(path), 'w') as f:
        json.dump(review_objs, f)
        
    # construct graph
    (user_dict, item_dict, adj_dict) = processor.construct_graph(user_objs, item_objs, review_objs)
    with open('{}/user_map.bin'.format(path), 'wb') as f:
        pkl.dump(user_dict, f)
    with open('{}/item_map.bin'.format(path), 'wb') as f:
        pkl.dump(item_dict, f)
    with open('{}/vocab_map.bin'.format(path), 'wb') as f:
        pkl.dump(processor.vocab, f)
    with open('{}/adj_all.bin'.format(path), 'wb') as f:
        pkl.dump(adj_dict, f)
    
    # process edge rate and text
    (edge_rate, edge_text) = processor.process_edge(review_objs, user_dict, item_dict)
    with open("{}/edge_rate.bin".format(path), 'wb') as f:
        pkl.dump(edge_rate, f)
    with open("{}/edge_text.bin".format(path), 'wb') as f:
        pkl.dump(edge_text, f)

    # process label of item, and aggregate to user
    (class_dict, y, y_uni) = processor.process_label(item_objs, adj_dict, user_dict, item_dict)    
    with open('{}/label_map.bin'.format(path), 'wb') as f:
        pkl.dump(class_dict, f)
    with open('{}/node_label.bin'.format(path), 'wb') as f:
        pkl.dump(y, f)
    with open('{}/node_label_uni.bin'.format(path), 'wb') as f:
        pkl.dump(y_uni, f)
    
