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
from collections import defaultdict
import csv

nltk.download('punkt')
nltk.download('stopwords')

class yelpProcessor(object):
    def __init__(self, folder, mode):
        if mode == 0:
            # read business.json
            self.item_objs = self.load_json('{}/business.json'.format(folder))
            # read user.json
            self.user_objs = self.load_json('{}/user.json'.format(folder))
            # read review.json
            self.review_objs = self.load_json('{}/review.json'.format(folder))
        else:
            with open('{}/business.json'.format(folder), 'r') as f:
                self.item_objs = json.load(f)
            with open('{}/user.json'.format(folder), 'r') as f:
                self.user_objs = json.load(f)
            with open('{}/review.json'.format(folder), 'r') as f:
                self.review_objs = json.load(f)

        print ('------ load data -----')
        print ("item: {}".format(len(self.item_objs)))
        print ("user: {}".format(len(self.user_objs)))
        print ("review: {}".format(len(self.review_objs)))
        
        # read vocab
        self.vocab = {}
        maxsize = 2000
        with open('{}/vocab.txt'.format(folder), 'r') as f:
            for line in f:
                if len(self.vocab) >= maxsize:
                    break
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
        
    # construct graph as adjacent dict, with edge rate and text
    # mind!!!
    # duplicate edges between (u, i) pair: concatenate all reviews
    # short or empty edge text: remove the edges with len(text)<=5
    def construct_graph(self, user_objs, item_objs, review_objs):
        print ('----- constructing graph -----')
        # process text and rating (remove len(text)<=5 edges)
        edges = {}
        user_dict = {}
        item_dict = {}
        removed = 0
        try:
            with tqdm(review_objs) as objs:
                for obj in objs:
                    # process text first and remove len(text)<=5 edges
                    tokens = self.string2gram(obj['text'], 2)
                    feat = defaultdict(int)
                    for t in tokens:
                        if t in self.vocab:
                            feat[self.vocab[t]] += 1
                    if len(feat) <= 5:
                        removed += 1
                        continue
                    # rating
                    rate = int(obj['stars'])
                    # user and item id
                    u_id = obj['user_id']
                    i_id = obj['business_id']
                    if u_id not in user_dict:
                        user_dict[u_id] = len(user_dict)
                    if i_id not in item_dict:
                        item_dict[i_id] = len(item_dict)
                    # store and combine duplicated edge text/rating
                    pair = (u_id, i_id)
                    if pair not in edges:
                        edges[pair] = ([rate], feat)
                    else:
                        edges[pair][0].append(rate)
                        for k, v in feat.items():
                            edges[pair][1][k] += v
        except KeyboardInterrupt:
            objs.close()
            raise
        objs.close()
        
        # construct user/item map
        shift = len(user_dict)
        for k, v in item_dict.items():
            item_dict[k] += shift
        
        # construct graph as adj_dict with edge content and rating
        adj_dict = {}
        edge_rate = defaultdict()
        edge_text = defaultdict()
        duplicates = 0
        for idpair, content in edges.items():
            (u_id, i_id) = idpair
            (rate, feat) = content
            u_idx = user_dict[u_id]
            i_idx = item_dict[i_id]
            # construct adjacency dict
            if u_idx not in adj_dict:
                adj_dict[u_idx] = [i_idx]
            else:
                adj_dict[u_idx] = list(set(adj_dict[u_idx] + [i_idx]))
            if i_idx not in adj_dict:
                adj_dict[i_idx] = [u_idx]
            else:
                adj_dict[i_idx] = list(set(adj_dict[i_idx] + [u_idx]))
            # construct edges
            pairs = [(u_idx, i_idx), (i_idx, u_idx)]
            ave_rate = round(sum(rate)/len(rate))
            for p in pairs:
                edge_rate[p] = ave_rate
                edge_text[p] = feat
            duplicates += len(rate)-1
        
        # node statistics
        print ('node stats: all={}, user={}, item={}'.format(len(adj_dict), len(user_dict), len(item_dict)))
        
        # edge statistics
        lens = []
        for k, v in adj_dict.items():
            lens.append(len(v))
        print ("edge stats: all={}(+ {} duplicates + {} short ={}),".format(int(sum(lens)/2), duplicates, removed, 
                                                                           int(sum(lens)/2)+duplicates+removed),
               'ave={:.3f}, max={}, min={}'.format(sum(lens)/len(lens), max(lens), min(lens)))
        
        # rating statistics
        rates = {}
        for k, v in edge_rate.items():
            if v not in rates:
                rates[v] = 1
            else:
                rates[v] += 1
        print ("rating stats: {}".format(rates))
        
        # text statistics
        lens = []
        for k, v in edge_text.items():
            lens.append(len(v))
        print ("text stats: ave={}, max={}, min={}, zeros={}".format(sum(lens)/len(lens), max(lens), min(lens), 
                                                                   len(lens) - np.count_nonzero(lens)))

        return (user_dict, item_dict, adj_dict, edge_rate, edge_text)
    
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

def process_yelp():
    """ processing yelp data"""
    # build from scratch
#     # input folder
#     infolder = "../../dataset/yelp"
#     processor = yelpProcessor(infolder, 0)
#     # filter dense graph
#     (user_objs, item_objs, review_objs) = processor.filter_dense(30, 35, 8)
#     # output folder
#     path = infolder + "/sample-" + str(len(review_objs))
#     if not os.path.exists(path):
#         os.makedirs(path)
    
#     with open('{}/business.json'.format(path), 'w') as f:
#         json.dump(item_objs, f)
#     with open('{}/user.json'.format(path), 'w') as f:
#         json.dump(user_objs, f)
#     with open('{}/review.json'.format(path), 'w') as f:
#         json.dump(review_objs, f)
        
    # build from sampled jsonobjs
    infolder = "../../dataset/yelp/sample-641938"
    processor = yelpProcessor(infolder, 1)
    user_objs, item_objs, review_objs = processor.user_objs, processor.item_objs, processor.review_objs
    path = infolder
        
    # construct graph
    (user_dict, item_dict, adj_dict, edge_rate, edge_text) = processor.construct_graph(user_objs, item_objs, review_objs)
    with open('{}/user_map.bin'.format(path), 'wb') as f:
        pkl.dump(user_dict, f)
    with open('{}/item_map.bin'.format(path), 'wb') as f:
        pkl.dump(item_dict, f)
    with open('{}/adj_all.bin'.format(path), 'wb') as f:
        pkl.dump(adj_dict, f)
    with open("{}/edge_rate.bin".format(path), 'wb') as f:
        pkl.dump(edge_rate, f)
    with open("{}/edge_text.bin".format(path), 'wb') as f:
        pkl.dump(edge_text, f)
    with open('{}/vocab_map.bin'.format(path), 'wb') as f:
        pkl.dump(processor.vocab, f)

    # process label of item, and aggregate to user
    (class_dict, y, y_uni) = processor.process_label(item_objs, adj_dict, user_dict, item_dict)    
    with open('{}/label_map.bin'.format(path), 'wb') as f:
        pkl.dump(class_dict, f)
    with open('{}/node_label.bin'.format(path), 'wb') as f:
        pkl.dump(y, f)
    with open('{}/node_label_uni.bin'.format(path), 'wb') as f:
        pkl.dump(y_uni, f)
        
    
class stackoverflowProcessor(yelpProcessor):
    def __init__(self, folder):
        question_user_map = {} # key: ID, value: userID
        with open('{}/Questions.csv'.format(folder), encoding = "ISO-8859-1") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if not (line_count == 0 or row[0] == 'NA' or row[1] == 'NA'):
                    question_user_map[int(row[0])] = int(row[1])
                line_count += 1
            print('Processed {} questions.'.format(len(question_user_map)))
        
        self.adj = {}
        missing = 0
        self_loop = 0
        with open('{}/Answers.csv'.format(folder), encoding = "ISO-8859-1") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            with tqdm(csv_reader) as rows:
                for row in rows:
                    if not (line_count == 0 or row[1] == 'NA' or row[3] == 'NA'):
                        u1 = int(row[1])
                        p = int(row[3])
                        if not p in question_user_map:
                            missing += 1
                            continue
                        u2 = question_user_map[p]
                        if u1 == u2:
                            self_loop += 1
                            continue
                        # construct adj dict
                        if u1 not in self.adj:
                            self.adj[u1] = [u2]
                        else:
                            self.adj[u1].append(u2)
                        if u2 not in self.adj:
                            self.adj[u2] = [u1]
                        else:
                            self.adj[u2].append(u1)
                    line_count += 1
        print('Processed {} users from {} edges with {} missing {} self_loop'.format(len(self.adj), line_count, missing, self_loop))
        
        # filter dense
        self.filter_dense(10, 5)
        print ('Filtered and obtain {} users'.format(len(self.adj)))
        
        # read vocab
        self.vocab = {}
        maxsize = 2000
        with open('{}/vocab.txt'.format(folder), 'r') as f:
            for line in f:
                if len(self.vocab) >= maxsize:
                    break
                if line[0] != '#':
                    self.vocab[line.strip()] = len(self.vocab)
        print ("vocab size: {}".format(len(self.vocab)))
                    
        # read doc and filter out small doc (some edge may removed)
        self.edge_texts = {}
        self.edge_texts2 = {}
        small = 0
        self_loop = 0
        with open('{}/Answers.csv'.format(folder), encoding = "ISO-8859-1") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            with tqdm(csv_reader) as rows:
                for row in rows:
                    if not (line_count == 0 or row[1] == 'NA' or row[3] == 'NA'):
                        u1 = int(row[1])
                        p = int(row[3])
                        if not p in question_user_map:
                            continue
                        u2 = question_user_map[p]
                        
                        if not (u1 in self.adj and u2 in self.adj):
                            continue
                        if u1 == u2:
                            self_loop += 1
                            continue
                        
                        tokens = self.string2gram(row[5], 2)
                        doc = defaultdict(int)
                        for tk in tokens:
                            if tk in self.vocab:
                                doc[self.vocab[tk]] += 1
                        if len(doc) <= 10:
                            small += 1
                            continue
                        
                        pairs = [(u1, u2), (u2, u1)]
                        for p in pairs:
                            if p not in self.edge_texts:
                                self.edge_texts[p] = doc
                            else:
                                for k, v in doc.items():
                                    self.edge_texts[p][k] += v
                                   
                            if p not in self.edge_texts2:
                                self.edge_texts2[p] = [doc]
                            else:
                                self.edge_texts2[p].append(doc)
                    
                    line_count += 1
        print('Processed {} edges with {} small {} self_loop'.format(len(self.edge_texts), small, self_loop))
        
        # generate user_idxs that appear in edge_texts
        self.user_dict = {}
        for p in self.edge_texts.keys():
            (u1, u2) = p
            if u1 not in self.user_dict:
                self.user_dict[u1] = len(self.user_dict)
            if u2 not in self.user_dict:
                self.user_dict[u2] = len(self.user_dict)
        print ('GET {} users'.format(len(self.user_dict)))
        
        # regenerate edge_text and adj with user idx
        self.edge_texts_new = {}
        self.edge_texts2_new = {}
        self.adj_all = {}
        self_loop = 0
        for pair in self.edge_texts.keys():
            (u1, u2) = pair
            u1 = self.user_dict[u1]
            u2 = self.user_dict[u2]
            if u1 == u2:
                self_loop += 1
                continue
            # construct adj (two direction are considered in edge_texts already)
            if u1 not in self.adj_all:
                self.adj_all[u1] = [u2]
            else:
                self.adj_all[u1].append(u2)
            # construct edge text
            p = (u1, u2)
            self.edge_texts_new[p] = self.edge_texts[pair]
            self.edge_texts2_new[p] = self.edge_texts2[pair]
        print ('GET {} users in adj, {} edges, {} self_loops'.format(len(self.adj_all), len(self.edge_texts_new), self_loop))
        
        #statistic
        lens = []
        for k, v in self.adj_all.items():
            lens.append(len(v))
        lens = np.array(lens)
        print ('edge: mean={}, max={}, min={}'.format(np.sum(lens)/len(lens), np.max(lens), np.min(lens)))

        lens = []
        for k, v in self.edge_texts_new.items():
            lens.append(len(v))
        lens = np.array(lens)
        print ('edge texts: {}, mean_len={}, max_len={}, min_len={}'.format(len(self.edge_texts_new), np.sum(lens)/len(lens),
                                                                  np.max(lens), np.min(lens)))
        
        path = folder + "/sample-" + str(len(self.edge_texts_new))
        if not os.path.exists(path):
            os.makedirs(path)
        with open('{}/user_map.bin'.format(path), 'wb') as f:
            pkl.dump(self.user_dict, f)
        with open('{}/item_map.bin'.format(path), 'wb') as f:
            pkl.dump(self.user_dict, f)
        with open('{}/adj_all.bin'.format(path), 'wb') as f:
            pkl.dump(self.adj_all, f)
        with open("{}/edge_text.bin".format(path), 'wb') as f:
            pkl.dump(self.edge_texts_new, f)
        with open('{}/vocab_map.bin'.format(path), 'wb') as f:
            pkl.dump(self.vocab, f)
        with open("{}/edge_text2.bin".format(path), 'wb') as f:
            pkl.dump(self.edge_texts2_new, f)
        
        # filter the raw data to obtain a dense subgraph
    def filter_dense(self, user_lim=30, iter_lim=10):
        print ('----- filtering graph -----')
        print ('set: user_lim={}, iter_lim={}'.format(user_lim, iter_lim))
        
        # filter sparse nodes
        for i in range(iter_lim):
            for k in list(self.adj):
                if len(self.adj[k]) < user_lim:
                    del self.adj[k]
            print ('-- iter {} --'.format(i))
            print ("user: {}".format(len(self.adj)))

            # reset and recount
            empty = 0
            adj_new = {}
            for k, nei in self.adj.items():
                new_nei = []
                for v in nei:
                    if v in self.adj:
                        new_nei.append(v)
                if len(new_nei) > 0:
                    adj_new[k] = new_nei
                else:
                    empty += 1
            self.adj = adj_new
            print ('empty: {}, new user: {}'.format(empty, len(self.adj)))

def process_stackoverflow():
    folder = "../../dataset/stackoverflow"
    processor = stackoverflowProcessor(folder)
    
if __name__ == "__main__":
    """ processing yelp data"""
#     process_yelp()
        
    """ process stackoverflow data"""
    process_stackoverflow()