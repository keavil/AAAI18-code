import numpy as np
import tensorflow as tf
import json, random

class DataManager(object):
    def __init__(self, dataset):
        '''
        Read the data from dir "dataset"
        '''
        self.origin = {}
        for fname in ['train', 'dev', 'test']:
            data = []
            for line in open('%s/%s.res' % (dataset, fname)):
                s = json.loads(line.strip())
                if len(s) > 0:
                    data.append(s)
            self.origin[fname] = data
    def getword(self):
        '''
        Get the words that appear in the data.
        Sorted by the times it appears.
        {'ok': 1, 'how': 2, ...}
        Never run this function twice.
        '''
        wordcount = {}
        def dfs(node):
            if node.has_key('children'):
                dfs(node['children'][0])
                dfs(node['children'][1])
            else:
                word = node['word'].lower()
                wordcount[word] = wordcount.get(word, 0) + 1
        for fname in ['train', 'dev', 'test']:
            for sent in self.origin[fname]:
                dfs(sent)
        words = wordcount.items()
        words.sort(key = lambda x : x[1], reverse = True)
        self.words = words
        self.wordlist = {item[0]: index+1 for index, item in enumerate(words)}
        return self.wordlist
    
    def getdata(self, grained, maxlenth):
        '''
        Get all the data, divided into (train,dev,test).
        For every sentence, {'words':[1,3,5,...], 'solution': [0,1,0,0,0]}
        For each data, [sentence1, sentence2, ...]
        Never run this function twice.
        '''
        def one_hot_vector(r):
            s = np.zeros(grained, dtype=np.float32)
            s[r] += 1.0
            return s
        def dfs(node, words):
            if node.has_key('children'):
                dfs(node['children'][0], words)
                dfs(node['children'][1], words)
                node['size'] = node['children'][0]['size'] + node['children'][1]['size']
            else:
                word = self.wordlist[node['word'].lower()]
                words.append(word)
                node['size'] = 1
        def look_action(node, action, ulen):
            if node['size'] <= ulen:
                action += [0] * (node['size'] - 1)
                action.append(1)
            elif node.has_key('children'):
                look_action(node['children'][0], action, ulen)
                look_action(node['children'][1], action, ulen)
        self.getword()
        self.data = {}
        for fname in ['train', 'dev', 'test']:
            self.data[fname] = []
            for sent in self.origin[fname]:
                words, action = [], []
                dfs(sent, words)
                lens = len(words)
                words += [0] * (maxlenth - lens)
                solution = one_hot_vector(int(sent['rating']))
                look_action(sent, action, int(np.sqrt(lens) + 0.5))
                now = {'words': np.array(words), \
                        'solution': solution,\
                        'lenth': lens, \
                        'action': action}
                self.data[fname].append(now)
        return self.data['train'], self.data['dev'], self.data['test']
    
    def get_wordvector(self, name):
        fr = open(name)
        n, dim = map(int, fr.readline().split())
        self.wv = {}
        for i in range(n - 1):
            vec = fr.readline().split()
            word = vec[0].lower()
            vec = map(float, vec[1:])
            if self.wordlist.has_key(word):
                self.wv[self.wordlist[word]] = vec
        self.wordvector = []
        losscnt = 0
        for i in range(len(self.wordlist) + 1):
            if self.wv.has_key(i):
                self.wordvector.append(self.wv[i])
            else:
                losscnt += 1
                self.wordvector.append(np.random.uniform(-0.1,0.1,[dim]))
        self.wordvector = np.array(self.wordvector, dtype=np.float32)
        print losscnt, "words not find in wordvector"
        print len(self.wordvector), "words in total"
        return self.wordvector

#datamanager = DataManager("../TrainData/SUBJ")
#train_data, test_data, dev_data = datamanager.getdata(2, 200)
#wv = datamanager.get_wordvector("../WordVector/vector.25dim")
#mxlen = 0
#for item in test_data:
#    print item['action'], item['lenth']
#    if item['lenth'] > mxlen:
#        mxlen = item['lenth']
#print mxlen

#datamanager = DataManager("../TrainData/MR")
#train_data, dev_data, test_data = datamanager.getdata(2,70);
#for item in dev_data:
#    print json.dumps(item['action'][:item['lenth']])
#    print json.dumps([datamanager.words[i-1][0] for i in item['words']][:item['lenth']])

