import numpy as np
import tensorflow as tf
import random
import sys, os
import json
import argparse
from parser import Parser
from datamanager import DataManager
from actor import ActorNetwork
from LSTM_critic import LSTM_CriticNetwork
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#get parse
argv = sys.argv[1:]
parser = Parser().getParser()
args, _ = parser.parse_known_args(argv)
random.seed(args.seed)

#get data
dataManager = DataManager(args.dataset)
train_data, dev_data, test_data = dataManager.getdata(args.grained, args.maxlenth)
word_vector = dataManager.get_wordvector(args.word_vector)

if args.fasttest == 1:
    train_data = train_data[:100]
    dev_data = dev_data[:20]
    test_data = test_data[:20]
print "train_data ", len(train_data)
print "dev_data", len(dev_data)
print "test_data", len(test_data)

def sampling_RL(sess, actor, inputs, vec, lenth, Random=True):
    current_lower_state = np.zeros((1, 2*args.dim), dtype=np.float32)
    actions = []
    states = []
    #sampling actions

    for pos in range(lenth):
        predicted = actor.predict_target(current_lower_state, [vec[0][pos]])
        #print predicted 
        states.append([current_lower_state, [vec[0][pos]]])
        if Random:
            action = (0 if random.random() < predicted[0] else 1)
        else:
            action = np.argmax(predicted)
        actions.append(action)
        if action == 1:
            out_d, current_lower_state = critic.lower_LSTM_target(current_lower_state, [[inputs[pos]]])
    
    Rinput = []
    for (i, a) in enumerate(actions):
        if a == 1:
            Rinput.append(inputs[i])
    Rlenth = len(Rinput)
    if Rlenth == 0:
        actions[lenth-2] = 1
        Rinput.append(inputs[lenth-2])
        Rlenth = 1
    Rinput += [0] * (args.maxlenth - Rlenth)
    return actions, states, Rinput, Rlenth

def test(sess, actor, critic, test_data, noRL=False):
    acc = 0
    total_lenth = 0
    total_dis = 0
    rwords = {}
    owords = {}
    for i in range(len(test_data)):
        #prepare
        data = test_data[i]
        inputs, solution, lenth = data['words'], data['solution'], data['lenth']
        
        #predict
        if noRL:
            out = critic.predict_target([inputs], [lenth])
        else:
            actions, states, Rinput, Rlenth = sampling_RL(sess, actor, inputs, critic.wordvector_find([inputs]), lenth, Random=False)
            out = critic.predict_target([Rinput], [Rlenth])
            #print json.dumps(actions)
            #print Rinput
            #print json.dumps([dataManager.words[i-1][0] for i in inputs][:lenth])
            #print [dataManager.words[i-1][0] for i in Rinput][:Rlenth]
            #print out, solution
            #print (float(Rlenth)/lenth) * 0.05 * args.grained

        if np.argmax(out) == np.argmax(solution):
            acc += 1

        total_lenth += lenth
        total_dis += Rlenth
        for i in range(lenth):
            wd = dataManager.words[inputs[i]-1][0]
            if owords.has_key(wd):
                owords[wd] = owords[wd] + 1
            else:
                owords[wd] = 1
            if actions[i] == 0:
                if rwords.has_key(wd):
                    rwords[wd] = rwords[wd] + 1
                else:
                    rwords[wd] = 1
    ratewords = {}
    for (key, value) in rwords.items():
        ratewords[key] = float(value) / owords[key]
    rdwords = ratewords.items()
    rdwords.sort(key = lambda x : x[1], reverse = True)
    outcnt = 0
    for i in range(len(rdwords)):
        if owords[rdwords[i][0]] > 20:
            print rdwords[i], owords[rdwords[i][0]]
            outcnt += 1
        if outcnt > 20:
            break;
    avelenth = float(total_lenth) / float(len(test_data))
    avedis = float(total_dis) / float(len(test_data))
    #print "average length", avelenth
    #print "average distilled length", avedis
    return float(acc) / len(test_data)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config = config) as sess:
    #model
    critic = LSTM_CriticNetwork(sess, args.dim, args.optimizer, args.lr, args.tau, args.grained, args.maxlenth, args.dropout, word_vector) 
    actor = ActorNetwork(sess, args.dim, args.optimizer, args.lr, args.tau)
    #print variables
    for item in tf.trainable_variables():
        print (item.name, item.get_shape())
    
    saver = tf.train.Saver()
    
    saver.restore(sess, "checkpoints/best816")

    print test(sess, actor, critic, dev_data)

