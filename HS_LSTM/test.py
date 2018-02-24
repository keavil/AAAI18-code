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

print "train_data ", len(train_data)
print "dev_data", len(dev_data)
print "test_data", len(test_data)
if args.fasttest == 1:
    train_data = train_data[:100]
    dev_data = dev_data[:20]
    test_data = test_data[:20]

def sampling_RL(sess, actor, inputs, lenth, Random=True):
    current_lower_state = np.zeros((1, state_size), dtype=np.float32)
    current_upper_state = np.zeros((1, state_size), dtype=np.float32)
    actions = []
    states = []
    #sampling actions
    
    for pos in range(lenth):
        out_d, current_lower_state = critic.lower_LSTM_target(current_lower_state, [[inputs[pos]]])
        predicted = actor.predict_target(current_upper_state, current_lower_state)
        #print predicted
        states.append([current_upper_state, current_lower_state])
        if Random:
            action = (0 if random.random() < predicted[0] else 1)
        else:
            action = np.argmax(predicted)
        actions.append(action)
        if action == 1:
            current_upper_state = critic.upper_LSTM_target(current_upper_state, out_d)
            current_lower_state = np.zeros_like(current_lower_state)
               
    #pad zeros
    actions += [0] * (args.maxlenth - lenth)
    actions[lenth-1] = 1
    #get the position of action 1
    action_pos = []
    for (i, j) in enumerate(actions):
        if j == 1:
            action_pos.append(i)
    return actions, states, action_pos


def test(sess, actor, critic, test_data, Random=False):
    acc = 0
    total_lenth = 0
    total_phrase_count = 0
    for i in range(len(test_data)):
        #prepare
        data = test_data[i]
        inputs, solution, lenth, paction = data['words'], data['solution'], data['lenth'], data['action'] 
        #get sampling
        if Random == False:
            actions, _, action_pos = sampling_RL(sess, actor, inputs, lenth, Random=False)
        else:
            actions, action_pos = sampling_random(lenth, postag, paction)
        
        #predict
        out = critic.predict_target([inputs], [actions], [action_pos], [lenth], [len(action_pos)])
        if np.argmax(out) == np.argmax(solution):
            acc += 1

        print json.dumps(actions[:lenth])
        print json.dumps([dataManager.words[i-1][0] for i in inputs][:lenth])
        #print out, solution

        total_lenth += lenth
        total_phrase_count += len(action_pos)
    
    avelenth = float(total_lenth) / float(len(test_data))
    avephrase= float(total_phrase_count) / float(len(test_data))
    avephraselenth = avelenth / avephrase

    #print "average length :", avelenth
    #print "average phrase number :", avephrase
    #print "average phrase length :", avephraselenth

    return float(acc) / len(test_data)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config = config) as sess:
    #model
    critic = LSTM_CriticNetwork(sess, args.dim, args.optimizer, args.lr, args.tau, args.grained, args.attention, args.maxlenth, args.dropout, word_vector) 
    actor = ActorNetwork(sess, args.dim, args.optimizer, args.lr, args.tau, critic.get_num_trainable_vars())
    state_size = critic.state_size

    #print variables
    for item in tf.trainable_variables():
        print (item.name, item.get_shape())
    
    saver = tf.train.Saver()
    
    saver.restore(sess, "checkpoints/best821")
    test(sess, actor, critic, dev_data)
