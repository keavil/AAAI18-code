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

def sampling_random(lenth, p_action = None):
    actions = []
    typ = args.pretype
    actions = np.copy(p_action).tolist()
    actions += [0] * (args.maxlenth - lenth)
    action_pos = []
    for (i, j) in enumerate(actions):
        if j == 1:
            action_pos.append(i)
    if len(action_pos) == 0:
        actions[lenth-1] = 1
        action_pos.append(lenth-1)
    if len(actions) != args.maxlenth:
        print lenth, p_action
    return actions, action_pos

def train(sess, actor, critic, train_data, batch_size, samplecnt, LSTM_trainable=True, RL_trainable=True):
    print "training : total ", len(train_data), "nodes. ", len(train_data)/batch_size, " batchs." 
    random.shuffle(train_data)
    for b in range(len(train_data)/batch_size):
        datas = train_data[b * batch_size: (b+1) * batch_size]
        totloss = 0.
        actor.assign_active_network()
        critic.assign_active_network()
        for i in range(batch_size):
            #prepare
            data = datas[i]
            inputs, solution, lenth, p_action = data['words'], data['solution'], data['lenth'], data['action']
            aveloss = 0.
            statelist, actionlist, losslist = [], [], []
            #get sampling
            if RL_trainable:
                for sp in range(samplecnt):
                    actions, states, action_pos = sampling_RL(sess, actor, inputs, lenth)
                    statelist.append(states)
                    actionlist.append(actions) 
                    out, loss = critic.getloss([inputs], [actions], [action_pos], [lenth], [len(action_pos)], [solution])
                    # control loss of lenth
                    _x = float(len(action_pos)) /  lenth
                    loss += (1 * _x + 0.1 / _x - 0.6) * 0.1 * args.grained
                    #
                    aveloss += loss
                    losslist.append(loss)
            else :
                actions, action_pos = sampling_random(lenth, p_action) 
            #train the predict network
            if LSTM_trainable:
                out, loss, _ = critic.train([inputs], [actions], [action_pos], [lenth], [len(action_pos)], [solution])
                if not RL_trainable:
                    totloss += loss
            #train the actor network
            if RL_trainable:
                aveloss /= samplecnt
                totloss += aveloss
                grad = None
                for sp in range(samplecnt):
                    for pos in range(lenth):
                        rr = [0.,0.]
                        rr[actionlist[sp][pos]] = (losslist[sp] - aveloss) * args.alpha
                        
                        g = actor.get_gradient(statelist[sp][pos][0], statelist[sp][pos][1], rr)
                        if grad == None:
                            grad = g
                        else:
                            grad[0] += g[0]
                            grad[1] += g[1]
                            grad[2] += g[2]
                actor.train(grad)
                
        if RL_trainable:
            actor.update_target_network()
        if LSTM_trainable:
            if RL_trainable:
                critic.update_target_network()
            else:
                critic.assign_target_network()
        if (b + 1) % 500 == 0:
            acc_test = test(sess, actor, critic, test_data, not RL_trainable)
            acc_dev  = test(sess, actor, critic, dev_data, not RL_trainable)
            print "batch ",b , "total loss ", totloss, "----test: ", acc_test, "| dev: ", acc_dev

def test(sess, actor, critic, test_data, Random=False):
    acc = 0
    for i in range(len(test_data)):
        #prepare
        data = test_data[i]
        inputs, solution, lenth, paction = data['words'], data['solution'], data['lenth'], data['action'] 
        #get sampling
        if Random == False:
            actions, _, action_pos = sampling_RL(sess, actor, inputs, lenth, Random=False)
        else:
            actions, action_pos = sampling_random(lenth, paction)
        
        if len(actions) != args.maxlenth:
            print inputs
        #predict
        out = critic.predict_target([inputs], [actions], [action_pos], [lenth], [len(action_pos)])
        if np.argmax(out) == np.argmax(solution):
            acc += 1
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
    
    #LSTM pretrain
    if args.RLpretrain != '':
        pass
    elif args.LSTMpretrain == '':
        sess.run(tf.global_variables_initializer())
        for i in range(0,2):
            train(sess, actor, critic, train_data, args.batchsize, args.samplecnt, RL_trainable=False)
            critic.assign_target_network()
            acc_test = test(sess, actor, critic, test_data, True)
            acc_dev = test(sess, actor, critic, dev_data, True)
            print "LSTM_only ",i, "----test: ", acc_test, "| dev: ", acc_dev
            saver.save(sess, "checkpoints/"+args.name+"_base", global_step=i)
        print "LSTM pretrain OK"
    else:
        print "Load LSTM from ", args.LSTMpretrain
        saver.restore(sess, args.LSTMpretrain)
        pass
    #RL pretrain
    if args.RLpretrain == '':
        for i in range(0,5):
            train(sess, actor, critic, train_data, args.batchsize, args.samplecnt, LSTM_trainable=False)
            acc_test = test(sess, actor, critic, test_data)
            acc_dev = test(sess, actor, critic, dev_data)
            print "RL pretrain ", i, "----test: ", acc_test, "| dev: ", acc_dev
            saver.save(sess, "checkpoints/"+args.name+"_RL", global_step=i)
        print "RL pretrain OK"
    else:
        print "Load RL from ", args.RLpretrain
        saver.restore(sess, args.RLpretrain)
    #train
    results = []
    for e in range(args.epoch):
        train(sess, actor, critic, train_data, args.batchsize, args.samplecnt)
        acc_test = test(sess, actor, critic, test_data)
        acc_dev  = test(sess, actor, critic, dev_data)
        print "epoch ", e, "---- test: ", acc_test, "| dev: ", acc_dev
        saver.save(sess, "checkpoints/"+args.name, global_step=e)


