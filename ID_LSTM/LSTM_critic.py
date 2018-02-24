import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
import tflearn
import numpy as np

class LSTM_CriticNetwork(object):
    """
    predict network.
    use the word vector and actions(sampled from actor network)
    get the final prediction.
    """
    def __init__(self, sess, dim, optimizer, learning_rate, tau, grained, max_lenth, dropout, wordvector):
        self.global_step = tf.Variable(0, trainable=False, name="LSTMStep")
        self.sess = sess
        self.max_lenth = max_lenth
        self.dim = dim
        self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 10000, 0.95, staircase=True)
        self.tau = tau
        self.grained = grained
        self.dropout = dropout
        self.init = tf.random_uniform_initializer(-0.05, 0.05, dtype=tf.float32)
        self.L2regular = 0.00001 # add to parser
        print "optimizer: ", optimizer
        if optimizer == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif optimizer == 'Adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif optimizer == 'Adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
        self.keep_prob = tf.placeholder(tf.float32, name="keepprob")
        self.num_other_variables = len(tf.trainable_variables())
        self.wordvector = tf.get_variable('wordvector', dtype=tf.float32, initializer=wordvector, trainable=True)

        #lstm cells
        self.lower_cell_state, self.lower_cell_input, self.lower_cell_output, self.lower_cell_state1 = self.create_LSTM_cell('Lower/Active')

        #critic network (updating)
        self.inputs, self.lenth, self.out = self.create_critic_network("Active")
        self.network_params = tf.trainable_variables()[self.num_other_variables:]
        
        self.target_wordvector = tf.get_variable('wordvector_target', dtype=tf.float32, initializer=wordvector, trainable=True)

        #lstm cells
        self.target_lower_cell_state, self.target_lower_cell_input, self.target_lower_cell_output, self.target_lower_cell_state1 = self.create_LSTM_cell('Lower/Target')
        
        #critic network (delayed updating)
        self.target_inputs, self.target_lenth, self.target_out = self.create_critic_network("Target")
        self.target_network_params = tf.trainable_variables()[len(self.network_params)+self.num_other_variables:]

        #delayed updating critic network ops
        self.update_target_network_params = \
                [self.target_network_params[i].assign(\
                tf.multiply(self.network_params[i], self.tau)+\
                tf.multiply(self.target_network_params[i], 1 - self.tau))\
                for i in range(len(self.target_network_params))]
        
        self.assign_target_network_params = \
                [self.target_network_params[i].assign(\
                self.network_params[i]) for i in range(len(self.target_network_params))]
        self.assign_active_network_params = \
                [self.network_params[i].assign(\
                self.target_network_params[i]) for i in range(len(self.network_params))]

        self.ground_truth = tf.placeholder(tf.float32, [1,self.grained], name="ground_truth")
        
        
        #calculate loss
        self.loss_target = tf.nn.softmax_cross_entropy_with_logits(labels=self.ground_truth, logits=self.target_out)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.ground_truth, logits=self.out)
        self.loss2 = 0
        with tf.variable_scope("Lower/Active", reuse=True):
            self.loss2+= tf.nn.l2_loss(tf.get_variable('lstm_cell/kernel'))
        with tf.variable_scope("Active/pred", reuse=True):
            self.loss2+= tf.nn.l2_loss(tf.get_variable('W'))
        self.loss += self.loss2 * self.L2regular
        self.loss_target += self.loss2 * self.L2regular
        self.gradients = tf.gradients(self.loss_target, self.target_network_params)
        self.optimize = self.optimizer.apply_gradients(zip(self.gradients, self.network_params), global_step = self.global_step)
        
        #total variables
        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)
        
        #wordvector look for
        self.WVinput, self.WVvec = self.create_wordvector_find()


    def create_critic_network(self, Scope):
        inputs = tf.placeholder(shape=[1, self.max_lenth], dtype=tf.int32, name="inputs")
        lenth = tf.placeholder(shape=[1], dtype=tf.int32, name="lenth")
       
        #Lower network
        if Scope[-1] == 'e':
            vec = tf.nn.embedding_lookup(self.wordvector, inputs)
        else:
            vec = tf.nn.embedding_lookup(self.target_wordvector, inputs)
        cell = LSTMCell(self.dim, initializer=self.init, state_is_tuple=False)
        
        with tf.variable_scope("Lower", reuse=True):
            out, _ = tf.nn.dynamic_rnn(cell, vec, lenth, dtype=tf.float32, scope=Scope)
        out = tf.gather(out[0], lenth-1)
        
        out = tflearn.dropout(out, self.keep_prob)
        out = tflearn.fully_connected(out, self.grained, scope=Scope+"/pred", name="get_pred")
        return inputs, lenth, out
    
    def create_LSTM_cell(self,Scope):
        cell = LSTMCell(self.dim, initializer=self.init, state_is_tuple=False)
        state = tf.placeholder(tf.float32, shape = [1, cell.state_size], name="cell_state")
        inputs = tf.placeholder(tf.int32, shape = [1, 1], name="cell_input")
        if Scope[-1] == 'e':
            vec = tf.nn.embedding_lookup(self.wordvector, inputs)
        else:
            vec = tf.nn.embedding_lookup(self.target_wordvector, inputs)
        with tf.variable_scope(Scope, reuse=False):
            out, state1 = cell(vec[:,0,:], state)
        return state, inputs, out, state1
    
    def create_wordvector_find(self):
        inputs = tf.placeholder(tf.int32, shape=[1, self.max_lenth], name="WVtofind")
        vec = tf.nn.embedding_lookup(self.target_wordvector, inputs)
        return inputs, vec

    def getloss(self, inputs, lenth, ground_truth):
        return self.sess.run([self.target_out, self.loss_target], feed_dict={
            self.target_inputs: inputs,
            self.target_lenth: lenth,
            self.ground_truth: ground_truth,
            self.keep_prob: 1.0})

    def train(self, inputs, lenth, ground_truth):
        return self.sess.run([self.target_out, self.loss_target, self.optimize], feed_dict={
            self.target_inputs: inputs,
            self.target_lenth: lenth,
            self.ground_truth: ground_truth,
            self.keep_prob: self.dropout})

    def predict_target(self, inputs, lenth):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_lenth: lenth,
            self.keep_prob: 1.0})

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
    
    def assign_target_network(self):
        self.sess.run(self.assign_target_network_params)
    
    def assign_active_network(self):
        self.sess.run(self.assign_active_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
    
    def lower_LSTM_target(self, state, inputs):
        return self.sess.run([self.target_lower_cell_output, self.target_lower_cell_state1], feed_dict={
            self.target_lower_cell_state: state,
            self.target_lower_cell_input: inputs})

    def wordvector_find(self, inputs):
        return self.sess.run(self.WVvec, feed_dict={
            self.WVinput :inputs})
