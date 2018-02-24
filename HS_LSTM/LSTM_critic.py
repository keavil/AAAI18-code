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
    def __init__(self, sess, dim, optimizer, learning_rate, tau, grained, isAttention, max_lenth, dropout, wordvector):
        self.global_step = tf.Variable(0, trainable=False, name="LSTMStep")
        self.sess = sess
        self.dim = dim
        self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 10000, 0.95, staircase=True)
        self.tau = tau
        self.grained = grained
        self.isAttention = isAttention
        self.max_lenth = max_lenth
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
        self.upper_cell_state, self.upper_cell_input, self.upper_cell_output = self.create_Upper_LSTM_cell('Upper/Active')
        self.lower_cell_state, self.lower_cell_input, self.lower_cell_output, self.lower_cell_state1 = self.create_Lower_LSTM_cell('Lower/Active')

        #critic network (updating)
        self.inputs, self.action, self.action_pos, self.lenth, self.lenth_up, self.out = self.create_critic_network("Active")
        self.network_params = tf.trainable_variables()[self.num_other_variables:]
        
        self.target_wordvector = tf.get_variable('wordvector_target', dtype=tf.float32, initializer=wordvector, trainable=True)

        #lstm cells
        self.target_upper_cell_state, self.target_upper_cell_input, self.target_upper_cell_output = self.create_Upper_LSTM_cell('Upper/Target')
        self.target_lower_cell_state, self.target_lower_cell_input, self.target_lower_cell_output, self.target_lower_cell_state1 = self.create_Lower_LSTM_cell('Lower/Target')
        
        #critic network (delayed updating)
        self.target_inputs, self.target_action, self.target_action_pos, self.target_lenth, self.target_lenth_up, self.target_out = self.create_critic_network("Target")
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
        #with tf.variable_scope("Upper/Active", reuse=True):
        #    self.loss2 = tf.nn.l2_loss(tf.get_variable('lstm_cell/kernel'))
        #with tf.variable_scope("Lower/Active", reuse=True):
        #    self.loss2+= tf.nn.l2_loss(tf.get_variable('lstm_cell/kernel'))
        #with tf.variable_scope("Active/pred", reuse=True):
        #    self.loss2+= tf.nn.l2_loss(tf.get_variable('W'))
        #self.loss += self.loss2 * self.L2regular
        #self.loss_target += self.loss2 * self.L2regular
        self.gradients = tf.gradients(self.loss_target, self.target_network_params)
        self.optimize = self.optimizer.apply_gradients(zip(self.gradients, self.network_params), global_step = self.global_step)
        #self.optimize = self.optimizer.minimize(self.loss)
        
        #total variables
        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_critic_network(self, Scope):
        inputs = tf.placeholder(shape=[1, self.max_lenth], dtype=tf.int32, name="inputs")
        action = tf.placeholder(shape=[1, self.max_lenth], dtype=tf.int32, name="action")
        action_pos = tf.placeholder(shape=[1, None], dtype=tf.int32, name="action_pos")
        lenth = tf.placeholder(shape=[1], dtype=tf.int32, name="lenth")
        lenth_up = tf.placeholder(shape=[1], dtype=tf.int32, name="lenth_up")
       
        #Lower network
        if Scope[-1] == 'e':
            vec = tf.nn.embedding_lookup(self.wordvector, inputs)
            print "active"
        else:
            vec = tf.nn.embedding_lookup(self.target_wordvector, inputs)
            print "target"
        cell = LSTMCell(self.dim, initializer=self.init, state_is_tuple=False)
        self.state_size = cell.state_size
        actions = tf.to_float(action)
        h = cell.zero_state(1, tf.float32)
        embedding = []
        for step in range(self.max_lenth):
            with tf.variable_scope("Lower/"+Scope, reuse=True):
                o, h = cell(vec[:,step,:], h)
            embedding.append(o[0])
            h = h *(1.0 - actions[0,step])

        #Upper network
        embedding = tf.stack(embedding)
        embedding = tf.gather(embedding, action_pos, name="Upper_input")
        with tf.variable_scope("Upper", reuse=True):
            out, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, embedding, lenth_up, dtype=tf.float32, scope=Scope)

        if self.isAttention:
            out = tf.concat(out, 2)
            out = out[0,:,:]
            tmp = tflearn.fully_connected(out, self.dim, scope=Scope, name="att")
            tmp = tflearn.tanh(tmp)
            with tf.variable_scope(Scope):
                v_T = tf.get_variable("v_T", dtype=tf.float32, shape=[self.dim, 1], trainable=True)
            a = tflearn.softmax(tf.matmul(tmp,v_T))
            out = tf.reduce_sum(out * a, 0)
            out = tf.expand_dims(out, 0)
        else:
            #out = embedding[:, -1, :]
            out = tf.concat((out[0][:,-1,:], out[1][:,0,:]), 1)

        out = tflearn.dropout(out, self.keep_prob)
        out = tflearn.fully_connected(out, self.grained, scope=Scope+"/pred", name="get_pred")
        return inputs, action, action_pos, lenth, lenth_up, out
    
    def create_Lower_LSTM_cell(self,Scope):
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
    
    def create_Upper_LSTM_cell(self, Scope):
        cell = LSTMCell(self.dim, initializer=self.init, state_is_tuple=False)
        state_l = tf.placeholder(tf.float32, shape = [1, cell.state_size], name="cell_state_l")
        state_d = tf.placeholder(tf.float32, shape = [1, self.dim], name="cell_state_d")
        with tf.variable_scope(Scope, reuse=False):
            _, out = cell(state_d, state_l)
        return state_l, state_d, out

    def getloss(self, inputs, action, action_pos, lenth, lenth_up, ground_truth):
        return self.sess.run([self.target_out, self.loss_target], feed_dict={
            self.target_inputs: inputs,
            self.target_action: action,
            self.target_action_pos: action_pos,
            self.target_lenth: lenth,
            self.target_lenth_up: lenth_up,
            self.ground_truth: ground_truth,
            self.keep_prob: 1.0})

    def train(self, inputs, action, action_pos, lenth, lenth_up, ground_truth):
        return self.sess.run([self.target_out, self.loss_target, self.optimize], feed_dict={
            self.target_inputs: inputs,
            self.target_action: action,
            self.target_action_pos: action_pos,
            self.target_lenth: lenth,
            self.target_lenth_up: lenth_up,
            self.ground_truth: ground_truth,
            self.keep_prob: self.dropout})

    def predict_target(self, inputs, action, action_pos, lenth, lenth_up):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action,
            self.target_action_pos: action_pos,
            self.target_lenth: lenth,
            self.target_lenth_up: lenth_up,
            self.keep_prob: 1.0})

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
    
    def assign_target_network(self):
        self.sess.run(self.assign_target_network_params)
    
    def assign_active_network(self):
        self.sess.run(self.assign_active_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
    
    def upper_LSTM_target(self, state, inputs):
        return self.sess.run(self.target_upper_cell_output, feed_dict={
            self.target_upper_cell_state: state,
            self.target_upper_cell_input: inputs})

    def lower_LSTM_target(self, state, inputs):
        return self.sess.run([self.target_lower_cell_output, self.target_lower_cell_state1], feed_dict={
            self.target_lower_cell_state: state,
            self.target_lower_cell_input: inputs})
