import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    
# import tensorflow as tf
import tensorflow as tf
import numpy as np
import pickle as pickle
import io_1 as io
# from tensorflow import keras
from tqdm import tqdm_notebook


class Network(object):

    def __init__(self, input_size, latent_size, input2_size, output_size,
                 encoder_num_units=[100, 100], decoder_num_units=[100, 100], name='Unnamed',
                 tot_epochs=0, load_file=None):
        # Parameters:
        # input_size: length of a single data vector.
        # latent_size: number of latent neurons to be used.
        # input2_size: number of neurons for 2nd input into decoder.
        # output_size: length of a single label vector.
        # encoder_num_units, decoder_num_units: Number of neurons in encoder and decoder hidden layers. Everything is fully connected.
        # name: Used for tensorboard
        # tot_epochs and  load_file are used internally for loading and saving, don't pass anything to them manually.

        self.graph = tf.Graph()
        self.input_size = input_size
        self.latent_size = latent_size
        self.input2_size = input2_size
        self.output_size = output_size
        self.encoder_num_units = encoder_num_units
        self.decoder_num_units = decoder_num_units
        self.name = name
        self.tot_epochs = tot_epochs

        # Set up neural network
        self.graph_setup()
        self.session = tf.compat.v1.Session(graph=self.graph)
        with self.graph.as_default():
            initialize_uninitialized(self.session)

        # Load saved network
        self.load_file = load_file
        if self.load_file is not None:
            self.load(self.load_file)

    #########################################
    #           Public interface            #
    #########################################

    # Train the network.
    def train(self, epoch_num, batch_size, learning_rate, training_data, validation_data,
              beta_fun=lambda x: 0.001, test_step=None):
        
        with self.graph.as_default():
            initialize_uninitialized(self.session)

            for epoch_iter in tqdm_notebook(range(epoch_num)):
                self.tot_epochs += 1
                current_beta = beta_fun(self.tot_epochs)

                if test_step is not None and self.tot_epochs > 0 and self.tot_epochs % test_step == 0:
                    self.test(validation_data, beta=current_beta)

                for step, data_dict in enumerate(self.gen_batch(training_data, batch_size)):

                    parameter_dict = {self.learning_rate: learning_rate, self.beta: current_beta}
                    feed_dict = dict(data_dict, **parameter_dict)

                    self.session.run(self.training_op, feed_dict=feed_dict)

    #Test the network.
    def test(self, data, beta=0):
        
        with self.graph.as_default():
            data_dict = self.gen_data_dict(data, random_epsilon=False)
            parameter_dict = {self.beta: beta}
            summary = self.session.run(self.all_summaries, feed_dict=dict(data_dict, **parameter_dict))
            self.summary_writer.add_summary(summary, global_step=self.tot_epochs)

    #Run the network.
    def run(self, data, layer, random_epsilon=False, additional_params={}):
       
        with self.graph.as_default():
            data_dict = self.gen_data_dict(data, random_epsilon)
            return self.session.run(layer, feed_dict=dict(data_dict, **additional_params))
    
    #Save the network.
    def save(self, file_name):

        with self.graph.as_default():
            saver = tf.compat.v1.train.Saver()
            saver.save(self.session, io.tf_save_path + file_name + '.ckpt')
            params = {'latent_size': self.latent_size,
                      'input_size': self.input_size,
                      'input2_size': self.input2_size,
                      'output_size': self.output_size,
                      'encoder_num_units': self.encoder_num_units,
                      'decoder_num_units': self.decoder_num_units,
                      'tot_epochs': self.tot_epochs,
                      'name': self.name}
            with open(io.tf_save_path + file_name + '.pkl', 'wb') as f:
                pickle.dump(params, f)
            print("Saved network to file " + file_name)

    #Initialize a new network from saved data.
    @classmethod
    def from_saved(cls, file_name, change_params={}):
        """
        Initializes a new network from saved data.
        file_name (str): model is loaded from tf_save/file_name.ckpt
        """
        with open(io.tf_save_path + file_name + '.pkl', 'rb') as f:
            params = pickle.load(f)
        params['load_file'] = file_name
        for p in change_params:
            params[p] = change_params[p]
        print(params)
        return cls(**params)

    #########################################
    #        Private helper functions       #
    ########################################

    # Set up the TensorFlow graph.
    def graph_setup(self):
        with self.graph.as_default():

            #######################
            # Define placeholders #
            #######################
            # self.input = tf.keras.Input(tf.int32, [None, self.input_size], name='input')
            # self.epsilon = tf.keras.Input(tf.float32, [None, self.latent_size], name='epsilon')
            # self.learning_rate = tf.keras.Input(tf.float32, shape=[], name='learning_rate')
            # self.beta = tf.keras.Input(tf.float32, shape=[], name='beta')
            # self.input2 = tf.keras.Input(tf.float32, shape=[None, self.input2_size], name='input2')
            # self.labels = tf.keras.Input(tf.float32, shape=[None, self.output_size], name='labels')
            self.input = tf.compat.v1.placeholder(tf.float32, [None, self.input_size], name='input')
            self.epsilon = tf.compat.v1.placeholder(tf.float32, [None, self.latent_size], name='epsilon')
            self.learning_rate = tf.compat.v1.placeholder(tf.float32, shape=[], name='learning_rate')
            self.beta = tf.compat.v1.placeholder(tf.float32, shape=[], name='beta')
            self.input2 = tf.compat.v1.placeholder(tf.float32, shape=[None, self.input2_size], name='input2')
            self.labels = tf.compat.v1.placeholder(tf.float32, shape=[None, self.output_size], name='labels')


            ##########################################
            # Set up variables and computation graph #
            ##########################################
            with tf.compat.v1.variable_scope('encoder'):
                # input and output dimensions for each of the weight tensors
                enc_in_dims = [self.input_size] + self.encoder_num_units
                enc_out_dims = self.encoder_num_units + [2 * self.latent_size]
                temp_layer = self.input

                for k in range(len(enc_in_dims)):
                    with tf.compat.v1.variable_scope('{}th_enc_layer'.format(k)):
                        w = tf.compat.v1.get_variable('w', [enc_in_dims[k], enc_out_dims[k]],
                                            initializer=tf.compat.v1.initializers.random_normal(stddev=2. / np.sqrt(enc_in_dims[k] + enc_out_dims[k])))
                        b = tf.compat.v1.get_variable('b', [enc_out_dims[k]], initializer=tf.compat.v1.zeros_initializer())
                        squash = ((k + 1) != len(enc_in_dims))  # don't squash latent layer
                        temp_layer = forwardprop(temp_layer, w, b, name='enc_layer_{}'.format(k), squash=squash)



            with tf.compat.v1.name_scope('latent_layer'):
                self.log_sigma = temp_layer[:, :self.latent_size]
                self.mu = temp_layer[:, self.latent_size:]
                self.mu_sample = tf.add(self.mu, tf.exp(self.log_sigma) * self.epsilon, name='add_noise')
                self.mu_with_input2 = tf.concat([self.mu_sample, self.input2], axis=1)

            with tf.compat.v1.name_scope('kl_loss'):
                self.kl_loss = kl_divergence(self.mu, self.log_sigma, dim=self.latent_size)

            with tf.compat.v1.variable_scope('decoder'):
                temp_layer = self.mu_with_input2

                dec_in_dims = [self.latent_size + self.input2_size] + self.decoder_num_units
                dec_out_dims = self.decoder_num_units + [self.output_size]
                for k in range(len(dec_in_dims)):
                    with tf.compat.v1.variable_scope('{}th_dec_layer'.format(k)):
                        w = tf.Variable(tf.random.normal(tf.cast([dec_in_dims[k], dec_out_dims[k]], tf.int32), stddev=2. / tf.sqrt(tf.cast(dec_in_dims[k] + dec_out_dims[k], tf.float32))), name='w')
                        b = tf.Variable(tf.zeros([dec_out_dims[k]]), name='b')
                        squash = ((k + 1) != len(dec_in_dims))  # don't squash latent layer
                        temp_layer = forwardprop(temp_layer, w, b, name='dec_layer_{}'.format(k), squash=squash)


                self.output = temp_layer

            with tf.compat.v1.name_scope('recon_loss'):
                self.recon_loss = tf.reduce_mean(tf.reduce_sum(tf.math.squared_difference(self.labels, self.output), axis=1))

            #####################
            # Cost and training #
            #####################
            with tf.compat.v1.name_scope('cost'):
                self.cost = self.recon_loss + self.beta * self.kl_loss
            with tf.compat.v1.name_scope('optimizer'):
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
                gvs = optimizer.compute_gradients(self.cost)
                capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
                self.training_op = optimizer.apply_gradients(capped_gvs)

            #########################
            # Tensorboard summaries #
            #########################
            tf.compat.v1.summary.histogram('latent_means', self.mu)
            tf.compat.v1.summary.histogram('latent_log_sigma', self.log_sigma)
            tf.compat.v1.summary.histogram('ouput_means', self.output)
            tf.compat.v1.summary.scalar('recon_loss', self.recon_loss)
            tf.compat.v1.summary.scalar('kl_loss', self.kl_loss)
            tf.compat.v1.summary.scalar('cost', self.cost)
            tf.compat.v1.summary.scalar('beta', self.beta)

            self.summary_writer = tf.compat.v1.summary.FileWriter(io.tf_log_path + self.name + '/', graph=self.graph)
            self.summary_writer.flush()  # write out graph
            self.all_summaries = tf.compat.v1.summary.merge_all()

    # Generate batches of data.
    def gen_batch(self, data, batch_size, shuffle=True, random_epsilon=True):
       
        epoch_size = len(data[0]) / batch_size
        if shuffle:
            p = np.random.permutation(len(data[0]))
            data = [data[i][p] for i in [0, 1, 2]]
        for i in range(epoch_size):
            batch_slice = slice(i * batch_size, (i + 1) * batch_size)
            batch = [data[j][batch_slice] for j in [0, 1, 2]]
            yield self.gen_data_dict(batch, random_epsilon=random_epsilon)

    #Generate data dictionary for feeding into the network.
    def gen_data_dict(self, data, random_epsilon=True):
        
        if random_epsilon is True:
            eps = np.random.normal(size=[len(data[0]), self.latent_size])
        else:
            eps = np.zeros([len(data[0]), self.latent_size])
        return {self.input: data[0],
                self.input2: data[1],
                self.labels: data[2],
                self.epsilon: eps}

    #Load the network.
    def load(self, file_name):
        
        with self.graph.as_default():
            saver = tf.compat.v1.train.Saver()
            saver.restore(self.session, io.tf_save_path + file_name + '.ckpt')
            print("Loaded network from file " + file_name)


###########
# Helpers #
###########


def forwardprop(x, w, b, squash=True, act_fun=tf.nn.elu, name=''):
    
    if name != '':
        name = '_' + name
    pre_act = tf.add(tf.matmul(x, w, name=('w_mul' + name)), b, name=('b_add' + name))
    if name != '':
        tf.compat.v1.summary.histogram('pre-act' + name, pre_act)
    if squash:
        return act_fun(pre_act, name=('act_fun' + name))
    else:
        return pre_act


def initialize_uninitialized(sess):
    global_vars = tf.compat.v1.global_variables()
    is_not_initialized = sess.run([tf.compat.v1.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    if len(not_initialized_vars):
        sess.run(tf.compat.v1.variables_initializer(not_initialized_vars))


def kl_divergence(means, log_sigma, dim, target_sigma=0.1):
    # KL divergence between given distribution and unit Gaussian
    target_sigma = tf.constant(target_sigma, shape=[dim])
    return 1 / 2. * tf.reduce_mean(tf.reduce_sum(1 / target_sigma**2 * means**2 +
                                                 tf.exp(2 * log_sigma) / target_sigma**2 - 2 * log_sigma + 2 * tf.math.log(target_sigma), axis=1) - dim)
