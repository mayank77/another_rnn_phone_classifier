
# coding: utf-8

# In[62]:

import numpy as np
import pickle, os, random, math, sys

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

from sklearn.metrics import confusion_matrix

import dnnutil
import importlib
importlib.reload(dnnutil)

phone_stash = dnnutil.phone_stash

corpus = "en_uk_kids_align_from_clean"
pickle_dir='../features/work_in_progress/'+corpus+'/pickles'



# In[64]:

traindata = phone_stash([ os.path.join(pickle_dir, 'train-0.pickle'),
                          os.path.join(pickle_dir, 'train-1.pickle') ])
evaldata = phone_stash([ os.path.join(pickle_dir, 'train-2.pickle') ], zmean=traindata.mean, zstd=traindata.std, max_len=traindata.max_len)

'''
traindata = phone_stash([ os.path.join(pickle_dir, 'train-0.pickle'),
                           os.path.join(pickle_dir, 'train-1.pickle'),
                           os.path.join(pickle_dir, 'train-2.pickle'),
                           os.path.join(pickle_dir, 'train-3.pickle'),
                           os.path.join(pickle_dir, 'train-4.pickle'),
                           os.path.join(pickle_dir, 'train-5.pickle'),
                           os.path.join(pickle_dir, 'train-6.pickle'),
                           os.path.join(pickle_dir, 'train-7.pickle') ])

evaldata = phone_stash([ os.path.join(pickle_dir, 'test-0.pickle') ], zmean=traindata.mean, zstd=traindata.std, max_len=traindata.max_len)
'''


hm_epochs = 40
n_classes = traindata.num_classes
batch_size = 128
chunk_size = traindata.max_len #30
n_chunks = 129 #28
rnn_size = 512
n_hidden = rnn_size

num_layers = 5

dropoutval=0.8


# # 1 Autoencoder training #
# 
# Nicked from https://github.com/rajarsheem/libsdae/blob/master/deepautoencoder/stacked_autoencoder.py and heavily adapted:

# In[70]:

import numpy as np
import tensorflow as tf

allowed_activations = ['sigmoid', 'tanh', 'softmax', 'relu', 'linear']
allowed_noises = [None, 'gaussian', 'mask']
allowed_losses = ['rmse', 'cross-entropy']


def get_batch(X, X_, size):
    a = np.random.choice(len(X), size, replace=False)
    return X[a], X_[a]


class StackedAutoEncoder:
    """A deep autoencoder with denoising capability"""

    def assertions(self):
        global allowed_activations, allowed_noises, allowed_losses
        assert self.loss in allowed_losses, 'Incorrect loss given'
        assert 'list' in str(
            type(self.dims)), 'dims must be a list even if there is one layer.'
        assert len(self.epoch) == len(
            self.dims), "No. of epochs must equal to no. of hidden layers"
        assert len(self.activations) == len(
            self.dims), "No. of activations must equal to no. of hidden layers"
        assert all(
            True if x > 0 else False
            for x in self.epoch), "No. of epoch must be atleast 1"
        assert set(self.activations + allowed_activations) == set(
            allowed_activations), "Incorrect activation given."
        #assert utils.noise_validator(
        #    self.noise, allowed_noises), "Incorrect noise given"

    def __init__(self, dims, activations, epoch=1000, noise=None, loss='rmse',
                 lr=0.001, batch_size=100, print_step=50):
        self.print_step = print_step
        self.batch_size = batch_size
        self.lr = lr
        self.loss = loss
        self.activations = activations
        self.noise = noise
        self.epoch = epoch
        self.dims = dims
        self.assertions()
        self.depth = len(dims)
        self.weights, self.biases = [], []

    def add_noise(self, x):
        if self.noise == 'gaussian':
            n = np.random.normal(0, 0.1, (len(x), len(x[0])))
            return x + n
        if 'mask' in self.noise:
            frac = float(self.noise.split('-')[1])
            temp = np.copy(x)
            for i in temp:
                n = np.random.choice(len(i), round(
                    frac * len(i)), replace=False)
                i[n] = 0
            return temp
        if self.noise == 'sp':
            pass

    def fit(self, noisy_x, clean_x):
        for i in range(self.depth):
            print('Layer {0}'.format(i + 1))
            #if self.noise is None:
            #    x = self.run(data_x=x, activation=self.activations[i],
            #                 data_x_=x,
            #                 hidden_dim=self.dims[i], epoch=self.epoch[
            #                     i], loss=self.loss,
            #                 batch_size=self.batch_size, lr=self.lr,
            #                 print_step=self.print_step)
            #else:
            if 1 == 1:
                #temp = np.copy(x)
                x = clean_x
                x = self.run(data_x = noisy_x, #data_x=self.add_noise(temp),
                             activation=self.activations[i], data_x_=x,
                             hidden_dim=self.dims[i],
                             epoch=self.epoch[
                                 i], loss=self.loss,
                             batch_size=self.batch_size,
                             lr=self.lr, print_step=self.print_step)

    def transform(self, data):
        tf.reset_default_graph()
        sess = tf.Session()
        x = tf.constant(data, dtype=tf.float32)
        for w, b, a in zip(self.weights, self.biases, self.activations):
            weight = tf.constant(w, dtype=tf.float32)
            bias = tf.constant(b, dtype=tf.float32)
            layer = tf.matmul(x, weight) + bias
            x = self.activate(layer, a)
        return x.eval(session=sess)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def run(self, data_x, data_x_, hidden_dim, activation, loss, lr,
            print_step, epoch, batch_size=100):
        tf.reset_default_graph()
        input_dim = len(data_x[0])
        sess = tf.Session()
        x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x')
        x_ = tf.placeholder(dtype=tf.float32, shape=[
                            None, input_dim], name='x_')
        encode = {'weights': tf.Variable(tf.truncated_normal(
            [input_dim, hidden_dim], dtype=tf.float32)),
            'biases': tf.Variable(tf.truncated_normal([hidden_dim],
                                                      dtype=tf.float32))}
        decode = {'biases': tf.Variable(tf.truncated_normal([input_dim],
                                                            dtype=tf.float32)),
                  'weights': tf.transpose(encode['weights'])}
        encoded = self.activate(
            tf.matmul(x, encode['weights']) + encode['biases'], activation)
        decoded = tf.matmul(encoded, decode['weights']) + decode['biases']

        # reconstruction loss
        if loss == 'rmse':
            loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(x_, decoded))))
        elif loss == 'cross-entropy':
            loss = -tf.reduce_mean(x_ * tf.log(decoded))
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)

        sess.run(tf.initialize_all_variables())
        for i in range(epoch):
            b_x, b_x_ = get_batch(
                data_x, data_x_, batch_size)
            sess.run(train_op, feed_dict={x: b_x, x_: b_x_})
            if (i + 1) % print_step == 0:
                l = sess.run(loss, feed_dict={x: data_x, x_: data_x_})
                print('epoch {0}: global loss = {1}'.format(i, l))
        # debug
        # print('Decoded', sess.run(decoded, feed_dict={x: self.data_x_})[0])
        self.weights.append(sess.run(encode['weights']))
        self.biases.append(sess.run(encode['biases']))
        return sess.run(encoded, feed_dict={x: data_x_})

    def activate(self, linear, name):
        if name == 'sigmoid':
            return tf.nn.sigmoid(linear, name='encoded')
        elif name == 'softmax':
            return tf.nn.softmax(linear, name='encoded')
        elif name == 'linear':
            return linear
        elif name == 'tanh':
            return tf.nn.tanh(linear, name='encoded')
        elif name == 'relu':
            return tf.nn.relu(linear, name='encoded')


# In[85]:


#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#data, target = mnist.train.images, mnist.train.labels

# train / test  split
#idx = np.random.rand(data.shape[0]) < 0.8
#train_X, train_Y = data[idx], target[idx]
#test_X, test_Y = data[~idx], target[~idx]

noisy_X = traindata.noisydata 
clean_X = traindata.cleandata 

#noisy_X = np.reshape(traindata.noisydata,(-1,129,1))
#clean_X = np.reshape(traindata.cleandata,(-1,129,1))

model = StackedAutoEncoder(dims=[96, 64, 32], activations=['relu', 'relu', 'relu'], epoch=[
                           300, 300, 300], loss='rmse', lr=0.007, batch_size=128, print_step=100)
model.fit(noisy_X, clean_X)


# In[95]:

model.weights[1].shape


# In[86]:

#est_X = np.reshape(evaldata.noisydata,(-1,129,1))
#rint (test_X[0,:,:])
test_X = evaldata.noisydata[0:129:]
test_X_ = model.transform(test_X)


# In[59]:



x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network(x, keep_prob):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(0, n_chunks, x)

    #GRU

    gru_cell = rnn_cell.GRUCell(rnn_size)
    gru_cell = rnn_cell.DropoutWrapper(gru_cell, output_keep_prob=keep_prob)
    gru_cell = rnn_cell.MultiRNNCell([gru_cell] * num_layers)
    outputs, states = rnn.rnn(gru_cell, x, dtype=tf.float32)    

    ''''
    # Standard LSTM:

    #lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
    #lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    #lstm_cell = rnn_cell.MultiRNNCell([lstm_cell] * num_layers)
    #outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    '''


    '''
    # Bidirectional LSTM; needs 
    layer['weights'] = tf.Variable(tf.random_normal([2*rnn_size,n_classes]))

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn_cell.BasicLSTMCell(rnn_size, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn_cell.BasicLSTMCell(rnn_size, forget_bias=1.0)

    # Get lstm cell output
    try:
        outputs, states, extras = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], layer['weights']) + layer['biases']
        

    #outputs, states  = tf.nn.bidirectional_dynamic_rnn(
    #    cell_fw=lstm_cell,
    #    cell_bw=lstm_cell,
    #    dtype=tf.float32,
    #    #sequence_length=X_lengths,
    #    inputs=x)
    
    #output_fw, output_bw = outputs
    #states_fw, states_bw = states
    '''
    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output
                         


# In[3]:


def train_neural_network(x):
    keep_prob = tf.placeholder(tf.float32)

    prediction = recurrent_neural_network(x, keep_prob)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    

    eval_x, eval_y, eval_len = evaldata.next_batch(evaldata.num_examples)                 
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        #train_x, train_y = traindata.next_batch(traindata.num_examples)                 
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print("")
        for epoch in range(hm_epochs):
            epoch_loss = 0
            traindata.next_train_round()
            num_rounds = float(traindata.num_examples/batch_size)
            for i in range(int(traindata.num_examples/batch_size)):
                epoch_x, epoch_y, batch_len = traindata.next_batch(batch_size)
                #print (epoch_x.shape)
                #print ((batch_size,n_chunks,chunk_size))
                epoch_x = epoch_x.reshape((batch_size,n_chunks,batch_len))

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y, keep_prob: dropoutval})
                epoch_loss += c
                sys.stderr.write("\rEpoch %i %0.2f%s" % ( epoch, i/num_rounds ,"%") )  
       
            print("")
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

            #print('Train accuracy:',accuracy.eval({x:train_x.reshape((-1, n_chunks, chunk_size)), y:train_y, keep_prob:1.0}))         
            print('Eval accuracy:',accuracy.eval({x:eval_x.reshape((-1, n_chunks, eval_len)), y:eval_y, keep_prob:1.0}))         

        y_p = tf.argmax( prediction, 1)
        val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x:eval_x.reshape((-1, n_chunks, chunk_size)), y:eval_y, keep_prob:1.0})
        #val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x:train_x.reshape((-1, n_chunks, chunk_size)), y:train_y, keep_prob:1.0})
        #y_true = np.argmax( eval_y ,1)
        y_true = np.argmax( train_y ,1)
        print ("confusion_matrix in /tmp/tf.training.confusion_matrix")
        np.savetxt('/tmp/tf.training.confusion_matrix',confusion_matrix(y_true, y_pred), "%i")                 
        #print ("Accuracy: ",accuracy)
       
        #print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))



# In[ ]:

train_neural_network(x)

