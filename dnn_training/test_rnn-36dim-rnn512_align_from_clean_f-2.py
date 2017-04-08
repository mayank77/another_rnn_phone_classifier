
# coding: utf-8

# In[1]:

import numpy as np
import pickle, os, random, math, sys

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.contrib.tensorboard.plugins import projector

from sklearn.metrics import confusion_matrix

from dnnutil import phonestashwithoutnoise as phonestash

#import importlib
#importlib.reload(dnnutil)

phone_stash = phonestash.phone_stash

en_corpus = "en_uk_kids_align_from_clean-2"
en_pickle_dir='../features/work_in_progress/'+en_corpus+'/pickles'

en_atr_corpus = 'en_uk_atr_kids_kaldi-align-2'
en_atr_pickle_dir='../features/work_in_progress/'+en_atr_corpus+'/pickles'


fi_corpus = "speecon_kids_kaldi_align"
fi_pickle_dir='../features/work_in_progress/'+fi_corpus+'/pickles'

test_corpus = "more_fysiak-gamedata-2-aligned_with_clean_f"
test_pickle_dir='../features/work_in_progress/'+test_corpus+'/pickles'

logs_path = '../models/rnn512-e'
LOG_DIR = logs_path

checkpoint=48055

EVAL_DIR = '../models/rnn512-e/testscores-%i/' % checkpoint

#
# A function that will be useful:
#

def mkdir(path):
    try:
        os.makedirs(path)        
    except OSError as exc:  # Python >2.5
        #print ("dir %s exists" % path)
        dummy = 1


mkdir(EVAL_DIR)

modelfile=os.path.join(LOG_DIR, 'model2.ckpt-%i' % checkpoint) #model2.ckpt-31069')
my_meta = modelfile+".meta"

# # 0 Load data #

zmean = np.loadtxt( os.path.join(logs_path, "traindata.mean" )) #traindata.mean
zstd = np.loadtxt( os.path.join(logs_path, "traindata.std" ))# traindata.std
zmax_len = 62#traindata.max_len
znum_classes = 119 #traindata.num_classes

traindata = None

players_bad_data = phone_stash([ os.path.join(test_pickle_dir, 'disqualified_melbin36_and_f0_alldata.pickle2'),
                             ], zmean=zmean, zstd=zstd, max_len=zmax_len)

players_ok_data = phone_stash([ os.path.join(test_pickle_dir, 'some_stars_melbin36_and_f0_alldata.pickle2'),
                            ], zmean=zmean, zstd=zstd, max_len=zmax_len)

players_good_data =  phone_stash([ os.path.join(test_pickle_dir, 'lots_of_stars_melbin36_and_f0_alldata.pickle2'),
                               ], zmean=zmean, zstd=zstd, max_len=zmax_len)


players_native_data =  phone_stash([ os.path.join(test_pickle_dir, 'native_or_nativelike_melbin36_and_f0_alldata.pickle2'),
                               ], zmean=zmean, zstd=zstd, max_len=zmax_len)

'''

eval_uk_data = phone_stash([ os.path.join(en_pickle_dir, 'test.00.a_mspec66_and_f0_alldata.pickle2'),
                                os.path.join(en_pickle_dir, 'test.00.b_mspec66_and_f0_alldata.pickle2'),
                                 os.path.join(en_pickle_dir, 'test.00.c_mspec66_and_f0_alldata.pickle2'),
                                 os.path.join(en_pickle_dir, 'test.00.d_mspec66_and_f0_alldata.pickle2'),
                                 os.path.join(en_pickle_dir, 'test.00.e_mspec66_and_f0_alldata.pickle2'),
                                 os.path.join(en_pickle_dir, 'test.01.a_mspec66_and_f0_alldata.pickle2'),
                                 os.path.join(en_pickle_dir, 'test.01.b_mspec66_and_f0_alldata.pickle2'),
                                 os.path.join(en_pickle_dir, 'test.01.c_mspec66_and_f0_alldata.pickle2'),
                                 os.path.join(en_pickle_dir, 'test.01.d_mspec66_and_f0_alldata.pickle2'),
                                 os.path.join(en_atr_pickle_dir, 'Participant5_mspec66_and_f0_alldata.pickle2'),
                                 os.path.join(en_atr_pickle_dir, 'Participant9_mspec66_and_f0_alldata.pickle2'), 
                                 os.path.join(en_atr_pickle_dir, 'Participant34_mspec66_and_f0_alldata.pickle2'),
                                 os.path.join(en_atr_pickle_dir, 'Participant37_mspec66_and_f0_alldata.pickle2'),
                                 os.path.join(en_atr_pickle_dir, 'Participant41_mspec66_and_f0_alldata.pickle2'),                          
                                 
                         ],  zmean=zmean, zstd=zstd, max_len=zmax_len)

eval_fi_data = phone_stash([  os.path.join(fi_pickle_dir, '041.recipe_mspec66_and_f0_alldata.pickle2'),
                             os.path.join(fi_pickle_dir, '042.recipe_mspec66_and_f0_alldata.pickle2'),
                             os.path.join(fi_pickle_dir, '043.recipe_mspec66_and_f0_alldata.pickle2'),
                             os.path.join(fi_pickle_dir, '044.recipe_mspec66_and_f0_alldata.pickle2'),
                             os.path.join(fi_pickle_dir, '045.recipe_mspec66_and_f0_alldata.pickle2'),
                             os.path.join(fi_pickle_dir, '046.recipe_mspec66_and_f0_alldata.pickle2'),
                             os.path.join(fi_pickle_dir, '047.recipe_mspec66_and_f0_alldata.pickle2'),
                             os.path.join(fi_pickle_dir, '048.recipe_mspec66_and_f0_alldata.pickle2'),
                             os.path.join(fi_pickle_dir, '049.recipe_mspec66_and_f0_alldata.pickle2'),
                             os.path.join(fi_pickle_dir, '050.recipe_mspec66_and_f0_alldata.pickle2')
                         ],  zmean=zmean, zstd=zstd, max_len=zmax_len)

'''


for stash in [players_bad_data, players_ok_data,  players_good_data, players_native_data]: #, eval_uk_data, eval_fi_data]:
    stash.usedim = np.arange(1,37)
    stash.featdim = 36



featdim=36

hm_epochs = 40
n_classes = znum_classes
batch_size = 128
n_chunks = featdim #28
rnn_size = 384 #512
n_hidden = rnn_size

train_len = zmax_len #30

print("\n\nMax train seq. length: %i" % train_len)

num_layers = 5

dropoutval=0.6


'''
[ eval_uk_x, 
  eval_uk_y, 
  eval_uk_items, 
  eval_uk_len, 
  eval_uk_x_len,
  eval_uk_keys ] = eval_uk_data.get_eval_batch( 50 )

[ eval_fi_x, 
  eval_fi_y, 
  eval_fi_items, 
  eval_fi_len, 
  eval_fi_x_len,
  eval_fi_keys ] = eval_fi_data.get_eval_batch( 50 )
'''

[ players_bad_x, 
  players_bad_y, 
  num_items, 
  players_bad_len, 
  players_bad_x_len,
  players_bad_keys ] = players_bad_data.next_batch( players_bad_data.num_examples  ) 


[ players_ok_x, 
  players_ok_y, 
  num_items, 
  players_ok_len, 
  players_ok_x_len,
  players_ok_keys] = players_ok_data.next_batch( players_ok_data.num_examples  ) 


[ players_good_x, 
  players_good_y, 
  num_items, 
  players_good_len, 
  players_good_x_len,
  players_good_keys ] = players_good_data.next_batch( players_good_data.num_examples  ) 


[ players_native_x, 
  players_native_y, 
  num_items, 
  players_native_len, 
  players_native_x_len,
  players_native_keys ] = players_native_data.next_batch( players_native_data.num_examples  ) 

players_bad_writer = False
players_bad_embedding_writer =False
players_ok_writer = False
players_ok_embedding_writer = False
players_good_writer = False
players_good_embedding_writer = False
players_native_writer = False
players_native_embedding_writer = False


keep_prob = tf.placeholder(tf.float32)

x = tf.placeholder('float', [None, None,featdim])
y = tf.placeholder('float')
early_stop = tf.placeholder(tf.int32, [None])

'''
From https://danijar.com/variable-sequence-lengths-in-tensorflow/

"I will assume that the sequences are padded with zero vectors to fill 
up the remaining time steps in the batch. To pass sequence lengths to 
TensorFlow, we have to compute them from the batch. While we could do 
this in Numpy in a pre-processing step, let’s do it on the fly as part 
of the compute graph!"
'''
def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant






def recurrent_neural_network(x, keep_prob):
    # Bidirectional LSTM; needs 
    layer = {'weights':  tf.Variable(tf.random_normal([2*rnn_size ,n_classes])),
             'biases':  tf.Variable(tf.random_normal([n_classes])) }

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True, forget_bias=1.0)
    lstm_fw_cell = rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=keep_prob)
    lstm_fw_cell = rnn_cell.MultiRNNCell([lstm_fw_cell] * num_layers)
    
    # Backward direction cell
    lstm_bw_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True,  forget_bias=1.0)
    lstm_bw_cell = rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=keep_prob)
    lstm_bw_cell = rnn_cell.MultiRNNCell([lstm_bw_cell] * num_layers)
    
    
    # Get lstm cell output
    #try:
    outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, 
                                                      lstm_bw_cell, 
                                                      x,
                                                      dtype=tf.float32,
                                                      sequence_length=length(x))
                                                      #sequence_length=early_stop)
    #except Exception: # Old TensorFlow version only returns outputs not states
    #    outputs = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x,
    #                                    dtype=tf.float32, sequence_length=early_stop)

    output_fw, output_bw = outputs
    
    last = last_relevant(output_fw, length(x))
    first = last_relevant(output_fw, length(x))

    return tf.matmul(tf.concat(1,[first,last]) , layer['weights']) + layer['biases']




with tf.Session() as sess:
    #sess.run(init)


    prediction = recurrent_neural_network(x, keep_prob)

    print("Init saver with meta graph")
    restorer = tf.train.Saver(tf.global_variables())
    restorer.restore(sess, modelfile)
    print("Done!")


    for tests in [ #[ "eval_uk", eval_uk_x, eval_uk_y, eval_uk_keys ],
                   #[ "eval_fi", eval_fi_x, eval_fi_y, eval_fi_keys ],
                   [ "players_bad", players_bad_x, players_bad_y, players_bad_keys ],
                   [ "players_ok", players_ok_x, players_ok_y, players_ok_keys ],
                   [ "players_good", players_good_x, players_good_y, players_good_keys ],
                   [ "players_native", players_native_x, players_native_y, players_native_keys ] ]:

        testname = tests[0]
        test_x = tests[1]
        test_y = tests[2]
        testkeys = tests[3]

        print("Let's get summary for %s" % testname)


        # Sample of test data:              
        EMB = sess.run([ prediction ], 
                            feed_dict={ x: test_x, 
                                        y: test_y, 
                                        keep_prob: 1 })


        print (np.where(  (np.argmax(test_y, 1) - np.argmax(EMB,2).reshape([-1]))==0)[0].shape[0]/test_x.shape[0])

        #print (testkeys.shape)
        #print  (np.argmax(test_y, 1).shape)
        #print (np.argmax(EMB,2).shape)

        
        key_and_class_stack = np.vstack( (testkeys,
                                          np.argmax(test_y, 1)))

        # Write [EMB, np.argmax(eval_y) ]to file!
        np.savetxt(os.path.join(EVAL_DIR, '%s_id_y_and_prediction' % (testname)),
                   np.vstack( (key_and_class_stack, 
                               np.argmax(EMB,2))).T.astype(int), 
                   fmt='%i',)












