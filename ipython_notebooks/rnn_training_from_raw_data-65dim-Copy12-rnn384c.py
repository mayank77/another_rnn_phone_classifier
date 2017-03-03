
# coding: utf-8

# In[1]:

import numpy as np
import pickle, os, random, math, sys

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.contrib.tensorboard.plugins import projector

from sklearn.metrics import confusion_matrix

from dnnutil import phonestash

#import importlib
#importlib.reload(dnnutil)

phone_stash = phonestash.phone_stash

en_corpus = "en_uk_kids_align_from_clean-2"
en_pickle_dir='../features/work_in_progress/'+en_corpus+'/pickles'

en_atr_corpus = 'en_uk_atr_kids_kaldi-align-2'
en_atr_pickle_dir='../features/work_in_progress/'+en_atr_corpus+'/pickles'


fi_corpus = "speecon_kids_kaldi_align"
fi_pickle_dir='../features/work_in_progress/'+fi_corpus+'/pickles'

test_corpus = "fysiak-gamedata-2"
test_pickle_dir='../features/work_in_progress/'+test_corpus+'/pickles'

logs_path = '/tmp/tensorflow_logs/copy12-rnn384-h'
LOG_DIR=logs_path
#
# A function that will be useful:
#

def mkdir(path):
    try:
        os.makedirs(path)        
    except OSError as exc:  # Python >2.5
        #print ("dir %s exists" % path)
        dummy = 1


mkdir(logs_path)

# # 0 Load data #

# In[2]:


if False:
    traindata = phone_stash([ os.path.join(en_pickle_dir, 'train.00.a_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.00.b_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.00.c_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '001.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '002.recipe_mspec66_and_f0_alldata.pickle2')
                          ])


    eval_uk_data = phone_stash([ os.path.join(en_pickle_dir, 'eval.00.a_mspec66_and_f0_alldata.pickle2'),
                                 os.path.join(en_pickle_dir, 'eval.00.b_mspec66_and_f0_alldata.pickle2') 
                             ], zmean=traindata.mean, zstd=traindata.std, max_len=traindata.max_len)

    
    eval_fi_data = phone_stash( [os.path.join(fi_pickle_dir, '047.recipe_mspec66_and_f0_alldata.pickle2'),
                                 os.path.join(fi_pickle_dir, '048.recipe_mspec66_and_f0_alldata.pickle2')
                             ], zmean=traindata.mean, zstd=traindata.std, max_len=traindata.max_len)

    players_bad_data = phone_stash([ os.path.join(test_pickle_dir, 'disqualified-32smoothed_mspec66_and_f0_alldata.pickle2'),
                             ], zmean=traindata.mean, zstd=traindata.std, max_len=traindata.max_len)

    players_ok_data = phone_stash([ os.path.join(test_pickle_dir, 'some_stars-32smoothed_mspec66_and_f0_alldata.pickle2'),
                            ], zmean=traindata.mean, zstd=traindata.std, max_len=traindata.max_len)

    players_good_data =  phone_stash([ os.path.join(test_pickle_dir, 'lots_of_stars-32smoothed_mspec66_and_f0_alldata.pickle2'),
                               ], zmean=traindata.mean, zstd=traindata.std, max_len=traindata.max_len)


    players_native_data =  phone_stash([ os.path.join(test_pickle_dir, 'native_or_nativelike-32smoothed_mspec66_and_f0_alldata.pickle2'),
                                 ], zmean=traindata.mean, zstd=traindata.std, max_len=traindata.max_len)

    #traindata.usedims=np.arange(1,66)
    #eval_uk_data.usedims=np.arange(1,66)
    #eval_fi_data.usedims=np.arange(1,66)    
    #players_bad_data.usedims=np.arange(1,66)
    #players_ok_data.usedims=np.arange(1,66)
    #players_good_data.usedims=np.arange(1,66)
    
    


else:
    traindata = phone_stash([ os.path.join(en_pickle_dir, 'train.00.a_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.00.b_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.00.c_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.00.d_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.00.e_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.01.a_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.01.b_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.01.c_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.01.d_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.01.e_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.01.f_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.02.a_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.02.b_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.02.c_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.02.d_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.02.e_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.03.a_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.03.b_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.03.c_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.03.d_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.03.e_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.04.a_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.04.b_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.04.c_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.04.d_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.05.a_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.05.b_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.05.c_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.05.d_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.06.a_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.06.b_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.06.c_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.06.d_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.07.a_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.07.b_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.07.c_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'train.07.d_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '001.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '002.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '003.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '004.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '005.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '006.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '007.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '008.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '009.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '010.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '011.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '012.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '013.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '014.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '015.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '016.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '017.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '018.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '019.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '020.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '021.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '022.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '023.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '024.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '025.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '026.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '027.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '028.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '029.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '030.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '031.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '032.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '033.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '034.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '035.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '036.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '037.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '038.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '039.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(fi_pickle_dir, '040.recipe_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_atr_pickle_dir, 'Participant10_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_atr_pickle_dir, 'Participant11_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_atr_pickle_dir, 'Participant12_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_atr_pickle_dir, 'Participant13_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_atr_pickle_dir, 'Participant14_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_atr_pickle_dir, 'Participant15_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_atr_pickle_dir, 'Participant17_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_atr_pickle_dir, 'Participant18_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_atr_pickle_dir, 'Participant19_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_atr_pickle_dir, 'Participant20_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_atr_pickle_dir, 'Participant21_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_atr_pickle_dir, 'Participant22_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_atr_pickle_dir, 'Participant25_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_atr_pickle_dir, 'Participant26_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_atr_pickle_dir, 'Participant29_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_atr_pickle_dir, 'Participant30_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_atr_pickle_dir, 'Participant33_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'eval.00.a_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'eval.00.b_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'eval.00.c_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'eval.00.d_mspec66_and_f0_alldata.pickle2'),
                              os.path.join(en_pickle_dir, 'eval.00.e_mspec66_and_f0_alldata.pickle2'),
                          ])



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
                                 
                         ], zmean=traindata.mean, zstd=traindata.std, max_len=traindata.max_len)

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
                         ], zmean=traindata.mean, zstd=traindata.std, max_len=traindata.max_len)

    players_bad_data = phone_stash([ os.path.join(test_pickle_dir, 'disqualified-32smoothed_mspec66_and_f0_alldata.pickle2'),
                             ], zmean=traindata.mean, zstd=traindata.std, max_len=traindata.max_len)

    players_ok_data = phone_stash([ os.path.join(test_pickle_dir, 'some_stars-32smoothed_mspec66_and_f0_alldata.pickle2'),
                            ], zmean=traindata.mean, zstd=traindata.std, max_len=traindata.max_len)

    players_good_data =  phone_stash([ os.path.join(test_pickle_dir, 'lots_of_stars-32smoothed_mspec66_and_f0_alldata.pickle2'),
                               ], zmean=traindata.mean, zstd=traindata.std, max_len=traindata.max_len)


    players_native_data =  phone_stash([ os.path.join(test_pickle_dir, 'native_or_nativelike-32smoothed_mspec66_and_f0_alldata.pickle2'),
                                 ], zmean=traindata.mean, zstd=traindata.std, max_len=traindata.max_len)
    
    

for stash in [traindata, eval_uk_data,  eval_fi_data, players_bad_data, players_ok_data,  players_good_data, players_native_data]:
    stash.usedim = np.arange(1,66)
    stash.featdim = 65

np.savetxt(os.path.join(logs_path, "traindata.std"), traindata.std)
np.savetxt(os.path.join(logs_path, "traindata.mean"), traindata.mean)
np.savetxt(os.path.join(logs_path, "traindata.num_classes"), np.array([traindata.num_classes]))
np.savetxt(os.path.join(logs_path, "traindata.max_len"), np.array([traindata.max_len]))

featdim=65

hm_epochs = 40
n_classes = traindata.num_classes
batch_size = 128
n_chunks = featdim #28
rnn_size = 384
n_hidden = rnn_size

train_len = traindata.max_len #30

print("\n\nMax train seq. length: %i" % train_len)

num_layers = 5

dropoutval=0.6




# In[3]:

# Print class counts:

print ("class train% test%")

for i in range(1, traindata.num_classes+1):
    traincount=np.where(traindata.classes==i)[0].shape[0]
    testukcount=np.where(eval_uk_data.classes==i)[0].shape[0]
    testficount=np.where(eval_fi_data.classes==i)[0].shape[0]
    playersbadcount=np.where(players_bad_data.classes==i)[0].shape[0]
    playersokcount=np.where(players_ok_data.classes==i)[0].shape[0]
    playersgoodcount=np.where(players_ok_data.classes==i)[0].shape[0]
    playersnativecount=np.where(players_ok_data.classes==i)[0].shape[0]

    print ("%i\t%i=>%0.3f%s\t%i=>%0.3f%s\t%i=>%0.3f%s\t%i=>%0.3f%s\t%i=>%0.3f%s\t%i=>%0.3f%s\t%i=>%0.3f%s" % ( i,
                                                           traincount,
                                                           traincount/traindata.classes.shape[0]*100,
                                                           "%",
                                                           testukcount,      
                                                           testukcount/eval_uk_data.classes.shape[0]*100,
                                                           "%",
                                                           testficount,      
                                                           testficount/eval_fi_data.classes.shape[0]*100,
                                                           "%",
                                                           playersbadcount,      
                                                           playersbadcount/players_bad_data.classes.shape[0]*100,
                                                           "%",
                                                           playersokcount,      
                                                           playersokcount/players_ok_data.classes.shape[0]*100,
                                                           "%",
                                                           playersgoodcount,      
                                                           playersgoodcount/players_good_data.classes.shape[0]*100,
                                                           "%",
                                                           playersnativecount,      
                                                           playersnativecount/players_native_data.classes.shape[0]*100,
                                                           "%"
                                                       ))


# In[4]:

np.median(np.bincount(traindata.classes))


# # 1 RNN training #
# 
# 

# In[5]:

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
    #layer['weights'] = tf.Variable(tf.random_normal([2*rnn_size,n_classes]))

    #layer = {'weights': weights['weight1'],
    #         'biases': weights['bias1'] }
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

#weights = {
#    'weight1': tf.Variable(tf.random_normal([2*rnn_size ,n_classes])),
#    'bias1' : tf.Variable(tf.random_normal([n_classes]))
#}


# # Visualisation on Tensorboard#
# 
# Run 
# 
# **`tensorboard --logdir=/tmp/tensorflow_logs/copy3-rnn512`**
# 
# and navigate to http://127.0.1.1:6006.
# 
# This bit is mostly nicked from
# https://github.com/oduerr/dl_tutorial/blob/master/tensorflow/debugging/embedding.ipynb

# In[6]:


y_true=[]
y_pred=[]
lstm_variables = []


keep_prob = tf.placeholder(tf.float32)

# Construct model and encapsulating all ops into scopes, making
# Tensorboard's Graph visualization more convenient

#with tf.name_scope('Model'):
    # Model
prediction = recurrent_neural_network(x, keep_prob)


#with tf.name_scope('Loss'):
    # Minimize error using cross entropy        
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )

#with tf.name_scope('Optimiser'):
    # Gradient Descent
optimizer = tf.train.AdamOptimizer().minimize(cost)

#with tf.name_scope('Accuracy'):
    # Accuracy
acc = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
acc = tf.reduce_mean(tf.cast(acc, tf.float32))



[ eval_uk_x, 
  eval_uk_y, 
  eval_uk_items, 
  eval_uk_len, 
  eval_uk_x_len ] = eval_uk_data.get_eval_batch( 50 )

[ eval_fi_x, 
  eval_fi_y, 
  eval_fi_items, 
  eval_fi_len, 
  eval_fi_x_len ] = eval_fi_data.get_eval_batch( 50 )


[ train_sample_x, 
  train_sample_y, 
  num_items, 
  train_sample_len, 
  train_sample_x_len ] = traindata.get_eval_batch( 50 ) 


[ players_bad_x, 
  players_bad_y, 
  num_items, 
  players_bad_len, 
  players_bad_x_len ] = players_bad_data.next_batch( players_bad_data.num_examples  ) 


[ players_ok_x, 
  players_ok_y, 
  num_items, 
  players_ok_len, 
  players_ok_x_len ] = players_ok_data.next_batch( players_ok_data.num_examples  ) 


[ players_good_x, 
  players_good_y, 
  num_items, 
  players_good_len, 
  players_good_x_len ] = players_good_data.next_batch( players_good_data.num_examples  ) 


[ players_native_x, 
  players_native_y, 
  num_items, 
  players_native_len, 
  players_native_x_len ] = players_native_data.next_batch( players_native_data.num_examples  ) 


# Initializing the variables
init = tf.global_variables_initializer() #initialize_all_variables()


# Summaries for tensor board from 
# https://github.com/aymericdamien/TensorFlow-Examples/\
#                       blob/master/examples/4_Utils/tensorboard_basic.py
# Create a summary to monitor cost tensor
tf.summary.scalar("loss", cost)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", acc)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Retrieve just the LSTM variables.
model_saver = tf.train.Saver(max_to_keep=None)

with tf.Session() as sess:
    #sess.run(tf.initialize_all_variables())
    sess.run(init)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path+ '/train', graph=tf.get_default_graph())


    train_sample_writer = tf.summary.FileWriter(logs_path+ '/train-sample')
    eval_uk_writer      = tf.summary.FileWriter(logs_path+ '/eval_uk')
    eval_fi_writer      = tf.summary.FileWriter(logs_path+ '/eval_fi')
    players_bad_writer  = tf.summary.FileWriter(logs_path+ '/players_bad')
    players_ok_writer   = tf.summary.FileWriter(logs_path+ '/players_ok')
    players_good_writer = tf.summary.FileWriter(logs_path+ '/players_good')
    players_native_writer = tf.summary.FileWriter(logs_path+ '/players_native')

    test_embedding_writer = tf.summary.FileWriter(logs_path+ '/test-embed')        


    train_sample_embedding_writer = tf.summary.FileWriter(logs_path+ '/train-embed')
    eval_uk_embedding_writer = tf.summary.FileWriter(logs_path+ '/eval_uk')
    eval_fi_embedding_writer = tf.summary.FileWriter(logs_path+ '/eval_fi')
    players_bad_embedding_writer = tf.summary.FileWriter(logs_path+ '/players_bad')
    players_ok_embedding_writer = tf.summary.FileWriter(logs_path+ '/players_ok')
    players_good_embedding_writer = tf.summary.FileWriter(logs_path+ '/players_good')
    players_native_embedding_writer = tf.summary.FileWriter(logs_path+ '/players_native')

    #train_embedding_writer = tf.summary.FileWriter(logs_path+ '/train-embed')
    #train_embedding_saver = tf.train.Saver()



    #train_x, train_y = traindata.next_batch(traindata.num_examples)                 
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    traindata.next_balanced_round()
    #total_batch = math.ceil(traindata.balanced_data_size/batch_size)
    total_batch = math.ceil((traindata.balanced_data_order.shape[0])/batch_size)
    print("")
    for epoch in range(hm_epochs):

        epoch_loss = 0
        for i in range(total_batch):
            if (i % 1000 == 0):

                save_path = model_saver.save(sess, os.path.join(logs_path, 'model2.ckpt'), (epoch * total_batch + i))
                sys.stderr.write("\nSaved model parameters to %s\n" % save_path) 

                #meta_graph_def = tf.train.export_meta_graph(filename=os.path.join(LOG_DIR,'my-model.meta'))
                #sys.stderr.write("\nSaved meta graph to %s\n" % os.path.join(LOG_DIR,'my-model.meta')) 




                for tests in [ [ "train_sample", train_sample_x, train_sample_y, train_sample_writer, train_sample_embedding_writer],
                               [ "eval_uk", eval_uk_x, eval_uk_y , eval_uk_writer, eval_uk_embedding_writer],
                               [ "eval_fi", eval_fi_x, eval_fi_y, eval_fi_writer, eval_fi_embedding_writer ],
                               [ "players_bad", players_bad_x, players_bad_y, players_bad_writer, players_bad_embedding_writer ],
                               [ "players_ok", players_ok_x, players_ok_y, players_ok_writer, players_ok_embedding_writer ],
                               [ "players_good", players_good_x, players_good_y, players_good_writer, players_good_embedding_writer ],
                               [ "players_native", players_native_x, players_native_y, players_native_writer, players_native_embedding_writer ] ]:

                    testname = tests[0]
                    test_x = tests[1]
                    test_y = tests[2]

                    test_writer = tests[3]
                    embedding_writer = tests[4]
                    # Sample of test data:              
                    summary, acc, EMB = sess.run([ merged_summary_op, 
                                                   accuracy, 
                                                   prediction ], 
                                                 feed_dict={ x: test_x, 
                                                             y: test_y, 
                                                             keep_prob: 1 })

                    test_writer.add_summary(summary,  epoch * total_batch + i )

                    # Write [EMB, np.argmax(eval_y) ]to file!
                    np.savetxt(os.path.join(LOG_DIR, '%s_y_and_prediction.%i' % (testname, epoch * total_batch + i)),
                               np.vstack((np.argmax(test_y, 1), 
                                          np.argmax(EMB,1))).T.astype(int), 
                               fmt='%i',)



                    np.savetxt(os.path.join(LOG_DIR, '%s_y_and_posterior.%i' % (testname, epoch * total_batch + i)),
                               np.hstack( 
                                   (np.argmax(test_y, 1).reshape([-1,1]), 
                                    EMB )
                               ), fmt='%0.2f',)

                    sys.stderr.write("Acc: %0.2f Test: %s\t \n" % (acc, testname) )




            [epoch_x, 
             epoch_y, 
             num_items, 
             batch_len, 
             epoch_x_len] = traindata.next_balanced_batch(batch_size)

            if batch_len == 0:
                continue
            _, c, summary = sess.run([optimizer, cost,  merged_summary_op], 
                                         feed_dict={x: epoch_x, 
                                                    y: epoch_y, 
                                                    keep_prob: dropoutval})
            summary_writer.add_summary(summary, epoch * total_batch + i)
            epoch_loss += c
            sys.stderr.write("\rEpoch %i %0.2f%s (global batch %i, epoch batch count: %i) " % 
                             ( epoch, (i+1)/total_batch*100 ,
                               "%", 
                               epoch * total_batch + i, 
                               total_batch 
                               #traindata.balanced_data_order.shape[0]/batch_size
                           ))


        traindata.next_balanced_round()    
        print("")
        print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)






with tf.Session() as sess2:
    #sess.run(init)


    prediction = recurrent_neural_network(x, keep_prob)



    LOG_DIR = logs_path
    EVAL_DIR = './dnnscores_copy12-rnn384/'

    modelfile=os.path.join(LOG_DIR, 'model2.ckpt-0') #model2.ckpt-31069')
    my_meta = modelfile+".meta"


    print("Init saver with meta graph")
    restorer = tf.train.Saver(tf.global_variables())
    restorer.restore(sess, modelfile)
    print("Done!")


    for tests in [ [ "players_bad", players_bad_x, players_bad_y ],
                   [ "players_ok", players_ok_x, players_ok_y ],
                   [ "players_good", players_good_x, players_good_y],
                   [ "players_native", players_native_x, players_native_y ] ]:

        testname = tests[0]
        test_x = tests[1]
        test_y = tests[2]

        print("Let's get summary for %s" % testname)


        # Sample of test data:              
        EMB = sess2.run([ accuracy, 
                                       prediction ], 
                                     feed_dict={ x: test_x, 
                                                 y: test_y, 
                                                 keep_prob: 1 })

        print (acc)
        print (np.where(  (np.argmax(test_y, 1) - np.argmax(EMB,1))==0)[0].shape[0]/test_x.shape[0])

        '''
        # Write [EMB, np.argmax(eval_y) ]to file!
        np.savetxt(os.path.join(EVAL_DIR, '%s_y_and_prediction.%i' % (testname, epoch * total_batch + i)),
                   np.vstack((np.argmax(test_y, 1), 
                              np.argmax(EMB,1))).T.astype(int), 
                   fmt='%i',)

        #print(np.argmax(test_y, 1).reshape([-1,1]).shape)
        #print(EMB.shape)

        np.savetxt(os.path.join(EVAL_DIR, '%s_y_and_posterior.%i' % (testname, epoch * total_batch + i)),
                   np.hstack( 
                       (np.argmax(test_y, 1).reshape([-1,1]), 
                        EMB )
                   ), fmt='%0.2f',)

        '''












# In[ ]:

print(y_true[1:10])
print(y_pred[1:10])

