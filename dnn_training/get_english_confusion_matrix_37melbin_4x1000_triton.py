#!/usr/bin/python3
# coding: utf-8

import numpy as np
import pickle, os, random, math, sys, datetime

#
#   Tensorflow probably needs to be version rc 0.12
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.contrib.tensorboard.plugins import projector

#
# Some help from scikit 
#from sklearn.metrics import confusion_matrix
def confusion_matrix(p1,p2):
    cf = np.zeros([int(np.max(p1)+1),  int(np.max(p2)+1)])
    for i in np.arange(0,p1.shape[0], dtype='int'):
        cf[  int(p1[i]), int(p2[i]) ] += 1
    return cf

# Some data handling I wrote:
from dnnutil import phonestashwithoutnoise as phonestash
phone_stash = phonestash.phone_stash

langtype="fi_only"


# Which model checkpoint to use?
checkpoint=6000 # 29478

#
#  Define training parameters:
#


rnn_size = 1000   # The LSTM width
num_layers = 4    # The LSTM depth

# For the Adam optimiser:
learning_rate=0.0005 # Was 0.001
beta1=0.9
beta2=0.999
epsilon=0.001, # was 1e-8


# Specs on the data:
featdim=36
class_map = "None"

# Batching stuff:
hm_epochs = 10
batch_size = 128
n_chunks = featdim #28

# Dropout keep value:
dropoutval=0.6

# Compensate data unbalance by repeating data from less frequent classes in training
# 0.0 = No compensation at all
# 1.0 = Full compensation, identical count of data from all classes in one epoch  
class_balancing = 0.5

n_hidden = rnn_size

# Training and devel data definitions:

en_corpus = "/en_uk_all_kids_aligned_with_clean-f/" #en_uk_kids_align_from_clean-2"
en_pickle_dir='../features/melbin36_and_f0/'+en_corpus+'/pickles'

fi_corpus = "speecon_kids_kaldi_align"
fi_pickle_dir='../features/melbin36_and_f0/'+fi_corpus+'/pickles'

test_corpus = "more_fysiak-gamedata-2-aligned_with_mc_b"
test_pickle_dir='../features/melbin36_and_f0/'+test_corpus+'/pickles'



train_files = [
os.path.join(en_pickle_dir, 'test_017f10nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_022m11bh_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_023m11nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_026f10nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_032m11nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_040f07nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_040f09nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_041f07nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_041f08nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_041f09nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_041m06nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_042f07nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_042f08nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_042m06nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_043f07nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_043f09nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_043m06bh_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_043m07nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_044m06nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_044m07nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_045f07nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_045m06nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_045m08nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_054m06nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_082f07nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_083m07nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_120f06nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_120m06nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_123f07nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_162f09sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_166f09sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_168f09sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_174m09sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_177f09sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_178m09sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_180f09sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_187f08sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_188f08sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_189f08sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_195m08sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'test_197m08sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_001f12bh_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_001m14bh_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_002f12bh_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_002m14bh_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_003m13bh_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_004f05bh_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_004m04bh_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_004m13bh_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_005m10bh_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_005m10nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_005m14bh_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_006m10nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_007f13bh_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_007m10nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_016f10nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_019f10nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_020m10nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_020m12bh_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_021f11nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_024m10nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_025m10nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_027m10nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_028m10nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_029m10nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_030m10nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_031f10nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_033m10nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_034m11nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_035m10nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_036f10nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_037f11nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_038m11nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_039f11nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_042f06nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_042m10nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_081m06nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_082f06nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_083f07nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_084m07nl_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_161m10sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_163f10sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_164m10sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_165m10sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_167m10sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_169f10sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_170m10sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_171m10sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_172f10sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_173m10sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_175m10sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_176m10sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_179m09sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_181f10sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_182f09sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_183f10sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_185f10sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_186m10sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_190f08sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_191f07sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_192f08sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_193f08sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_194f08sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_196f08sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_198f08sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_199m08sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_200f08sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_201m08sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_202f07sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_203f08sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_204m08sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_205m08sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_206f08sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_207f08sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_208f08sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_210f08sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_211m08sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_212m08sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_213m08sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_216m10sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'train_248f08sp_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant10-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant11-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant12-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant13-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant14a-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant14b-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant14c-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant15-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant16-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant17-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant18-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant19-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant1-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant20-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant21-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant22-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant23-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant24-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant25-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant26-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant27-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant28-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant29-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant2-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant30-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant31-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant32-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant33a-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant33b-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant34-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant35-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant36-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant37-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant38-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant39-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant3-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant40-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant41-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant42-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant43-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant44-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant4-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant5-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant6-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant7-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant8-_melbin36_and_f0_alldata.pickle2'),
                               os.path.join(en_pickle_dir, 'wavfiles_Participant9-_melbin36_and_f0_alldata.pickle2')
    # os.path.join(fi_pickle_dir, '001.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '002.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '003.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '004.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '005.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '006.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '007.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '008.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '009.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '010.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '011.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '012.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '013.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '014.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '015.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '016.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '017.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '018.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '019.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '020.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '021.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '022.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '023.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '024.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '025.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '026.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '027.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '028.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '029.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '030.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '031.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '032.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '033.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '034.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '035.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '036.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '037.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '038.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '039.recipe_melbin36_and_f0_alldata.pickle2'),
    # os.path.join(fi_pickle_dir, '040.recipe_melbin36_and_f0_alldata.pickle2'),
]

'''
eval_uk_files = [ os.path.join(en_pickle_dir, 'test_001m10nl_melbin36_and_f0_alldata.pickle2'),
                                 os.path.join(en_pickle_dir, 'test_002m10nl_melbin36_and_f0_alldata.pickle2'),
                                 os.path.join(en_pickle_dir, 'test_003m10bh_melbin36_and_f0_alldata.pickle2'),
                                 os.path.join(en_pickle_dir, 'test_003m10nl_melbin36_and_f0_alldata.pickle2'),
                                 os.path.join(en_pickle_dir, 'test_004f06bh_melbin36_and_f0_alldata.pickle2'),
                                 os.path.join(en_pickle_dir, 'test_004f11nl_melbin36_and_f0_alldata.pickle2'),
                                 os.path.join(en_pickle_dir, 'test_004m08bh_melbin36_and_f0_alldata.pickle2'),
                                 os.path.join(en_pickle_dir, 'test_004m10bh_melbin36_and_f0_alldata.pickle2'),
                                 os.path.join(en_pickle_dir, 'test_005f06bh_melbin36_and_f0_alldata.pickle2'),
                                 os.path.join(en_pickle_dir, 'test_005m08bh_melbin36_and_f0_alldata.pickle2'),
                                 os.path.join(en_pickle_dir, 'test_006f10bh_melbin36_and_f0_alldata.pickle2'),
                                 os.path.join(en_pickle_dir, 'test_008m11nl_melbin36_and_f0_alldata.pickle2'),
                                 os.path.join(en_pickle_dir, 'test_009f11nl_melbin36_and_f0_alldata.pickle2'),
                                 os.path.join(en_pickle_dir, 'test_010m11nl_melbin36_and_f0_alldata.pickle2'),
                                 os.path.join(en_pickle_dir, 'test_011f10nl_melbin36_and_f0_alldata.pickle2'),
                                 os.path.join(en_pickle_dir, 'test_012f11nl_melbin36_and_f0_alldata.pickle2'),
                                 os.path.join(en_pickle_dir, 'test_013f11nl_melbin36_and_f0_alldata.pickle2'),
                                 os.path.join(en_pickle_dir, 'test_014f11nl_melbin36_and_f0_alldata.pickle2'),
                                 os.path.join(en_pickle_dir, 'test_015f10nl_melbin36_and_f0_alldata.pickle2')                                                          
                             ]

eval_fi_files = [  os.path.join(fi_pickle_dir, '041.recipe_melbin36_and_f0_alldata.pickle2'),
                             os.path.join(fi_pickle_dir, '042.recipe_melbin36_and_f0_alldata.pickle2'),
                             os.path.join(fi_pickle_dir, '043.recipe_melbin36_and_f0_alldata.pickle2'),
                             os.path.join(fi_pickle_dir, '044.recipe_melbin36_and_f0_alldata.pickle2'),
                             os.path.join(fi_pickle_dir, '045.recipe_melbin36_and_f0_alldata.pickle2'),
                             os.path.join(fi_pickle_dir, '046.recipe_melbin36_and_f0_alldata.pickle2'),
                             os.path.join(fi_pickle_dir, '047.recipe_melbin36_and_f0_alldata.pickle2'),
                             os.path.join(fi_pickle_dir, '048.recipe_melbin36_and_f0_alldata.pickle2'),
                             os.path.join(fi_pickle_dir, '049.recipe_melbin36_and_f0_alldata.pickle2'),
                             os.path.join(fi_pickle_dir, '050.recipe_melbin36_and_f0_alldata.pickle2')
                         ]

players_bad_files = [ os.path.join(test_pickle_dir, 'disqualified-mc_b_melbin36_and_f0_alldata.pickle2'),
                             ]

players_ok_files = [ os.path.join(test_pickle_dir, 'some_stars-mc_b_melbin36_and_f0_alldata.pickle2'),
                            ]

players_good_files = [ os.path.join(test_pickle_dir, 'lots_of_stars-mc_b_melbin36_and_f0_alldata.pickle2'),
                               ]

players_native_files = [ os.path.join(test_pickle_dir, 'native_or_nativelike-mc_b_melbin36_and_f0_alldata.pickle2'),
                                 ]
'''



#Specify where to keep log file:
logs_path = '../models/%s-rnn%ix%i-learningrate%0.5f-dropout%0.1f-classbalance%0.1f-triton-a' % (langtype, num_layers, rnn_size, learning_rate, dropoutval, class_balancing)
LOG_DIR=logs_path
modelfile=os.path.join(LOG_DIR, 'model2.ckpt-%i' % checkpoint) #model2.ckpt-31069')


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

logfile = open(os.path.join(logs_path, "confusion_log"), 'w')

def print_to_log(text):
    print (text)
    #if len(text) > 0:
        #if text[0] != '\r':
    logfile.write(text+"\n")
    logfile.flush()


# # Record parameters:

print_to_log("Training parameters:")


print_to_log("%s:\t%i" % ("rnn_width", rnn_size))
print_to_log("%s:\t%i" % ("rnn_depth", num_layers))
print_to_log("%s:\t%s" % ("Optimiser", "Adam"))
print_to_log("%s:\t%0.6f" % ("learning rate", learning_rate))
print_to_log("%s:\t%0.3f" % ("beta1", beta1))
print_to_log("%s:\t%0.3f" % ("beta2", beta2))
print_to_log("%s:\t%s" % ("epsilon", str(epsilon)))
print_to_log("%s:\t%i" % ("training epochs", hm_epochs))
print_to_log("%s:\t%i" % ("minibatch size", batch_size))
print_to_log("%s:\t%0.2f" % ("dropout keep value", dropoutval))


print_to_log("%s:\t%s" % ("\nclass mapping", class_map))
print_to_log("%s:\t%0.2f" % ("\nclass balancing factor", class_balancing))
   
# # 0 Load data #

zmean = np.loadtxt( os.path.join(logs_path, "traindata.mean" )) #traindata.mean
zstd = np.loadtxt( os.path.join(logs_path, "traindata.std" ))# traindata.std
zmax_len = 62#traindata.max_len

traindata = phone_stash(train_files, class_map=class_map, class_balancing=class_balancing, zmean=zmean, zstd=zstd, max_len=zmax_len)
print_to_log("\nTraining data:")
print_to_log('\n'.join(train_files))

'''
eval_uk_data = phone_stash(eval_uk_files, zmean=traindata.mean, zstd=traindata.std, max_len=traindata.max_len, class_map=class_map)
print_to_log("\nEval uk data:")
print_to_log('\n'.join(eval_uk_files))

eval_fi_data = phone_stash(eval_fi_files, zmean=traindata.mean, zstd=traindata.std, max_len=traindata.max_len, class_map=class_map)
print_to_log("\nEval fi data:")
print_to_log('\n'.join(eval_fi_files))

players_bad_data = phone_stash(players_bad_files, zmean=traindata.mean, zstd=traindata.std, max_len=traindata.max_len, class_map=class_map)
print_to_log("\nPlayers bad data:")
print_to_log('\n'.join(players_bad_files))

players_ok_data = phone_stash(players_ok_files, zmean=traindata.mean, zstd=traindata.std, max_len=traindata.max_len, class_map=class_map)
print_to_log("\nPlayers ok data:")
print_to_log('\n'.join(players_ok_files))

players_good_data =  phone_stash(players_good_files, zmean=traindata.mean, zstd=traindata.std, max_len=traindata.max_len, class_map=class_map)
print_to_log("\nPlayers good data:")
print_to_log('\n'.join(players_good_files))

players_native_data =  phone_stash(players_native_files, zmean=traindata.mean, zstd=traindata.std, max_len=traindata.max_len, class_map=class_map)
print_to_log("\nPlayers native data:")
print_to_log('\n'.join(players_native_files))
'''    
    

for stash in [traindata]: #, eval_uk_data,  eval_fi_data, players_bad_data, players_ok_data,  players_good_data, players_native_data]:
    stash.usedim = np.arange(1,37)
    stash.featdim = 36


n_classes = traindata.num_classes
train_len = traindata.max_len #30


print_to_log("%s:\t%i" % ("number of classes", n_classes))



#np.savetxt(os.path.join(logs_path, "traindata.std"), traindata.std)
#np.savetxt(os.path.join(logs_path, "traindata.mean"), traindata.mean)
#np.savetxt(os.path.join(logs_path, "traindata.num_classes"), np.array([traindata.num_classes]))
#np.savetxt(os.path.join(logs_path, "traindata.max_len"), np.array([traindata.max_len]))


print_to_log("\n\nMax train seq. length: %i" % train_len)





# In[3]:

# Print class counts:

print ("class train% test%")
'''
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

'''

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
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#with tf.name_scope('Accuracy'):
    # Accuracy
acc = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
acc = tf.reduce_mean(tf.cast(acc, tf.float32))


[ train_sample_x, 
  train_sample_y, 
  num_items, 
  train_sample_len, 
  train_sample_x_len, 
  train_sample_keys ] = traindata.next_batch( traindata.num_examples ) 


'''
[ eval_uk_x, 
  eval_uk_y, 
  eval_uk_items, 
  eval_uk_len, 
  eval_uk_x_len, 
  eval_uk_keys ] = eval_uk_data.next_batch( eval_uk_data.num_examples  )

[ eval_fi_x, 
  eval_fi_y, 
  eval_fi_items, 
  eval_fi_len, 
  eval_fi_x_len, 
  eval_fi_keys ] = eval_fi_data.next_batch( eval_fi_data.num_examples  )


[ train_sample_x, 
  train_sample_y, 
  num_items, 
  train_sample_len, 
  train_sample_x_len, 
  train_sample_keys ] = traindata.get_eval_batch( 50 ) 


[ players_bad_x, 
  players_bad_y, 
  num_items, 
  players_bad_len, 
  players_bad_x_len,
  players_bad_keys  ] = players_bad_data.next_batch( players_bad_data.num_examples  ) 


[ players_ok_x, 
  players_ok_y, 
  num_items, 
  players_ok_len, 
  players_ok_x_len,
  players_ok_keys ] = players_ok_data.next_batch( players_ok_data.num_examples  ) 


[ players_good_x, 
  players_good_y, 
  num_items, 
  players_good_len, 
  players_good_x_len,
  players_good_keys  ] = players_good_data.next_batch( players_good_data.num_examples  ) 


[ players_native_x, 
  players_native_y, 
  num_items, 
  players_native_len, 
  players_native_x_len,
  players_native_keys  ] = players_native_data.next_batch( players_native_data.num_examples  ) 
'''

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


# From http://stackoverflow.com/questions/37564901/regularly-evaluating-large-test-sets-during-convolutional-neural-network-trainin
test_accuracy_placeholder = tf.placeholder(tf.float32, name="test_accuracy")
test_summary = tf.summary.scalar("accuracy", test_accuracy_placeholder)


balanced_accuracy_placeholder = tf.placeholder(tf.float32, name="balanced_test_accuracy")
balanced_test_summary = tf.summary.scalar("accuracy", balanced_accuracy_placeholder)

# Retrieve just the LSTM variables.


with tf.Session() as sess:
    #sess.run(init)


    #prediction = recurrent_neural_network(x, keep_prob)

    print("Init saver with meta graph")
    restorer = tf.train.Saver(tf.global_variables())
    restorer.restore(sess, modelfile)
    print("Done!")




    for tests in [ [ "english_train_data", train_sample_x, train_sample_y, train_sample_keys] ]:
                   #[ "eval_uk", eval_uk_x, eval_uk_y, eval_uk_keys, eval_uk_writer, balanced_eval_uk_writer],
                   ## [ "eval_fi", eval_fi_x, eval_fi_y, eval_fi_keys, eval_fi_writer, balanced_eval_fi_writer ],
                   #[ "players_bad", players_bad_x, players_bad_y, players_bad_keys, players_bad_writer, balanced_players_bad_writer ],
                   #[ "players_ok", players_ok_x, players_ok_y, players_ok_keys,players_ok_writer, balanced_players_ok_writer ],
                   #[ "players_good", players_good_x, players_good_y, players_good_keys, players_good_writer, balanced_players_good_writer ],
                   #[ "players_native", players_native_x, players_native_y, players_native_keys, players_native_writer, balanced_players_native_writer ] ]:

                    testname = tests[0]
                    test_x = tests[1]
                    test_y = tests[2]
                    testkeys = tests[3]

                    testlen =  len(tests[2])

                    #print_to_log("Let's get summary for %s" % testname)


                    eval_batch_size = 512

                    #print ("Test shape:")
                    #print (test_x.shape)

                    confmatrix = np.zeros([n_classes, n_classes])
                    predictions = np.zeros([testlen, 3])
                    activations = np.zeros([testlen, n_classes + 2])

                    #print_to_log("Start test %s" % testname)
                    
                    for m in range(math.ceil(testlen/eval_batch_size)):

                        start=min(m * eval_batch_size, testlen)
                        end=min(start+eval_batch_size, testlen)

                        if (start < end):

                            #sys.stderr.write("\rTesting %s %i:%i/%i" % (testname, start, end, testlen))
                            #print_to_log("Testing %s %i:%i/%i" % (testname, start, end, testlen))



                            batch_x = test_x[start:end,:,:];
                            batch_y = test_y[start:end]


                            EMB = sess.run( prediction , 
                                            feed_dict={ x: batch_x, 
                                                        y: batch_y, 
                                                        keep_prob: 1.0 })

                            predictions[start:end,0] = testkeys[start:end]
                            predictions[start:end,1] = np.argmax(batch_y, 1)
                            predictions[start:end,2] = np.argmax(EMB,1).reshape([-1])

                            activations[start:end,0] = testkeys[start:end]
                            activations[start:end,1] = np.argmax(batch_y, 1)
                            activations[start:end,2:] = EMB

                            cf = confusion_matrix( np.argmax(batch_y, 1), np.argmax(EMB,1).reshape([-1]) )

                            confmatrix[:cf.shape[0],:cf.shape[1]] += cf

                            np.savetxt(os.path.join(LOG_DIR, 'cp%i_%s_confusion_matrix_work_in_progress' % (checkpoint, testname)),
                                       confmatrix,
                                       fmt='%i',)

                    np.savetxt(os.path.join(LOG_DIR, 'cp%i_%s_confusion_matrix' % (checkpoint, testname)),
                               confmatrix,
                               fmt='%i',)

                    np.savetxt(os.path.join(LOG_DIR, 'cp%i_%s_id_y_and_prediction' % (checkpoint, testname)),
                               predictions,
                               fmt='%i',)

                    np.savetxt(os.path.join(LOG_DIR, 'cp%i_%s_id_y_and_activations' % (checkpoint, testname)),
                               activations,
                               fmt='%i',)

                    biased_performance = np.diag(confmatrix).sum(-1) / confmatrix.sum(-1).sum(-1)

                    sum_m=confmatrix.sum(1)
                    sum_m[sum_m==0]=1      
                    norm_m=(confmatrix.T/sum_m.T).T
                    norm_m_diag=np.diag(norm_m)

                    balanced_performance = np.mean(norm_m_diag[norm_m_diag>0])

                    #print ("\r all: %0.2f balanced: %0.2f %s" % (biased_performance, balanced_performance, testname))

                    '''
                    test_writer = tests[4]
                    balanced_test_writer = tests[5]

                    
                    # "calculate" and log the average accuracy over all test batches
                    summarytensor = sess.run(test_summary,
                                             feed_dict={
                                                 test_accuracy_placeholder: biased_performance })


                    balanced_summarytensor = sess.run(balanced_test_summary,
                                                      feed_dict={
                                                          balanced_accuracy_placeholder: balanced_performance })

                    test_writer.add_summary(summarytensor,  epoch * total_batch + i )
                    balanced_test_writer.add_summary(balanced_summarytensor,  epoch * total_batch + i )

                    '''
                    
                    '''
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
                    '''
                    #sys.stderr.write
                    print_to_log("Acc: %0.2f/%0.2f balanced, Test: %s" % (biased_performance, balanced_performance, testname) )



