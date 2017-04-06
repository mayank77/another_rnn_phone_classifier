#!/usr/bin/python
# coding: utf-8

from os.path import join

import traceback
import os
import sys

import timeit
import time

import numpy

import binascii
import socket
import struct
import sys

HOST = 'localhost'        # Symbolic name meaning all available interfaces
PORT = 50007              # Arbitrary non-privileged port

debugging = True

import time, datetime, sys
import numpy as np
#import h5py

import pickle

import numpy as np
import pickle, os, random, math, sys

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell


'''  1.  Define RNN '''



zmean = np.loadtxt( sys.argv[3] ) #traindata.mean
zstd = np.loadtxt( sys.argv[4] )# traindata.std
zmax_len = 62#traindata.max_len
znum_classes = 119 #traindata.num_classes

usedim = np.arange(1,66)
featdim = 65

hm_epochs = 40
n_classes = znum_classes
batch_size = 128
n_chunks = featdim #28
rnn_size = 384
n_hidden = rnn_size

train_len = zmax_len #30

print("\n\nMax train seq. length: %i" % train_len)

num_layers = 5



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


''' 2 Define dictionaries '''


class_def = {
"sil" : {"count" : 6611, "probability" :0.04628649220995, "sqrt_probability" :0.2151429576118, "class" :45},
"P" : {"count" : 306, "probability" :1, "sqrt_probability" :1, "class" :44},
"Z" : {"count" : 311, "probability" :0.98392282958199, "sqrt_probability" :0.99192884300336, "class" :43},
"Å" : {"count" : 344, "probability" :0.88953488372093, "sqrt_probability" :0.94315156985552, "class" :42},
"U" : {"count" : 497, "probability" :0.61569416498994, "sqrt_probability" :0.78466181568236, "class" :41},
"W" : {"count" : 501, "probability" :0.61077844311377, "sqrt_probability" :0.78152315583978, "class" :40},
"ö" : {"count" : 573, "probability" :0.53403141361257, "sqrt_probability" :0.73077452994242, "class" :39},
"å" : {"count" : 601, "probability" :0.50915141430948, "sqrt_probability" :0.71354846668568, "class" :38},
"S" : {"count" : 641, "probability" :0.47737909516381, "sqrt_probability" :0.69092625884663, "class" :37},
"C" : {"count" : 746, "probability" :0.41018766756032, "sqrt_probability" :0.64045895072231, "class" :36},
"J" : {"count" : 757, "probability" :0.40422721268164, "sqrt_probability" :0.63578865409949, "class" :35},
"A" : {"count" : 958, "probability" :0.31941544885177, "sqrt_probability" :0.56516851367692, "class" :34},
"N" : {"count" : 958, "probability" :0.31941544885177, "sqrt_probability" :0.56516851367692, "class" :33},
"H" : {"count" : 963, "probability" :0.31775700934579, "sqrt_probability" :0.56369939626169, "class" :32},
"R" : {"count" : 1050, "probability" :0.29142857142857, "sqrt_probability" :0.53984124650546, "class" :31},
"T" : {"count" : 1057, "probability" :0.28949858088931, "sqrt_probability" :0.53805072334243, "class" :30},
"j" : {"count" : 1132, "probability" :0.27031802120141, "sqrt_probability" :0.5199211682567, "class" :29},
"Y" : {"count" : 1354, "probability" :0.22599704579025, "sqrt_probability" :0.47539146583658, "class" :28},
"g" : {"count" : 1420, "probability" :0.21549295774648, "sqrt_probability" :0.46421219043287, "class" :27},
"u" : {"count" : 1621, "probability" :0.18877236273905, "sqrt_probability" :0.4344794157829, "class" :26},
"o" : {"count" : 1648, "probability" :0.18567961165049, "sqrt_probability" :0.4309055716169, "class" :25},
"w" : {"count" : 1675, "probability" :0.18268656716418, "sqrt_probability" :0.42741849183696, "class" :24},
"ä" : {"count" : 1742, "probability" :0.17566016073479, "sqrt_probability" :0.41911831352828, "class" :23},
"p" : {"count" : 1773, "probability" :0.17258883248731, "sqrt_probability" :0.41543812112914, "class" :22},
"E" : {"count" : 1807, "probability" :0.16934144991699, "sqrt_probability" :0.41151117836213, "class" :21},
"D" : {"count" : 1819, "probability" :0.16822429906542, "sqrt_probability" :0.4101515562148, "class" :20},
"O" : {"count" : 1843, "probability" :0.16603364080304, "sqrt_probability" :0.4074722577097, "class" :19},
"b" : {"count" : 2129, "probability" :0.14372945044622, "sqrt_probability" :0.379116671285, "class" :18},
"m" : {"count" : 2177, "probability" :0.140560404226, "sqrt_probability" :0.37491386240842, "class" :17},
"v" : {"count" : 2227, "probability" :0.13740458015267, "sqrt_probability" :0.37068123792913, "class" :16},
"e" : {"count" : 2298, "probability" :0.1331592689295, "sqrt_probability" :0.36490994632855, "class" :15},
"d" : {"count" : 2410, "probability" :0.12697095435685, "sqrt_probability" :0.35632983927374, "class" :14},
"a" : {"count" : 2582, "probability" :0.11851278079009, "sqrt_probability" :0.34425685293119, "class" :13},
"I" : {"count" : 2608, "probability" :0.11733128834356, "sqrt_probability" :0.3425365503761, "class" :12},
"f" : {"count" : 2690, "probability" :0.11375464684015, "sqrt_probability" :0.33727532794462, "class" :11},
"Ä" : {"count" : 2795, "probability" :0.10948121645796, "sqrt_probability" :0.33087945910552, "class" :10},
"z" : {"count" : 3066, "probability" :0.09980430528376, "sqrt_probability" :0.31591819397394, "class" :9},
"k" : {"count" : 3292, "probability" :0.09295261239368, "sqrt_probability" :0.30488130869845, "class" :8},
"l" : {"count" : 3619, "probability" :0.08455374412821, "sqrt_probability" :0.2907812650915, "class" :7},
"r" : {"count" : 3630, "probability" :0.08429752066116, "sqrt_probability" :0.29034035313948, "class" :6},
"Q" : {"count" : 4684, "probability" :0.06532877882152, "sqrt_probability" :0.25559495069645, "class" :5},
"s" : {"count" : 4867, "probability" :0.06287240599959, "sqrt_probability" :0.25074370580254, "class" :4},
"i" : {"count" : 5140, "probability" :0.05953307392996, "sqrt_probability" :0.24399400388116, "class" :3},
"t" : {"count" : 5996, "probability" :0.05103402268179, "sqrt_probability" :0.22590711073755, "class" :2},
"n" : {"count" : 6811, "probability" :0.04492732344736, "sqrt_probability" :0.21196066485875, "class" :1},
}

words = {
    u'finn': [ 'f', 'i', 'n'],
    u'june': [ 'J', 'u', 'n'],
    u'age': [ 'E', 'J'],
    u'air': [ 'W'],
    u'am': [ 'ä', 'm'],
    u'and': [ 'ä', 'n', 'd'],
    u'ant': [ 'ä', 'n', 't'],
    u'apple': [ 'ä', 'p', 'l'],
    u'arm': [ 'A', 'm'],
    u'art': [ 'A', 't'],
    u'at': [ 'ä', 't'],
    u'baby': [ 'b', 'E', 'b', 'I'],
    u'back': [ 'b', 'ä', 'k'],
    u'bad': [ 'b', 'ä', 'd'],
    u'bag': [ 'b', 'ä', 'g'],
    u'bake': [ 'b', 'E', 'k'],
    u'ball': [ 'b', 'O', 'l'],
    u'bark': [ 'b', 'A', 'k'],
    u'baseball': [ 'b', 'E', 's', 'b', 'O', 'l'],
    u'basketball': [ 'b', 'A', 's', 'k', 'i', 't', 'b', 'O', 'l'],
    u'bath': [ 'b', 'A', 'T'],
    u'be': [ 'b', 'I'],
    u'bear': [ 'b', 'W'],
    u'bed': [ 'b', 'e', 'd'],
    u'bedroom': [ 'b', 'e', 'd', 'r', 'u', 'm'],
    u'bee': [ 'b', 'I'],
    u'belt': [ 'b', 'e', 'l', 't'],
    u'berry': [ 'b', 'e', 'r', 'I'],
    u'best': [ 'b', 'e', 's', 't'],
    u'big': [ 'b', 'i', 'g'],
    u'bird': [ 'b', 'ö', 'd'],
    u'black': [ 'b', 'l', 'ä', 'k'],
    u'blue': [ 'b', 'l', 'u'],
    u'body': [ 'b', 'Y', 'd', 'I'],
    u'book': [ 'b', 'U', 'k'],
    u'boy': [ 'b', 'O', 'I'],
    u'bread': [ 'b', 'r', 'e', 'd'],
    u'breakfast': [ 'b', 'r', 'e', 'k', 'f', 'Q', 's', 't'],
    u'brother': [ 'b', 'r', 'a', 'D', 'Q'],
    u'brown': [ 'b', 'r', 'å', 'n'],
    u'bug': [ 'b', 'a', 'g'],
    u'bush': [ 'b', 'U', 'S'],
    u'butter': [ 'b', 'a', 't', 'Q'],
    u'butterfly': [ 'b', 'a', 't', 'Q', 'f', 'l', 'Ä'],
    u'buy': [ 'b', 'Ä'],
    u'bye': [ 'b', 'Ä'],
    u'bye_bye': [ 'b', 'Ä', 'b', 'Ä'],
    u'cake': [ 'k', 'E', 'k'],
    u'candy': [ 'k', 'ä', 'n', 'd', 'I'],
    u'cap': [ 'k', 'ä', 'p'],
    u'carpet': [ 'k', 'A', 'p', 'i', 't'],
    u'carrot': [ 'k', 'ä', 'r', 'Q', 't'],
    u'cat': [ 'k', 'ä', 't'],
    u'chair': [ 'C', 'W'],
    u'chat': [ 'C', 'ä', 't'],
    u'cheap': [ 'C', 'I', 'p'],
    u'cheek': [ 'C', 'I', 'k'],
    u'cheese': [ 'C', 'I', 'z'],
    u'cherry': [ 'C', 'e', 'r', 'I'],
    u'chew': [ 'C', 'u'],
    u'chick': [ 'C', 'i', 'k'],
    u'chicken': [ 'C', 'i', 'k', 'i', 'n'],
    u'child': [ 'C', 'Ä', 'l', 'd'],
    u'children': [ 'C', 'i', 'l', 'd', 'r', 'Q', 'n'],
    u'chin': [ 'C', 'i', 'n'],
    u'chip': [ 'C', 'i', 'p'],
    u'chocolate': [ 'C', 'Y', 'k', 'l', 'Q', 't'],
    u'choose': [ 'C', 'u', 'z'],
    u'city': [ 's', 'i', 't', 'I'],
    u'classmate': [ 'k', 'l', 'A', 's', 'm', 'E', 't'],
    u'close': [ 'k', 'l', 'o', 's'],
    u'clothe': [ 'k', 'l', 'o', 'D'],
    u'clothes': [ 'k', 'l', 'o', 'D', 'z'],
    u'coat': [ 'k', 'o', 't'],
    u'cold': [ 'k', 'o', 'l', 'd'],
    u'colour': [ 'k', 'a', 'l', 'Q'],
    u'come': [ 'k', 'a', 'm'],
    u'come_here': [ 'k', 'a', 'm', 'H', 'R'],
    u'come_on': [ 'k', 'a', 'm', 'Q', 'n'],
    u'cook': [ 'k', 'U', 'k'],
    u'cool': [ 'k', 'u', 'l'],
    u'country': [ 'k', 'a', 'n', 't', 'r', 'I'],
    u'cow': [ 'k', 'å'],
    u'cucumber': [ 'k', 'j', 'u', 'k', 'a', 'm', 'b', 'Q'],
    u'cup': [ 'k', 'a', 'p'],
    u'curl': [ 'k', 'ö', 'l'],
    u'cut': [ 'k', 'a', 't'],
    u'dark': [ 'd', 'A', 'k'],
    u'day': [ 'd', 'E'],
    u'deer': [ 'd', 'R'],
    u'dessert': [ 'd', 'i', 'z', 'ö', 't'],
    u'do': [ 'd', 'u'],
    u'do_you_want_to_play_with_us': [ 'd', 'u', 'j', 'Q', 'w', 'Y', 'n', 't', 't', 'Q', 'p', 'l', 'E', 'w', 'i', 'D', 'a', 's'],
    u'dog': [ 'd', 'Y', 'g'],
    u'dolphin': [ 'd', 'Y', 'l', 'f', 'i', 'n'],
    u'don_t': [ 'd', 'Y', 'n', 't', 'I'],
    u'door': [ 'd', 'O'],
    u'draw': [ 'd', 'r', 'O'],
    u'dream': [ 'd', 'r', 'I', 'm'],
    u'drink': [ 'd', 'r', 'i', 'N', 'k'],
    u'dry': [ 'd', 'r', 'Ä'],
    u'duck': [ 'd', 'a', 'k'],
    u'ear': [ 'R'],
    u'eat': [ 'e', 't'],
    u'eight': [ 'E', 't'],
    u'elbow': [ 'e', 'l', 'b', 'o'],
    u'elephant': [ 'e', 'l', 'i', 'f', 'Q', 'n', 't'],
    u'elk': [ 'e', 'l', 'k'],
    u'england': [ 'i', 'N', 'g', 'l', 'Q', 'n', 'd'],
    u'english': [ 'i', 'N', 'g', 'l', 'i', 'S'],
    u'europe': [ 'j', 'P', 'r', 'Q', 'p'],
    u'evening': [ 'I', 'v', 'n', 'i', 'N'],
    u'eye': [ 'Ä'],
    u'eyes': [ 'Ä', 'z'],
    u'face': [ 'f', 'E', 's'],
    u'fall': [ 'f', 'O', 'l'],
    u'family': [ 'f', 'ä', 'm', 'Q', 'l', 'I'],
    u'fan': [ 'f', 'ä', 'n'],
    u'farmer': [ 'f', 'A', 'm', 'Q'],
    u'fat': [ 'f', 'ä', 't'],
    u'father': [ 'f', 'A', 'D', 'Q'],
    u'feather': [ 'f', 'e', 'D', 'Q'],
    u'feel': [ 'f', 'I', 'l'],
    u'feet': [ 'f', 'I', 't'],
    u'field': [ 'f', 'I', 'l', 'd'],
    u'fin': [ 'f', 'i', 'n'],
    u'fine': [ 'f', 'Ä', 'n'],
    u'finger': [ 'f', 'i', 'N', 'g', 'Q'],
    u'finland': [ 'f', 'i', 'n', 'l', 'Q', 'n', 'd'],
    u'finn': [ 'f', 'i', 'n'],
    u'finnish': [ 'f', 'i', 'n', 'i', 'S'],
    u'first': [ 'f', 'ö', 's', 't'],
    u'fish': [ 'f', 'i', 'S'],
    u'fit': [ 'f', 'i', 't'],
    u'flower': [ 'f', 'l', 'å', 'Q'],
    u'food': [ 'f', 'u', 'd'],
    u'forest': [ 'f', 'Y', 'r', 'i', 's', 't'],
    u'fork': [ 'f', 'O', 'k'],
    u'fox': [ 'f', 'Y', 'k', 's'],
    u'friend': [ 'f', 'r', 'e', 'n', 'd'],
    u'full': [ 'f', 'U', 'l'],
    u'fun': [ 'f', 'a', 'n'],
    u'fur': [ 'f', 'ö'],
    u'game': [ 'g', 'E', 'm'],
    u'garage': [ 'g', 'ä', 'r', 'A', 'Z'],
    u'girl': [ 'g', 'ö', 'l'],
    u'glass': [ 'g', 'l', 'A', 's'],
    u'goal': [ 'g', 'o', 'l'],
    u'goat': [ 'g', 'o', 't'],
    u'gold': [ 'g', 'o', 'l', 'd'],
    u'good': [ 'g', 'U', 'd'],
    u'good_evening': [ 'g', 'U', 'd', 'I', 'v', 'n', 'i', 'N'],
    u'good_morning': [ 'g', 'U', 'd', 'm', 'O', 'n', 'i', 'N'],
    u'good_night': [ 'g', 'U', 'd', 'n', 'Ä', 't'],
    u'grass': [ 'g', 'r', 'A', 's'],
    u'green': [ 'g', 'r', 'I', 'n'],
    u'grey': [ 'g', 'r', 'E'],
    u'hair': [ 'H', 'W'],
    u'hairy': [ 'H', 'W', 'r', 'I'],
    u'hand': [ 'H', 'ä', 'n', 'd'],
    u'hard': [ 'H', 'A', 'd'],
    u'hare': [ 'H', 'W'],
    u'hat': [ 'H', 'ä', 't'],
    u'hay': [ 'H', 'E'],
    u'he_goes': [ 'H', 'I', 'g', 'o', 'z'],
    u'healthy': [ 'H', 'e', 'l', 'T', 'I'],
    u'hear': [ 'H', 'R'],
    u'heart': [ 'H', 'A', 't'],
    u'hello': [ 'H', 'Q', 'l', 'o'],
    u'hen': [ 'H', 'e', 'n'],
    u'here': [ 'H', 'R'],
    u'here_you_are': [ 'H', 'R', 'j', 'Q', 'A'],
    u'high': [ 'H', 'Ä'],
    u'home': [ 'H', 'o', 'm'],
    u'horse': [ 'H', 'O', 's'],
    u'house': [ 'H', 'å', 's'],
    u'hurry_up': [ 'H', 'a', 'r', 'I', 'a', 'p'],
    u'i_am': [ 'Ä', 'ä', 'm'],
    u'i_am_sorry': [ 'Ä', 'ä', 'm', 's', 'Y', 'r', 'I'],
    u'i_go': [ 'Ä', 'g', 'o'],
    u'i_like': [ 'Ä', 'l', 'Ä', 'k'],
    u'i_live_in_finland': [ 'Ä', 'l', 'i', 'v', 'i', 'n', 'f', 'i', 'n', 'l', 'Q', 'n', 'd'],
    u'i_speak_finnish': [ 'Ä', 's', 'p', 'I', 'k', 'f', 'i', 'n', 'i', 'S'],
    u'i_was': [ 'Ä', 'w', 'Q', 'z'],
    u'ice': [ 'Ä', 's'],
    u'ice_hockey': [ 'Ä', 's', 'H', 'Y', 'k', 'I'],
    u'ice_skating': [ 'Ä', 's', 's', 'k', 'E', 't', 'i', 'N'],
    u'it': [ 'i', 't'],
    u'jam': [ 'J', 'ä', 'm'],
    u'jaw': [ 'J', 'O'],
    u'jet': [ 'J', 'e', 't'],
    u'juice': [ 'J', 'u', 's'],
    u'jump': [ 'J', 'a', 'm', 'p'],
    u'june': [ 'J', 'u', 'n'],
    u'kitchen': [ 'k', 'i', 'C', 'i', 'n'],
    u'knee': [ 'n', 'I'],
    u'knife': [ 'n', 'Ä', 'f'],
    u'lake': [ 'l', 'E', 'k'],
    u'lamp': [ 'l', 'ä', 'm', 'p'],
    u'language': [ 'l', 'ä', 'N', 'g', 'w', 'i', 'J'],
    u'lead': [ 'l', 'e', 'd'],
    u'leaf': [ 'l', 'I', 'f'],
    u'learn': [ 'l', 'ö', 'n'],
    u'leave': [ 'l', 'I', 'v'],
    u'legs': [ 'l', 'e', 'g', 'z'],
    u'lemon': [ 'l', 'e', 'm', 'Q', 'n'],
    u'lets_go': [ 'l', 'e', 't', 's', 'g', 'o'],
    u'lick': [ 'l', 'i', 'k'],
    u'life': [ 'l', 'Ä', 'f'],
    u'light': [ 'l', 'Ä', 't'],
    u'lion': [ 'l', 'Ä', 'Q', 'n'],
    u'lips': [ 'l', 'i', 'p', 's'],
    u'listen': [ 'l', 'i', 's', 'n'],
    u'live': [ 'l', 'Ä', 'v'],
    u'living_room': [ 'l', 'i', 'v', 'i', 'N', 'r', 'u', 'm'],
    u'lock': [ 'l', 'Y', 'k'],
    u'log': [ 'l', 'Y', 'g'],
    u'long': [ 'l', 'Y', 'N'],
    u'look': [ 'l', 'U', 'k'],
    u'loud': [ 'l', 'å', 'd'],
    u'love': [ 'l', 'a', 'v'],
    u'low': [ 'l', 'o'],
    u'lunch': [ 'l', 'a', 'n', 'C'],
    u'make': [ 'm', 'E', 'k'],
    u'man': [ 'm', 'ä', 'n'],
    u'maths': [ 'm', 'ä', 'T', 's'],
    u'mats': [ 'm', 'ä', 't', 's'],
    u'meat': [ 'm', 'I', 't'],
    u'milk': [ 'm', 'i', 'l', 'k'],
    u'money': [ 'm', 'a', 'n', 'I'],
    u'monkey': [ 'm', 'a', 'N', 'k', 'I'],
    u'month': [ 'm', 'a', 'n', 'T'],
    u'moon': [ 'm', 'u', 'n'],
    u'moose': [ 'm', 'u', 's'],
    u'more': [ 'm', 'O'],
    u'morning': [ 'm', 'O', 'n', 'i', 'N'],
    u'mother': [ 'm', 'a', 'D', 'Q'],
    u'mouse': [ 'm', 'å', 's'],
    u'mouth': [ 'm', 'å', 'D'],
    u'move': [ 'm', 'u', 'v'],
    u'movie': [ 'm', 'u', 'v', 'I'],
    u'music': [ 'm', 'j', 'u', 'z', 'i', 'k'],
    u'my_name_is': [ 'm', 'Ä', 'n', 'E', 'm', 'i', 'z'],
    u'neck': [ 'n', 'e', 'k'],
    u'neighbour': [ 'n', 'E', 'b', 'Q'],
    u'nice_to_meet_you': [ 'n', 'Ä', 's', 't', 'Q', 'm', 'I', 't', 'j', 'Q'],
    u'night': [ 'n', 'Ä', 't'],
    u'no_thank_you': [ 'n', 'o', 'T', 'ä', 'N', 'k', 'j', 'Q'],
    u'north': [ 'n', 'O', 'T'],
    u'nose': [ 'n', 'o', 'z'],
    u'now': [ 'n', 'å'],
    u'of': [ 'Q', 'v'],
    u'off': [ 'O', 'f'],
    u'old': [ 'o', 'l', 'd'],
    u'once': [ 'w', 'a', 'n', 's'],
    u'one': [ 'w', 'a', 'n'],
    u'ones': [ 'w', 'a', 'n', 'z'],
    u'orange': [ 'Y', 'r', 'i', 'n', 'J'],
    u'owl': [ 'å', 'l'],
    u'pack': [ 'p', 'ä', 'k'],
    u'page': [ 'p', 'E', 'J'],
    u'paint': [ 'p', 'E', 'n', 't'],
    u'parents': [ 'p', 'W', 'r', 'Q', 'n', 't', 's'],
    u'park': [ 'p', 'A', 'k'],
    u'parrot': [ 'p', 'ä', 'r', 'Q', 't'],
    u'party': [ 'p', 'A', 't', 'I'],
    u'pays': [ 'p', 'E', 'z'],
    u'pea': [ 'p', 'I'],
    u'pear': [ 'p', 'W'],
    u'peas': [ 'p', 'I', 'z'],
    u'pepper': [ 'p', 'e', 'p', 'Q'],
    u'pet': [ 'p', 'e', 't'],
    u'phone': [ 'f', 'o', 'n'],
    u'pie': [ 'p', 'Ä'],
    u'piece': [ 'p', 'I', 's'],
    u'pig': [ 'p', 'i', 'g'],
    u'pink': [ 'p', 'i', 'N', 'k'],
    u'plant': [ 'p', 'l', 'A', 'n', 't'],
    u'plate': [ 'p', 'l', 'E', 't'],
    u'play': [ 'p', 'l', 'E'],
    u'please': [ 'p', 'l', 'I', 'z'],
    u'pool': [ 'p', 'u', 'l'],
    u'potato': [ 'p', 'Q', 't', 'E', 't', 'o'],
    u'pull': [ 'p', 'U', 'l'],
    u'purple': [ 'p', 'ö', 'p', 'l'],
    u'quiet': [ 'k', 'w', 'Ä', 'Q', 't'],
    u'rat': [ 'r', 'ä', 't'],
    u'read': [ 'r', 'e', 'd'],
    u'real': [ 'r', 'E', 'A', 'l'],
    u'red': [ 'r', 'e', 'd'],
    u'rest': [ 'r', 'e', 's', 't'],
    u'ride': [ 'r', 'Ä', 'd'],
    u'right': [ 'r', 'Ä', 't'],
    u'river': [ 'r', 'i', 'v', 'Q'],
    u'rock': [ 'r', 'Y', 'k'],
    u'room': [ 'r', 'u', 'm'],
    u'round': [ 'r', 'å', 'n', 'd'],
    u'run': [ 'r', 'a', 'n'],
    u'safe': [ 's', 'E', 'f'],
    u'salad': [ 's', 'ä', 'l', 'Q', 'd'],
    u'salt': [ 's', 'O', 'l', 't'],
    u'sand': [ 's', 'ä', 'n', 'd'],
    u'save': [ 's', 'E', 'v'],
    u'say': [ 's', 'E'],
    u'school': [ 's', 'k', 'u', 'l'],
    u'sea': [ 's', 'I'],
    u'seal': [ 's', 'I', 'l'],
    u'seat': [ 's', 'I', 't'],
    u'seed': [ 's', 'I', 'd'],
    u'sees': [ 's', 'I', 'z'],
    u'set': [ 's', 'e', 't'],
    u'she': [ 'S', 'I'],
    u'she_is': [ 'S', 'I', 'i', 'z'],
    u'she_likes': [ 'S', 'I', 'l', 'Ä', 'k', 's'],
    u'sheep': [ 'S', 'I', 'p'],
    u'sheet': [ 'S', 'I', 't'],
    u'shine': [ 'S', 'Ä', 'n'],
    u'ship': [ 'S', 'i', 'p'],
    u'shirt': [ 'S', 'ö', 't'],
    u'shoe': [ 'S', 'u'],
    u'shop': [ 'S', 'Y', 'p'],
    u'shoulder': [ 'S', 'o', 'l', 'd', 'Q'],
    u'shower': [ 'S', 'å', 'Q'],
    u'shows': [ 'S', 'o', 'z'],
    u'sing': [ 's', 'i', 'N'],
    u'sister': [ 's', 'i', 's', 't', 'Q'],
    u'sit': [ 's', 'i', 't'],
    u'ski': [ 's', 'k', 'I'],
    u'sleep': [ 's', 'l', 'I', 'p'],
    u'smell': [ 's', 'm', 'e', 'l'],
    u'snack': [ 's', 'n', 'ä', 'k'],
    u'snake': [ 's', 'n', 'E', 'k'],
    u'socks': [ 's', 'Y', 'k', 's'],
    u'soft': [ 's', 'Y', 'f', 't'],
    u'soon': [ 's', 'u', 'n'],
    u'sorry': [ 's', 'Y', 'r', 'I'],
    u'sound': [ 's', 'å', 'n', 'd'],
    u'soup': [ 's', 'u', 'p'],
    u'speak': [ 's', 'p', 'I', 'k'],
    u'spoon': [ 's', 'p', 'u', 'n'],
    u'squirrel': [ 's', 'k', 'w', 'i', 'r', 'Q', 'l'],
    u'strawberry': [ 's', 't', 'r', 'O', 'b', 'r', 'I'],
    u'sun': [ 's', 'a', 'n'],
    u'swedish': [ 's', 'w', 'I', 'd', 'i', 'S'],
    u'sweet': [ 's', 'w', 'I', 't'],
    u'swim': [ 's', 'w', 'i', 'm'],
    u'table': [ 't', 'A', 'b', 'l'],
    u'taste': [ 't', 'E', 's', 't'],
    u'teacher': [ 't', 'I', 'C', 'Q'],
    u'teas': [ 't', 'I', 'z'],
    u'teeth': [ 't', 'I', 'T'],
    u'ten': [ 't', 'e', 'n'],
    u'than': [ 'D', 'ä', 'n'],
    u'thank_you': [ 'T', 'ä', 'N', 'k', 'j', 'Q'],
    u'that': [ 'D', 'ä', 't'],
    u'the': [ 'D', 'Q'],
    u'theatre': [ 'T', 'R', 't', 'Q'],
    u'their': [ 'D', 'W'],
    u'then': [ 'D', 'e', 'n'],
    u'there': [ 'D', 'W'],
    u'these': [ 'D', 'I', 'z'],
    u'they': [ 'D', 'E'],
    u'thick': [ 'T', 'i', 'k'],
    u'thin': [ 'T', 'i', 'n'],
    u'thing': [ 'T', 'i', 'N'],
    u'think': [ 'T', 'i', 'N', 'k'],
    u'thirst': [ 'T', 'ö', 's', 't'],
    u'thirsty': [ 'T', 'ö', 's', 't', 'I'],
    u'this': [ 'D', 'i', 's'],
    u'those': [ 'D', 'o', 'z'],
    u'three': [ 'T', 'r', 'I'],
    u'throw': [ 'T', 'r', 'o'],
    u'time': [ 't', 'Ä', 'm'],
    u'toast': [ 't', 'o', 's', 't'],
    u'today': [ 't', 'Q', 'd', 'E'],
    u'toe': [ 't', 'o'],
    u'toilet': [ 't', 'Å', 'l', 'i', 't'],
    u'tomato': [ 't', 'Q', 'm', 'A', 't', 'o'],
    u'tomorrow': [ 't', 'Q', 'm', 'Y', 'r', 'o'],
    u'tongue': [ 't', 'a', 'N'],
    u'too': [ 't', 'u'],
    u'tortoise': [ 't', 'O', 't', 'Q', 's'],
    u'town': [ 't', 'å', 'n'],
    u'tree': [ 't', 'r', 'I'],
    u'two': [ 't', 'u'],
    u'use': [ 'j', 'u', 's'],
    u'van': [ 'v', 'ä', 'n'],
    u'vest': [ 'v', 'e', 's', 't'],
    u'vet': [ 'v', 'e', 't'],
    u'village': [ 'v', 'i', 'l', 'i', 'J'],
    u'walk': [ 'w', 'O', 'k'],
    u'wall': [ 'w', 'O', 'l'],
    u'want': [ 'w', 'Y', 'n', 't'],
    u'was': [ 'w', 'Q', 'z'],
    u'wash': [ 'w', 'Y', 'S'],
    u'watch': [ 'w', 'Y', 'C'],
    u'water': [ 'w', 'O', 't', 'Q'],
    u'we_go': [ 'w', 'I', 'g', 'o'],
    u'we_were': [ 'w', 'I', 'w', 'ö'],
    u'wear': [ 'w', 'W'],
    u'week': [ 'w', 'I', 'k'],
    u'well_done': [ 'w', 'e', 'l', 'd', 'a', 'n'],
    u'were': [ 'w', 'ö'],
    u'west': [ 'w', 'e', 's', 't'],
    u'wet': [ 'w', 'e', 't'],
    u'what': [ 'w', 'Y', 't'],
    u'wheel': [ 'w', 'I', 'l'],
    u'where': [ 'w', 'W'],
    u'which': [ 'w', 'i', 'C'],
    u'white': [ 'w', 'Ä', 't'],
    u'who': [ 'H', 'u'],
    u'why': [ 'w', 'Ä'],
    u'win': [ 'w', 'i', 'n'],
    u'window': [ 'w', 'i', 'n', 'd', 'o'],
    u'wish': [ 'w', 'i', 'S'],
    u'with': [ 'w', 'i', 'D'],
    u'woman': [ 'w', 'U', 'm', 'Q', 'n'],
    u'wool': [ 'w', 'U', 'l'],
    u'worm': [ 'w', 'ö', 'm'],
    u'would_you_like_some_juice': [ 'w', 'Q', 'd', 'j', 'Q', 'l', 'Ä', 'k', 's', 'a', 'm', 'J', 'u', 's'],
    u'would_you_like_some_more': [ 'w', 'Q', 'd', 'j', 'Q', 'l', 'Ä', 'k', 's', 'a', 'm', 'm', 'O'],
    u'write': [ 'r', 'Ä', 't'],
    u'wrong': [ 'r', 'Y', 'N'],
    u'year': [ 'j', 'ö'],
    u'yellow': [ 'j', 'e', 'l', 'o'],
    u'yes_please': [ 'j', 'e', 's', 'p', 'l', 'I', 'z'],
    u'yesterday': [ 'j', 'e', 's', 't', 'Q', 'd', 'E'],
    u'yet': [ 'j', 'e', 't'],
    u'you_are': [ 'j', 'Q', 'A'],
    u'you_go': [ 'j', 'Q', 'g', 'o'],
    u'you_re_welcome': [ 'j', 'Q', 'r', 'E', 'w', 'e', 'l', 'k', 'Q', 'm'],
    u'you_were': [ 'j', 'Q', 'w', 'ö'],
    u'young': [ 'j', 'a', 'N'],
    u'your': [ 'j', 'O'],
    u'zoo': [ 'z', 'u'],
    u'zoos': [ 'z', 'u', 'z']  }


# some extra words:

for word in ['feather', 'healthy', 'it', 'eat', 'jet', 'yet']:
    for n in [ '1','2','3','4','5','6','7','8','9']:
        words[ word + n ] = words[word]
        words[ word + n + '_txt'] = words[word]

for word in ['juice', 'use', 'your', 'jaw', 'jump', 'jam', 'age', 'orange']:
    words[ word + '_txt'] = words[word]

'''  3.  Define server loop '''



def run_server_loop(portfilename, modelfile, meanfile, stdfile, lsqweightfile):


    ackvalue  = -1
    acklength=4


    single_integer_packer = struct.Struct('i')
    single_float_packer = struct.Struct('f')
    ackpacket = single_integer_packer.pack(ackvalue)

    datadim = 66
    timesteps = 62
    nb_classes = 119
    
    #batch_size=32
    #nb_epoch=30


    loadstartmoment= time.clock()

    with tf.Session() as sess:
        #sess.run(init)


        prediction = recurrent_neural_network(x, keep_prob)

        print("Init saver with meta graph from %s" % modelfile)
        restorer = tf.train.Saver(tf.global_variables())
        restorer.restore(sess, modelfile)
        print("Done!")



        print ( "Load normalisation stats..." )
        
        zmean = np.loadtxt( meanfile ) #traindata.mean
        zstd = np.loadtxt( stdfile )# traindata.std

        lsq_weights = np.loadtxt( lsqweightfile ).reshape(45, 120)

        loadtime = time.clock()-loadstartmoment
        print ("Loading took %0.1f seconds!"%loadtime)


        # Connection handling! #
        # Get data from socket and do it!
        
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((HOST, 0))
        s.listen(1)

        
        port = s.getsockname()[1]
        text_file = open(portfilename, "w")
        text_file.write(str(port))
        text_file.close()

        single_integer_packer = struct.Struct('i')

        print ("Waiting for connection in port %i!" % port)

        while 1:
            connectionstartstartmoment = time.clock()

            conn, addr = s.accept()


            print ('Connected by', addr)

            print ('Length of word to be classified:')
            datalen = conn.recv(single_integer_packer.size)

            if not datalen: break

            print (datalen)

            unpacked_datalen = single_integer_packer.unpack(datalen)
            unpacked_datalen = int(unpacked_datalen[0])
            print(unpacked_datalen)
            print ("Going to read unicode data of length %i" % unpacked_datalen)

            char_packer = struct.Struct( unpacked_datalen*'c')   
            conn.send( ackpacket )

            print ('Word to be classified:')

            worddata = conn.recv(char_packer.size)
            if not worddata: break

            print(worddata)

            try:
                word = worddata.decode("utf-8").replace('\x00', '') #char_packer.unpack( worddata )
                #word = ''.join(word)
                print (word)
                #word = word.decode('utf-16')
                word_data_ok = True           
            except:
                print ("Something went wrong, let's print stack trace:")
                traceback.print_exc()
                conn.close()
                word_data_ok = False
            
            if word_data_ok:

                print("Got word \"%s\""%word)

                print(words[word])

                if word not in words:
                    print ("Word not in dictionary: Closing connection")
                    conn.close()
                    word_data_ok = False
                    continue

                phones = words[word]            
                classes = [ class_def[phone]['class'] for phone in phones  ]

                conn.send( ackpacket )


                print ('Will now classify using models %s' % modelfile)

                datalen = conn.recv(single_integer_packer.size)
                if not datalen: break


                unpacked_datalen = single_integer_packer.unpack(datalen)

                print ("Going to read float data of length %i" % unpacked_datalen)


                float_packer = struct.Struct( unpacked_datalen[0]*'f')   
                conn.send( ackpacket )

                MSGLEN = unpacked_datalen[0] * 4
                chunks = []
                bytes_recd = 0
                while bytes_recd < MSGLEN:
                    chunk = conn.recv(min(MSGLEN - bytes_recd, 2048))
                    if chunk == b'':
                        raise RuntimeError("socket connection broken")
                    chunks.append(chunk)
                    bytes_recd = bytes_recd + len(chunk)
                data = b''.join(chunks)


                #data = conn.recv(float_packer.size)
                if not data: break

                try:
                    unpacked_data = float_packer.unpack( data )
                    unpacked_data_ok = True
                except:
                    print ("Something went wrong, let's print stack trace:")
                    print ("Size of data: " + str(len(data)))

                    traceback.print_exc()
                    conn.close()
                    unpacked_data_ok = False
                    continue

                if unpacked_data_ok:

                    featdim=len(unpacked_data)

                    test_x = np.array(unpacked_data).reshape(-1, timesteps, datadim)

                    if debugging:
                        for n in range(0,test_x.shape[0]):
                            np.savetxt('/tmp/got_features%i' % n,test_x[n,:,1:])


                    np.savetxt("/tmp/feat_orig", test_x[0,:,:])

                    for n in range(test_x.shape[0]):
                        non_empty_rows = np.where(np.abs(test_x[n,:,:]).sum(-1)>0)[0]
                        print("Non-empty rows for sample %i:" %n)
                        print(non_empty_rows)
                        test_x[n,non_empty_rows,:] = np.log( test_x[n,non_empty_rows,:] + 0.001 )
                        #np.savetxt("/tmp/feat_log", test_x[0,:,:])

                        np.savetxt('/tmp/log_features%i' % n,test_x[n,:,1:])

                        test_x[n,non_empty_rows,:] = ( test_x[n,non_empty_rows,:] - zmean ) / zstd

                    np.savetxt("/tmp/feat_normalised", test_x[0,:,:])

                    test_y = np.zeros(test_x.shape[0]);

                    if debugging:
                        for n in range(0,test_x.shape[0]):
                            np.savetxt('/tmp/norm_features%i' % n,test_x[n,:,1:])


                    #  Classification here!  #

                    teststartmoment= time.clock()

                    #return_data = model.predict(test_x, batch_size=batch_size)
                    [EMB] = sess.run([ prediction ], 
                                   feed_dict={ x: test_x[:,:,1:], 
                                               y: test_y, 
                                               keep_prob: 1 })

                    #return_data = EMB # np.argmax(EMB,2).reshape([-1])

                    print ("EMB shape")
                    print (EMB.shape)             

                    guesses = np.argmax(EMB,1)
                    print ("guesses shape")
                    print (guesses.shape)

                    print ("Classified:")
                    print (guesses)

                    ranking_matrix = np.zeros([45,120])

                    for i in range(test_y.shape[0]):
                        guess = guesses[i]
                        wanted = classes[i]

                        ranking_matrix[ wanted, guess ] += 1

                    if (ranking_matrix).sum(-1).sum(-1) > 0:
                        ranking_matrix /= (ranking_matrix).sum(-1).sum(-1)

                    score = (lsq_weights*ranking_matrix).sum(-1).sum(-1)

                    if score < 1:
                        rounded_score = -2

                    else:
                        rounded_score = round(score)
                        if rounded_score > 5:
                            rounded_score = 5

                    print ("Score %0.2f -> %i" % (score, rounded_score))
                    testtime = time.clock()-teststartmoment

                    # Classification done, let's go back
                    # to handling data transfer:

                    encoded_length=single_integer_packer.pack( len(guesses)+len(classes)+1 )

                    conn.send( encoded_length  )

                    print ("Waiting for acc:")
                    client_ack = conn.recv( acklength )

                    try: 
                        ack = (single_integer_packer.unpack(client_ack))
                        ack_ok = True
                    except:
                        ack_ok = False
                        print ("Something went wrong, let's print stack trace:")
                        traceback.print_exc()
                        conn.close()

                    if ack_ok:

                        print ("client_ack: " + str(ack) )

                        if single_integer_packer.unpack(client_ack)[0] == ackvalue:

                            packable = [rounded_score]
                            for cl in classes:
                                packable.append( cl )
                            for gu in guesses:
                                packable.append( gu )

                            print(packable)

                            result_packer = struct.Struct(len(packable) * 'f')
                            encoded_score_data = result_packer.pack( *packable )
                            print ('Encoded data: "%s"' % binascii.hexlify(encoded_score_data))

                            

                            conn.send(encoded_score_data)

                        conn.close()


                        connectiontime = time.clock() - connectionstartstartmoment

                        print ("Processing took %f s " % (connectiontime))

            sys.stdout.flush()
            


if __name__ == '__main__':

    portfilename=sys.argv[1]
    modelfile=sys.argv[2]
    mean=sys.argv[3]
    std = sys.argv[4]
    lsq =  sys.argv[5]

    run_server_loop(portfilename, modelfile, mean, std, lsq)




