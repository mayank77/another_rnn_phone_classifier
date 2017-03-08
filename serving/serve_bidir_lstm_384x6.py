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
import h5py

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
    'finn': [ 'f', 'i', 'n'],
    'june': [ 'J', 'u', 'n'],
    'age': [ 'E', 'J'],
    'air': [ 'W'],
    'am': [ 'ä', 'm'],
    'and': [ 'ä', 'n', 'd'],
    'ant': [ 'ä', 'n', 't'],
    'apple': [ 'ä', 'p', 'l'],
    'arm': [ 'A', 'm'],
    'art': [ 'A', 't'],
    'at': [ 'ä', 't'],
    'baby': [ 'b', 'E', 'b', 'I'],
    'back': [ 'b', 'ä', 'k'],
    'bad': [ 'b', 'ä', 'd'],
    'bag': [ 'b', 'ä', 'g'],
    'bake': [ 'b', 'E', 'k'],
    'ball': [ 'b', 'O', 'l'],
    'bark': [ 'b', 'A', 'k'],
    'baseball': [ 'b', 'E', 's', 'b', 'O', 'l'],
    'basketball': [ 'b', 'A', 's', 'k', 'i', 't', 'b', 'O', 'l'],
    'bath': [ 'b', 'A', 'T'],
    'be': [ 'b', 'I'],
    'bear': [ 'b', 'W'],
    'bed': [ 'b', 'e', 'd'],
    'bedroom': [ 'b', 'e', 'd', 'r', 'u', 'm'],
    'bee': [ 'b', 'I'],
    'belt': [ 'b', 'e', 'l', 't'],
    'berry': [ 'b', 'e', 'r', 'I'],
    'best': [ 'b', 'e', 's', 't'],
    'big': [ 'b', 'i', 'g'],
    'bird': [ 'b', 'ö', 'd'],
    'black': [ 'b', 'l', 'ä', 'k'],
    'blue': [ 'b', 'l', 'u'],
    'body': [ 'b', 'Y', 'd', 'I'],
    'book': [ 'b', 'U', 'k'],
    'boy': [ 'b', 'O', 'I'],
    'bread': [ 'b', 'r', 'e', 'd'],
    'breakfast': [ 'b', 'r', 'e', 'k', 'f', 'Q', 's', 't'],
    'brother': [ 'b', 'r', 'a', 'D', 'Q'],
    'brown': [ 'b', 'r', 'å', 'n'],
    'bug': [ 'b', 'a', 'g'],
    'bush': [ 'b', 'U', 'S'],
    'butter': [ 'b', 'a', 't', 'Q'],
    'butterfly': [ 'b', 'a', 't', 'Q', 'f', 'l', 'Ä'],
    'buy': [ 'b', 'Ä'],
    'bye': [ 'b', 'Ä'],
    'bye_bye': [ 'b', 'Ä', 'b', 'Ä'],
    'cake': [ 'k', 'E', 'k'],
    'candy': [ 'k', 'ä', 'n', 'd', 'I'],
    'cap': [ 'k', 'ä', 'p'],
    'carpet': [ 'k', 'A', 'p', 'i', 't'],
    'carrot': [ 'k', 'ä', 'r', 'Q', 't'],
    'cat': [ 'k', 'ä', 't'],
    'chair': [ 'C', 'W'],
    'chat': [ 'C', 'ä', 't'],
    'cheap': [ 'C', 'I', 'p'],
    'cheek': [ 'C', 'I', 'k'],
    'cheese': [ 'C', 'I', 'z'],
    'cherry': [ 'C', 'e', 'r', 'I'],
    'chew': [ 'C', 'u'],
    'chick': [ 'C', 'i', 'k'],
    'chicken': [ 'C', 'i', 'k', 'i', 'n'],
    'child': [ 'C', 'Ä', 'l', 'd'],
    'children': [ 'C', 'i', 'l', 'd', 'r', 'Q', 'n'],
    'chin': [ 'C', 'i', 'n'],
    'chip': [ 'C', 'i', 'p'],
    'chocolate': [ 'C', 'Y', 'k', 'l', 'Q', 't'],
    'choose': [ 'C', 'u', 'z'],
    'city': [ 's', 'i', 't', 'I'],
    'classmate': [ 'k', 'l', 'A', 's', 'm', 'E', 't'],
    'close': [ 'k', 'l', 'o', 's'],
    'clothe': [ 'k', 'l', 'o', 'D'],
    'clothes': [ 'k', 'l', 'o', 'D', 'z'],
    'coat': [ 'k', 'o', 't'],
    'cold': [ 'k', 'o', 'l', 'd'],
    'colour': [ 'k', 'a', 'l', 'Q'],
    'come': [ 'k', 'a', 'm'],
    'come_here': [ 'k', 'a', 'm', 'H', 'R'],
    'come_on': [ 'k', 'a', 'm', 'Q', 'n'],
    'cook': [ 'k', 'U', 'k'],
    'cool': [ 'k', 'u', 'l'],
    'country': [ 'k', 'a', 'n', 't', 'r', 'I'],
    'cow': [ 'k', 'å'],
    'cucumber': [ 'k', 'j', 'u', 'k', 'a', 'm', 'b', 'Q'],
    'cup': [ 'k', 'a', 'p'],
    'curl': [ 'k', 'ö', 'l'],
    'cut': [ 'k', 'a', 't'],
    'dark': [ 'd', 'A', 'k'],
    'day': [ 'd', 'E'],
    'deer': [ 'd', 'R'],
    'dessert': [ 'd', 'i', 'z', 'ö', 't'],
    'do': [ 'd', 'u'],
    'do_you_want_to_play_with_us': [ 'd', 'u', 'j', 'Q', 'w', 'Y', 'n', 't', 't', 'Q', 'p', 'l', 'E', 'w', 'i', 'D', 'a', 's'],
    'dog': [ 'd', 'Y', 'g'],
    'dolphin': [ 'd', 'Y', 'l', 'f', 'i', 'n'],
    'don_t': [ 'd', 'Y', 'n', 't', 'I'],
    'door': [ 'd', 'O'],
    'draw': [ 'd', 'r', 'O'],
    'dream': [ 'd', 'r', 'I', 'm'],
    'drink': [ 'd', 'r', 'i', 'N', 'k'],
    'dry': [ 'd', 'r', 'Ä'],
    'duck': [ 'd', 'a', 'k'],
    'ear': [ 'R'],
    'eat': [ 'e', 't'],
    'eight': [ 'E', 't'],
    'elbow': [ 'e', 'l', 'b', 'o'],
    'elephant': [ 'e', 'l', 'i', 'f', 'Q', 'n', 't'],
    'elk': [ 'e', 'l', 'k'],
    'england': [ 'i', 'N', 'g', 'l', 'Q', 'n', 'd'],
    'english': [ 'i', 'N', 'g', 'l', 'i', 'S'],
    'europe': [ 'j', 'P', 'r', 'Q', 'p'],
    'evening': [ 'I', 'v', 'n', 'i', 'N'],
    'eye': [ 'Ä'],
    'eyes': [ 'Ä', 'z'],
    'face': [ 'f', 'E', 's'],
    'fall': [ 'f', 'O', 'l'],
    'family': [ 'f', 'ä', 'm', 'Q', 'l', 'I'],
    'fan': [ 'f', 'ä', 'n'],
    'farmer': [ 'f', 'A', 'm', 'Q'],
    'fat': [ 'f', 'ä', 't'],
    'father': [ 'f', 'A', 'D', 'Q'],
    'feather': [ 'f', 'e', 'D', 'Q'],
    'feel': [ 'f', 'I', 'l'],
    'feet': [ 'f', 'I', 't'],
    'field': [ 'f', 'I', 'l', 'd'],
    'fin': [ 'f', 'i', 'n'],
    'fine': [ 'f', 'Ä', 'n'],
    'finger': [ 'f', 'i', 'N', 'g', 'Q'],
    'finland': [ 'f', 'i', 'n', 'l', 'Q', 'n', 'd'],
    'finn': [ 'f', 'i', 'n'],
    'finnish': [ 'f', 'i', 'n', 'i', 'S'],
    'first': [ 'f', 'ö', 's', 't'],
    'fish': [ 'f', 'i', 'S'],
    'fit': [ 'f', 'i', 't'],
    'flower': [ 'f', 'l', 'å', 'Q'],
    'food': [ 'f', 'u', 'd'],
    'forest': [ 'f', 'Y', 'r', 'i', 's', 't'],
    'fork': [ 'f', 'O', 'k'],
    'fox': [ 'f', 'Y', 'k', 's'],
    'friend': [ 'f', 'r', 'e', 'n', 'd'],
    'full': [ 'f', 'U', 'l'],
    'fun': [ 'f', 'a', 'n'],
    'fur': [ 'f', 'ö'],
    'game': [ 'g', 'E', 'm'],
    'garage': [ 'g', 'ä', 'r', 'A', 'Z'],
    'girl': [ 'g', 'ö', 'l'],
    'glass': [ 'g', 'l', 'A', 's'],
    'goal': [ 'g', 'o', 'l'],
    'goat': [ 'g', 'o', 't'],
    'gold': [ 'g', 'o', 'l', 'd'],
    'good': [ 'g', 'U', 'd'],
    'good_evening': [ 'g', 'U', 'd', 'I', 'v', 'n', 'i', 'N'],
    'good_morning': [ 'g', 'U', 'd', 'm', 'O', 'n', 'i', 'N'],
    'good_night': [ 'g', 'U', 'd', 'n', 'Ä', 't'],
    'grass': [ 'g', 'r', 'A', 's'],
    'green': [ 'g', 'r', 'I', 'n'],
    'grey': [ 'g', 'r', 'E'],
    'hair': [ 'H', 'W'],
    'hairy': [ 'H', 'W', 'r', 'I'],
    'hand': [ 'H', 'ä', 'n', 'd'],
    'hard': [ 'H', 'A', 'd'],
    'hare': [ 'H', 'W'],
    'hat': [ 'H', 'ä', 't'],
    'hay': [ 'H', 'E'],
    'he_goes': [ 'H', 'I', 'g', 'o', 'z'],
    'healthy': [ 'H', 'e', 'l', 'T', 'I'],
    'hear': [ 'H', 'R'],
    'heart': [ 'H', 'A', 't'],
    'hello': [ 'H', 'Q', 'l', 'o'],
    'hen': [ 'H', 'e', 'n'],
    'here': [ 'H', 'R'],
    'here_you_are': [ 'H', 'R', 'j', 'Q', 'A'],
    'high': [ 'H', 'Ä'],
    'home': [ 'H', 'o', 'm'],
    'horse': [ 'H', 'O', 's'],
    'house': [ 'H', 'å', 's'],
    'hurry_up': [ 'H', 'a', 'r', 'I', 'a', 'p'],
    'i_am': [ 'Ä', 'ä', 'm'],
    'i_am_sorry': [ 'Ä', 'ä', 'm', 's', 'Y', 'r', 'I'],
    'i_go': [ 'Ä', 'g', 'o'],
    'i_like': [ 'Ä', 'l', 'Ä', 'k'],
    'i_live_in_finland': [ 'Ä', 'l', 'i', 'v', 'i', 'n', 'f', 'i', 'n', 'l', 'Q', 'n', 'd'],
    'i_speak_finnish': [ 'Ä', 's', 'p', 'I', 'k', 'f', 'i', 'n', 'i', 'S'],
    'i_was': [ 'Ä', 'w', 'Q', 'z'],
    'ice': [ 'Ä', 's'],
    'ice_hockey': [ 'Ä', 's', 'H', 'Y', 'k', 'I'],
    'ice_skating': [ 'Ä', 's', 's', 'k', 'E', 't', 'i', 'N'],
    'it': [ 'i', 't'],
    'jam': [ 'J', 'ä', 'm'],
    'jaw': [ 'J', 'O'],
    'jet': [ 'J', 'e', 't'],
    'juice': [ 'J', 'u', 's'],
    'jump': [ 'J', 'a', 'm', 'p'],
    'june': [ 'J', 'u', 'n'],
    'kitchen': [ 'k', 'i', 'C', 'i', 'n'],
    'knee': [ 'n', 'I'],
    'knife': [ 'n', 'Ä', 'f'],
    'lake': [ 'l', 'E', 'k'],
    'lamp': [ 'l', 'ä', 'm', 'p'],
    'language': [ 'l', 'ä', 'N', 'g', 'w', 'i', 'J'],
    'lead': [ 'l', 'e', 'd'],
    'leaf': [ 'l', 'I', 'f'],
    'learn': [ 'l', 'ö', 'n'],
    'leave': [ 'l', 'I', 'v'],
    'legs': [ 'l', 'e', 'g', 'z'],
    'lemon': [ 'l', 'e', 'm', 'Q', 'n'],
    'lets_go': [ 'l', 'e', 't', 's', 'g', 'o'],
    'lick': [ 'l', 'i', 'k'],
    'life': [ 'l', 'Ä', 'f'],
    'light': [ 'l', 'Ä', 't'],
    'lion': [ 'l', 'Ä', 'Q', 'n'],
    'lips': [ 'l', 'i', 'p', 's'],
    'listen': [ 'l', 'i', 's', 'n'],
    'live': [ 'l', 'Ä', 'v'],
    'living_room': [ 'l', 'i', 'v', 'i', 'N', 'r', 'u', 'm'],
    'lock': [ 'l', 'Y', 'k'],
    'log': [ 'l', 'Y', 'g'],
    'long': [ 'l', 'Y', 'N'],
    'look': [ 'l', 'U', 'k'],
    'loud': [ 'l', 'å', 'd'],
    'love': [ 'l', 'a', 'v'],
    'low': [ 'l', 'o'],
    'lunch': [ 'l', 'a', 'n', 'C'],
    'make': [ 'm', 'E', 'k'],
    'man': [ 'm', 'ä', 'n'],
    'maths': [ 'm', 'ä', 'T', 's'],
    'mats': [ 'm', 'ä', 't', 's'],
    'meat': [ 'm', 'I', 't'],
    'milk': [ 'm', 'i', 'l', 'k'],
    'money': [ 'm', 'a', 'n', 'I'],
    'monkey': [ 'm', 'a', 'N', 'k', 'I'],
    'month': [ 'm', 'a', 'n', 'T'],
    'moon': [ 'm', 'u', 'n'],
    'moose': [ 'm', 'u', 's'],
    'more': [ 'm', 'O'],
    'morning': [ 'm', 'O', 'n', 'i', 'N'],
    'mother': [ 'm', 'a', 'D', 'Q'],
    'mouse': [ 'm', 'å', 's'],
    'mouth': [ 'm', 'å', 'D'],
    'move': [ 'm', 'u', 'v'],
    'movie': [ 'm', 'u', 'v', 'I'],
    'music': [ 'm', 'j', 'u', 'z', 'i', 'k'],
    'my_name_is': [ 'm', 'Ä', 'n', 'E', 'm', 'i', 'z'],
    'neck': [ 'n', 'e', 'k'],
    'neighbour': [ 'n', 'E', 'b', 'Q'],
    'nice_to_meet_you': [ 'n', 'Ä', 's', 't', 'Q', 'm', 'I', 't', 'j', 'Q'],
    'night': [ 'n', 'Ä', 't'],
    'no_thank_you': [ 'n', 'o', 'T', 'ä', 'N', 'k', 'j', 'Q'],
    'north': [ 'n', 'O', 'T'],
    'nose': [ 'n', 'o', 'z'],
    'now': [ 'n', 'å'],
    'of': [ 'Q', 'v'],
    'off': [ 'O', 'f'],
    'old': [ 'o', 'l', 'd'],
    'once': [ 'w', 'a', 'n', 's'],
    'one': [ 'w', 'a', 'n'],
    'ones': [ 'w', 'a', 'n', 'z'],
    'orange': [ 'Y', 'r', 'i', 'n', 'J'],
    'owl': [ 'å', 'l'],
    'pack': [ 'p', 'ä', 'k'],
    'page': [ 'p', 'E', 'J'],
    'paint': [ 'p', 'E', 'n', 't'],
    'parents': [ 'p', 'W', 'r', 'Q', 'n', 't', 's'],
    'park': [ 'p', 'A', 'k'],
    'parrot': [ 'p', 'ä', 'r', 'Q', 't'],
    'party': [ 'p', 'A', 't', 'I'],
    'pays': [ 'p', 'E', 'z'],
    'pea': [ 'p', 'I'],
    'pear': [ 'p', 'W'],
    'peas': [ 'p', 'I', 'z'],
    'pepper': [ 'p', 'e', 'p', 'Q'],
    'pet': [ 'p', 'e', 't'],
    'phone': [ 'f', 'o', 'n'],
    'pie': [ 'p', 'Ä'],
    'piece': [ 'p', 'I', 's'],
    'pig': [ 'p', 'i', 'g'],
    'pink': [ 'p', 'i', 'N', 'k'],
    'plant': [ 'p', 'l', 'A', 'n', 't'],
    'plate': [ 'p', 'l', 'E', 't'],
    'play': [ 'p', 'l', 'E'],
    'please': [ 'p', 'l', 'I', 'z'],
    'pool': [ 'p', 'u', 'l'],
    'potato': [ 'p', 'Q', 't', 'E', 't', 'o'],
    'pull': [ 'p', 'U', 'l'],
    'purple': [ 'p', 'ö', 'p', 'l'],
    'quiet': [ 'k', 'w', 'Ä', 'Q', 't'],
    'rat': [ 'r', 'ä', 't'],
    'read': [ 'r', 'e', 'd'],
    'real': [ 'r', 'E', 'A', 'l'],
    'red': [ 'r', 'e', 'd'],
    'rest': [ 'r', 'e', 's', 't'],
    'ride': [ 'r', 'Ä', 'd'],
    'right': [ 'r', 'Ä', 't'],
    'river': [ 'r', 'i', 'v', 'Q'],
    'rock': [ 'r', 'Y', 'k'],
    'room': [ 'r', 'u', 'm'],
    'round': [ 'r', 'å', 'n', 'd'],
    'run': [ 'r', 'a', 'n'],
    'safe': [ 's', 'E', 'f'],
    'salad': [ 's', 'ä', 'l', 'Q', 'd'],
    'salt': [ 's', 'O', 'l', 't'],
    'sand': [ 's', 'ä', 'n', 'd'],
    'save': [ 's', 'E', 'v'],
    'say': [ 's', 'E'],
    'school': [ 's', 'k', 'u', 'l'],
    'sea': [ 's', 'I'],
    'seal': [ 's', 'I', 'l'],
    'seat': [ 's', 'I', 't'],
    'seed': [ 's', 'I', 'd'],
    'sees': [ 's', 'I', 'z'],
    'set': [ 's', 'e', 't'],
    'she': [ 'S', 'I'],
    'she_is': [ 'S', 'I', 'i', 'z'],
    'she_likes': [ 'S', 'I', 'l', 'Ä', 'k', 's'],
    'sheep': [ 'S', 'I', 'p'],
    'sheet': [ 'S', 'I', 't'],
    'shine': [ 'S', 'Ä', 'n'],
    'ship': [ 'S', 'i', 'p'],
    'shirt': [ 'S', 'ö', 't'],
    'shoe': [ 'S', 'u'],
    'shop': [ 'S', 'Y', 'p'],
    'shoulder': [ 'S', 'o', 'l', 'd', 'Q'],
    'shower': [ 'S', 'å', 'Q'],
    'shows': [ 'S', 'o', 'z'],
    'sing': [ 's', 'i', 'N'],
    'sister': [ 's', 'i', 's', 't', 'Q'],
    'sit': [ 's', 'i', 't'],
    'ski': [ 's', 'k', 'I'],
    'sleep': [ 's', 'l', 'I', 'p'],
    'smell': [ 's', 'm', 'e', 'l'],
    'snack': [ 's', 'n', 'ä', 'k'],
    'snake': [ 's', 'n', 'E', 'k'],
    'socks': [ 's', 'Y', 'k', 's'],
    'soft': [ 's', 'Y', 'f', 't'],
    'soon': [ 's', 'u', 'n'],
    'sorry': [ 's', 'Y', 'r', 'I'],
    'sound': [ 's', 'å', 'n', 'd'],
    'soup': [ 's', 'u', 'p'],
    'speak': [ 's', 'p', 'I', 'k'],
    'spoon': [ 's', 'p', 'u', 'n'],
    'squirrel': [ 's', 'k', 'w', 'i', 'r', 'Q', 'l'],
    'strawberry': [ 's', 't', 'r', 'O', 'b', 'r', 'I'],
    'sun': [ 's', 'a', 'n'],
    'swedish': [ 's', 'w', 'I', 'd', 'i', 'S'],
    'sweet': [ 's', 'w', 'I', 't'],
    'swim': [ 's', 'w', 'i', 'm'],
    'table': [ 't', 'A', 'b', 'l'],
    'taste': [ 't', 'E', 's', 't'],
    'teacher': [ 't', 'I', 'C', 'Q'],
    'teas': [ 't', 'I', 'z'],
    'teeth': [ 't', 'I', 'T'],
    'ten': [ 't', 'e', 'n'],
    'than': [ 'D', 'ä', 'n'],
    'thank_you': [ 'T', 'ä', 'N', 'k', 'j', 'Q'],
    'that': [ 'D', 'ä', 't'],
    'the': [ 'D', 'Q'],
    'theatre': [ 'T', 'R', 't', 'Q'],
    'their': [ 'D', 'W'],
    'then': [ 'D', 'e', 'n'],
    'there': [ 'D', 'W'],
    'these': [ 'D', 'I', 'z'],
    'they': [ 'D', 'E'],
    'thick': [ 'T', 'i', 'k'],
    'thin': [ 'T', 'i', 'n'],
    'thing': [ 'T', 'i', 'N'],
    'think': [ 'T', 'i', 'N', 'k'],
    'thirst': [ 'T', 'ö', 's', 't'],
    'thirsty': [ 'T', 'ö', 's', 't', 'I'],
    'this': [ 'D', 'i', 's'],
    'those': [ 'D', 'o', 'z'],
    'three': [ 'T', 'r', 'I'],
    'throw': [ 'T', 'r', 'o'],
    'time': [ 't', 'Ä', 'm'],
    'toast': [ 't', 'o', 's', 't'],
    'today': [ 't', 'Q', 'd', 'E'],
    'toe': [ 't', 'o'],
    'toilet': [ 't', 'Å', 'l', 'i', 't'],
    'tomato': [ 't', 'Q', 'm', 'A', 't', 'o'],
    'tomorrow': [ 't', 'Q', 'm', 'Y', 'r', 'o'],
    'tongue': [ 't', 'a', 'N'],
    'too': [ 't', 'u'],
    'tortoise': [ 't', 'O', 't', 'Q', 's'],
    'town': [ 't', 'å', 'n'],
    'tree': [ 't', 'r', 'I'],
    'two': [ 't', 'u'],
    'use': [ 'j', 'u', 's'],
    'van': [ 'v', 'ä', 'n'],
    'vest': [ 'v', 'e', 's', 't'],
    'vet': [ 'v', 'e', 't'],
    'village': [ 'v', 'i', 'l', 'i', 'J'],
    'walk': [ 'w', 'O', 'k'],
    'wall': [ 'w', 'O', 'l'],
    'want': [ 'w', 'Y', 'n', 't'],
    'was': [ 'w', 'Q', 'z'],
    'wash': [ 'w', 'Y', 'S'],
    'watch': [ 'w', 'Y', 'C'],
    'water': [ 'w', 'O', 't', 'Q'],
    'we_go': [ 'w', 'I', 'g', 'o'],
    'we_were': [ 'w', 'I', 'w', 'ö'],
    'wear': [ 'w', 'W'],
    'week': [ 'w', 'I', 'k'],
    'well_done': [ 'w', 'e', 'l', 'd', 'a', 'n'],
    'were': [ 'w', 'ö'],
    'west': [ 'w', 'e', 's', 't'],
    'wet': [ 'w', 'e', 't'],
    'what': [ 'w', 'Y', 't'],
    'wheel': [ 'w', 'I', 'l'],
    'where': [ 'w', 'W'],
    'which': [ 'w', 'i', 'C'],
    'white': [ 'w', 'Ä', 't'],
    'who': [ 'H', 'u'],
    'why': [ 'w', 'Ä'],
    'win': [ 'w', 'i', 'n'],
    'window': [ 'w', 'i', 'n', 'd', 'o'],
    'wish': [ 'w', 'i', 'S'],
    'with': [ 'w', 'i', 'D'],
    'woman': [ 'w', 'U', 'm', 'Q', 'n'],
    'wool': [ 'w', 'U', 'l'],
    'worm': [ 'w', 'ö', 'm'],
    'would_you_like_some_juice': [ 'w', 'Q', 'd', 'j', 'Q', 'l', 'Ä', 'k', 's', 'a', 'm', 'J', 'u', 's'],
    'would_you_like_some_more': [ 'w', 'Q', 'd', 'j', 'Q', 'l', 'Ä', 'k', 's', 'a', 'm', 'm', 'O'],
    'write': [ 'r', 'Ä', 't'],
    'wrong': [ 'r', 'Y', 'N'],
    'year': [ 'j', 'ö'],
    'yellow': [ 'j', 'e', 'l', 'o'],
    'yes_please': [ 'j', 'e', 's', 'p', 'l', 'I', 'z'],
    'yesterday': [ 'j', 'e', 's', 't', 'Q', 'd', 'E'],
    'yet': [ 'j', 'e', 't'],
    'you_are': [ 'j', 'Q', 'A'],
    'you_go': [ 'j', 'Q', 'g', 'o'],
    'you_re_welcome': [ 'j', 'Q', 'r', 'E', 'w', 'e', 'l', 'k', 'Q', 'm'],
    'you_were': [ 'j', 'Q', 'w', 'ö'],
    'young': [ 'j', 'a', 'N'],
    'your': [ 'j', 'O'],
    'zoo': [ 'z', 'u'],
    'zoos': [ 'z', 'u', 'z']  }



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



        print ( "Load normalisation stats fom h5py..." )
        
        zmean = np.loadtxt( meanfile ) #traindata.mean
        zstd = np.loadtxt( stdfile )# traindata.std

        lsqweights = np.loadtxt( lsqweightfile )

        loadtime = time.clock()-loadstartmoment
        print ("Loading took %0.1f seconds!"%loadtime)

        
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

            
            print ('Word to be classified:')


            datalen = conn.recv(single_integer_packer.size)
            if not datalen: break
            
            print ("Going to read ascii data of length %i" % unpacked_datalen)

            char_packer = struct.Struct( unpacked_datalen[0]*'c')   
            conn.send( ackpacket )


            data = conn.recv(char_packer.size)
            if not data: break

            try:
                word = char_packer.unpack( data )
                word_data_ok = True           
            except:
                print ("Something went wrong, let's print stack trace:")
                traceback.print_exc()
                conn.close()
                word_data_ok = False
            

            phones = words[word]

            if phone not in words.keys():
                print ("Word not in dictionary: Closing connection")
                conn.close()
                word_data_ok = False
                continue
            
            classes = [ classdict[phone]['class'] for phone in phones  ]

            conn.send( ackpacket )


            print ('Will now classify using models %s' % modelfile)

            datalen = conn.recv(single_integer_packer.size)
            if not datalen: break


            unpacked_datalen = single_integer_packer.unpack(datalen)

            print ("Going to read float data of length %i" % unpacked_datalen)

            float_packer = struct.Struct( unpacked_datalen[0]*'f')   
            conn.send( ackpacket )



            data = conn.recv(float_packer.size)
            if not data: break

            try:
                unpacked_data = float_packer.unpack( data )
                unpacked_data_ok = True
            except:
                print ("Something went wrong, let's print stack trace:")
                traceback.print_exc()
                conn.close()
                unpacked_data_ok = False
                continue

            if unpacked_data_ok:

                featdim=len(unpacked_data)

                test_x = np.array(unpacked_data).reshape(-1, timesteps, datadim)

                if debugging:
                    for n in range(0,test_x.shape[0]):
                        np.savetxt('/tmp/got_features%i' % n,test_x[n,:,:])



                test_x = (test_x-norm_means)/norm_stds
                test_y = np.zeros(test_x.shape[0]);

                if debugging:
                    for n in range(0,test_x.shape[0]):
                        np.savetxt('/tmp/norm_features%i' % n,test_x[n,:,:])


                #  Classification here!  #

                teststartmoment= time.clock()

                #return_data = model.predict(test_x, batch_size=batch_size)
                EMB = sess.run([ prediction ], 
                               feed_dict={ x: test_x, 
                                           y: test_y, 
                                           keep_prob: 1 })

                #return_data = EMB # np.argmax(EMB,2).reshape([-1])

                guesses = np.argmax(EMB,1)

                print ("Classified:")
                print (guesses)

                ranking_matrix = np.zeros([45,120])

                for i in range(EMB.shape[0]):
                    guess = guesses[i]
                    wanted = classes[i]
                    
                    ranking_matrix[ wanted, guess ] += 1

                if (ranking_matrix).sum(-1).sum(-1) > 0:
                    ranking_matrix /= (ranking_matrix).sum(-1).sum(-1)
                    
                score = (lsq_weights*ranking_matrix).sum(-1).sum(-1)

                testtime = time.clock()-teststartmoment

                # Classification done, let's go back
                # to handling data transfer:

                encoded_length=single_integer_packer.pack( 1 )

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
                        result_packer = struct.Struct((len(guesses)+1) * 'f')
                        encoded_score_data = single_result_packer.pack(*([score], guesses.tolist()))
                        #print 'Encoded data: "%s"' % binascii.hexlify(encoded_data)

                        conn.send(encoded_score_data)
                        conn.send(encoded_class_data)

                    conn.close()


                    connectiontime = time.clock() - connectionstartstartmoment

                    print ("Processing took %f s " % (connectiontime))

            sys.stdout.flush()


if __name__ == '__main__':

    portfilename=sys.argv[1]
    model_arch=sys.argv[2]
    model_weights=sys.argv[3]
    model_norm = sys.argv[4]
    lsq =  sys.argv[5]

    run_server_loop(portfilename, modelfile, mean, std, lsq)




