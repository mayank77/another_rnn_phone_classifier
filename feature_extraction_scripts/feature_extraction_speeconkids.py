
# coding: utf-8

# In[451]:

#!/usr/bin/env python



# This file is (was?) in /l/rkarhila/speecon_wsj_phoneme_dnn/data_preprocessing

#
#  1. Divide each data file into single phoneme chunks based on aliged labels
#
#  2. Run the chunks through feature extraction shell script
#
#  3. Store the features and their associated phoneme information in arrays
#
#  4. Pickle for future normalisation (with other corpora) 
#

import io
import os
import numpy as np
from subprocess import Popen, PIPE, STDOUT
import re
import math 
import struct
import time
import sys
import struct
import random

from dnnutil import process_pfstar_label
from dnnutil import get_labelstring
from dnnutil import chop_features
from dnnutil import get_features
from dnnutil import extract_collection_and_save
#
# Use some funky structure from tensor flow to store 3d-matrices of variable length more compactly.
#
import tensorflow as tf

#
# A function that will be useful:
#

def mkdir(path):
    try:
        os.makedirs(path)        
    except OSError as exc:  # Python >2.5
        #print ("dir %s exists" % path)
        dummy = 1
#
# Some more output?
#

debug=True




# In[452]:

preprocessing_scripts = {'none' :{'script': '../feature_extraction_scripts/preprocess_pfstar.sh', 'name' : 'clean', 'parameters': [[0,0], [0,0]] },
                         'overdrive' : {'script': '../feature_extraction_scripts/preprocess_pfstar_and_overdrive.sh', 'name' : 'overdrive', 'parameters': [[1,10], [-20,0]] },
                         'underdrive' : {'script': '../feature_extraction_scripts/preprocess_pfstar_and_overdrive.sh', 'name' : 'underdrive', 'parameters': [[-40,-20], [0,0]] },
                         'babble' : {'script': '../feature_extraction_scripts/preprocess_pfstar_and_add_babble.sh', 'name' : 'babbled', 'parameters': [[-40,-20],[-20,0]] },
                         'humming' : {'script': '../feature_extraction_scripts/preprocess_pfstar_and_add_humming.sh', 'name' : 'volvo', 'parameters': [[-30,-20],[-20,0]] } }

#feature_extraction_script = '../feature_extraction_scripts/extract_5500hz_spec_with_start_end.sh'
feature_extraction_script = '../feature_extraction_scripts/extract_8000hz_mspec_with_start_end.sh'
featuretype = "mspec_and_f0"

global quality_control_wavdir
quality_control_wavdir = ""
global statistics_handle
statistics_handle = ""

# In[453]:


# For AaltoASR UK English:
vowels = ['a','A','å','Å','ä','Ä','e','E','f','i','I','o','O','ö','u','U']

nonvow = ['b','C','d','D','g','H','j','J','k','l','m','n','N','p','P','Q','r','R','s','S','t','T','v','w','W','Y','z','Z']

combinations = []

# For Finnish speecon:

vowels = [ 'a', 'ä', 'e', 'i', 'o', 'ö', 'u', 'y' ]

nonvow = [ 'd', 'f', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v']

combinations = ['aa','ai','ao','ae',
                'au','ea','ee','ei','eo','eu','ey','eä','ia','ie','ii',
                'io','iu','iy','iä','iö','oa','oe','oi','oo','ou','ua','ue',
                'ui','uo','uu','yi','yy','yä','yö','äe','äi','äy','ää','äö',
                'öi','öy','öä','öö','ng','nn','mm','kk','pp','hh','ll','pp',
                'rr','ss','tt' ]

used_classes = vowels+nonvow+combinations
classes_name = all


used_classes = vowels+nonvow+combinations

#classes_name = "mc_en_uk_all"
classes_name = "mc_fi_fi_all"


#
# Settings for feature extraction:
#

datatypelength = 2 # 16 bits = 2 bytes, no?


# For 16 kHz samples:

audio_fs = 16000
'''
frame_length = 400
frame_step = 128
'''


padding_array = bytearray()

progress_length = 80

max_num_samples=8000 # 0.5 should be enough for any reasonable phoneme, right?

max_num_classes = 10000

feature_dimension=130
#feature_dimension=30

'''
# For 8 kHz samples:
'''
global fs

feature_fs = 8000
fs = feature_fs
frame_length = 256
frame_step = 64

frame_leftovers = frame_length-frame_step



max_num_frames=50
max_phone_length=max_num_frames * 128 # ( 128 being the frame step in alignments)

max_num_monoclasses = 200
max_num_monoclasses = 9


#max_num_samples=100160
assigned_num_samples=100

global tmpfilecounter
tmpfilecounter = 0

# tmp directory for feature extraction.
# This should reside in memory (tempfs or whatever it's called, often under /dev/shm/)

tmp_dir="/dev/shm/siak-feat-extract-python-"+str(time.time())
try:
    os.makedirs(tmp_dir)
except OSError as exc:  # Python >2.5
    if exc.errno == errno.EEXIST and os.path.isdir(tmp_dir):
        pass
    else:
        raise   


print ('using tmp dir %s' % tmp_dir)


# ## Classes and probabilities ##
# 

# In[454]:

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


# ##  Dataset definitions ##
# *In a very awkward manner, we'll specify some local files that contain list of audio and transcription files*

# In[455]:


#
#   Data collection defitinions - train, dev and eval sets:
#


corpus = "en_uk_kids_align_from_clean"
pickle_dir='../features/work_in_progress/'+corpus+'/pickles'
statistics_dir = '../features/work_in_progress/'+corpus+'/statistics/'

collections = [          
    { 'name' : 'train-0',
      'recipe' : '/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/recipe.speakers.train.00',
      'condition' : 'clean',
      'numlines': 878 },
    { 'name' : 'train-1',
      'recipe' : '/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/recipe.speakers.train.01',
      'condition' : 'clean',
      'numlines': 1083 },
    { 'name' : 'train-2',
      'recipe' : '/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/recipe.speakers.train.02',
      'condition' : 'clean',
      'numlines': 946 },
    { 'name' : 'train-3',
      'recipe' : '/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/recipe.speakers.train.03',
      'condition' : 'clean',
      'numlines': 870 },
    { 'name' : 'train-4',
      'recipe' : '/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/recipe.speakers.train.04',
      'condition' : 'clean',
      'numlines': 651 },
    { 'name' : 'train-5',
      'recipe' : '/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/recipe.speakers.train.05',
      'condition' : 'clean',
      'numlines': 785},
    { 'name' : 'train-6',
      'recipe' : '/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/recipe.speakers.train.06',
      'condition' : 'clean',
      'numlines': 699 },
    { 'name' : 'train-7',
      'recipe' : '/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/recipe.speakers.train.07',
      'condition' : 'clean',
      'numlines': 699 },
    { 'name' : 'test-0',
      'recipe' : '/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/recipe.speakers.test.00',
      'condition' : 'clean',
      'numlines': 852 },
    { 'name' : 'test-1',
      'recipe' : '/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/recipe.speakers.test.01',
      'condition' : 'clean',
      'numlines': 752 },
    { 'name' : 'test-2',
      'recipe' : '/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/recipe.speakers.test.02',
      'condition' : 'clean',
      'numlines': 594 },
    { 'name' : 'test-3',
      'recipe' : '/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/recipe.speakers.test.03',
      'condition' : 'clean',
      'numlines': 758 },
    { 'name' : 'test-4',
      'recipe' : '/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/recipe.speakers.test.04',
      'condition' : 'clean',
      'numlines': 734 },
    { 'name' : 'eval-1',
      'recipe' : '/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/recipe.speakers.test.05',
      'condition' : 'clean',
      'numlines': 393},
    { 'name' : 'eval-0',
      'recipe' : '/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/recipe.speakers.eval.00',
      'condition' : 'clean',
      'numlines': 837 }
]


featdim1 = -1;
featdim2 = -1;

means_set = False
means = -1;
stds = -1;
new_pickle_dir = "-1"

classes = {}




# In[483]:

print ("start!")

debug=True
debug=False
#print (preprocessing_scripts)
discard_counter=0

for collection in collections:
    extract_collection_and_save(collection)


# In[481]:

print(collection['name'] )


# ## Pickle/save ##
# 
# Next we'll save the audio data into variable length tensor flow thingies:

# In[ ]:



