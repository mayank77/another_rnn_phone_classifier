
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

from dnnutil import preprocessing
#from dnnutil import get_labelstring
#from dnnutil import chop_features
#from dnnutil import get_features
#from dnnutil import extract_collection_and_save

get_labelstring = preprocessing.get_labelstring
chop_features =  preprocessing.chop_features
get_features = preprocessing.get_features
extract_collection_and_save = preprocessing.extract_collection_and_save
#
#
# Use some funky structure from tensor flow to store 3d-matrices of variable length more compactly.
#
#import tensorflow as tf#
# Some more output?
#


class extract_config:
    debug = False
    preprocessing_scipts = {}
    feature_extraction_script = ""
    featuretype = ""
    quality_control_wavdir = ""
    statistics_handle = ""
    vowels = []
    nonvow = []
    combinations = []
    used_classes = []
    classes_name = ""
    datatypelength = -1
    audio_fs = -1
    progress_length = -1
    max_num_samples=-1
    max_num_classes =-1
    feature_dimension=-1


    feature_fs = -1
    fs = -1
    frame_length = -1 
    frame_step = -1
    frame_leftovers = -1 
    
    max_num_frames=-1
    max_phone_length=-1
    max_num_monoclasses = -1
    
    assigned_num_samples=100
    
    discard_counter=0
    tmpfilecounter = 0
    tmp_dir="/tmp/"

    class_def={}
    corpus = ""
    pickle_dir=''
    statistics_dir = ''
    
    lastframeindex = 0
    extraframes = 4
    

    def __init__(self):
        dummy=1
    






conf = extract_config()



conf.preprocessing_scripts = {#'none' :{'script': '../feature_extraction_scripts/preprocess_pfstar.sh', 'name' : 'clean', 'parameters': [[0,0], [0,0]] },
                              'none' :{'script': '../feature_extraction_scripts/preprocess_pfstar.sh', 'name' : 'clean', 'parameters': [[0,0], [0,0]] },
                              'overdrive' : {'script': '../feature_extraction_scripts/preprocess_pfstar_and_overdrive.sh', 'name' : 'overdrive', 'parameters': [[1,10], [-20,0]] },
                              'babble' : {'script': '../feature_extraction_scripts/preprocess_pfstar_and_add_babble.sh', 'name' : 'babbled', 'parameters': [[-40,-25],[-10,0]] },
                              'humming' : {'script': '../feature_extraction_scripts/preprocess_pfstar_and_add_humming.sh', 'name' : 'volvo', 'parameters': [[-20,-10],[-10,0]] } }

#feature_extraction_script = '../feature_extraction_scripts/extract_5500hz_spec_with_start_end.sh'
conf.feature_extraction_script = '../feature_extraction_scripts/extract_8000hz_mspec66_with_start_end.sh'
conf.featuretype = "mspec66_and_f0_alldata"

conf.quality_control_wavdir = ""
conf.statistics_handle = ""

# In[453]:


conf.vowels = ['a','A','å','Å','ä','Ä','e','E','f','i','I','o','O','ö','u','U']

conf.nonvow = ['b','C','d','D','g','H','j','J','k','l','m','n','N','p','P','Q','r','R','s','S','t','T','v','w','W','Y','z','Z']

conf.combinations = []


conf.used_classes = conf.vowels+conf.nonvow+conf.combinations
conf.classes_name = "mc_en_uk_all"


#
# Settings for feature extraction:
#

conf.datatypelength = 2 # 16 bits = 2 bytes, no?


# For 16 kHz samples:

conf.audio_fs = 16000
'''
frame_length = 400
frame_step = 128
'''

conf.progress_length = 80

conf.max_num_samples=8000 # 0.5 should be enough for any reasonable phoneme, right?

conf.max_num_classes = 10000

conf.feature_dimension=66#130

conf.extraframes = 5


'''
# For 8 kHz samples:
'''

conf.feature_fs = 8000
conf.fs = 8000
conf.frame_length = 256
conf.frame_step = 64

conf.frame_leftovers = conf.frame_length-conf.frame_step



conf.max_num_frames=50
conf.max_phone_length=conf.max_num_frames * 128 # ( 128 being the frame step in alignments)

conf.max_num_monoclasses = 200
conf.max_num_monoclasses = 9


#max_num_samples=100160
conf.assigned_num_samples=100

global tmpfilecounter
conf.tmpfilecounter = 0

# tmp directory for feature extraction.
# This should reside in memory (tempfs or whatever it's called, often under /dev/shm/)

conf.tmp_dir="/dev/shm/siak-feat-extract-pfs-python-"+str(time.time())
try:
    os.makedirs(conf.tmp_dir)
except OSError as exc:  # Python >2.5
    if exc.errno == errno.EEXIST and os.path.isdir(tmp_dir):
        pass
    else:
        raise   


print ('using tmp dir %s' % conf.tmp_dir)


# ## Classes and probabilities ##
# 

# In[454]:

conf.labeltype = "pfstar"


conf.class_def = {
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


conf.corpus = "en_uk_kids_align_from_clean-2"
conf.pickle_dir='../features/work_in_progress/'+conf.corpus+'/pickles'
conf.statistics_dir = '../features/work_in_progress/'+conf.corpus+'/statistics/'

collections = [        
{ "name" : "train.00.a",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.00.a",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.00.b",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.00.b",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.00.c",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.00.c",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.00.d",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.00.d",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.00.e",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.00.e",
  "numlines": 78 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.01.a",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.01.a",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.01.b",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.01.b",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.01.c",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.01.c",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.01.d",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.01.d",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.01.e",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.01.e",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.01.f",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.01.f",
  "numlines": 83 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.02.a",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.02.a",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.02.b",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.02.b",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.02.c",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.02.c",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.02.d",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.02.d",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.02.e",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.02.e",
  "numlines": 146 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.03.a",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.03.a",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.03.b",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.03.b",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.03.c",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.03.c",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.03.d",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.03.d",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.03.e",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.03.e",
  "numlines": 70 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.04.a",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.04.a",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.04.b",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.04.b",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.04.c",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.04.c",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.04.d",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.04.d",
  "numlines": 51 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.05.a",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.05.a",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.05.b",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.05.b",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.05.c",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.05.c",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.05.d",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.05.d",
  "numlines": 185 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.06.a",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.06.a",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.06.b",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.06.b",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.06.c",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.06.c",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.06.d",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.06.d",
  "numlines": 99 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.07.a",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.07.a",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.07.b",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.07.b",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.07.c",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.07.c",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "train.07.d",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.train.07.d",
  "numlines": 99 , "condition" : "mixed", "training" : True },  
{ "name" : "eval.00.a",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.eval.00.a",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "eval.00.b",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.eval.00.b",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "eval.00.c",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.eval.00.c",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "eval.00.d",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.eval.00.d",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "eval.00.e",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.eval.00.e",
  "numlines": 37 , "condition" : "mixed", "training" : True }, 
{ "name" : "test.00.a",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.test.00.a",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "test.00.b",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.test.00.b",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "test.00.c",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.test.00.c",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "test.00.d",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.test.00.d",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "test.00.e",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.test.00.e",
  "numlines": 52 , "condition" : "mixed", "training" : True }, 
{ "name" : "test.01.a",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.test.01.a",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "test.01.b",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.test.01.b",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "test.01.c",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.test.01.c",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "test.01.d",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.test.01.d",
  "numlines": 152 , "condition" : "mixed", "training" : True }, 
{ "name" : "test.02.a",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.test.02.a",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "test.02.b",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.test.02.b",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "test.02.c",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.test.02.c",
  "numlines": 194 , "condition" : "mixed", "training" : True }, 
{ "name" : "test.03.a",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.test.03.a",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "test.03.b",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.test.03.b",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "test.03.c",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.test.03.c",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "test.03.d",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.test.03.d",
  "numlines": 158 , "condition" : "mixed", "training" : True }, 
{ "name" : "test.04.a",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.test.04.a",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "test.04.b",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.test.04.b",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "test.04.c",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.test.04.c",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "test.04.d",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.test.04.d",
  "numlines": 134 , "condition" : "mixed", "training" : True }, 
{ "name" : "test.05.a",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.test.05.a",
  "numlines": 200 , "condition" : "mixed", "training" : True }, 
{ "name" : "test.05.b",
  "recipe" : "/l/rkarhila/speecon_wsj_phoneme_dnn/kids_en_uk/leave_one_out_recipes/aged_recipe.speakers.test.05.b",
  "numlines": 193, "condition" : "mixed", "training" : True   }
]



#conf.debug=True

numcores=1
batchid=0


if len(sys.argv)>2:
    numcores=int(sys.argv[1])
    batchid=int(sys.argv[2])


counter=1
for collection in collections:
    print("counter %i %s numcores %i == batchid %i?" % ( counter, "%s", numcores, batchid ) )
    if (counter%numcores) == batchid:
        extract_collection_and_save(conf, collection)
    counter +=1


