
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

from dnnutil import preprocessingwithnoise as preprocessing


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

    ## boolean debug
    # Output lots of progress information on screen:
    debug = False

    ## dictionary preprocessing_scipts
    # A list of preprocessing scripts; Each audio file will be preprocessed with
    # a randomly selected preprocessing script, which will normalise audio by 
    # peak value and convert it into standard format readable by the next
    # script, possibly adding noise or degrading the signal somehow while doing
    # that.
    preprocessing_scipts = {}

    ## string feature_extraction_script
    # This is 
    feature_extraction_script = ""

    ## string featuretype
    # A descriptor that is only used to name the file where features are saved.
    featuretype = ""


    
    datatypelength = -1

    ## int audio_fs
    # sampling rate of data used
    audio_fs = -1

    ## int max_num_classes
    # Basically the class count for the data ie. the maximum
    # class index (+1) in the data
    max_num_classes =-1

    ## int feature_dimension
    # Dimensionality of the features (For example 37 for the 36 Mel-banks +f0)
    feature_dimension=-1

    ## int feature_fs
    # Sampling rate for feature extractor - The preprocessor should output data
    # in this fs (8kHz in the 36 Mel-bank example)
    feature_fs = -1

    ## int fs
    # fs of source data (often 16kHz)
    fs = -1

    ## int frame_length
    # frame length for feature extraction     
    frame_length = -1 

    ## int frame_step
    # frame step for feature extraction        
    frame_step = -1

    ## int frame_leftovers
    # How much longer frame is compared to frame step.
    # Usually computed from lenght and step
    frame_leftovers = -1 
    
    ## int max_num_frames
    # A limit on how long a segment can be (in frames). Anything longer will be
    # cropped or discarded (depending on feature extraction script version)
    max_num_frames=-1

    ## int max_phone_length
    # A limit on how long a segment can be (in samples). Anything longer will be
    # cropped or discarded (depending on feature extraction script version)
    # Should be computed from max_num_frames and frame length
    max_phone_length=-1



    ## int discard_counter
    # A place to store count of discarded phones
    discard_counter=0

    ## int tmpfilecounter
    # A place to store count of tmp files (is it still used?)
    tmpfilecounter = 0

    ## string tmp_dir
    # A place where to create tmp files
    tmp_dir="/tmp/"

    ## dictionary class_def
    # A very important map from phone name (in alignment files) to a class number 
    # for training and operating the DNN classifier.
    class_def={}

    ## string corpus
    # The name of the corpus
    corpus = ""

    ## string pickle_dir
    # Location (ie. directoiy) where features will be saved
    pickle_dir=''

    ## string statistics_dir
    # Location (ie. directoiy) where statistics of phone segments were 
    # saved (not saved anymore, I think)
    statistics_dir = ''
    
    ## int lastframeindex
    # Used in bookkeeping in the extraction process. Is it still used?
    lastframeindex = 0

    ## int extraframes
    # Number of frames borrowed from phones before and after the phone itself
    # to give a little bit of context for the classifier to use
    extraframes = 5





    ### The following seem to be deprecated (May 15 2017) ###

    ## string quality_control_wavdir 
    # Deprecated: All the audio bits of different classes used to be saved so
    # the developer could listen if correct bits of audio where put into right
    # classes. Now it is easier to plot the features to do the same check.
    quality_control_wavdir = ""

    ## string statistics_handle
    # Deprecated: Used to be filename (or handle?) where statistics of segment
    # lenghts were saved.
    statistics_handle = ""

    ## array vowels
    # Deprecated: List of vowels used in data
    vowels = []

    ## array nonvowels
    # Deprecated: List of consonants used in data
    nonvow = []

    ## array combinations
    # Deprecated: List of allowed phoneme merges in data
    combinations = []

    ## array used_classes
    # Deprecated? List of classes that are present in data
    used_classes = []

    ## string classes_name
    # Deprecated? What on earth was this?
    classes_name = ""

    ## int progress_length
    # Deprecated: used to display progress of processing
    progress_length = -1

    ## int max_num_samples
    # An upper limit of samples for an individual segment
    # (deprecated?)
    max_num_samples=-1

    ## int max_num_monoclasses
    # Deprecated: (What did this do earlier?)
    max_num_monoclasses = -1
    
    ## int assigned_num_samples
    # Deprecated: (What did this do earlier?)   
    assigned_num_samples=100
    


    

    def __init__(self):
        dummy=1
    
debug=True




# In[452]:

conf = extract_config()

conf.preprocessing_options = ['none','overdrive','babble', 'humming'] 

conf.preprocessing_scripts = {#'none' :{'script': '../feature_extraction_scripts/preprocess_pfstar.sh', 'name' : 'clean', 'parameters': [[0,0], [0,0]] },
                              'none' :{'script': '../feature_extraction_scripts/preprocess_speeconkids.sh', 'name' : 'clean', 'parameters': [[0,0], [0,0]] },
                              'overdrive' : {'script': '../feature_extraction_scripts/preprocess_speeconkids_and_overdrive.sh', 'name' : 'overdrive', 'parameters': [[1,10], [-20,0]] },
                              'babble' : {'script': '../feature_extraction_scripts/preprocess_speeconkids_and_add_babble.sh', 'name' : 'babbled', 'parameters': [[-40,-25],[-10,0]] },
                              'humming' : {'script': '../feature_extraction_scripts/preprocess_speeconkids_and_add_humming.sh', 'name' : 'volvo', 'parameters': [[-20,-10],[-10,0]] } }

#feature_extraction_script = '../feature_extraction_scripts/extract_5500hz_spec_with_start_end.sh'
conf.feature_extraction_script = '../feature_extraction_scripts/extract_8000hz_melbin26_with_start_end.sh'
conf.featuretype = "melbin36_and_f0_alldata"

conf.quality_control_wavdir = ""
conf.statistics_handle = ""

# In[453]:


conf.vowels = ['a','A','å','Å','ä','Ä','e','E','f','i','I','o','O','ö','u','U']

conf.nonvow = ['b','C','d','D','g','H','j','J','k','l','m','n','N','p','P','Q','r','R','s','S','t','T','v','w','W','Y','z','Z']

conf.combinations = []


conf.used_classes = conf.vowels+conf.nonvow+conf.combinations
conf.classes_name = "clean_fi_kids_all"


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

conf.feature_dimension=37#130

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

conf.tmp_dir="/dev/shm/siak-feat-extract-speecon-python-"+str(time.time())
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

conf.labeltype = "aaltolike-stateless"


conf.class_def = {
u'sil' : { "class" : 45},
u'a' : { "class" : 46},
u'ä' : { "class" : 47},
u'aa' : { "class" : 48},
u'ää' : { "class" : 49},
u'ae' : { "class" : 50},
u'äe' : { "class" : 51},
u'ai' : { "class" : 52},
u'äi' : { "class" : 53},
u'ao' : { "class" : 54},
u'au' : { "class" : 55},
u'äy' : { "class" : 56},
u'b' : { "class" : 57},
u'd' : { "class" : 58},
u'e' : { "class" : 59},
u'ea' : { "class" : 60},
u'eä' : { "class" : 61},
u'ee' : { "class" : 62},
u'ei' : { "class" : 63},
u'eo' : { "class" : 64},
u'eu' : { "class" : 65},
u'ey' : { "class" : 66},
u'f' : { "class" : 67},
u'g' : { "class" : 68},
u'h' : { "class" : 69},
u'hh' : { "class" : 70},
u'i' : { "class" : 71},
u'ia' : { "class" : 72},
u'iä' : { "class" : 73},
u'ie' : { "class" : 74},
u'ii' : { "class" : 75},
u'io' : { "class" : 76},
u'iu' : { "class" : 77},
u'j' : { "class" : 78},
u'k' : { "class" : 79},
u'kk' : { "class" : 80},
u'l' : { "class" : 81},
u'll' : { "class" : 82},
u'm' : { "class" : 83},
u'mm' : { "class" : 84},
u'n' : { "class" : 85},
u'ng' : { "class" : 86},
u'ngng' : { "class" : 87},
u'nn' : { "class" : 88},
u'o' : { "class" : 89},
u'ö' : { "class" : 90},
u'oa' : { "class" : 91},
u'oe' : { "class" : 92},
u'oi' : { "class" : 93},
u'öi' : { "class" : 94},
u'oo' : { "class" : 95},
u'öö' : { "class" : 96},
u'ou' : { "class" : 97},
u'öy' : { "class" : 98},
u'p' : { "class" : 99},
u'pp' : { "class" : 100},
u'r' : { "class" : 101},
u'rr' : { "class" : 102},
u's' : { "class" : 103},
u'ss' : { "class" : 104},
u't' : { "class" : 105},
u'tt' : { "class" : 106},
u'u' : { "class" : 107},
u'ua' : { "class" : 108},
u'ue' : { "class" : 109},
u'ui' : { "class" : 110},
u'uo' : { "class" : 111},
u'uu' : { "class" : 112},
u'v' : { "class" : 113},
u'y' : { "class" : 114},
u'yä' : { "class" : 115},
u'yi' : { "class" : 116},
u'yö' : { "class" : 117},
u'yy' : { "class" : 118},
}


# ##  Dataset definitions ##
# *In a very awkward manner, we'll specify some local files that contain list of audio and transcription files*

# In[455]:


#
#   Data collection defitinions - train, dev and eval sets:
#


conf.corpus = "speecon_kids_kaldi_align"
conf.pickle_dir='../features/work_in_progress/'+conf.corpus+'/pickles'
conf.statistics_dir = '../features/work_in_progress/'+conf.corpus+'/statistics/'



collections = [
{ "name" : "001.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/001.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "002.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/002.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "003.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/003.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "004.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/004.recipe",
  "condition" : "clean",
  "numlines":  217},
{ "name" : "005.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/005.recipe",
  "condition" : "clean",
  "numlines":  215},
{ "name" : "006.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/006.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "007.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/007.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "008.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/008.recipe",
  "condition" : "clean",
  "numlines":  217},
{ "name" : "009.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/009.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "010.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/010.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "011.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/011.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "012.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/012.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "013.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/013.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "014.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/014.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "015.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/015.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "016.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/016.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "017.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/017.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "018.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/018.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "019.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/019.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "020.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/020.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "021.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/021.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "022.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/022.recipe",
  "condition" : "clean",
  "numlines":  217},
{ "name" : "023.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/023.recipe",
  "condition" : "clean",
  "numlines":  217},
{ "name" : "024.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/024.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "025.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/025.recipe",
  "condition" : "clean",
  "numlines":  222},
{ "name" : "026.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/026.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "027.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/027.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "028.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/028.recipe",
  "condition" : "clean",
  "numlines":  217},
{ "name" : "029.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/029.recipe",
  "condition" : "clean",
  "numlines":  217},
{ "name" : "030.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/030.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "031.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/031.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "032.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/032.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "033.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/033.recipe",
  "condition" : "clean",
  "numlines":  217},
{ "name" : "034.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/034.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "035.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/035.recipe",
  "condition" : "clean",
  "numlines":  217},
{ "name" : "036.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/036.recipe",
  "condition" : "clean",
  "numlines":  217},
{ "name" : "037.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/037.recipe",
  "condition" : "clean",
  "numlines":  217},
{ "name" : "038.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/038.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "039.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/039.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "040.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/040.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "041.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/041.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "042.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/042.recipe",
  "condition" : "clean",
  "numlines":  217},
{ "name" : "043.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/043.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "044.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/044.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "045.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/045.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "046.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/046.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "047.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/047.recipe",
  "condition" : "clean",
  "numlines":  217},
{ "name" : "048.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/048.recipe",
  "condition" : "clean",
  "numlines":  215},
{ "name" : "049.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/049.recipe",
  "condition" : "clean",
  "numlines":  216},
{ "name" : "050.recipe",
  "recipe" : "/l/rkarhila/speecon_kids_labels/recipes/050.recipe",
  "condition" : "clean",
  "numlines":  223}
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



#for collection in collections:
#    extract_collection_and_save(conf, collection)



