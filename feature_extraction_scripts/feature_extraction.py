
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


vowels = ['a','A','å','Å','ä','Ä','e','E','f','i','I','o','O','ö','u','U']

nonvow = ['b','C','d','D','g','H','j','J','k','l','m','n','N','p','P','Q','r','R','s','S','t','T','v','w','W','Y','z','Z']

combinations = []


used_classes = vowels+nonvow+combinations
classes_name = "mc_en_uk_all"


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



# ## Some helper functions ##
# *Label processing etc.*

# In[456]:

def process_label( labelfile ):
    global discard_counter

    if not os.path.isfile(labelfile):
        print ("Can't find labelfile %s" % labelfile)
        return False
    
    with io.open(labelfile ,'r',encoding='iso-8859-15') as f:

        new_align = []

        current_start = 0
        current_end = 0
        current_model = False
        current_premodel = False
        current_postmodel = False

        skip = False

        phonect = 0
        statect = 0

        lcounter = 0

        # For printing the phoneme sequences into a log:
        skipmark=False

        startmark=-1
        endmark = -1

        discard = False
        sildone = False
        phone={}

        for l in  f.readlines():
            
            
            # If we have a short pause model:
            #if '+' not in l:
            #    no_skipping = True
            #    skipmark = True

            # We'll process the label line by line with a two-phone delay:

            if '+' in l and not discard:
                #print "Looking at %s"%(l)
                [start, 
                 end, 
                 premodel, 
                 model, 
                 postmodel, 
                 state] = re.split(r'[ .+-]', l.strip() ) #, l.encode('utf-8').strip() )

                if state=='0':

                    # Let's give a 5 frame buffer to the beginning and end (to get some coarticulation info)
                    phone = {'start': max(int(start) - 5 * 128, 0),
                             'premodel':premodel, 
                             'model': model,
                             'postmodel':postmodel,
                             'state':state,
                             'triphone': "%s-%s+%s" % (premodel, model, postmodel) }

                if state=='2':
                    phone['end'] = int(end) + 5 * 128

                    if (phone['model'] != '__'):

                        if (int(phone['end'])-int(phone['start']))/frame_step < 13 or (int(phone['end'])-int(phone['start']) > max_phone_length ):
                            discard_counter+=1
                            #print "Discarding %i/%i: %s: (Too short! Discards: %0.2f%s)" % (recipefilecounter, collection['numlines'], labelfile, 100.0*discard_counter/collection['numlines'],"%" )

                            discard = True

                        #elif (int(phone['end'])-int(phone['start']))/frame_step > 40 and '_' not in phone['triphone']:
                        #    #print "Discarding %i/%i: %s (Too Long! Discards: %0.2f%s)" % (recipefilecounter, collection['numlines'], labelfile, 100.0*discard_counter/collection['numlines'],"%" )
                        #    discard_counter+=1
                        #    discard = True

                        #if debug:
                        #    print ("saving %s-%s+%s " %  (phone['premodel'], phone['model'],phone['postmodel']))
                        else:
                            new_align.append({'pre' : phone['premodel'],
                                          'model' : phone['model'],
                                          'post' : phone['postmodel'],
                                          'start' : phone['start'],
                                          'end' : phone['end'],
                                          'triphone' : phone['triphone'],
                                          'sortable': "%s--%s++%s" % (phone['model'] , phone['premodel'], phone['postmodel'])
                                      })
            elif not sildone:
                [start, 
                 end, 
                 model, 
                 state] = re.split(r'[ .]', l.strip() )
                
                if state=='2':
                    new_align.append( {'pre' : '?',
                                       'model' : 'sil',
                                       'post' : '?',
                                       'start' : max(0, int(end)-16000 ),
                                       'end' : int(end),
                                       'triphone' : 'sil',
                                       'sortable': 'sil-?+?' })
                    sildone = True
            else:
                [foo1, 
                 fileend, 
                 foo2, 
                 foo3] = re.split(r'[ .]', l.strip() )
    
        if len(new_align)>1:
            if new_align[-1]['end'] >= int(fileend):
                discard_counter += 1
                del new_align[-1]
            

    return new_align
    


# In[457]:

def get_labelstring( new_align ):
    labelstring = ''
    for phone in new_align:
        labelstring += '.'+phone['model']
    return labelstring    


# In[474]:

def chop_features( cleanaudiodata, noisyaudiodata, noisy_feature_array, clean_feature_array, new_align, speedup, preprocessor_string, 
                   cleanphonedata, noisyphonedata, phoneclasses, phone_indices, segment_details, segment_lengths, quality_control_audio_files ):
    #global debug
    
    count = 0
    startmark = int(math.ceil(float(new_align[0]['start'])*(feature_fs/audio_fs)/speedup))
    #endmark= int(math.floor(float(new_align[-1]['end'])*(feature_fs/audio_fs)/speedup))
    
    tooshortcount=0
    
    for l in new_align:                

        lkey = l['sortable']
        mkey = l['model']
        
        if 1 == 1:
            tp = l['triphone']

            l_start = math.ceil((float(l['start'])/speedup-startmark)*(feature_fs/audio_fs)/frame_step)        
            l_end =  math.floor((float(l['end'])/speedup-startmark)*(feature_fs/audio_fs)/frame_step)  

            
           
            l_length = l_end - l_start
            
            if debug:
                print ("Segment length: %i" % l_length)
            if (l_length < 4):
                tooshortcount+=1
                continue

            # For debugging, let's write this stuff to disk:
            if mkey not in quality_control_audio_files.keys():
                qual_file = os.path.join(quality_control_wavdir,  mkey+".raw-"+str(feature_fs)+"hz-16bit-signed-integer")
                quality_control_audio_files[mkey] = open( qual_file , 'wb')

            win_i=0
            win_len=256
            max_val=32000
            
            audio_start = int(math.floor(float(l['start'])/speedup*(feature_fs/audio_fs)))
            audio_end = int(math.ceil(float(l['end'])/speedup*(feature_fs/audio_fs)))
            
            if debug:
                print("start: %i end: %i audiodata len: %i" %(audio_start, audio_end, len(noisyaudiodata)))
            if audio_end > len(noisyaudiodata):
                raise ValueError("Can't access %i:%i in audiofile of length %i (%s)"% 
                                 ( audio_start, audio_end , len(noisyaudiodata), preprocessor_string )) 
                
            norm=20000.0/max(abs(noisyaudiodata[audio_start:audio_end]))
            #print norm

            for val in noisyaudiodata[audio_start:audio_start+win_len]:
                (quality_control_audio_files[mkey]).write( 
                        struct.pack( 'h', int( max( -max_val, min( max_val,norm * val * win_i / win_len ) ) ) ) )
                win_i+=1

            for val in noisyaudiodata[audio_start+win_len:audio_end-win_len]:
                (quality_control_audio_files[mkey]).write(
                        struct.pack( 'h', int(max( -max_val, min(max_val,norm * val ) ) ) ) )

            for val in noisyaudiodata[audio_end-win_len:audio_end]:
                (quality_control_audio_files[mkey]).write(
                        struct.pack( 'h', int(max( -max_val,min(max_val,norm * val * win_i / win_len ) ) ) )  )
                win_i-=1

            for val in range(0,1024):
                (quality_control_audio_files[mkey]).write(
                        struct.pack( 'h', 0 ) ) 



            if (noisy_feature_array.shape[0] < l_end):
                print ("Not enough features: %i < %i" % (noisy_feature_array.shape[0], l_end))
                continue

            statistics_handle.write("%i\t%s\n" % (l_length, tp))

            if debug:
                print ("----------- "+l['triphone'] +" ----------------")
                print ("Array stats: start %i -> %i length ?? -> %i end %i -> %i" % (
                            int(l['start'])-startmark, 
                            l_start, 
                            l_length, 
                            int(l['end'])-startmark, 
                            l_end ))
                print ("      phone data size: %i x %i" % (noisy_feature_array[l_start:l_end, :]).shape)
                print ("Data size: %i x %i" % noisy_feature_array.shape)

            index = np.count_nonzero(phoneclasses)

            #if index >= len(phoneclasses):
            #    print("Adding 1000 more entries to phonedata")
            #    phoneclasses = np.concatenate( (phoneclasses, np.zeros([1000], dtype='uint8') ) )
            #    segment_lengths = np.concatenate( (segment_lengths, np.zeros([1000], dtype='uint32') ) )
            #    phone_indices = np.concatenate( (phone_indices, np.zeros([1000,2], dtype='uint32') ) )
                
            segment_lengths[index] = l_length
            phoneclasses[index] =  class_def[l['model']]['class']
            
            segment_details.append( l['triphone'] + ' ' + preprocessor_string )
            
            if index == 0:
                phone_indices[index, :] = [0, l_length]
            else:
                phone_indices[index, :] = [ phone_indices[index-1][1] + 1, 
                                            phone_indices[index-1][1] + 1 + l_length ]
                
            #if phone_indices[index, 1] > phonedata.shape[0]:
            #    print("Adding 50000 more entries to phonedata")
            #    phonedata = np.concatenate( (phonedata, np.zeros([50000, feature_dimension], dtype='float32')), axis=0)
                
            noisyphonedata[ phone_indices[index][0]:phone_indices[index][1], :] = noisy_feature_array[l_start:l_end, :]
            cleanphonedata[ phone_indices[index][0]:phone_indices[index][1], :] = clean_feature_array[l_start:l_end, :]
            
            count += 1
            #phonedata.append(feature_array[l_start:l_end, :])
            #triphoneclasses.append( )
            #segment_lengths.append( l_length  )
            #triphonedata.append ({ 'data': feature_array[l_start:l_start+max_num_frames, :],
            #                        'counter': 0,
            #                        'mono' :l['model'],
            #                        'triphone' : l['triphone'],
            #                        'sorting' : l['sortable'] })

    return count        
    #return { "data" : triphonedata, 
    #         "classes" : triphoneclasses, 
    #         "segment_lengths" : segment_lengths }


# ## Feature extraction (spectral/vocoder parameters) ##

# In[475]:

def get_features( audiofile, align,  preprocessors, cleanphonedata, noisyphonedata, phoneclasses, phone_indices, 
                  segment_details, segment_lengths, quality_control_audio_files, training=True):
    
    global tmpfilecounter
    count = 0
    errors = []
    
    if training:
        speedups =  [0.94, 1.0, 1.06 ]      
    else:
        speedups = [1.0]
        
    for speedup in speedups:

        new_align = [];
        for l in align:                
            if random.random() < class_def[l['model']]["probability"]:
                #if len(new_align) == 0:
                #    startmark = math.floor(float(l['start'])/speedup*fs/16000)
                #endmark= math.ceil(float(l['end'])/speedup*fs/16000)

                #l['start'] -= startmark
                #l['end'] -= startmark
                new_align.append(l)
        if debug:
            print(new_align)
        
        if len(new_align) > 0:
            
            preprocessor_key = preprocessors[random.randint( 0, len(preprocessors)-1 ) ]
            
            noisy_preprocessor = preprocessing_scripts[preprocessor_key]
                                                                    
            noisy_preprocessed_audio=os.path.join(tmp_dir,
                                            str(tmpfilecounter)+"_"+str(speedup)+"_preprocessed")

            
            clean_preprocessor = preprocessing_scripts['none']

            clean_preprocessed_audio = os.path.join(tmp_dir,
                                            str(tmpfilecounter)+"_"+str(speedup)+"_clean")
            #for preprocessor in preprocessors:
            #    preprocessor =  preprocessing_scripts[random.randint(0, len(preprocessing_scripts)-1)]
    
            if 1 == 1:
                if debug:
                    print ("Preprocessor : %s" % noisy_preprocessor['name'])
    
                if noisy_preprocessor['parameters'][0][1] - noisy_preprocessor['parameters'][0][0] > 0:
                    param1 = str(random.randint( noisy_preprocessor['parameters'][0][0],
                                                 noisy_preprocessor['parameters'][0][1]) )
                else:
                    param1 = str(noisy_preprocessor['parameters'][0][0])
    

                if noisy_preprocessor['parameters'][1][1] - noisy_preprocessor['parameters'][1][0] > 0:                    
                    param2 = str(random.randint( noisy_preprocessor['parameters'][1][0],
                                                 noisy_preprocessor['parameters'][1][1]) )
                else:
                    param2 = str(noisy_preprocessor['parameters'][1][0])
    
                preprocessor_string = ("%s %s %s %.1f %s" % (preprocessor_key , param1, param2, speedup, audiofile))
        
                if debug:                    
                    print (' '.join([noisy_preprocessor['script'],
                                     audiofile,
                                     noisy_preprocessed_audio,
                                     str(speedup),
                                     param1,
                                     param2
                                 ]))

                preprocess_progress = Popen([noisy_preprocessor['script'],
                                             audiofile,
                                             noisy_preprocessed_audio,
                                             str(speedup),
                                             param1,
                                             param2
                                        ], stdout=PIPE, stdin=PIPE, stderr=STDOUT).communicate()

                
                clean_preprocess_progress = Popen([clean_preprocessor['script'],
                                             audiofile,
                                             clean_preprocessed_audio,
                                             str(speedup),
                                             param1,
                                             param2
                                        ], stdout=PIPE, stdin=PIPE, stderr=STDOUT).communicate()
                

                noisy_audiodata = np.fromfile( noisy_preprocessed_audio, 'int16', -1)
                clean_audiodata = np.fromfile( clean_preprocessed_audio, 'int16', -1)

                startmark = 0 #math.floor(float(new_align[0]['start'])/speedup)
                endmark= math.ceil(float(new_align[-1]['end'])/speedup)

                if debug:
                    print ("start feature extraction at %s (%f s) and end at %s (%f s) ==> %i frames"  % (
                            startmark, 
                            (float(startmark)/16000), 
                            endmark, (float(endmark)/16000), 
                            (endmark-startmark)/frame_step) )

                # Communication from: 
                # http://stackoverflow.com/questions/163542/python-how-do-i-pass-a-string-into-subprocess-popen-using-the-stdin-argument

                noisy_tmp_input=os.path.join(tmp_dir,str(tmpfilecounter)+"_noisy_in")
                noisy_tmp_output=os.path.join(tmp_dir,str(tmpfilecounter)+"_noisy_out")
                noisy_audiodata.tofile(noisy_tmp_input, "")

                clean_tmp_input=os.path.join(tmp_dir,str(tmpfilecounter)+"_clean_in")
                clean_tmp_output=os.path.join(tmp_dir,str(tmpfilecounter)+"_clean_out")

                clean_audiodata.tofile(clean_tmp_input, "")

                process_progress = Popen([
                        feature_extraction_script, 
                        noisy_tmp_input, 
                        noisy_tmp_output, 
                        str(startmark), 
                        str(endmark+frame_leftovers) #, '/tmp/test_noisy.wav'
                ], 
                                         stdout=PIPE, stdin=PIPE, stderr=STDOUT).communicate()

                process_progress = Popen([
                        feature_extraction_script, 
                        clean_tmp_input, 
                        clean_tmp_output, 
                        str(startmark), 
                        str(endmark+frame_leftovers) # , '/tmp/test_clean.wav' 
                ], stdout=PIPE, stdin=PIPE, stderr=STDOUT).communicate()

                noisy_feature_list = np.fromfile(noisy_tmp_output, dtype='float32', count=-1)
                noisy_feature_array = noisy_feature_list.reshape([-1,feature_dimension])

                clean_feature_list = np.fromfile(clean_tmp_output, dtype='float32', count=-1)
                clean_feature_array = clean_feature_list.reshape([-1,feature_dimension])

                
                f_end =  math.floor((endmark-startmark)/frame_step)

                if debug:
                    print ("Utterance data size: %i x %i" % (noisy_feature_array).shape)

                if (noisy_feature_array.shape[0] < f_end/16000*fs):
                        print ("Not enough noisy features for file %s: %i < %i" % 
                               (audiofile, noisy_feature_array.shape[0], f_end))
                        print ("panic save to /tmp/this_is_not_good")
                        np.savetxt('/tmp/this_is_not_good', noisy_feature_array, delimiter='\t')
                        raise ValueError("Not enough features for file %s: %i < %i" % (
                                audiofile, 
                                noisy_feature_array.shape[0], 
                                f_end) )
                        
                if (clean_feature_array.shape[0] < f_end/16000*fs):
                        print ("Not enough clean features for file %s: %i < %i" % 
                               (audiofile, clean_feature_array.shape[0], f_end))
                        print ("panic save to /tmp/this_is_not_good")
                        np.savetxt('/tmp/this_is_not_good', clean_feature_array, delimiter='\t')
                        raise ValueError("Not enough features for file %s: %i < %i" % (
                                audiofile, 
                                clean_feature_array.shape[0], 
                                f_end) )                        
                        
                else:
                    try: 
                        count += chop_features( clean_audiodata,
                                               noisy_audiodata,
                                               clean_feature_array,
                                                          noisy_feature_array, 
                                                          new_align, 
                                                          speedup,
                                                          preprocessor_string,
                                                          cleanphonedata, 
                                                          noisyphonedata,
                                                          phoneclasses, 
                                                          phone_indices,
                                                          segment_details,
                                                          segment_lengths,
                                                          quality_control_audio_files)
                        #triphonedata += data_and_classes["data"]
                        #triphoneclasses += data_and_classes["classes"]
                        #segment_lengths += data_and_classes["segment_lengths"]

                    except ValueError as error:   
                        print (error)
                        errors.append("Bad amount of data! in %s speedup %s" % (audiofile, speedup))
                        continue

                os.remove(clean_preprocessed_audio)
                os.remove(clean_tmp_input)
                os.remove(clean_tmp_output)
                
                os.remove(noisy_preprocessed_audio)
                os.remove(noisy_tmp_input)
                os.remove(noisy_tmp_output)

    if len(errors)>1:
        print (errors[-1])
    return { "errors": errors, "count" : count }


# In[482]:



def extract_collection_and_save( collection ):
    cleanphonedata = np.zeros([250000,feature_dimension] ,dtype='float32')
    noisyphonedata = np.zeros([250000,feature_dimension] ,dtype='float32')
    phoneclasses = np.zeros([20000], dtype='uint8')
    phone_indices = np.zeros([20000,2], dtype='uint32')
    segment_lengths = np.zeros([20000], dtype='uint32')
    segment_details = []
    
    errors = []
    quality_control_audio_files = []
    discard_counter = 0

    recipefile = open( collection['recipe'] , 'r')
    recipefilecounter = 0
    too_long_counter = 0
    all_trips_counter = 0

    tmpfilecounter = 0

    progress_interval = math.ceil(collection['numlines']/1000.0)

    statistics_file=statistics_dir+"/"+corpus+"-"+collection['condition']+"-"+collection['name']+".triphone-frame-counts"
    global statistics_handle
    statistics_handle = open(statistics_file, 'w')

    class_file=statistics_dir+"/"+corpus+"-"+collection['condition']+"-"+collection['name']+".triphone-classes"
    class_handle= open(class_file, 'w')

    phone_merge_file=statistics_dir+"/"+corpus+"-"+collection['condition']+"-"+collection['name']+".phone-merge"
    phone_merge_handle = open(phone_merge_file, 'w')

    global quality_control_wavdir
    quality_control_wavdir = os.path.join(pickle_dir, 'control-wav', collection['condition']+"-"+collection['name']+"-classes_"+classes_name)

    mkdir(quality_control_wavdir)

    quality_control_audio_files = {}

    tooshortcount=0

    for r in recipefile.readlines():
        
        preprocessors = ['none','overdrive', 'underdrive', 'babble', 'humming']
        
        recipefilecounter += 1
        if debug:
            print ("Item %i/%i" % (recipefilecounter, collection['numlines']) )

        audiofile = re.sub('audio=', r'',  re.findall('audio=/[^ ]+', r)[0]).strip()
        labelfile = re.sub(r'transcript=', r'', re.findall('transcript=/[^ ]+', r)[0]).strip()
    
        new_align = process_label(labelfile)
        labelstring = get_labelstring( new_align )
        
        phone_merge_handle.write("%s\t%s\n" % (labelfile, labelstring))

        # OK, label file done.
        # Now it's time to process the audio.
        # We'll send to the feature extractor the bits of the file that 
        # match the speech segments.

        if len(new_align) > 0:
            errors_and_count = get_features( audiofile, 
                                   new_align, 
                                   preprocessors,
                                   cleanphonedata, 
                                   noisyphonedata,
                                   phoneclasses, 
                                   phone_indices, 
                                   segment_details,
                                   segment_lengths,
                                   quality_control_audio_files)
            #triphonedata += data_classes_and_errors["data"]
            #triphoneclasses += data_classes_and_errors["classes"]
            #errors += data_classes_and_errors["errors"]
            #segment_lengths += data_classes_and_errors["segment_lengths"]
            errors += errors_and_count["errors"]
            all_trips_counter += errors_and_count["count"]
            #if len(errors_and_count["errors"] ) > 0:
            #    print ( errors_and_count["errors"] )

        if not debug:
            if (recipefilecounter % int(progress_interval)) == 0:
                sys.stderr.write("\r%0.2f%s %s %s (%i phones, %i discarded, %i errors)" % (
                        100.0*recipefilecounter/collection['numlines'], 
                        "%",
                        collection['condition'], 
                        collection['name'],
                        all_trips_counter,
                        discard_counter,
                        len(errors)))
                sys.stderr.flush()

        if (recipefilecounter == collection['numlines']):
            print ("That's enough!")
            print ("recipefilecounter %i  == collection['numlines'] %i" % ( 
                    recipefilecounter, 
                    collection['numlines'] ))

    itemcount= np.count_nonzero(phoneclasses)
    indexcount = phone_indices[itemcount-1,1]
    
    # Save to a pickle:
    import pickle

    picklefile = os.path.join(pickle_dir,  collection['name'] + '_' + featuretype + ".pickle")

    mkdir(pickle_dir)            

    print ("pickling %i phones / %i frames to %s" % ( itemcount, indexcount, picklefile))
                        
    outf = open(picklefile, 'wb')
    # Pickle the list using the highest protocol available.
    pickle.dump( {  'cleandata': cleanphonedata[0:indexcount], 
                    'noisydata': noisyphonedata[0:indexcount], 
                    'classes': phoneclasses[0:itemcount],
                    'indices' : phone_indices[0:itemcount,:],
                    'lengths' : segment_lengths[0:itemcount],
                    'details' : segment_details }, outf, protocol=pickle.HIGHEST_PROTOCOL)
                
    


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



