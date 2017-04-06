# coding: utf-8


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
import pprint 
pp = pprint.PrettyPrinter(indent=4)



#
# A function that will be useful:
#

def mkdir(path):
    try:
        os.makedirs(path)        
    except OSError as exc:  # Python >2.5
        #print ("dir %s exists" % path)
        dummy = 1


def process_other_aaltoasrlike_label( conf, labelfile ):


    if not os.path.isfile(labelfile):
        print ("Can't find labelfile %s" % labelfile)
        return False
    
    with io.open(labelfile ,'r',encoding='utf-8') as f:

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

        fileend = -1

        #print (labelfile)
        for l in  f.readlines():

            #print(l)
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
                 postmodel] = re.split(r'[\t .+-]', l.strip() ) #, l.encode('utf-8').strip() )

                #start = math.floor(int(start)/128)*128
                #end = math.floor(int(end)/128)*128

                # Let's give a 5 frame buffer to the beginning and end (to get some coarticulation info)
                phone = {'start': int(start), #max(int(start) - 5 * 128, 0),
                         'premodel':premodel, 
                         'model': model,
                         'postmodel':postmodel,
                         'triphone': "%s-%s+%s" % (premodel, model, postmodel) }

                if True:
                    phone['end'] = int(end)# + 5 * 128

                    if (phone['model'] != '__'):

                        discard = False                    

                        if (int(phone['end'])-int(phone['start']))/conf.frame_step < 3 :
                            if conf.debug:
                                print("Phone %s: Too short (%i frames)!" % (phone['triphone'],(int(phone['end'])-int(phone['start']))/conf.frame_step))
                            conf.discard_counter+=1
                            discard = True
                        
                        elif int(phone['start']) < conf.extraframes * conf.frame_step * 1.05:
                            if conf.debug:
                                print("Phone %s: Too little beginning silence!" % phone['triphone'])
                            conf.discard_counter+=1
                            discard = True

                        elif (int(phone['end'])-int(phone['start']) > conf.max_phone_length ):
                            if conf.debug:
                                print("Phone %s: Too long (%i frames)" % (phone['triphone'],(int(phone['end'])-int(phone['start']))/conf.frame_step))
                            conf.discard_counter+=1
                            discard = True

                        if not discard:
                            if conf.debug:
                                print ("Adding model %s start %s end %s" % (phone['triphone'], phone['start'], phone['end']))

                            new_align.append({#'pre' : phone['premodel'],
                                          'model' : phone['model'],
                                          #'post' : phone['postmodel'],
                                          'start' : max(phone['start'], (conf.extraframes +1) * conf.frame_step ),
                                          'end' : phone['end'],
                                          'triphone' : phone['triphone'],
                                          #'sortable': "%s--%s++%s" % (phone['model'] , phone['premodel'], phone['postmodel'])
                                      })
                            
                            #print ("adding %i %i %s" % (int(phone['start']), int(phone['end']), phone['triphone'] ))
            elif not sildone:
                [start, 
                 end, 
                 model] = re.split(r'[\t .]', l.strip() )
                
                if True:
                    if int(start) > conf.extraframes * conf.frame_step:
                        if  int(end) > max( conf.extraframes * conf.frame_step  , int(start),  int(end) - conf.frame_step * conf.max_num_frames):
                            new_align.append( {#'pre' : '?',
                                               'model' : 'sil',
                                               #'post' : '?',
                                               'start' : max( (conf.extraframes  + 1) * conf.frame_step, int(start),  int(end) - conf.frame_step * conf.max_num_frames),
                                               'end' : int(end),
                                               'triphone' : 'sil'
                                               #'sortable': 'sil-?+?'
                                           })
                            sildone = True
            else:
                [foo1, 
                 fileend, 
                 foo2] = re.split(r'[\t .]', l.strip() )
    
        if len(new_align)>1:
            #new_align[-1]['end'] -= 4 * 128
            
            if new_align[-1]['end'] >= int(fileend) or new_align[-1]['end'] >= new_align[-1]['start']:
                conf.discard_counter += 1
                del new_align[-1]

        #
        #if len(new_align)>1:
        #    if new_align[-1]['end'] >= int(fileend) or new_align[-1]['end'] >= new_align[-1]['start']:
        #        conf.discard_counter += 1
        #        del new_align[-1]
     

    return new_align



def process_pfstar_label( conf, labelfile ):

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

        fileend = -1

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
                    phone = {'start': int(start), #max(int(start) - 5 * 128, 0),
                             'premodel':premodel, 
                             'model': model,
                             'postmodel':postmodel,
                             'state':state,
                             'triphone': "%s-%s+%s" % (premodel, model, postmodel) }

                if state=='2':
                    phone['end'] = int(end)# + 5 * 128

                    if (phone['model'] != '__'):

                        discard = False                    

                        if (int(phone['end'])-int(phone['start']))/conf.frame_step < 3 :
                            if conf.debug:
                                print("Phone %s: Too short (%i frames)!" % (phone['triphone'],(int(phone['end'])-int(phone['start']))/conf.frame_step))
                            conf.discard_counter+=1
                            discard = True
                        
                        elif int(phone['start']) < conf.extraframes * conf.frame_step * 1.05:
                            if conf.debug:
                                print("Phone %s: Too little beginning silence!" % phone['triphone'])
                            conf.discard_counter+=1
                            discard = True

                        elif (int(phone['end'])-int(phone['start']) > conf.max_phone_length ):
                            if conf.debug:
                                print("Phone %s: Too long (%i frames)" % (phone['triphone'],(int(phone['end'])-int(phone['start']))/conf.frame_step))
                            conf.discard_counter+=1
                            discard = True

                        if not discard:
                            if conf.debug:
                                print ("Adding model %s start %s end %s" % (phone['triphone'], phone['start'], phone['end']))

                            new_align.append({#'pre' : phone['premodel'],
                                          'model' : phone['model'],
                                          #'post' : phone['postmodel'],
                                          'start' : max( ((conf.extraframes +1) * conf.frame_step), phone['start']),
                                          'end' : phone['end'],
                                          'triphone' : phone['triphone'],
                                          #'sortable': "%s--%s++%s" % (phone['model'] , phone['premodel'], phone['postmodel'])
                                      })
            elif not sildone:
                [start, 
                 end, 
                 model, 
                 state] = re.split(r'[ .]', l.strip() )
                
                if state=='2':
                    if int(start) > conf.extraframes * conf.frame_step:
                        if  int(end) > max( conf.extraframes * conf.frame_step  , int(start),  int(end) - conf.frame_step * conf.max_num_frames):
                            new_align.append( {#'pre' : '?',
                                               'model' : 'sil',
                                               #'post' : '?',
                                               'start' : max( (conf.extraframes+1) * conf.frame_step , int(start),  int(end) - conf.frame_step * conf.max_num_frames),
                                               'end' : int(end),
                                               'triphone' : 'sil'
                                               #'sortable': 'sil-?+?'
                                           })
                            sildone = True
            else:
                [foo1, 
                 fileend, 
                 foo2, 
                 foo3] = re.split(r'[ .]', l.strip() )
    
        if len(new_align)>1:
            if new_align[-1]['end'] >= int(fileend):
                conf.discard_counter += 1
                del new_align[-1]
            

    return new_align
    




def get_labelstring( new_align ):
    labelstring = ''
    for phone in new_align:
        labelstring += '.'+phone['model']
    return labelstring    




def chop_features( conf, cleanaudiodata, clean_feature_array, new_align, speedup, preprocessor_string, 
                   cleanphonedata, phoneclasses, phone_indices, segment_details, segment_lengths, quality_control_audio_files, startbyte ):
    
    count = 0
    startmark =  0 #startbyte#int(math.ceil(float(new_align[0]['start'])*(conf.feature_fs/conf.audio_fs)/speedup))
    #endmark= int(math.floor(float(new_align[-1]['end'])*(conf.feature_fs/conf.audio_fs)/speedup))
    
    tooshortcount=0

    # Get first and last frames:



    # Then list the models that we have just copied, using overlapping indexing:

    for l in new_align:                

        #lkey = l['sortable']
        mkey = l['model']
        
        if mkey in conf.class_def:
            tp = l['triphone']

            #l_start =  max( math.ceil((float(l['start']) / speedup - startmark) * (conf.feature_fs / conf.audio_fs) / conf.frame_step), 0)
            #l_end =   math.floor((float(l['end']) / speedup - startmark) * (conf.feature_fs / conf.audio_fs) / conf.frame_step)

            #l_start =  max( math.ceil((float(l['start']) / speedup - startmark)  / conf.frame_step), 0)

            l_start =  max( math.ceil((float(l['start']) - startmark)  / conf.frame_step), 0)
            #print( l['end'])
            #print( (float( l['end'] ) ))
            #print(  (float( l['end'] ) / speedup ))
            #print(  (float( l['end'] ) / speedup - startmark) )
            #print(  (float( l['end'] ) / speedup - startmark) / conf.frame_step )
            #print( math.floor(  (float( l['end'] ) / speedup - startmark) / conf.frame_step) )
            #l_end =   math.floor(  (float( l['end'] ) / speedup - startmark) / conf.frame_step)
            l_end =   math.floor(  (float( l['end'] ) - startmark) / conf.frame_step)
        
            l_length = l_end - l_start
            
            if conf.debug:
                print ("Segment length: %i" % l_length)
            if (l_length < 2 * conf.extraframes):
                tooshortcount+=1
                continue

            # For conf.debugging, let's write this stuff to disk:
            if mkey not in quality_control_audio_files.keys():
                qual_file = os.path.join(conf.quality_control_wavdir,  mkey+".raw-"+str(conf.feature_fs)+"hz-16bit-signed-integer")
                quality_control_audio_files[mkey] = open( qual_file , 'wb')

            win_i=0
            win_len=conf.frame_length
            max_val=32000
            
            audio_start =  l_start  * conf.frame_step 
            audio_end = l_end * conf.frame_step + conf.frame_length
            
            if conf.debug:
                print("framestart: %i end: %i framedata len: %i" %(l_start, l_end, l_end-l_start))
                print(" bytestart: %i end: %i audiodata len: %i" %(audio_start, audio_end, len(cleanaudiodata)))
                print(" limits like in original labels: %i %i" % ( l_start * conf.frame_step + startbyte , l_end * conf.frame_step + startbyte ) )

            if conf.debug:
                print ("----------- "+l['triphone'] +" ----------------")
                print ("Array stats: start %i -> %i length ?? -> %i end %i -> %i" % (
                            int(l['start'])-startmark, 
                            l_start, 
                            l_length, 
                            int(l['end'])-startmark, 
                            l_end ))
                print ("      phone data size: %i x %i" % (clean_feature_array[l_start:l_end, :]).shape)
                print ("Data size: %i x %i" % clean_feature_array.shape)



            if (clean_feature_array.shape[0] < l_end):
                print ("\nNot enough features: %i < %i (speed: %f)" % (clean_feature_array.shape[0], l_end, speedup))
                return count

            conf.statistics_handle.write("%i\t%s\n" % (l_length, tp))


            count += 1

            index = np.count_nonzero(phoneclasses)

            #if index >= len(phoneclasses):
            #    print("Adding 1000 more entries to phonedata")
            #    phoneclasses = np.concatenate( (phoneclasses, np.zeros([1000], dtype='uint8') ) )
            #    segment_lengths = np.concatenate( (segment_lengths, np.zeros([1000], dtype='uint32') ) )
            #    phone_indices = np.concatenate( (phone_indices, np.zeros([1000,2], dtype='uint32') ) )
                
            segment_lengths[index] = l_length
            phoneclasses[index] =  conf.class_def[mkey]['class']
            
            segment_details.append( l['triphone'] + ' ' + preprocessor_string )
            
            phone_indices[index, :] = [ conf.lastframeindex + l_start  , conf.lastframeindex + l_end]

            

            #phonedata.append(feature_array[l_start:l_end, :])
            #triphoneclasses.append( )
            #segment_lengths.append( l_length  )
            #triphonedata.append ({ 'data': feature_array[l_start:l_start+max_num_frames, :],
            #                        'counter': 0,
            #                        'mono' :l['model'],
            #                        'triphone' : l['triphone'],
            #                        'sorting' : l['sortable'] })
        else:
            #print ("%s not in model keys list!" % mkey)
            dummy=1
    return count        
    #return { "data" : triphonedata, 
    #         "classes" : triphoneclasses, 
    #         "segment_lengths" : segment_lengths }


# ## Feature extraction (spectral/vocoder parameters) ##

def get_features( conf, audiofile, align,  preprocessors, cleanphonedata,  phoneclasses, phone_indices, 
                  segment_details, segment_lengths, quality_control_audio_files, training=True):
    
    preprocessor_string=""

    count = 0
    errors = []
    
    if training:
        speedups =  [0.96, 1.0, 1.04 ]
    else:
        speedups = [1.0]
        
    for speedup in speedups:

        if conf.debug:
            print ("align[0]:")
            print (align[0])
            print ("align[-1]:")
            print (align[-1])
            
            print ("start and end:")
            print ( [ align[0]['start'], align[-1]['end'] ])
            print ("extra padding bytes: %i "  % (conf.extraframes * conf.frame_step))
            print ("Starting byte: %i - %i" % ((align[ 0]['start']) / speedup * conf.feature_fs /  16000, (conf.extraframes * conf.frame_step) ) )

        startbyte = max(0, math. ceil( float( align[ 0]['start']) / speedup * conf.feature_fs /  16000) - (conf.extraframes * conf.frame_step))
        endbyte   = math. ceil( float( align[-1][  'end']) / speedup * conf.feature_fs / 16000) + (conf.extraframes * conf.frame_step)

        if conf.debug:
            print ("startbyte: %i endbyte: %i" % (startbyte, endbyte))

        new_align = [];
        for l in align:

            p = {'triphone': l['triphone'], 'model':l['model']}

            # Start from zero! (As the extra padding is already calculated in 'startbyte'
            p['start'] = math.floor(float(l['start']) / speedup * conf.feature_fs / 16000) - (conf.extraframes * conf.frame_step) -  startbyte
            #p['start'] = math.floor(float(l['start']) / speedup ) - startbyte 
            
            # Add 2 * extra frames to the end mark:
            p['end'] = math.ceil(float(l['end']) / speedup * conf.feature_fs / 16000) + (conf.extraframes * conf.frame_step) - startbyte
            #p['end'] = math.floor(float(l['start']) / speedup ) - startbyte  + 2 * conf.extraframes * conf.frame_step

            new_align.append(p)

        if conf.debug:
            pp.pprint(new_align)
        
        if len(new_align) > 0:
            
            preprocessor_key = preprocessors[random.randint( 0, len(preprocessors)-1 ) ]
                       
            clean_preprocessor = conf.preprocessing_scripts['none']

            clean_preprocessed_audio = os.path.join(conf.tmp_dir,
                                            str(conf.tmpfilecounter)+"_"+str(speedup)+"_clean")
    
            if 1 == 1:

                clean_preprocess_progress = Popen([clean_preprocessor['script'],
                                             audiofile,
                                             clean_preprocessed_audio,
                                             str(speedup)
                                        ], stdout=PIPE, stdin=PIPE, stderr=STDOUT).communicate()
                

                try:
                    clean_audiodata = np.fromfile( clean_preprocessed_audio, 'int16', -1)
                except:
                    print("\nPROBLEM!")
                    print("Could not find cleandata features: %s" % clean_preprocessed_audio )
                    continue
                    

                startmark = startbyte # 0  #math.floor(float(new_align[0]['start'])/speedup)
                endmark= endbyte # math.ceil(float(new_align[-1]['end'])/speedup)

                if conf.debug:
                    print ("start feature extraction at %s (%f s) and end at %s (%f s) ==> %i frames"  % (
                            startmark, 
                            (float(startmark)/16000), 
                            endmark, (float(endmark)/16000), 
                            (endmark-startmark)/conf.frame_step) )

                # Communication from: 
                # http://stackoverflow.com/questions/163542/python-how-do-i-pass-a-string-into-subprocess-popen-using-the-stdin-argument


                clean_tmp_input=os.path.join(conf.tmp_dir,str(conf.tmpfilecounter)+"_clean_in")
                clean_tmp_output=os.path.join(conf.tmp_dir,str(conf.tmpfilecounter)+"_clean_out")

                clean_audiodata.tofile(clean_tmp_input, "")

                process_progress = Popen([
                        conf.feature_extraction_script, 
                        clean_tmp_input, 
                        clean_tmp_output, 
                        str(startmark), 
                        str(endmark+conf.frame_leftovers) # , '/tmp/test_clean.wav' 
                ], stdout=PIPE, stdin=PIPE, stderr=STDOUT).communicate()
                

                # Read features from disk and copy all the features into our buffers:

                # What's our startindex?
                # Let's keep track of it in conf.lastframeindex
                
                
                
                if True:
                    clean_feature_list = np.fromfile(clean_tmp_output, dtype='float32', count=-1)

                    try:
                        clean_feature_array = clean_feature_list.reshape([-1,conf.feature_dimension])
                    except:
                        
                        print(process_progress[0].decode('utf-8'))

                        print("Not very compatible: %s" % audiofile)
                        print("clean_feature_list.shape:")
                        print(clean_feature_list.shape)
                        print ("Apparently not divisible by %i" % conf.feature_dimension )

                        sys.exit()

                    features_length_in_frames = clean_feature_array.shape[0]


                cleanphonedata[ conf.lastframeindex : conf.lastframeindex + features_length_in_frames, :] = clean_feature_array[:features_length_in_frames,:]
                


                f_end =  math.floor((endmark-startmark)/conf.frame_step)

                if conf.debug:
                    print ("Utterance data size: %i x %i" % (clean_feature_array).shape)
                '''
                        
                if (clean_feature_array.shape[0] < f_end / 16000 * conf.fs):
                        print ("Not enough clean features for file %s: %i < %i" % 
                               (audiofile, clean_feature_array.shape[0], f_end))
                        #print ("panic save to /tmp/this_is_not_good")
                        np.savetxt('/tmp/this_is_not_good', clean_feature_array, delimiter='\t')
                        #raise ValueError("Not enough features for file %s: %i < %i" % (
                        #        audiofile, 
                        #        clean_feature_array.shape[0], 
                        #        f_end) )                        
                '''
                if False:
                    nothing=1
                else:
                    try: 
                        count += chop_features( conf,
                                                clean_audiodata,
                                                clean_feature_array,
                                                new_align, 
                                                speedup,
                                                preprocessor_string,
                                                cleanphonedata, 
                                                phoneclasses, 
                                                phone_indices,
                                                segment_details,
                                                segment_lengths,
                                                quality_control_audio_files,
                                                startbyte)
                        
                        itemcount=np.count_nonzero(phone_indices[:,1])
                        conf.lastframeindex = phone_indices[ itemcount-1 , 1] + 1


                    except ValueError as error:   
                        print (error)
                        errors.append("Bad amount of data! in %s speedup %s" % (audiofile, speedup))
                        continue

 
                os.remove(clean_preprocessed_audio)
                os.remove(clean_tmp_input)
                os.remove(clean_tmp_output)



    if len(errors)>1:
        print (errors[-1])
    return { "errors": errors, "count" : count }




def extract_collection_and_save( conf, collection, training=True ):

    cleanphonedata = np.zeros([250000,conf.feature_dimension] ,dtype='float32')
    phoneclasses = np.zeros([20000], dtype='uint8')
    phone_indices = np.zeros([20000,2], dtype='uint32')
    segment_lengths = np.zeros([20000], dtype='uint32')
    segment_details = []

    conf.lastframeindex = 0

    errors = []
    quality_control_audio_files = []
    conf.discard_counter = 0

    recipefile = open( collection['recipe'] , 'r')
    recipefilecounter = 0
    too_long_counter = 0
    all_trips_counter = 0

    conf.tmpfilecounter = 0

    progress_interval = math.ceil(collection['numlines']/1000.0)

    statistics_file=conf.statistics_dir+"/"+conf.corpus+"-"+collection['condition']+"-"+collection['name']+".triphone-frame-counts"
    conf.statistics_handle = open(statistics_file, 'w')

    class_file=conf.statistics_dir+"/"+conf.corpus+"-"+collection['condition']+"-"+collection['name']+".triphone-classes"
    class_handle= open(class_file, 'w')

    phone_merge_file=conf.statistics_dir+"/"+conf.corpus+"-"+collection['condition']+"-"+collection['name']+".phone-merge"
    conf.phone_merge_handle = open(phone_merge_file, 'w')

    conf.quality_control_wavdir = os.path.join(conf.pickle_dir, 'control-wav', collection['condition']+"-"+collection['name']+"-classes_"+conf.classes_name)

    mkdir(conf.quality_control_wavdir)

    quality_control_audio_files = {}

    tooshortcount=0


    for r in recipefile.readlines():
        
        preprocessors = ['none']
        
        recipefilecounter += 1
        if conf.debug:
            print ("Item %i/%i" % (recipefilecounter, collection['numlines']) )
            print (r)
        audiofile = re.sub('audio=', r'',  re.findall('audio=/[^ ]+', r)[0]).strip()
        labelfile = re.sub(r'transcript=', r'', re.findall('transcript=/[^ ]+', r)[0]).strip()
    
        if conf.labeltype=="pfstar":
            new_align = process_pfstar_label(conf, labelfile)
        else:
            new_align = process_other_aaltoasrlike_label(conf, labelfile)

        labelstring = get_labelstring( new_align )
        
        conf.phone_merge_handle.write("%s\t%s\n" % (labelfile, labelstring))

        # OK, label file done.
        # Now it's time to process the audio.
        # We'll send to the feature extractor the bits of the file that 
        # match the speech segments.

        if len(new_align) > 0:
            errors_and_count = get_features( conf,
                                             audiofile, 
                                             new_align, 
                                             preprocessors,
                                             cleanphonedata, 
                                             phoneclasses, 
                                             phone_indices, 
                                             segment_details,
                                             segment_lengths,
                                             quality_control_audio_files,
                                             training)
            #triphonedata += data_classes_and_errors["data"]
            #triphoneclasses += data_classes_and_errors["classes"]
            #errors += data_classes_and_errors["errors"]
            #segment_lengths += data_classes_and_errors["segment_lengths"]
            errors += errors_and_count["errors"]
            all_trips_counter += errors_and_count["count"]
            #if len(errors_and_count["errors"] ) > 0:
            #    print ( errors_and_count["errors"] )

        if not conf.debug:
            if (recipefilecounter % int(progress_interval)) == 0:
                sys.stderr.write("\r%0.2f%s %s %s (%i phones, %i discarded, %i errors)" % (
                        100.0*recipefilecounter/collection['numlines'], 
                        "%",
                        collection['condition'], 
                        collection['name'],
                        all_trips_counter,
                        conf.discard_counter,
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

    picklefile = os.path.join(conf.pickle_dir,  collection['name'] + '_' + conf.featuretype + ".pickle2")

    mkdir(conf.pickle_dir)            

    print ("pickling %i phones / %i frames to %s" % ( itemcount, indexcount, picklefile))
                        
    outf = open(picklefile, 'wb')
    # Pickle the list using the highest protocol available.
    pickle.dump( {  'cleandata': cleanphonedata[0:indexcount], 
                    'classes': phoneclasses[0:itemcount],
                    'indices' : phone_indices[0:itemcount,:],
                    'lengths' : segment_lengths[0:itemcount],
                    'details' : segment_details }, outf, protocol=pickle.HIGHEST_PROTOCOL)
                
    
