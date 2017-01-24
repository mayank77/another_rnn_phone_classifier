

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
# A function that will be useful:
#

def mkdir(path):
    try:
        os.makedirs(path)        
    except OSError as exc:  # Python >2.5
        #print ("dir %s exists" % path)
        dummy = 1





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

                        if (int(phone['end'])-int(phone['start']))/conf.frame_step < 13 or (int(phone['end'])-int(phone['start']) > conf.max_phone_length ):
                            conf.discard_counter+=1
                            #print "Discarding %i/%i: %s: (Too short! Discards: %0.2f%s)" % (recipefilecounter, collection['numlines'], labelfile, 100.0*conf.discard_counter/collection['numlines'],"%" )

                            discard = True

                        #elif (int(phone['end'])-int(phone['start']))/conf.frame_step > 40 and '_' not in phone['triphone']:
                        #    #print "Discarding %i/%i: %s (Too Long! Discards: %0.2f%s)" % (recipefilecounter, collection['numlines'], labelfile, 100.0*conf.discard_counter/collection['numlines'],"%" )
                        #    conf.discard_counter+=1
                        #    discard = True

                        #if conf.debug:
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
                conf.discard_counter += 1
                del new_align[-1]
            

    return new_align
    


# In[457]:

def get_labelstring( new_align ):
    labelstring = ''
    for phone in new_align:
        labelstring += '.'+phone['model']
    return labelstring    


# In[474]:

def chop_features( conf, cleanaudiodata, noisyaudiodata, noisy_feature_array, clean_feature_array, new_align, speedup, preprocessor_string, 
                   cleanphonedata, noisyphonedata, phoneclasses, phone_indices, segment_details, segment_lengths, quality_control_audio_files ):
    
    count = 0
    startmark = int(math.ceil(float(new_align[0]['start'])*(conf.feature_fs/conf.audio_fs)/speedup))
    #endmark= int(math.floor(float(new_align[-1]['end'])*(conf.feature_fs/conf.audio_fs)/speedup))
    
    tooshortcount=0
    
    for l in new_align:                

        lkey = l['sortable']
        mkey = l['model']
        
        if 1 == 1:
            tp = l['triphone']

            l_start = math.ceil((float(l['start'])/speedup-startmark)*(conf.feature_fs/conf.audio_fs)/conf.frame_step)        
            l_end =  math.floor((float(l['end'])/speedup-startmark)*(conf.feature_fs/conf.audio_fs)/conf.frame_step)  

            
           
            l_length = l_end - l_start
            
            if conf.debug:
                print ("Segment length: %i" % l_length)
            if (l_length < 4):
                tooshortcount+=1
                continue

            # For conf.debugging, let's write this stuff to disk:
            if mkey not in quality_control_audio_files.keys():
                qual_file = os.path.join(conf.quality_control_wavdir,  mkey+".raw-"+str(conf.feature_fs)+"hz-16bit-signed-integer")
                quality_control_audio_files[mkey] = open( qual_file , 'wb')

            win_i=0
            win_len=256
            max_val=32000
            
            audio_start = int(math.floor(float(l['start'])/speedup*(conf.feature_fs/conf.audio_fs)))
            audio_end = int(math.ceil(float(l['end'])/speedup*(conf.feature_fs/conf.audio_fs)))
            
            if conf.debug:
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

            conf.statistics_handle.write("%i\t%s\n" % (l_length, tp))

            if conf.debug:
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
            phoneclasses[index] =  conf.class_def[l['model']]['class']
            
            segment_details.append( l['triphone'] + ' ' + preprocessor_string )
            
            if index == 0:
                phone_indices[index, :] = [0, l_length]
            else:
                phone_indices[index, :] = [ phone_indices[index-1][1] + 1, 
                                            phone_indices[index-1][1] + 1 + l_length ]
                
            #if phone_indices[index, 1] > phonedata.shape[0]:
            #    print("Adding 50000 more entries to phonedata")
            #    phonedata = np.concatenate( (phonedata, np.zeros([50000, conf.feature_dimension], dtype='float32')), axis=0)
                
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

def get_features( conf, audiofile, align,  preprocessors, cleanphonedata, noisyphonedata, phoneclasses, phone_indices, 
                  segment_details, segment_lengths, quality_control_audio_files, training=True):
    
    count = 0
    errors = []
    
    if training:
        speedups = [0.96, 1.0, 1.04 ]
    else:
        speedups = [1.0]
        
    for speedup in speedups:

        new_align = [];
        for l in align:                
            #if random.random() < conf.class_def[l['model']]["probability"]:
                #if len(new_align) == 0:
                #    startmark = math.floor(float(l['start'])/speedup*fs/16000)
                #endmark= math.ceil(float(l['end'])/speedup*fs/16000)

                #l['start'] -= startmark
                #l['end'] -= startmark
                new_align.append(l)
        if conf.debug:
            print(new_align)
        
        if len(new_align) > 0:
            
            preprocessor_key = preprocessors[random.randint( 0, len(preprocessors)-1 ) ]
            
            noisy_preprocessor = conf.preprocessing_scripts[preprocessor_key]
                                                                    
            noisy_preprocessed_audio=os.path.join(conf.tmp_dir,
                                            str(conf.tmpfilecounter)+"_"+str(speedup)+"_preprocessed")

            
            clean_preprocessor = conf.preprocessing_scripts['none']

            clean_preprocessed_audio = os.path.join(conf.tmp_dir,
                                            str(conf.tmpfilecounter)+"_"+str(speedup)+"_clean")
            #for preprocessor in preprocessors:
            #    preprocessor =  conf.preprocessing_scripts[random.randint(0, len(conf.preprocessing_scripts)-1)]
    
            if 1 == 1:
                if conf.debug:
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
        
                if conf.debug:                    
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

                if conf.debug:
                    print ("start feature extraction at %s (%f s) and end at %s (%f s) ==> %i frames"  % (
                            startmark, 
                            (float(startmark)/16000), 
                            endmark, (float(endmark)/16000), 
                            (endmark-startmark)/conf.frame_step) )

                # Communication from: 
                # http://stackoverflow.com/questions/163542/python-how-do-i-pass-a-string-into-subprocess-popen-using-the-stdin-argument

                noisy_tmp_input=os.path.join(conf.tmp_dir,str(conf.tmpfilecounter)+"_noisy_in")
                noisy_tmp_output=os.path.join(conf.tmp_dir,str(conf.tmpfilecounter)+"_noisy_out")
                noisy_audiodata.tofile(noisy_tmp_input, "")

                clean_tmp_input=os.path.join(conf.tmp_dir,str(conf.tmpfilecounter)+"_clean_in")
                clean_tmp_output=os.path.join(conf.tmp_dir,str(conf.tmpfilecounter)+"_clean_out")

                clean_audiodata.tofile(clean_tmp_input, "")

                process_progress = Popen([
                        conf.feature_extraction_script, 
                        noisy_tmp_input, 
                        noisy_tmp_output, 
                        str(startmark), 
                        str(endmark+conf.frame_leftovers) #, '/tmp/test_noisy.wav'
                ], 
                                         stdout=PIPE, stdin=PIPE, stderr=STDOUT).communicate()

                process_progress = Popen([
                        conf.feature_extraction_script, 
                        clean_tmp_input, 
                        clean_tmp_output, 
                        str(startmark), 
                        str(endmark+conf.frame_leftovers) # , '/tmp/test_clean.wav' 
                ], stdout=PIPE, stdin=PIPE, stderr=STDOUT).communicate()

                noisy_feature_list = np.fromfile(noisy_tmp_output, dtype='float32', count=-1)
                noisy_feature_array = noisy_feature_list.reshape([-1,conf.feature_dimension])

                clean_feature_list = np.fromfile(clean_tmp_output, dtype='float32', count=-1)
                clean_feature_array = clean_feature_list.reshape([-1,conf.feature_dimension])

                
                f_end =  math.floor((endmark-startmark)/conf.frame_step)

                if conf.debug:
                    print ("Utterance data size: %i x %i" % (noisy_feature_array).shape)

                if (noisy_feature_array.shape[0] < f_end/16000*conf.fs):
                        print ("Not enough noisy features for file %s: %i < %i" % 
                               (audiofile, noisy_feature_array.shape[0], f_end))
                        print ("panic save to /tmp/this_is_not_good")
                        np.savetxt('/tmp/this_is_not_good', noisy_feature_array, delimiter='\t')
                        raise ValueError("Not enough features for file %s: %i < %i" % (
                                audiofile, 
                                noisy_feature_array.shape[0], 
                                f_end) )
                        
                if (clean_feature_array.shape[0] < f_end/16000*conf.fs):
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
                        count += chop_features( conf,
                                                clean_audiodata,
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



def extract_collection_and_save( conf, collection ):

    cleanphonedata = np.zeros([250000,conf.feature_dimension] ,dtype='float32')
    noisyphonedata = np.zeros([250000,conf.feature_dimension] ,dtype='float32')
    phoneclasses = np.zeros([20000], dtype='uint8')
    phone_indices = np.zeros([20000,2], dtype='uint32')
    segment_lengths = np.zeros([20000], dtype='uint32')
    segment_details = []
    
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
        
        preprocessors = ['overdrive','babble', 'humming'] # ['none','none','overdrive','babble', 'humming']
        
        recipefilecounter += 1
        if conf.debug:
            print ("Item %i/%i" % (recipefilecounter, collection['numlines']) )

        audiofile = re.sub('audio=', r'',  re.findall('audio=/[^ ]+', r)[0]).strip()
        labelfile = re.sub(r'transcript=', r'', re.findall('transcript=/[^ ]+', r)[0]).strip()
    
        new_align = process_pfstar_label(conf, labelfile)
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

    picklefile = os.path.join(conf.pickle_dir,  collection['name'] + '_' + conf.featuretype + ".pickle")

    mkdir(conf.pickle_dir)            

    print ("pickling %i phones / %i frames to %s" % ( itemcount, indexcount, picklefile))
                        
    outf = open(picklefile, 'wb')
    # Pickle the list using the highest protocol available.
    pickle.dump( {  'cleandata': cleanphonedata[0:indexcount], 
                    'noisydata': noisyphonedata[0:indexcount], 
                    'classes': phoneclasses[0:itemcount],
                    'indices' : phone_indices[0:itemcount,:],
                    'lengths' : segment_lengths[0:itemcount],
                    'details' : segment_details }, outf, protocol=pickle.HIGHEST_PROTOCOL)
                
    
