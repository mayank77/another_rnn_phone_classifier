#!/usr/bin/python3

import numpy as np
import pickle
import random
import math


from sklearn.preprocessing import normalize
from scipy.ndimage.interpolation import zoom


class phone_stash:
    
    data = False

    indices = False
    classes = False
    lengths = False

    batch_counter = 0
    num_examples = 0
    max_len = 0
    num_classes = 0

    
    featdim=66
    usedim=[   0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,
              20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,
              40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,
              60,61,62,63,64,65 ]
    min_samples=100

    data_order=False
    balanced_data_order = False
    classcounts = False
    ok_classes = False

    samplings = [ -4, 
                  -3, -3, 
                  -2, -2, -2, -2, 
                  -1, -1, -1, -1, -1, -1, -1, -1, 
                   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                   1,  1,  1,  1,  1,  1,  1,  1,
                   2,  2,  2,  2,  
                   3,  3,
                   4  ]


    mean=False
    std=False

    seed = 1000
    
    def __init__(self, pickles_array, zmean=[], zstd=[], max_len=0, use_clean=False, show_samples=False):
        self.data = np.zeros([0,self.featdim], dtype='float32')

        self.indices = np.zeros([0,2], dtype='uint32')
        self.classes = np.zeros([0], dtype='uint8')
        self.lengths = np.zeros([0], dtype='uint32')

        self.fig = False

        last_classindex = 0
        last_phoneindex = 0

        for picklefile in pickles_array:            
            data_and_classes = pickle.load( open(picklefile, 'rb'))
            #print (data_and_classes.keys())

            # Get rid of trailing zeros if there are any:
            datacount = np.count_nonzero(data_and_classes["classes"])
            #data_and_classes["data"] = data_and_classes["data"][:datacount]
            #data_and_classes["lengths"] = data_and_classes["lengths"][:datacount]
            #data_and_classes["indices"] = data_and_classes["indices"][:datacount,:]
            batch_lastphoneindex = data_and_classes["indices"][-1,1]
            #data_and_classes["data"] = data_and_classes["data"][:lastphoneindex,:]
            
            
            self.classes = np.concatenate( (self.classes, data_and_classes["classes"][:datacount]) )
            self.lengths = np.concatenate( (self.lengths, data_and_classes["lengths"][:datacount]) )
            self.indices = np.concatenate( (self.indices, data_and_classes["indices"][:datacount,:] + last_phoneindex) )
            
            #print ("Appending %i items to noisydata" % lastphoneindex)

            if (use_clean):
                self.data = np.concatenate( (self.data, data_and_classes["cleandata"][:batch_lastphoneindex,:]))
            else:
                self.data = np.concatenate( (self.data, data_and_classes["noisydata"][:batch_lastphoneindex,:]))

            #print ("Noisydata shape:")
            #print (self.data.shape)

            last_classindex = self.classes.shape[0]
            last_phoneindex = self.data.shape[0]

            print ("last_classindex %i last_phoneindex %i" % ( last_classindex, last_phoneindex  ) )

        # Go for the annoying but necessary task of z-normalising the data:

        self.data = np.log(self.data+0.001)

        if len(zmean):
            self.mean = zmean
        else:
            self.mean = np.mean(self.data, 0)

        if len(zstd):
            self.std = zstd
        else:
            self.std = np.std(self.data,0)

        self.data = (self.data-self.mean)/self.std

        self.num_examples = len(self.classes)
        if max_len > 0:
            self.max_len = max_len
        else:
            self.max_len = np.max(self.lengths)
        self.num_classes = np.max(self.classes)+1
        
        # Quick hack, let's not do more of these!
        self.num_classes = 119



        self.classcounts = np.bincount(self.classes)
        self.ok_classes = np.where(self.classcounts>self.min_samples)[0]

        print ("Lengths: classes %i, indices %i, lengths %i, data %i" % (
               self.classes.shape[0],
               self.indices.shape[0],
               self.lengths.shape[0],
               self.data.shape[0] ))

        self.data_order=np.arange(self.num_examples)

        # Sanity check: If we shuffle keys, we'd expect the system to work worse
        #random.shuffle( self.classes, self.batch_seed )

        if show_samples:
            import matplotlib.pyplot as plt

            sample_classes = np.arange(self.num_classes-1).tolist()
            random.shuffle(sample_classes)

            # Plot some log-features from random classes:

            fcount=1
            # row and column sharing
            for class_counter in [1,2,3]:                
                sample_class = sample_classes.pop()
                print ("showing samples from class %i" % sample_class)
                sample_selection = np.where(self.classes == sample_class)[0].tolist()
                random.shuffle(sample_selection)
                for sample_counter in [1,2,3]: 
                    plt.figure(fcount)
                    plt.ioff()
                    key = sample_selection.pop()                    
                    [datastart, dataend] = self.indices[ key ]
                    print ("   showing item with key %i, start: %i, end: %i" % (key, datastart, dataend) )                                    
                    sample = self.data[ datastart:dataend, :]
                    sample = np.log(sample * self.std + self.mean)[:,1:]
                    #sample[:5,:] = sample[:5,:]*0.5;
                    #sample[-5:,:] = sample[-5:,:]*0.5;

                    plt.pcolor( sample , cmap='jet')
                    plt.set_title = ('Class %i key %i' % (sample_class, key) )
                    plt.colorbar()
                    plt.show()

            


    def batch_seed(self):
        return self.seed * math.pi % 1
    
    def next_batch(self, batch_size, sequence_length=-1, test=False):
        #print ("batch starting at %i" % self.batch_counter)
        # Get the next batch_size samples and pad with zeros to make equal length
        #return train_data[train_batch_counter:train_batch_counter+batch_size]

        batch_start = self.batch_counter        
        batch_end = min(self.batch_counter +batch_size, self.classes.shape[0])
        
        batch_size = batch_end-batch_start

        keys=self.data_order[batch_start:batch_end]

        if (batch_start == batch_end):
            return False
        #batch_maxlen = self.max_len
        #batch_maxlen = np.max(self.lengths[keys])
        if (sequence_length > 0):
            batch_maxlen = sequence_length
        else:
            batch_maxlen = np.max(self.lengths[keys])



        batch_data = np.zeros([ batch_size, batch_maxlen, self.featdim ])
        batch_lens = np.zeros([ batch_size ])

        for i in range(batch_size):
            [datastart, dataend] = self.indices[ keys[i] ]
            #print("start %i, end %i" % (datastart, dataend))
            wanted_data = self.data[ datastart:dataend, :]
            batch_data[i,0:wanted_data.shape[0],:] = wanted_data[0: min(wanted_data.shape[0],batch_maxlen), self.usedim] 
            batch_lens[i]=min(wanted_data.shape[0],batch_maxlen)
         
        # No, let's not do batch normalisation of features
        #if test:
        #    batch_data = (batch_data-self.mean)/self.std
        #else:
        #    batch_data = normalize(batch_data, norm='l2', axis=0 )

        # One-hot encode classes:
        batch_classes = np.zeros([batch_size, self.num_classes])
        batch_classes[ np.arange(batch_size), self.classes[keys]] = 1 

        self.batch_counter += (batch_end-batch_start)

        return [ batch_data, batch_classes, batch_size, batch_maxlen , batch_lens]


    def get_eval_batch(self, samples_per_class, sequence_length=-1, test=False):
        #
        # Get a test batch with a balanced set of phones:
        #

        keys=np.zeros([0], dtype='int32')
        for n in self.ok_classes:
            new_keys = np.where(self.classes==n)[0]
            random.shuffle(new_keys)
            keys =  np.concatenate([keys,new_keys[:samples_per_class]])

        batch_size = len(keys)
        #batch_maxlen = np.max(self.lengths[ keys ])
        if (sequence_length > 0):
            batch_maxlen = sequence_length
        else:
            batch_maxlen = np.max(self.lengths[keys])


        batch_data = np.zeros([ batch_size, batch_maxlen, self.featdim ])
        batch_lens = np.zeros([ batch_size ])

        for i in range(batch_size):
            [datastart, dataend] = self.indices[ keys[i] ]
            #print("start %i, end %i" % (datastart, dataend))
            wanted_data = self.data[ datastart:dataend, :]
            batch_data[i,0:wanted_data.shape[0],:] = wanted_data[0: min(wanted_data.shape[0],batch_maxlen), self.usedim] 
            batch_lens[i]=min(wanted_data.shape[0],batch_maxlen)

        # One-hot encode classes:
        batch_classes = np.zeros([batch_size, self.num_classes])
        batch_classes[ np.arange(batch_size), self.classes[keys]] = 1 
        
        return [ batch_data, batch_classes, batch_size, batch_maxlen , batch_lens]



    def next_train_round(self):
        self.seed += 1
        random.shuffle( self.data_order, self.batch_seed )
        self.batch_counter=0



    
    def next_balanced_batch(self, batch_size, sequence_length=-1, test=False):
        #print ("batch starting at %i" % self.batch_counter)
        # Get the next batch_size samples and pad with zeros to make equal length
        #return train_data[train_batch_counter:train_batch_counter+batch_size]

        batch_start = self.balanced_batch_counter        
        batch_end = min(self.balanced_batch_counter + batch_size, self.balanced_data_order.shape[0])
        
        batch_size = batch_end-batch_start

        keys=self.balanced_data_order[batch_start:batch_end]

        if (batch_start == batch_end):
            return [ np.zeros([0,0,self.featdim]), np.zeros([0]), 0, 0 , np.zeros([0])]
        #batch_maxlen = self.max_len
        if (sequence_length > 0):
            batch_maxlen = sequence_length
        else:
            batch_maxlen = np.max(self.lengths[keys]+4)


        batch_data = np.zeros([ batch_size, batch_maxlen, self.featdim ])
        batch_lens = np.zeros([ batch_size ])

        for i in range(batch_size):
            [datastart, dataend] = self.indices[ keys[i] ] + np.round(np.random.normal(0, 1.5, self.indices[ keys[i] ].shape ))
            
            if datastart<0:
                datastart = 0

            if dataend > self.data.shape[0]:
                dataend = self.data.shape[0]

            if (dataend <= datastart):
                dataend = min(datastart +5,  self.data.shape[0])
                datastart = max(datastart-5, 0)

            #print("start %i, end %i" % (datastart, dataend))
            wanted_data = self.data[ int(datastart):int(dataend), :]
            #print("wanted data length: %i" %  wanted_data.shape[0]  )


            sampling = random.choice(self.samplings)
            if sampling != 0:
                wanted_data = zoom(wanted_data, zoom=( (wanted_data.shape[0]+sampling)/wanted_data.shape[0], 1), order=3)
            wanted_data += np.random.normal(0,0.2, wanted_data.shape)
            
            batch_data[i,0:wanted_data.shape[0],:] = wanted_data[0: min(wanted_data.shape[0],batch_maxlen), self.usedim] 
            batch_lens[i]=min(wanted_data.shape[0],batch_maxlen)
         
        # No, let's not do batch normalisation of features
        #if test:
        #    batch_data = (batch_data-self.mean)/self.std
        #else:
        #    batch_data = normalize(batch_data, norm='l2', axis=0 )

        # One-hot encode classes:
        batch_classes = np.zeros([batch_size, self.num_classes])
        #print(keys)
        #print(self.classes[keys])
        batch_classes[ np.arange(batch_size), self.classes[keys]] = 1 

        self.balanced_batch_counter += (batch_end-batch_start)

        return [ batch_data, batch_classes, batch_size, batch_maxlen , batch_lens]



    def next_balanced_round(self):
        self.seed += 1

        #samplecap = np.median(classcounts)

        self.balanced_data_order = np.zeros([ self.ok_classes.shape[0] * max(self.classcounts) ], dtype="int32")
        
        items_per_class = max(self.classcounts)
        itempointer = 0
        for i in self.ok_classes:            
            class_keys = np.where(self.classes==i)[0]
            random.shuffle(class_keys, self.batch_seed)
            class_keys = np.tile(class_keys , (math.ceil(items_per_class/self.classcounts[i]) ))
            class_keys =  class_keys[0:items_per_class]

            self.balanced_data_order[itempointer:itempointer+items_per_class] = class_keys
            itempointer+=items_per_class
        '''
        random.shuffle( self.balanced_data_order, self.batch_seed )
        self.seed += 1
        random.shuffle( self.balanced_data_order, self.batch_seed )
        self.seed += 1
        random.shuffle( self.balanced_data_order, self.batch_seed )
        self.seed += 1
        random.shuffle( self.balanced_data_order, self.batch_seed )
        self.seed += 1
        random.shuffle( self.balanced_data_order, self.batch_seed )
        '''
        #for i in range(5):
        random.shuffle( self.balanced_data_order)
        self.balanced_batch_counter=0

