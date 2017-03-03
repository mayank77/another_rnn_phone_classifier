#!/usr/bin/python3


'''

Relying on this file is probably not a good idea!

Look instead into ../ipython_notebooks/dnnutil/phonestash.py

'''



import numpy as np
import pickle
import random
import math

class phone_stash:
    
    noisydata = False
    cleandata = False
    indices = False
    classes = False
    lengths = False

    batch_counter = 0
    num_examples = 0
    max_len = 0
    num_classes = 0

    featdim=66

    data_order=False

    mean=False
    std=False

    seed = 1000
    
    def __init__(self, pickles_array, zmean=[], zstd=[], max_len=0, use_clean_references=False, show_samples=False):
        self.noisydata = np.zeros([0,self.featdim], dtype='float32')
        self.cleandata = np.zeros([0,self.featdim], dtype='float32')
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
            self.noisydata = np.concatenate( (self.noisydata, data_and_classes["noisydata"][:batch_lastphoneindex,:]))
            if (use_clean_references):
                self.cleandata = np.concatenate( (self.cleandata, data_and_classes["cleandata"][:batch_lastphoneindex,:]))

            #print ("Noisydata shape:")
            #print (self.noisydata.shape)

            last_classindex = self.classes.shape[0]
            last_phoneindex = self.noisydata.shape[0]

            print ("last_classindex %i last_phoneindex %i" % ( last_classindex, last_phoneindex  ) )

        # Go for the annoying but necessary task of z-normalising the data:

        if len(zmean):
            self.mean = zmean
        else:
            self.mean = np.mean(self.noisydata, 0)

        if len(zstd):
            self.std = zstd
        else:
            self.std = np.std(self.noisydata,0)

        self.noisydata = (self.noisydata-self.mean)/self.std

        if (use_clean_references):
            self.cleandata = (self.cleandata-self.mean)/self.std
        
        self.num_examples = len(self.classes)
        if max_len > 0:
            self.max_len = max_len
        else:
            self.max_len = np.max(self.lengths)
        self.num_classes = np.max(self.classes)+1
        
        print ("Lengths: classes %i, indices %i, lengths %i, data %i" % (
               self.classes.shape[0],
               self.indices.shape[0],
               self.lengths.shape[0],
               self.noisydata.shape[0] ))

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
                    sample = self.noisydata[ datastart:dataend, :]
                    sample = np.log(sample * self.std + self.mean)[:,1:]
                    #sample[:5,:] = sample[:5,:]*0.5;
                    #sample[-5:,:] = sample[-5:,:]*0.5;

                    plt.pcolor( sample , cmap='jet')
                    plt.set_title = ('Class %i key %i' % (sample_class, key) )
                    plt.colorbar()
                    plt.show()

            


    def batch_seed(self):
        return self.seed * math.pi % 1
    
    def next_batch(self, batch_size, get_clean_references=False, another_batch_array=None):
        #print ("batch starting at %i" % self.batch_counter)
        # Get the next batch_size samples and pad with zeros to make equal length
        #return train_data[train_batch_counter:train_batch_counter+batch_size]

        #batch_array.fill(0)

        batch_start = self.batch_counter
        batch_end = min(self.batch_counter +batch_size, self.classes.shape[0])
        
        keys=self.data_order[batch_start:batch_end]

        if (batch_start == batch_end):
            return False
        batch_maxlen = self.max_len
        #batch_maxlen = np.max(self.lengths[batch_start:batch_end])


        batch_data = np.zeros([ batch_size, batch_maxlen, self.featdim ])
        if get_clean_references:
            clean_batch_data = np.zeros([ batch_size, batch_maxlen, self.featdim ])
        for i in range(batch_size):
            [datastart, dataend] = self.indices[ keys[i] ]
            #print("start %i, end %i" % (datastart, dataend))

            wanted_data = self.noisydata[ datastart:dataend, :]

            batch_data[i,0:wanted_data.shape[0],:] = wanted_data[0: min(wanted_data.shape[0],batch_maxlen), :] 
            #batch_array[i, min(wanted_data.shape[0],batch_maxlen) ] = wanted_data[0: min(wanted_data.shape[0],batch_maxlen), :]
           
            if get_clean_references:
                wanted_clean_data = self.cleandata[ datastart:dataend, :]
                clean_batch_data[i,0:wanted_data.shape[0],:] = wanted_data[0: min(wanted_data.shape[0],batch_maxlen), :] 
                
    
        # One-hot encode classes:
        batch_classes = np.zeros([batch_size, self.num_classes])
        batch_classes[ np.arange(batch_size), self.classes[keys]] = 1 

        self.batch_counter += (batch_end-batch_start)
        if get_clean_references:
            return [ batch_data, batch_clean_data, batch_classes, batch_maxlen ]

        else:
            return [ batch_data, batch_classes, batch_maxlen ]

    def next_train_round(self):
        self.seed += 1
        random.shuffle( self.data_order, self.batch_seed )
        self.batch_counter=0

