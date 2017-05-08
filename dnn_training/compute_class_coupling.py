#!/usr/bin/python3




import numpy as np
import pickle, os, random, math, sys

#from sklearn.metrics import confusion_matrix
def confusion_matrix(p1,p2):
    cf = np.zeros([int(np.max(p1)+1),  int(np.max(p2)+1)])
    for i in np.arange(0,p1.shape[0], dtype='int'):
        cf[  int(p1[i]), int(p2[i]) ] += 1
    return cf


#statfile='/m/triton/scratch/work/rkarhila/another_rnn_phone_classifier//models/rnn4x1000-learningrate0.00050-dropout0.6-classbalance0.5-triton-a/cp28109_en_uk_id_y_and_prediction'




ref1=np.loadtxt('/m/triton/scratch/work/rkarhila/another_rnn_phone_classifier/models/en_only-rnn4x1000-learningrate0.00050-dropout0.6-classbalance0.5-triton-a/cp10000_eval_uk_confusion_matrix')

sum_r1=ref1.sum(1)
sum_r1[sum_r1==0]=1
norm_ref1=(ref1.T/sum_r1.T).T


cf1=np.loadtxt('/m/triton/scratch/work/rkarhila/another_rnn_phone_classifier/models/en_only-rnn4x1000-learningrate0.00050-dropout0.6-classbalance0.5-triton-a/cp10000_finnish_train_data_confusion_matrix_work_in_progress')

sum_m1=cf1.sum(1)
sum_m1[sum_m1==0]=1      
norm_m1=(cf1.T/sum_m1.T).T


ref2=np.loadtxt('/m/triton/scratch/work/rkarhila/another_rnn_phone_classifier/models/fi_only-rnn4x1000-learningrate0.00050-dropout0.6-classbalance0.5-triton-a/cp26000_eval_fi_confusion_matrix')

sum_r2=ref2.sum(1)
sum_r2[sum_r2==0]=1
norm_ref2=(ref2.T/sum_r2.T).T

cf2=np.loadtxt('/m/triton/scratch/work/rkarhila/another_rnn_phone_classifier/models/fi_only-rnn4x1000-learningrate0.00050-dropout0.6-classbalance0.5-triton-a/cp26000_english_train_data_confusion_matrix_work_in_progress')

sum_m2=cf2.sum(1)
sum_m2[sum_m2==0]=1      
norm_m2=(cf2.T/sum_m2.T).T


'''
predictions=np.loadtxt(statfile)

cf = confusion_matrix(predictions[:,1], predictions[:,2])#np.zeros([int(np.max(predictions[:,1])+1),  int(np.max(predictions[:,2])+1)])
#for i in np.arange(0,predictions.shape[0], dtype='int'):
#    cf[ int(predictions[i,1]), int(predictions[i,2]) ] += 1
              
print(cf.shape)

np.savetxt('/tmp/conf', cf)

#cf = np.loadtxt('/tmp/conf')
'''


np.savetxt('/tmp/confnorm1', norm_m1)
np.savetxt('/tmp/confnorm2', norm_m2)

threshold = 0.3


for i in range(46):
    bestmatch = np.argmax(norm_m2[i,:])
    if norm_m2[ i, bestmatch  ] > threshold and norm_m1[ bestmatch, i ] > threshold:
        
        rel1=(norm_m2[ i, bestmatch  ] - norm_ref2[bestmatch,bestmatch]) #/  norm_ref2[bestmatch,bestmatch]
        rel2=(norm_m1[ bestmatch, i ] - norm_ref1[i,i]) #/ norm_ref1[i,i]  
        print("%03i -> %03i\t _%i_ \t _%i_ \t->%0.2f\t<-%0.2f\t(%+0.2f\t%+0.2f)" % (i,bestmatch, i, bestmatch, norm_m2[ i, bestmatch ], norm_m1[ bestmatch, i ], rel1, rel2 ))
        
        

print ("=========")

for i in range(45,118):
    bestmatch = np.argmax(norm_m1[i,:])
    if norm_m1[i, bestmatch ] > threshold and norm_m2[ bestmatch, i  ] > threshold :
        rel1=norm_m1[ i, bestmatch  ] - norm_ref1[bestmatch,bestmatch] 
        rel2=norm_m2[ bestmatch, i ] - norm_ref2[i,i]      
        print("%03i -> %03i\t _%i_ \t _%i_ \t->%0.2f\t<-%0.2f\t(%0.2f\t%0.2f)" % (i, bestmatch, i, bestmatch, norm_m1[i, bestmatch ], norm_m2[ bestmatch, i ], rel1, rel2 ) )
