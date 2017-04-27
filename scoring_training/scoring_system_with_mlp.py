#!/usr/bin/python3
# coding: utf-8

#import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import re
#from scipy.optimize import lsq_linear
#from scipy.optimize import least_squares
from scipy import optimize
from scipy.linalg import lstsq
import scipy

classdict = {"_1_" : u"n",
             "_2_" : u"t",
             "_3_" : u"ɪ",
             "_4_" : u"s",
             "_5_" : u"ə",
             "_6_" : u"r",
             "_7_" : u"l",
             "_8_" : u"k",
             "_9_" : u"z",
             "_10_" : u"aɪ",
             "_11_" : u"f",
             "_12_" : u"iː",
             "_13_" : u"ɑː",
             "_14_" : u"d",
             "_15_" : u"ɛ",
             "_16_" : u"v",
             "_17_" : u"m",
             "_18_" : u"b",
             "_19_" : u"ɔː",
             "_20_" : u"ð",
             "_21_" : u"eɪ",
             "_22_" : u"p",
             "_23_" : u"æ",
             "_24_" : u"w",
             "_25_" : u"oʊ",
             "_26_" : u"uː",
             "_27_" : u"ɡ",
             "_28_" : u"ɒ",
             "_29_" : u"j",
             "_30_" : u"θ",
             "_31_" : u"ɪər",
             "_32_" : u"h",
             "_33_" : u"ŋ",
             "_34_" : u"ʌ",
             "_35_" : u"dʒ",
             "_36_" : u"tʃ",
             "_37_" : u"ʃ",
             "_38_" : u"aʊ",
             "_39_" : u"ɜː",
             "_40_" : u"ɛər",
             "_41_" : u"ʊ",
             "_42_" : u"oɪ",
             "_43_" : u"ʒ",
             "_44_" : u"ɔːr",
             "_45_" : u"sil" ,
             "_46_" : u'_a' ,
             "_47_" : u'_ä' ,
             "_48_" : u'_aa' ,
             "_49_" : u'_ää' ,
              "_50_" : u'_ae' ,
              "_51_" : u'_äe' ,
              "_52_" : u'_ai' ,
              "_53_" : u'_äi' ,
              "_54_" : u'_ao' ,
              "_55_" : u'_au' ,
              "_56_" : u'_äy' ,
              "_57_" : u'_b' ,
              "_58_" : u'_d' ,
              "_59_" : u'_e' ,
              "_60_" : u'_ea' ,
              "_61_" : u'_eä' ,
              "_62_" : u'_ee' ,
              "_63_" : u'_ei' ,
              "_64_" : u'_eo' ,
              "_65_" : u'_eu' ,
              "_66_" : u'_ey' ,
              "_67_" : u'_f' ,
              "_68_" : u'_g' ,
              "_69_" : u'_h' ,
              "_70_" : u'_hh' ,
              "_71_" : u'_i' ,
              "_72_" : u'_ia' ,
              "_73_" : u'_iä' ,
              "_74_" : u'_ie' ,
              "_75_" : u'_ii' ,
              "_76_" : u'_io' ,
              "_77_" : u'_iu' ,
              "_78_" : u'_j' ,
              "_79_" : u'_k' ,
              "_80_" : u'_kk' ,
              "_81_" : u'_l' ,
              "_82_" : u'_ll' ,
              "_83_" : u'_m' ,
              "_84_" : u'_mm' ,
              "_85_" : u'_n' ,
              "_86_" : u'_ng' ,
              "_87_" : u'_ngng' ,
              "_88_" : u'_nn' ,
              "_89_" : u'_o' ,
              "_90_" : u'_ö' ,
              "_91_" : u'_oa' ,
              "_92_" : u'_oe' ,
              "_93_" : u'_oi' ,
              "_94_" : u'_öi' ,
              "_95_" : u'_oo' ,
              "_96_" : u'_öö' ,
              "_97_" : u'_ou' ,
              "_98_" : u'_öy' ,
              "_99_" : u'_p' ,
              "_100_" : u'_pp' ,
              "_101_" : u'_r' ,
              "_102_" : u'_rr' ,
              "_103_" : u'_s' ,
              "_104_" : u'_ss' ,
              "_105_" : u'_t' ,
              "_106_" : u'_tt' ,
              "_107_" : u'_u' ,
              "_108_" : u'_ua' ,
              "_109_" : u'_ue' ,
              "_110_" : u'_ui' ,
              "_111_" : u'_uo' ,
              "_112_" : u'_uu' ,
              "_113_" : u'_v' ,
              "_114_" : u'_y' ,
              "_115_" : u'_yä' ,
              "_116_" : u'_yi' ,
              "_117_" : u'_yö' ,
              "_118_" : u'_yy' }

phonedict = {}
for key in classdict.keys():
    phonedict[ classdict[key] ] = int(key.replace('_',''))


#print(phonedict)

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



#LOG_DIR='/tmp/tensorflow_logs/copy12-rnn384-d/'

checkpoint=58055#48055#20250

LOG_DIR='../models/rnn512-e/'
LOG_DIR += 'testscores-mce_b-%i/' % checkpoint

prediction_files={ 'native' : LOG_DIR + "players_native_id_y_and_prediction",
                  'good': LOG_DIR + "players_good_id_y_and_prediction",
                  'ok': LOG_DIR + "players_ok_id_y_and_prediction",
                  'bad': LOG_DIR + "players_bad_id_y_and_prediction" }


test_corpus = "more_fysiak-gamedata-2-aligned_with_mc_b"
test_pickle_dir='../features/work_in_progress/'+test_corpus+'/pickles'

phone_pickles = { 'bad' : os.path.join(test_pickle_dir, 'disqualified-mc_b_melbin36_and_f0_alldata.pickle2'),
                  'ok' : os.path.join(test_pickle_dir, 'some_stars-mc_b_melbin36_and_f0_alldata.pickle2'),
                  'good' : os.path.join(test_pickle_dir, 'lots_of_stars-mc_b_melbin36_and_f0_alldata.pickle2'),
                  'native' : os.path.join(test_pickle_dir, 'native_or_nativelike-mc_b_melbin36_and_f0_alldata.pickle2') }

bonuspoints = 0

points_per_phone = { 'bad' :-4 + bonuspoints,  #-2,
                     'ok' : 2 + bonuspoints, # 2
                     'good' : 5 + bonuspoints, # 5,
                     'native' : 6 + bonuspoints }# 7 }




#points_per_phone = { 'bad' :0.7 + bonuspoints,  #-2,
#                     'ok' : 2.8 + bonuspoints, # 2
#                     'good' : 4.5 + bonuspoints, # 5,
#                     'native' : 5.5 + bonuspoints }# 7 }

prediction_array = []
classes_array = []


numranks=7


ranking_array = np.zeros([8611, 45*120+1])
score_array = np.zeros([8611,2])
speaker_array = np.zeros([8611])


starts={}
stops={}

uttcounter = 0


speakers = {}
speakercounter=0

confusion_matrix=np.zeros([45,120])

for category in ['native', 'good', 'ok', 'bad' ]:
    
    starts[category]=uttcounter


    class_and_pred=np.loadtxt(prediction_files[category])
    classes = class_and_pred[:,1]
    predictions = class_and_pred[:,2]

    prediction_array.append(predictions)
    classes_array.append(classes)
     
    data_and_classes = pickle.load( open(phone_pickles[category], 'rb'))
    details=data_and_classes['details']
    currentsourcefile = ''

    rowcounter = 0
    
    ranking_matrix = np.zeros([45,120])
    score_line = 0
    phonecounter=0
    for line in details:
        [phone, sourcefile, noise, noiseparam] = line.split(' ')
        if currentsourcefile != sourcefile:                            
            #print (sourcefile)
            speaker, junk = sourcefile.split('_',1)
            if (np.sum(np.abs(ranking_matrix)))>0:  
                #norm_ranking_matrix = ranking_matrix/np.sum(ranking_matrix.reshape([-1]))

                #for i in range(balancing[category]):
                ranking_array[uttcounter,1:] = ranking_matrix.reshape([-1])
                ranking_array[uttcounter,0] = phonecounter
                wlen=np.sum(ranking_matrix.reshape([-1]))
                score_array[uttcounter,:] = [wlen, points_per_phone[category]] #score_line
                if speaker not in speakers.keys():
                    speakers[speaker] = speakercounter
                    speakercounter += 1
                speaker_array[uttcounter] = speakers[speaker]
            #print (ranking_matrix)
            #print (score_line)

            ranking_matrix = np.zeros([45,120])
            score_line = 0
            phonecounter = 0
            uttcounter += 1

            currentsourcefile = sourcefile

        if phone != "sil":
            [pre, phone, post] = re.split('\-|\+', phone)
            cl = class_def[phone]['class']
            
            if cl != classes[rowcounter]:
                print("Something funny about data order!")

            guess = int(predictions[rowcounter])

            ranking_matrix[ cl, guess ] += 1
            phonecounter += 1
                
            if category == 'native':
                confusion_matrix[cl, guess] += 1

        rowcounter+=1
    stops[category]=uttcounter-1


print("Row: %i"%uttcounter)

#diagonal5=np.ones([45])
#for i in range(1,45):
#    if confusion_matrix[i,i] > 1:
#        diagonal5[i] = 5 / (confusion_matrix[i,i])

#confusion_matrix = ( confusion_matrix * diagonal5.reshape([-1,1]) ) .reshape([-1])

confusion_matrix[ np.where(confusion_matrix>0) ] = 5
np.savetxt('/tmp/conf', confusion_matrix)
confusion_matrix = confusion_matrix.reshape([-1])



score_array = score_array[:uttcounter,:].reshape([-1,2])

print(score_array.shape)


loss="not_user" #"linear" #"soft_l1"
endcondition=1e-4
max_num_eval=20


all_samples = np.mod(np.arange(uttcounter), 10)
all_samples = np.mod(speaker_array, 11)

np.savetxt('/tmp/speaker_array_mod.txt', all_samples)

lsq_weight_array=np.zeros([12,45*120+1])

all_scores={ 'native': np.zeros([0]),'good': np.zeros([0]), 'ok': np.zeros([0]), 'bad': np.zeros([0])  }

print ("Number of speakers: %i" % len(speakers))

scores = []
all_train_scores = []





from sklearn.neural_network import MLPRegressor
import pickle


# 11 rounds of leave-one-out testing:



for testround in range(12):
    print ("Test speakers:")
    for k in speakers.keys():
        if np.mod(speakers[k],11)==testround:
            print (k)
    
    train_samples = np.where(all_samples != testround)[0]
    test_samples = np.where(all_samples == testround)[0]
            
    print("Train samples: %i Test samples: %i" % (len(train_samples),len(test_samples)))

    leave_one_out_train_ranking_array = scipy.sparse.csr_matrix(ranking_array[train_samples,:])
    leave_one_out_train_score_array = (score_array[train_samples,0]*score_array[train_samples,1]).T.reshape([-1])
                                                                           
    leave_one_out_test_ranking_array = scipy.sparse.csr_matrix(ranking_array[test_samples,:])
    leave_one_out_test_score_array = (score_array[test_samples,0]*score_array[test_samples,1]).T.reshape([-1])     
    leave_one_out_test_len_array = score_array[test_samples,0].T.reshape([-1])
    leave_one_out_category_array = score_array[test_samples,1].T.reshape([-1])

    mlp_file=os.path.join( LOG_DIR, 'mlp_%i'% (testround))
    print("parameters in file %s ? " % mlp_file)
    if os.path.isfile(mlp_file):                                                                                             
        tester = pickle.load(open(mlp_file, 'br'))

    else:
        tester = MLPRegressor(hidden_layer_sizes=(50,20,10),
                          activation='relu', 
                          solver='lbfgs',#'adam', 
                          alpha=100, # 0.0001, 
                          batch_size='auto', 
                          learning_rate='constant', 
                          learning_rate_init=0.001, 
                          power_t=0.5, 
                          max_iter=200, 
                          shuffle=True, 
                          random_state=None, 
                          tol=0.0001, 
                          verbose=False, #True, 
                          warm_start=False, 
                          momentum=0.9, 
                          nesterovs_momentum=True, 
                          early_stopping=False, 
                          validation_fraction=0.1, 
                          beta_1=0.9, 
                          beta_2=0.999, 
                          epsilon=1e-08)

        tester.fit(leave_one_out_train_ranking_array, leave_one_out_train_score_array)
        pickle.dump(tester, open(mlp_file, 'bw'), protocol=-1)


    res = tester.predict(leave_one_out_test_ranking_array)
    
    print (res.shape)

    scores.append({})
    
    for category in ['native','good', 'ok', 'bad' ]:                                                                     
        target_score =  points_per_phone[category]                                                                       
        sub_test_samples = np.where(leave_one_out_category_array == target_score)[0]                                   
        
        scores[testround][category]= res[sub_test_samples] / leave_one_out_test_len_array[sub_test_samples]
                                                                                                                             
        print("round %i test samples: %s\tmean: %0.2f\tstd: %0.2f" % (testround,                                         
                                                                 category,                                                   
                                                                 np.mean(scores[testround][category]),                       
                                                                 np.std(scores[testround][category])))                       
      
        np.savetxt(os.path.join( LOG_DIR,
                                 'scores_mlp-%s_%02i_category%s' % (loss,testround, category)),  scores[testround][category])

    '''
    lsq_file=os.path.join( LOG_DIR, 'lsq_weights-%s_%i'% (loss, testround))
    print ("Lsq saved in file %s?" % lsq_file)

    if os.path.isfile(lsq_file):
        lsq_weights = np.loadtxt(lsq_file).reshape([-1])
    
    else:
        print ("nope ... maybe we have an initial guess on disk?")
        if os.path.isfile(lsq_file + ".initguess" ):
            initguess =  np.loadtxt(lsq_file + ".initguess")

        else:
            #initguess =np.zeros([45,120])

            #for i in range(45):
            #    initguess[ i, i ] = 7

            #initguess=np.concatenate((np.array([1,3]),initguess.reshape([-1])))
            initguess=np.concatenate( (np.array([2]) ,confusion_matrix ) )

        def costfunc(x):      
            costsum = leave_one_out_train_ranking_array.dot( x )
            costsum= np.abs( costsum - leave_one_out_train_score_array)
            costsum -= 1.49
            costsum[costsum<0.0] = 0.0
            return costsum

            
        #lsq_res = optimize.least_squares(costfunc,initguess , xtol=endcondition, ftol=endcondition, max_nfev=max_num_eval, jac='2-point', loss=loss, bounds=lsq_bounds, verbose=2)
        lsq_res = optimize.lsq_linear(leave_one_out_train_ranking_array, leave_one_out_train_score_array.reshape([-1]), lsq_bounds)

        lsq_weights = lsq_res['x']

        print( lsq_res['message'])
        np.savetxt(lsq_file, lsq_weights)

    test_ranking_array = ranking_array[test_samples,:]
    test_score_array = score_array[test_samples]
    
    scores.append({})
    all_train_scores.append({})


    if len(test_samples) > 0:

        test_scores = leave_one_out_test_ranking_array.dot( lsq_weights ) - bonuspoints

        for category in ['native','good', 'ok', 'bad' ]:
            target_score =  points_per_phone[category]        
            sub_test_samples = np.where(leave_one_out_test_score_array == target_score)[0]       
                 
            scores[testround][category]= test_scores[sub_test_samples]

            print("round %i test samples: %s\tmean: %0.2f\tstd: %0.2f" % (testround,
                                                                 category,
                                                                 np.mean(scores[testround][category]),
                                                                 np.std(scores[testround][category])))
        
            np.savetxt(os.path.join( LOG_DIR,
                                  'scores_lsq-%s_%02i_category%s' % (loss,testround, category)),  scores[testround][category])

            all_scores[category] = np.concatenate((  all_scores[category], scores[testround][category] ))

    train_scores = leave_one_out_train_ranking_array.dot( lsq_weights )- bonuspoints


    for category in ['native','good', 'ok', 'bad' ]:
        target_score =  points_per_phone[category]        
        sub_test_samples = np.where(leave_one_out_train_score_array == target_score)[0]       

        all_train_scores[testround][category]= train_scores[sub_test_samples]
        
        m=np.mean(all_train_scores[testround][category])
        st=np.std(all_train_scores[testround][category])
        print("       train samples: %s\tmean: %0.2f\tstd: %0.2f" % (
                                                                 category,m, st))
                                                                

    print ("------------")
'''
    
lsq_file=os.path.join(LOG_DIR, "lsq-%s_weights" % loss)
if os.path.isfile(lsq_file):
    all_lsq_weights = np.loadtxt(lsq_file)


else:
    train_ranking_array = scipy.sparse.csr_matrix(ranking_array)
    train_score_array = score_array.T.reshape([-1])

    initguess=np.concatenate(([3],confusion_matrix))

    #print("initguess least_sq_cost.shape:")
    #print(my_least_squares_costfunc(initguess).shape)



    endcondition=1e-3
    max_num_eval=20
        
    def costfunc(x):      
        #print (leave_one_out_train_ranking_array.shape)
        costsum = leave_one_out_train_ranking_array.dot( x )
        costsum= np.abs( costsum - leave_one_out_train_score_array)
        costsum -= 1.49
        costsum[costsum<0.0] = 0.0
        return costsum 

            
    lsq_res = optimize.least_squares(costfunc,initguess , xtol=endcondition, ftol=endcondition, max_nfev=max_num_eval, jac='2-point', loss=loss, bounds=lsq_bounds, verbose=2)


    all_lsq_weights = lsq_res['x']

    np.savetxt(lsq_file, all_lsq_weights)


'''
scores={}
for category in ['native', 'good', 'ok', 'bad' ]:        
    
    target_score =  points_per_phone[category]        
    test_samples = np.where(score_array == target_score)[0]      

    test_ranking_array = ranking_array[test_samples]
    
    scores[category]= (all_lsq_weights * test_ranking_array).sum(-1) - bonuspoints
    
    print("            All: %s\tmean: %0.2f\tstd: %0.2f" % (
                                                             category,
                                                             np.mean(all_scores[category]),
                                                             np.std(all_scores[category])))
'''


for category in ['native', 'good', 'ok', 'bad' ]:        
    print("All leave 1 out test scores:")
    print("            All: %s\tmean: %0.2f\tstd: %0.2f" % (
        category,
        np.mean(all_scores[category]),
        np.std(all_scores[category])))
    
