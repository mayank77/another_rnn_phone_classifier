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

classdict = { "_0_" : u'0',
              "_44_" : u"ʌ",
              "_43_" : u"ʒ",
              "_42_" : u"oɪ",
              "_41_" : u"uː",
              "_40_" : u"w",
              "_39_" : u"ɜː",
              "_38_" : u"aʊ",
              "_37_" : u"ʃ",
              "_36_" : u"d",
              "_35_" : u"dʒ",
              "_34_" : u"ɑː",
              "_33_" : u"n",
              "_32_" : u"iː",
              "_31_" : u"r",
              "_30_" : u"θ",
              "_29_" : u"k",
              "_28_" : u"ɒ",
              "_27_" : u"h",
              "_26_" : u"ʊ",
              "_25_" : u"ɔː",
              "_24_" : u"eɪ",
              "_23_" : u"ɑr",
              "_22_" : u"p",
              "_21_" : u"ð",
              "_20_" : u"tʃ",
              "_19_" : u"ɔːr",
              "_18_" : u"b",
              "_17_" : u"ŋ",
              "_16_" : u"v",
              "_15_" : u"f",
              "_14_" : u"ɛ",
              "_13_" : u"aɪ",
              "_12_" : u"ɪ",
              "_11_" : u"ɡ",
              "_10_" : u"æ",
              "_9_" : u"z",
              "_8_" : u"l",
              "_7_" : u"m",
              "_6_" : u"ɪər",
              "_5_" : u"ə",
              "_4_" : u"s",
              "_3_" : u"j",
              "_2_" : u"t",
              "_1_" : u"oʊ",
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


posterior_files={ 'eval': "/tmp/tensorflow_logs/copy13-rnn512-g/eval_uk_y_and_posterior.16100",
                  'good': "/tmp/tensorflow_logs/copy13-rnn512-g/players_good_y_and_posterior.16100",
                  'ok': "/tmp/tensorflow_logs/copy13-rnn512-g/players_ok_y_and_posterior.16100",
                  'bad': "/tmp/tensorflow_logs/copy13-rnn512-g/players_bad_y_and_posterior.16100" }


test_corpus = "fysiak-gamedata-2"
test_pickle_dir='../features/work_in_progress/'+test_corpus+'/pickles'

phone_pickles = { 'bad' : os.path.join(test_pickle_dir, 'disqualified_mspec66_and_f0_alldata.pickle2'),
                  'ok' : os.path.join(test_pickle_dir, 'few_stars_mspec66_and_f0_alldata.pickle2'),
                  'good' : os.path.join(test_pickle_dir, 'lots_of_stars_mspec66_and_f0_alldata.pickle2') }

points_per_phone = { 'bad' : 0,
                     'ok' : 3,
                     'good' : 5 }

posterior_array = []
prediction_array = []
classes_array = []


numranks=7


ranking_array = np.zeros([6000, 45*numranks])
score_array = np.zeros([6000])

starts={}
stops={}

uttcounter = 0



for category in ['good', 'ok', 'bad' ]:
    
    starts[category]=uttcounter

    class_and_post=np.loadtxt(posterior_files[category])
    classes = class_and_post[:,0]
    posteriors = class_and_post[:,1:]
    posteriors = posteriors - np.min(posteriors, axis=1).reshape([-1,1])
    posteriors = posteriors / np.sum(posteriors, axis=1).reshape([-1,1])

    predictions = np.argmax(posteriors, axis=1)

    posterior_array.append(posteriors)
    prediction_array.append(predictions)
    classes_array.append(classes)
     
    data_and_classes = pickle.load( open(phone_pickles[category], 'rb'))
    details=data_and_classes['details']
    currentsourcefile = ''

    rowcounter = 0
    
    ranking_line = np.zeros([45*numranks])
    score_line = 0

    for line in details:
        [phone, something, value1, value2, value3, sourcefile] = line.split(' ')
        if currentsourcefile != sourcefile:                

            if (np.sum(np.abs(ranking_line)))>0:
                ranking_array[uttcounter,:] = ranking_line/np.sum(np.abs(ranking_line))
                score_array[uttcounter] =  points_per_phone[category] #score_line

            #print (ranking_line)
            #print (score_line)

            ranking_line = np.zeros([45*numranks])
            score_line = 0

            uttcounter += 1

            currentsourcefile = sourcefile

        if phone != "sil":
            [pre, phone, post] = re.split('\-|\+', phone)
            cl = class_def[phone]['class']
            phonebase = cl * numranks
            row=posteriors[rowcounter,:]            
            ranks = (row).argsort()[-(numranks-1):][::-1]
            thisrank=np.where(ranks==cl)[0]
            
            if len(thisrank) > 0:
                #print(phonebase)
                #print(thisrank)
                ranking_line[ phonebase + thisrank[0] ] += 1
            else:
                ranking_line[ phonebase + numranks-1 ] -= 1
            score_line += points_per_phone[category]
            

            #print(phone)
                
        rowcounter+=1
    #print(np.sum(ranking_array[0:40,:], axis=1))
    #print(score_array[0:40])
    stops[category]=uttcounter


print("Row: %i"%uttcounter)

ranking_array = ranking_array[:uttcounter,:]
score_array = score_array[:uttcounter].reshape([-1,1])

print(score_array.shape)

bounds=np.zeros([45*numranks, 2])
bounds[:,1] = 20

#weights = lsq_linear(ranking_array, score_array, bounds=bounds)

def my_costfunc(x):
    #print("x.shape")
    #print(x.shape)
    #print("ranking_array.shape")
    #print(ranking_array.shape)
    for i in range(45):        
        for j in range(3,numranks+1):
            x[(i+1)*numranks -j ] = max(x[(i+1)*numranks -j ], x[(i+1)*numranks - j +1 ])
    #print ("np.matmul ... shape")    
    #print ((np.matmul( ranking_array, x)- score_array.T)  .shape)
    return np.sum(
                   np.power(
                             np.matmul( ranking_array, x) - score_array.T
                       ,2 ))
    


#weights = least_squares(ranking_array, score_array, bounds=bounds)

initguess = np.zeros([45*numranks])

print("initguess cost:")
print(my_costfunc(initguess))

for i in range(45):        
    #print("setting %i to random" % (i*numranks) )
    initguess[ i*numranks] = np.random.rand()
    for j in range(1, numranks):
        #print("setting %i to max of random or %i" % ( (i*numranks+j),(i*numranks+j-1)))
        initguess[int(i*numranks+j)] = min(np.random.rand(), initguess[int(i*numranks+j-1)])
        
#print (initguess)

#weights = least_squares(my_costfunc, initguess, bounds=bounds, verbose=2)


weights1 = optimize.fmin_bfgs(my_costfunc, initguess, disp=2)#, fprime=fprime)

print(weights1)

weights2 = optimize.fmin_l_bfgs_b(my_costfunc, weights1, bounds=bounds, approx_grad=True, disp=2)#, fprime=fprime)

print(weights2)

print(weights2[0])


use_indices = np.where( np.sum(ranking_array, axis=0) > 0  )[0]

scores=( np.matmul(ranking_array, weights2[0]))

print("Scores:")
print(scores)

for category in ['good', 'ok', 'bad' ]:
    print("%s: %i-%i" % (category, starts[category], stops[category] ))

np.savetxt('/tmp/scores', scores)
