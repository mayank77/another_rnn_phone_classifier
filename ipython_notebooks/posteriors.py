#!/usr/bin/python3
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np

#plt.rcParams["font.family"] = "Arial"
plt.rc('font', family='DejaVu Sans', size=10)

files={ 'eval': "/tmp/tensorflow_logs/copy13-rnn512-g/eval_uk_y_and_posterior.16100",
        'good': "/tmp/tensorflow_logs/copy13-rnn512-g/players_good_y_and_posterior.16100",
        'ok': "/tmp/tensorflow_logs/copy13-rnn512-g/players_ok_y_and_posterior.16100",
        'bad': "/tmp/tensorflow_logs/copy13-rnn512-g/players_bad_y_and_posterior.16100" }


tour=np.array([93,
     1,
    51,
    55,
    97,
    95,
    65,
    77,
    73,
    74,
    75,
   114,
   108,
    90,
    91,
    60,
    64,
    63,
    94,
    54,
    53,
    39,
    43,
    34,
     8,
     7,
    25,
    16,
    24,
    11,
    35,
     6,
    42,
    29,
    14,
    26,
    22,
    67,
    98,
    66,
    50,
    61,
    57,
   116,
    62,
   118,
   103,
   102,
    85,
    89,
    59,
    69,
    58,
   107,
    46,
    23,
     3,
    36,
    37,
   104,
    38,
    31,
    21,
    19 ])

tour=tour-1


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



posterior_array = []
prediction_array = []
classes_array = []

for category in ['good', 'ok', 'bad', 'eval' ]:
    class_and_post=np.loadtxt(files[category])
    classes = class_and_post[:,0]
    posteriors = class_and_post[:,1:]
    posteriors = posteriors - np.min(posteriors, axis=1).reshape([-1,1])
    posteriors = posteriors / np.sum(posteriors, axis=1).reshape([-1,1])

    predictions = np.argmax(posteriors, axis=1)

    posterior_array.append(posteriors)
    prediction_array.append(predictions)
    classes_array.append(classes)
    


rowcount=3
colcount=4
pagecount=16
cl=0
# row and column sharing
for page in range(pagecount):

        #plt.figure(figsize=[6,6])
        f, axarr = plt.subplots(rowcount, colcount, figsize=(21.841, 14.195), dpi=100)


        for i in range(rowcount):

            cl +=1
            order=[]

            print ("\n\nClass %i %s" % (cl, classdict[("_%i_"%cl) ]))

            for j in range(colcount):

                posteriors=posterior_array[j]
                predictions = prediction_array[j]
                classes= classes_array[j]
                
                rows = np.where(classes==cl)[0]
                if len(order)<1:
                    #order_bys = np.where(predictions != cl)[0]
                    #order_bys = np.intersect1d(order_bys, rows)

                    #order=np.argsort(-np.sum(posteriors[order_bys,:], axis=0))

                    order_bys = np.bincount(predictions[rows])
                    order=np.argsort(-order_bys, axis=0)
                    
                    #order=order[:10]
                    labels= [classdict['_%i_' % item] for item in order[:10]]

                    #print("Order:")
                    #print(order[:10])
                    #print(labels)


                counts=np.zeros(120);
                rankings = np.zeros(5);

                if len(rows)==0:
                    #skips += 1
                    axarr[i, j].set_title('Class %i %s samples: 0' % (cl, classdict[("_%i_"%cl) ]))
                    continue

                else:
                    alpha = min(8.0/len(rows), 1)


                    right=0
                    wrong=0

                    for row in posteriors[rows,:]:         
                        #print("%i %i " % (np.argmax(row), cl))

                        #row = row-np.min(row)
                        #row = row/np.sum(row)                    

                        if ( np.argmax(row) == cl ):
                            axarr[i, j].plot(row[order[:10]], linewidth=2,alpha=alpha, color='green')
                            right += 1
                        else:
                            axarr[i, j].plot(row[order[:10]], linewidth=2,alpha=alpha, color='red')
                            wrong += 1 
                        counts[np.argmax(row)]+=1

                        #print("Right: %i, Wrong: %i" % (right, wrong))

                        ranks = (row).argsort()[-5:][::-1]                      
                        #print(row[ranks[:6]])
                        #print( [np.argmax(row), ranks[0] , (row.argsort()[-1]) ]  )                        
                        rankings[ np.where(ranks==cl)[0]] +=1
                        

                    #print rankings/len(rows)

                    print ("%0.2f %0.2f %0.2f %0.2f %0.2f" % (
                           (rankings[0]) / len(rows),
                           (rankings[0]+rankings[1]) / len(rows), 
                           (rankings[0]+rankings[1]+rankings[2]) / len(rows),
                           (rankings[0]+rankings[1]+rankings[2]+rankings[3]) / len(rows),
                           (rankings[0]+rankings[1]+rankings[2]+rankings[3]+rankings[4]) / len(rows)
                    ))
                    counts=counts/np.max(counts)*0.005
                    axarr[i, j].plot(counts[order], linewidth=4,alpha=1, color='blue')

                    axarr[i, j].set_title('Class %i %s samples: %i correct:%0.2f%s' % (cl, classdict[("_%i_"%cl) ], right+wrong, 100.0*right/(right+wrong), "%"))





                    axarr[i, j].set_xticks(np.arange(0,10))
                    axarr[i, j].set_xticklabels(labels)
                    axarr[i, j].set_ylim([0.00,0.025])
                    axarr[i, j].set_xlim([-1,10])
                    axarr[i, j].grid(True)
                

        
        #f.set_figheight(115)
        #f.set_figwidth(215)

        #plt.show()
        
        #x = np.arange(0,100,0.00001)
        #y = x*np.sin(2*pi*x)
        #plt.plot(y)
        #plt.axis('off')
        #plt.gca().set_position([0, 0, 1, 1])
        plt.savefig("page_%i.svg"%page, format="svg", dpi=300, bbox_inches='tight')
    
'''            
    ax2.scatter(x, y)
    ax3.scatter(x, 2 * y ** 2 - 1, color='r')
    ax4.plot(x, 2 * y ** 2 - 1, color='r')
    


    for cl in [ 1, 2, 3, 4, 5 ]:
        rows = np.where(classes==cl)[0]
        for row in posteriors[rows,:]:
            row = row+np.min(row)
            row=row/np.sum(row)
            plt.plot(row,  linewidth=5,alpha=0.02, color='red')
            plt.ylabel('Class %i'%cl)
    

'''
