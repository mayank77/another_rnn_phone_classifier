#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys

modelmap = {
    "0": { "phone": "<eps>","aaltophone": "_", "position":""},
    "1": { "phone": "NSN","aaltophone": "__", "position":""},
    "2": { "phone": "NSN","aaltophone": "__", "position":"_B"},
    "3": { "phone": "NSN","aaltophone": "__", "position":"_E"},
    "4": { "phone": "NSN","aaltophone": "__", "position":"_I"},
    "5": { "phone": "NSN","aaltophone": "__", "position":"_S"},
    "6": { "phone": "SIL","aaltophone": "__", "position":""},
    "7": { "phone": "SIL","aaltophone": "__", "position":"_B"},
    "8": { "phone": "SIL","aaltophone": "__", "position":"_E"},
    "9": { "phone": "SIL","aaltophone": "__", "position":"_I"},
    "10": { "phone": "SIL","aaltophone": "__", "position":"_S"},
    "11": { "phone": "SPN","aaltophone": "__", "position":""},
    "12": { "phone": "SPN","aaltophone": "__", "position":"_B"},
    "13": { "phone": "SPN","aaltophone": "__", "position":"_E"},
    "14": { "phone": "SPN","aaltophone": "__", "position":"_I"},
    "15": { "phone": "SPN","aaltophone": "__", "position":"_S"},
    "16": { "phone": "2","aaltophone": "ö", "position":"_B"},
    "17": { "phone": "2","aaltophone": "ö", "position":"_E"},
    "18": { "phone": "2","aaltophone": "ö", "position":"_I"},
    "19": { "phone": "2","aaltophone": "ö", "position":"_S"},
    "20": { "phone": "A","aaltophone": "a", "position":"_B"},
    "21": { "phone": "A","aaltophone": "a", "position":"_E"},
    "22": { "phone": "A","aaltophone": "a", "position":"_I"},
    "23": { "phone": "A","aaltophone": "a", "position":"_S"},
    "24": { "phone": "I","aaltophone": "i", "position":"_B"},
    "25": { "phone": "I","aaltophone": "i", "position":"_E"},
    "26": { "phone": "I","aaltophone": "i", "position":"_I"},
    "27": { "phone": "I","aaltophone": "i", "position":"_S"},
    "28": { "phone": "U","aaltophone": "u", "position":"_B"},
    "29": { "phone": "U","aaltophone": "u", "position":"_E"},
    "30": { "phone": "U","aaltophone": "u", "position":"_I"},
    "31": { "phone": "U","aaltophone": "u", "position":"_S"},
    "32": { "phone": "b","aaltophone": "b", "position":"_B"},
    "33": { "phone": "b","aaltophone": "b", "position":"_E"},
    "34": { "phone": "b","aaltophone": "b", "position":"_I"},
    "35": { "phone": "b","aaltophone": "b", "position":"_S"},
    "36": { "phone": "d","aaltophone": "d", "position":"_B"},
    "37": { "phone": "d","aaltophone": "d", "position":"_E"},
    "38": { "phone": "d","aaltophone": "d", "position":"_I"},
    "39": { "phone": "d","aaltophone": "d", "position":"_S"},
    "40": { "phone": "e","aaltophone": "e", "position":"_B"},
    "41": { "phone": "e","aaltophone": "e", "position":"_E"},
    "42": { "phone": "e","aaltophone": "e", "position":"_I"},
    "43": { "phone": "e","aaltophone": "e", "position":"_S"},
    "44": { "phone": "f","aaltophone": "f", "position":"_B"},
    "45": { "phone": "f","aaltophone": "f", "position":"_E"},
    "46": { "phone": "f","aaltophone": "f", "position":"_I"},
    "47": { "phone": "f","aaltophone": "f", "position":"_S"},
    "48": { "phone": "g","aaltophone": "g", "position":"_B"},
    "49": { "phone": "g","aaltophone": "g", "position":"_E"},
    "50": { "phone": "g","aaltophone": "g", "position":"_I"},
    "51": { "phone": "g","aaltophone": "g", "position":"_S"},
    "52": { "phone": "h","aaltophone": "h", "position":"_B"},
    "53": { "phone": "h","aaltophone": "h", "position":"_E"},
    "54": { "phone": "h","aaltophone": "h", "position":"_I"},
    "55": { "phone": "h","aaltophone": "h", "position":"_S"},
    "56": { "phone": "j","aaltophone": "j", "position":"_B"},
    "57": { "phone": "j","aaltophone": "j", "position":"_E"},
    "58": { "phone": "j","aaltophone": "j", "position":"_I"},
    "59": { "phone": "j","aaltophone": "j", "position":"_S"},
    "60": { "phone": "k","aaltophone": "k", "position":"_B"},
    "61": { "phone": "k","aaltophone": "k", "position":"_E"},
    "62": { "phone": "k","aaltophone": "k", "position":"_I"},
    "63": { "phone": "k","aaltophone": "k", "position":"_S"},
    "64": { "phone": "l","aaltophone": "l", "position":"_B"},
    "65": { "phone": "l","aaltophone": "l", "position":"_E"},
    "66": { "phone": "l","aaltophone": "l", "position":"_I"},
    "67": { "phone": "l","aaltophone": "l", "position":"_S"},
    "68": { "phone": "m","aaltophone": "m", "position":"_B"},
    "69": { "phone": "m","aaltophone": "m", "position":"_E"},
    "70": { "phone": "m","aaltophone": "m", "position":"_I"},
    "71": { "phone": "m","aaltophone": "m", "position":"_S"},
    "72": { "phone": "n","aaltophone": "n", "position":"_B"},
    "73": { "phone": "n","aaltophone": "n", "position":"_E"},
    "74": { "phone": "n","aaltophone": "n", "position":"_I"},
    "75": { "phone": "n","aaltophone": "n", "position":"_S"},
    "76": { "phone": "o","aaltophone": "o", "position":"_B"},
    "77": { "phone": "o","aaltophone": "o", "position":"_E"},
    "78": { "phone": "o","aaltophone": "o", "position":"_I"},
    "79": { "phone": "o","aaltophone": "o", "position":"_S"},
    "80": { "phone": "p","aaltophone": "p", "position":"_B"},
    "81": { "phone": "p","aaltophone": "p", "position":"_E"},
    "82": { "phone": "p","aaltophone": "p", "position":"_I"},
    "83": { "phone": "p","aaltophone": "p", "position":"_S"},
    "84": { "phone": "r","aaltophone": "r", "position":"_B"},
    "85": { "phone": "r","aaltophone": "r", "position":"_E"},
    "86": { "phone": "r","aaltophone": "r", "position":"_I"},
    "87": { "phone": "r","aaltophone": "r", "position":"_S"},
    "88": { "phone": "s","aaltophone": "s", "position":"_B"},
    "89": { "phone": "s","aaltophone": "s", "position":"_E"},
    "90": { "phone": "s","aaltophone": "s", "position":"_I"},
    "91": { "phone": "s","aaltophone": "s", "position":"_S"},
    "92": { "phone": "t","aaltophone": "t", "position":"_B"},
    "93": { "phone": "t","aaltophone": "t", "position":"_E"},
    "94": { "phone": "t","aaltophone": "t", "position":"_I"},
    "95": { "phone": "t","aaltophone": "t", "position":"_S"},
    "96": { "phone": "v","aaltophone": "v", "position":"_B"},
    "97": { "phone": "v","aaltophone": "v", "position":"_E"},
    "98": { "phone": "v","aaltophone": "v", "position":"_I"},
    "99": { "phone": "v","aaltophone": "v", "position":"_S"},
    "100": { "phone": "y","aaltophone": "y", "position":"_B"},
    "101": { "phone": "y","aaltophone": "y", "position":"_E"},
    "102": { "phone": "y","aaltophone": "y", "position":"_I"},
    "103": { "phone": "y","aaltophone": "y", "position":"_S"},
    "104": { "phone": "{","aaltophone": "ä", "position":"_B"},
    "105": { "phone": "{","aaltophone": "ä", "position":"_E"},
    "106": { "phone": "{","aaltophone": "ä", "position":"_I"},
    "107": { "phone": "{","aaltophone": "ä", "position":"_S"},
    "108": { "phone": "#0","aaltophone": "__", "position":""},
    "109": { "phone": "#1","aaltophone": "__", "position":""},
    "110": { "phone": "#2","aaltophone": "__", "position":""},
    "111": { "phone": "#3","aaltophone": "__", "position":""},
    "112": { "phone": "#4","aaltophone": "__", "position":""},
    "113": { "phone": "#5","aaltophone": "__", "position":""},
}

combinations = ['aa','ai','ao','ae',
                'au','ea','ee','ei','eo','eu','ey','eä','ia','ie','ii',
                'io','iu','iy','iä','iö','oa','oe','oi','oo','ou','ua','ue',
                'ui','uo','uu','yi','yy','yä','yö','äe','äi','äy','ää','äö',
                'öi','öy','öä','öö','ngng','nn','mm','kk','pp','hh','ll','pp',
                'rr','ss','tt','ng' ]


labelfilehandle=open(sys.argv[1], 'r')

###
#Check for EOF or manually add
last_line=labelfilehandle.readlines()[-1]
if (last_line.split()[-1]!="6"):
    sec = ("%.3f" % float(os.popen("sh length.sh "+"/teamwork/t40511_asr/c/speecon-fi/adult/ADULT1FI/BLOCK"+sys.argv[1].split("/")[1].replace("SA","")[:-1]+"/"+sys.argv[1].split("/")[1].replace("SA","SES")+"/"+sys.argv[1].split("/")[2].replace("raw_label","FI0")).read().strip('\n')))
    with open(sys.argv[1], "a") as f:
        start_EOF = ("%.3f" % float(float(last_line.split()[0].strip('\n'))+float(last_line.split()[1].strip('\n'))) )
        end_EOF = float(sec) - float(start_EOF)
	end_EOF = ("%.3f" % float( 0.02000*int((end_EOF*10000)/200) ) )
        f.write(str(start_EOF)+" "+str(end_EOF)+" 6")
    f.close()
labelfilehandle.close()
###

labelfilehandle=open(sys.argv[1], 'r')

oldaaltophone = ""
oldstart = 0
oldend = 0

monoarray = []

for l in labelfilehandle.readlines():
    start, length, phonenr = l.strip().split(' ')
    position = modelmap[phonenr]['position']
    kaldiphone=modelmap[phonenr]['phone']
    aaltophone=modelmap[phonenr]['aaltophone']

    start=round(16000*float(start))
    length=round(16000*float(length))

    end=start+length

    #print("\t\t\t\t%s %s %s%s \t%s" % (start, length, kaldiphone,position, aaltophone))

    if oldaaltophone == "__" and aaltophone == "__":
        oldend = end
    elif oldaaltophone+aaltophone in combinations:
        #print ("%i\t%i\t%s" % ( oldstart, end, oldaaltophone+aaltophone ))
        monoarray.append([ oldstart, end, oldaaltophone+aaltophone ])
        oldaaltophone = ""
        #oldstart = ""
        oldend = end
    elif position == "_E":
        if len(oldaaltophone)>0:
            #print ("%i\t%i\t%s" % ( oldstart, oldend, oldaaltophone ))
            monoarray.append([ oldstart, oldend, oldaaltophone ])
        #print ("%i\t%i\t%s" % ( start, end, aaltophone ) )
        monoarray.append([ start, end, aaltophone ])
        oldaaltophone = ""
        oldstart = 0
        oldend = 0
    else:
        if len(oldaaltophone)>0:
            #print ("%i\t%i\t%s" % ( oldstart, oldend, oldaaltophone ))
            monoarray.append([ oldstart, oldend, oldaaltophone ])
        oldaaltophone = aaltophone
        oldstart = start
        oldend = end

if oldstart > 0:
    #print ("%i\t%i\t%s" % ( oldstart, oldend, oldaaltophone ))
    monoarray.append([ oldstart, oldend, oldaaltophone ])


for i in range(len(monoarray)):
    if monoarray[i][2]== "__":
        print ("%i\t%i\t%s" % ( monoarray[i][0], monoarray[i][1], monoarray[i][2] ))
    else:
        trip=""
        if (i > 0):
            if monoarray[i-1][2] == '__':
                trip += "_-"
            else:
                trip += monoarray[i-1][2] + "-"
        else:
            trip += "_-"
        
        trip +=  monoarray[i][2] 

        if i < len(monoarray)-1:
            if monoarray[i+1][2] == '__':
                trip += "+_"
            else:
                trip += "+" + monoarray[i+1][2] 
        else:
            trip += "+_"

        print ("%i\t%i\t%s" % ( monoarray[i][0], monoarray[i][1], trip))

    
