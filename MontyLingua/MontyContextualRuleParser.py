__author__="Hugo Liu <hugo@media.mit.edu>"
__version__="2.0"
import sys
import MontyUtils

class MontyContextualRuleParser:
    contextualrules_filename='CONTEXTUALRULEFILE.MDF'
    rules=[]
    rule_names=['PREVTAG','NEXTTAG','PREV1OR2TAG','NEXT1OR2TAG','PREV1OR2OR3TAG','NEXT1OR2OR3TAG','SURROUNDTAG','PREVBIGRAM','NEXTBIGRAM','PREV2TAG','NEXT2TAG']

    def __init__(self):
        self.contextualrules_filename=MontyUtils.MontyUtils().find_file(self.contextualrules_filename)

        if self.contextualrules_filename=='':
            print "ERROR: could not find %s" % self.contextualrules_filename
            print "in current dir, %MONTYLINGUA% or %PATH%"
        self.populate_from_file(self.contextualrules_filename)
        print "ContextualRuleParser OK!"
        return

    def apply_rules_to_all_words_brill(self,text_arr):
        user_dict=self.rules
        a_arr=self.apply_rule

        for the_parser_str in range(len(user_dict)):
            namess=user_dict[the_parser_str]

            for chroot_arr in range(len(text_arr)):
                a_arr(namess,text_arr,chroot_arr)
        return

    def apply_rules_to_all_words(self,text_arr,depth_or_breadth_first_firing='depth'):
        hostnames_arr=self.apply_rules_to_one_word

        if depth_or_breadth_first_firing=='breadth':
            _montylingua_arr=0
        else :
            _montylingua_arr=1
        inputs=1

        while (inputs):
            inputs=0

            for the_parser_str in range(len(text_arr)):
                b_arr=hostnames_arr(text_arr,the_parser_str,_montylingua_arr)

                if b_arr:
                    inputs=1
        return

    def apply_rules_to_one_word(self,text_arr,word_index,exhaustive_p=0):
        user_dict=self.rules
        a_arr=self.apply_rule
        b_arru=0
        inputs=1

        while (inputs):
            inputs=0

            for the_parser_str in range(len(user_dict)):
                namess=user_dict[the_parser_str]
                ps1=text_arr[word_index]['pos']
                b_arr=a_arr(namess,text_arr,word_index)

                if b_arr:
                    inputs=1
                    b_arru=1
                    print "DEBUG: POS of word",text_arr[word_index]['word'],"changed from",ps1,"to",text_arr[word_index]['pos']

                    if not exhaustive_p:
                        return b_arru
        return b_arru

    def apply_rule(self,rule,text_arr,word_index):
        inputs=0
        awk_dict=text_arr[word_index]['pos']
        hostname_arr=text_arr[word_index]['all_pos']
        dirname_p=rule[0]
        cksum_arrz=rule[1]
        arg_cleaned=cksum_arrz[0]
        a_cleaned=cksum_arrz[1]
        bs=cksum_arrz[3:]

        if arg_cleaned!=awk_dict:
            return inputs

        if 'UNK' not in hostname_arr and a_cleaned not in hostname_arr:
            return inputs
        buf_dict=''
        aliass=''
        filename=''
        values_dict=''
        built_ins=''
        j_arr=''

        if word_index>2:
            buf_dict=text_arr[word_index-3]['pos']

        if word_index>1:
            aliass=text_arr[word_index-2]['pos']

        if word_index>0:
            filename=text_arr[word_index-1]['pos']

        if word_index<len(text_arr)-3:
            j_arr=text_arr[word_index+3]['pos']

        if word_index<len(text_arr)-2:
            built_ins=text_arr[word_index+2]['pos']

        if word_index<len(text_arr)-1:
            values_dict=text_arr[word_index+1]['pos']

        if dirname_p=='PREVTAG':

            if bs[0]in[filename]:
                text_arr[word_index]['pos']=a_cleaned
                inputs=1
        elif dirname_p=='NEXTTAG':

            if bs[0]in[values_dict]:
                text_arr[word_index]['pos']=a_cleaned
                inputs=1
        elif dirname_p=='PREV1OR2TAG':

            if bs[0]in[filename,aliass]:
                text_arr[word_index]['pos']=a_cleaned
                inputs=1
        elif dirname_p=='NEXT1OR2TAG':

            if bs[0]in[values_dict,built_ins]:
                text_arr[word_index]['pos']=a_cleaned
                inputs=1
        elif dirname_p=='PREV1OR2OR3TAG':

            if bs[0]in[filename,aliass,buf_dict]:
                text_arr[word_index]['pos']=a_cleaned
                inputs=1
        elif dirname_p=='NEXT1OR2OR3TAG':

            if bs[0]in[values_dict,built_ins,j_arr]:
                text_arr[word_index]['pos']=a_cleaned
                inputs=1
        elif dirname_p=='SURROUNDTAG':

            if bs[0]is filename and bs[1]is values_dict:
                text_arr[word_index]['pos']=a_cleaned
                inputs=1
        elif dirname_p=='PREVBIGRAM':

            if [bs[0],bs[1]]is[aliass,filename]:
                text_arr[word_index]['pos']=a_cleaned
                inputs=1
        elif dirname_p=='NEXTBIGRAM':

            if [bs[0],bs[1]]is[values_dict,built_ins]:
                text_arr[word_index]['pos']=a_cleaned
                inputs=1
        elif dirname_p=='PREV2TAG':

            if bs[0]in[aliass]:
                text_arr[word_index]['pos']=a_cleaned
                inputs=1
        elif dirname_p=='NEXT2TAG':

            if bs[0]in[built_ins]:
                text_arr[word_index]['pos']=a_cleaned
                inputs=1
        return inputs

    def populate_from_file(self,filename):
        user_dict=self.rules

        try :
            input_str=open(filename,'r')
            chgrp_dictc=input_str.readline()

            while chgrp_dictc:
                chown_arr=chgrp_dictc.split()
                dirname_p=chown_arr[2]
                cksum_arrz=chown_arr
                namess=[dirname_p,cksum_arrz]
                user_dict.append(namess)
                chgrp_dictc=input_str.readline()
            input_str.close()
        except :
            print "Error parsing contextual rule file!"
            sys.exit(-1)
        return 