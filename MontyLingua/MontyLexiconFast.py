__author__="Hugo Liu <hugo@media.mit.edu>"
__version__="2.0"
import MontyLexiconCustom
import sys
import MontySettings,MontyUtils

if MontySettings.MontySettings().JYTHON_P:
    import jarray
    import struct
else :
    import array

class MontyLexiconFast:
    java_p=MontySettings.MontySettings().JYTHON_P
    auto_load_lexicon=1
    lexicon_filename='LEXICON.MDF'
    fast_lexicon_filename='FASTLEXICON'
    packed_words=""
    packed_pos=""

    if java_p:
        word_start_arr=jarray.array([],'l')
        word_end_arr=jarray.array([],'l')
        pos_start_arr=jarray.array([],'l')
        pos_end_arr=jarray.array([],'l')
    else :
        word_start_arr=array.array('L')
        word_end_arr=array.array('L')
        pos_start_arr=array.array('L')
        pos_end_arr=array.array('L')

    def __init__(self):
        self.lexicon_custom=MontyLexiconCustom.MontyLexiconCustom()

        if MontyUtils.MontyUtils().find_file(self.fast_lexicon_filename+'_1.MDF')!='':
            print "Fast Lexicon Found! Now Loading!"
            self.load_fastlexicon()
        elif self.auto_load_lexicon:
            print "No Fast Lexicon Detected...Now Building..."
            self.lexicon_filename=MontyUtils.MontyUtils().find_file(self.lexicon_filename)

            if self.lexicon_filename=='':
                print "ERROR: could not find %s" % self.lexicon_filename
                print "in current dir, %MONTYLINGUA% or %PATH%"
            self.populate_lexicon_from_file(self.lexicon_filename)
            self.make_fastlexicon()
            print "Finished building FASTLEXICON files!"
        else :
            print "No Fast Lexicon Detected. Standard Lexicon used."
            notify.append(-1)
            return
        print "Lexicon OK!"
        return

    def make_fastlexicon(self):
        res_arr=open(self.fast_lexicon_filename+'_1.MDF','w')
        res_arr.write(self.packed_words)
        res_arr.close()
        res_arr=open(self.fast_lexicon_filename+'_2.MDF','w')
        res_arr.write(self.packed_pos)
        res_arr.close()
        res_arr=open(self.fast_lexicon_filename+'_3.MDF','wb')
        self.word_start_arr.tofile(res_arr)
        res_arr.close()
        res_arr=open(self.fast_lexicon_filename+'_4.MDF','wb')
        self.word_end_arr.tofile(res_arr)
        res_arr.close()
        res_arr=open(self.fast_lexicon_filename+'_5.MDF','wb')
        self.pos_start_arr.tofile(res_arr)
        res_arr.close()
        res_arr=open(self.fast_lexicon_filename+'_6.MDF','wb')
        self.pos_end_arr.tofile(res_arr)
        res_arr.close()
        res_arr=open(self.fast_lexicon_filename+'_7.MDF','w')
        res_arr.write(str(len(self.word_start_arr))+'\n')
        res_arr.write(str(len(self.word_end_arr))+'\n')
        res_arr.write(str(len(self.pos_start_arr))+'\n')
        res_arr.write(str(len(self.pos_end_arr))+'\n')
        res_arr.close()
        return

    def load_fastlexicon(self):
        chown_p=MontyUtils.MontyUtils()
        names_p=chown_p.find_file(self.fast_lexicon_filename+'_1.MDF')
        res_arrk=chown_p.find_file(self.fast_lexicon_filename+'_2.MDF')
        aliass=chown_p.find_file(self.fast_lexicon_filename+'_3.MDF')
        dirname_dict=chown_p.find_file(self.fast_lexicon_filename+'_4.MDF')
        output_cleaned=chown_p.find_file(self.fast_lexicon_filename+'_5.MDF')
        pairs_cleaned=chown_p.find_file(self.fast_lexicon_filename+'_6.MDF')
        c_p=chown_p.find_file(self.fast_lexicon_filename+'_7.MDF')
        res_arr=open(c_p,'r')
        built_in_p,input_arr,chgrp1,cd_arr=map(lambda hostnames:int(hostnames),res_arr.read().split())
        res_arr.close()
        res_arr=open(names_p,'r')
        self.packed_words=res_arr.read()
        res_arr.close()
        res_arr=open(res_arrk,'r')
        self.packed_pos=res_arr.read()
        res_arr.close()
        res_arr=open(aliass,'rb')
        line1=self.array_fromfile(res_arr,self.word_start_arr,built_in_p,self.java_p,java_code='ws')
        res_arr.close()
        res_arr=open(dirname_dict,'rb')
        self.array_fromfile(res_arr,self.word_end_arr,input_arr,self.java_p,java_code='we')
        res_arr.close()
        res_arr=open(output_cleaned,'rb')
        self.array_fromfile(res_arr,self.pos_start_arr,chgrp1,self.java_p,java_code='ps')
        res_arr.close()
        res_arr=open(pairs_cleaned,'rb')
        self.array_fromfile(res_arr,self.pos_end_arr,cd_arr,self.java_p,java_code='pe')
        res_arr.close()

    def compare(self,element1,element2):

        if element1[0]<element2[0]:return-1
        elif element1[0]>element2[0]:return 1
        else :return 0

    def get(self,word,default):
        the_tokenizer1=self.packed_pos
        gawk=self.pos_start_arr
        built_in_dict=self.pos_end_arr
        cleaned_arr=self.word_start_arr
        hostname=self.word_end_arr
        domain_str=self.packed_words
        chroots=self.lexicon_custom.get(word,[])

        if len(chroots)>0:
            return chroots
        alias1=0
        alias_p,chgrp_str=[len(cleaned_arr)]*2

        while alias_p>=alias1 and alias1<chgrp_str:
            cron_cleanedo=(alias_p-alias1)/2+alias1
            cron_p=domain_str[cleaned_arr[cron_cleanedo]:hostname[cron_cleanedo]]

            if word<cron_p:pathname_cleaned=-1
            elif word>cron_p:pathname_cleaned=1
            else :pathname_cleaned=0

            if pathname_cleaned!=0 and alias1==alias_p:
                return default
            elif pathname_cleaned<0:
                alias_p=cron_cleanedo
            elif pathname_cleaned>0 and alias_p-alias1==1:
                alias1=alias_p
            elif pathname_cleaned>0:
                alias1=cron_cleanedo
            elif pathname_cleaned==0:
                awk_arr=the_tokenizer1[gawk[cron_cleanedo]:built_in_dict[cron_cleanedo]]
                return awk_arr.split()
            else :
                return default
        return default

    def primary_pos(self,word):
        names_pt=self.get(word,[])

        if names_pt==[]:
            return ""
        else :
            return names_pt[0]

    def all_pos(self,word):
        names_pt=self.get(word,[])
        return names_pt

    def has_pos(self,word,pos):
        return pos in self.get(word,[])

    def is_word(self,word,case_sensitivity=0):
        a_cleaned=self.get

        if case_sensitivity:
            return(a_cleaned(word,None)!=None)
        else :
            b_cleaned=word.capitalize()
            pathname_cleaned=(a_cleaned(word,None)!=None)or(a_cleaned(word.lower(),None)!=None)or(a_cleaned(word.upper(),None)!=None)or(a_cleaned(b_cleaned,None)!=None)
            return pathname_cleaned

    def populate_lexicon_from_file(self,filename):
        line_str=[]

        try :
            res_arr=open(filename,'r')
            alias_dict=res_arr.readline()

            while alias_dict:
                line_dict=alias_dict.find(' ')
                env_dict=alias_dict[:line_dict]
                awk_arr=alias_dict[line_dict+1:]
                line_str.append((env_dict,awk_arr))
                alias_dict=res_arr.readline()
            res_arr.close()
        except :
            print "Error parsing Lexicon!"
            sys.exit(-1)
        line_str.sort(self.compare)
        cleaned_p=''
        popd_dict=''
        cmp_arr=[]
        groups_arr=[]
        inputs=[]
        res_dict=[]
        the_parser_dict=0
        chmod_str=0
        alias_dicts=0

        for env_dict,awk_arr in line_str:
            the_parser_dict += 1

            if the_parser_dict % 100000==0:
                print the_parser_dict
            cmp_arr.append(chmod_str)
            chmod_str += len(env_dict)
            groups_arr.append(chmod_str)
            inputs.append(alias_dicts)
            alias_dicts += len(awk_arr)
            res_dict.append(alias_dicts)
        print the_parser_dict
        cleaned_p=''.join(map(lambda hostnames:hostnames[0],line_str))
        popd_dict=''.join(map(lambda hostnames:hostnames[1],line_str))
        self.packed_words=cleaned_p
        self.packed_pos=popd_dict
        self.word_start_arr.fromlist(cmp_arr)
        self.word_end_arr.fromlist(groups_arr)
        self.pos_start_arr.fromlist(inputs)
        self.pos_end_arr.fromlist(res_dict)
        return

    def array_fromfile(self,file_ptr,array_ptr,length,java_p=0,java_code='',endian_order='little'):
        history=4

        if java_p:

            if endian_order=='big':
                command_p='>'
            else :
                command_p='<'
            line1=struct.unpack(command_p+str(length)+'L',file_ptr.read())

            if java_code=='ws':
                self.word_start_arr=jarray.array(line1,'l')
            elif java_code=='we':
                self.word_end_arr=jarray.array(line1,'l')
            elif java_code=='ps':
                self.pos_start_arr=jarray.array(line1,'l')
            elif java_code=='pe':
                self.pos_end_arr=jarray.array(line1,'l')
            else :
                print "error! java code invalid!"
                sys.exit(-1)
        else :
            array_ptr.fromfile(file_ptr,length)

if __name__=="__main__":
    l=MontyLexiconFast()
    a="aberration bird ate an apple"

    for word in a.split():
        print l.get(word,'UNK')