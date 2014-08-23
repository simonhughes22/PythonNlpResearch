__author__="Hugo Liu <hugo@media.mit.edu>"
__version__="2.0"
import sys
import MontyUtils

class MontyLexicalRuleParser:
    lexicalrules_filename='LEXICALRULEFILE.MDF'
    lex_rules=[]
    rule_names=['char','hassuf','deletesuf','addsuf','haspref','deletepref','addpref','goodleft','goodright']
    rule_names +=['fchar','fhassuf','fdeletesuf','faddsuf','fhaspref','fdeletepref','faddpref','fgoodleft','fgoodright']

    def __init__(self,LexiconHandle):
        self.theLexicon=LexiconHandle
        self.lexicalrules_filename=MontyUtils.MontyUtils().find_file(self.lexicalrules_filename)

        if self.lexicalrules_filename=='':
            print "ERROR: could not find %s" % self.lexicalrules_filename
            print "in current dir, %MONTYLINGUA% or %PATH%"
        self.populate_from_file(self.lexicalrules_filename)
        print 'LexicalRuleParser OK!'
        return

    def apply_all_rules(self,text_arr,word_index):
        awk=self.theLexicon.is_word
        popd_dict=self.lex_rules
        chroot_p=self.apply_rule

        for command_cleaned in range(len(popd_dict)):
            res_p=popd_dict[command_cleaned]
            chroot_p(res_p,text_arr,word_index,awk)

    def apply_rule(self,rule,text_arr,word_index,is_word_handle):
        popdw=text_arr[word_index]['word']
        arg=text_arr[word_index]['pos']
        tagged_dict=is_word_handle
        cal_p=rule[0]
        alias_str=0

        if cal_p[0]=='f':
            cal_p=cal_p[1:]
            alias_str=1
        built_in_arro=rule[1]
        arg_str=''
        output_p=''

        if word_index>0:
            arg_str=text_arr[word_index-1]['word']

        if word_index<len(text_arr)-1:
            output_p=text_arr[word_index+1]['word']
        cp1=''

        if alias_str:
            cp1=built_in_arro[0]
            built_in_arro=built_in_arro[1:]
        popds=built_in_arro[0]
        cksum1=built_in_arro[-2]
        popds=popds.lower()
        popdw=popdw.lower()
        arg_str=arg_str.lower()
        output_p=output_p.lower()

        if alias_str and(arg!=cp1):
            return

        if cal_p=='char':

            if popds in popdw:
                text_arr[word_index]['pos']=cksum1
        elif cal_p=='hassuf':

            if popds is popdw[len(popds):]:
                text_arr[word_index]['pos']=cksum1
        elif cal_p=='deletesuf':

            if popds is popdw[len(popds):]and tagged_dict(popdw[:len(popds)]):
                text_arr[word_index]['pos']=cksum1
        elif cal_p=='addsuf':

            if tagged_dict(popdw+popds):
                text_arr[word_index]['pos']=cksum1
        elif cal_p=='haspref':

            if popds is popdw[:len(popds)]:
                text_arr[word_index]['pos']=cksum1
        elif cal_p=='deletepref':

            if popds is popdw[:len(popds)]and tagged_dict(popdw[len(popds)-1:]):
                text_arr[word_index]['pos']=cksum1
        elif cal_p=='addsuf':

            if tagged_dict(popds+popdw):
                text_arr[word_index]['pos']=cksum1
        elif cal_p=='goodleft':

            if arg_str==popds:
                text_arr[word_index]['pos']=cksum1
        elif cal_p=='goodright':

            if output_p==popds:
                text_arr[word_index]['pos']=cksum1
        return

    def populate_from_file(self,filename):
        cksum_p=self.rule_names
        enabled_dict=self.lex_rules

        try :
            a_arr=open(filename,'r')
            groupnames1=a_arr.readline()

            while groupnames1:
                _hugo_str=groupnames1.split()
                cal_p=''

                for gawks in cksum_p:

                    if gawks in _hugo_str:
                        cal_p=gawks
                        break
                built_in_arro=_hugo_str
                res_p=[cal_p,built_in_arro]
                enabled_dict.append(res_p)
                groupnames1=a_arr.readline()
            a_arr.close()
        except :
            print "Error parsing Lexical rule file!"
            sys.exit(-1)
        return 