__author__="Hugo Liu <hugo@media.mit.edu>"
__version__="2.0"
import sys,string,time
import MontyTokenizer,MontyLexicalRuleParser,MontyContextualRuleParser,MontyLexiconFast,MontyCommonsense,MontyLemmatiser

class MontyTagger:

    def __init__(self,trace_p=0,MontyLemmatiser_handle=None):

        if not MontyLemmatiser_handle:
            MontyLemmatiser_handle=MontyLemmatiser.MontyLemmatiser()
        self.theMontyLemmatiser=MontyLemmatiser_handle
        self.trace_p=trace_p
        self.theTokenizer=MontyTokenizer.MontyTokenizer()
        self.theLexicon=MontyLexiconFast.MontyLexiconFast()
        self.theLRP=MontyLexicalRuleParser.MontyLexicalRuleParser(self.theLexicon)
        self.theCRP=MontyContextualRuleParser.MontyContextualRuleParser()
        self.theMontyCommonsense=MontyCommonsense.MontyCommonsense(self.theMontyLemmatiser,self)

    def tag(self,text,expand_contractions_p=0,all_pos_p=0,commonsense_p=1):
        the_tokenizer1=self.theTokenizer.tokenize(text,expand_contractions_p)
        cp_cleaned=self.tag_tokenized(the_tokenizer1,all_pos_p,commonsense_p)
        return cp_cleaned

    def tag_tokenized(self,text,all_pos_p=0,commonsense_p=1):
        _montylingua_p=self.theLexicon.all_pos
        groups_cleaned=self.theLRP.apply_all_rules
        user1=string.uppercase
        _montylingua=[]
        chmod_dict=text.split()

        for cps in chmod_dict:

            if '/' in cps and cps[cps.index('/'):].upper()==cps[cps.index('/'):]:
                cps,cksum_cleaned=cps.split('/')
                the_tokenizer_dict=[cksum_cleaned]
            else :
                the_tokenizer_dict=_montylingua_p(cps)

            if the_tokenizer_dict==[]:
                chmods='UNK'
                the_tokenizer_dict.append('UNK')
            else :
                chmods=the_tokenizer_dict[0]
            _montylingua.append({'word':cps,'pos':chmods,'all_pos':the_tokenizer_dict})
        b_arr={'word':'S-T-A-R-T','pos':'STAART','all_pos':[]}
        _montylingua.insert(0,b_arr.copy())
        _montylingua.append(b_arr.copy())

        if self.trace_p:
            print "TRACE: [output after lexicon lookup]:\n  ",self.form_output(_montylingua)

        for _hugo_p in range(len(_montylingua)):
            hash=_montylingua[_hugo_p]

            if hash['pos']!='UNK':
                continue

            if (hash['word'][0]in user1):
                _montylingua[_hugo_p]['pos']='NNP'
            else :
                _montylingua[_hugo_p]['pos']='NN'
            groups_cleaned(_montylingua,_hugo_p)
            _montylingua[_hugo_p]['all_pos']=['UNK',_montylingua[_hugo_p]['pos']]

        if self.trace_p:
            print "TRACE: [output after lexical rules were applied]:\n  ",self.form_output(_montylingua)
        self.theCRP.apply_rules_to_all_words_brill(_montylingua)
        cp_cleaned=self.form_output(_montylingua,all_pos_p)

        if commonsense_p:
            cp_cleaned=self.theMontyCommonsense.cs_verify_tagged(cp_cleaned)
        return cp_cleaned

    def form_output(self,text_arr,all_pos_p=0):
        cp_cleaned=''

        for hash in text_arr[1:-1]:
            cps=hash['word']
            popd_po=hash['pos']

            if all_pos_p:
                the_tokenizer_dict=hash['all_pos']
                popd1=[]

                for pathname1 in the_tokenizer_dict:

                    if pathname1!=popd_po:
                        popd1.append(pathname1)
                the_tokenizer_dict=popd1
                chroot=[popd_po]+the_tokenizer_dict
                cp_cleaned += cps+'/'+'/'.join(chroot)+' '
            else :
                cp_cleaned += cps+'/'+popd_po+' '
        cp_cleaned=cp_cleaned.strip()
        return cp_cleaned

    def verify_and_repair(self,tagged):
        _montylingua_p=self.theLexicon.all_pos
        _montylingua=[]
        chmod_dict=tagged.split()

        for c in chmod_dict:
            alias_dict=c.split('/')
            cps=alias_dict[0]
            chmods=alias_dict[1]
            the_tokenizer_dict=_montylingua_p(cps)

            if the_tokenizer_dict==[]:
                the_tokenizer_dict.append('UNK')
            _montylingua.append({'word':cps,'pos':chmods,'all_pos':the_tokenizer_dict})
        b_arr={'word':'S-T-A-R-T','pos':'STAART','all_pos':[]}
        _montylingua.insert(0,b_arr.copy())
        _montylingua.append(b_arr.copy())

        if self.trace_p:
            print "TRACE: [inputted as]:\n  ",self.form_output(_montylingua)
        self.theCRP.apply_rules_to_all_words_brill(_montylingua)
        return self.form_output(_montylingua,all_pos_p)

if __name__=="__main__":

    if '/?' in sys.argv or '-?' in sys.argv:
        print """
        USAGE: >> python MontyTagger.py [-trace] [-allpos] [-repair]
        -trace   shows intermediary steps and debug messages
        -allpos  displays all plausible POS tags, ranked
        -repair  in repair mode, enter tagged text at the
                 prompt, monty will attempt to fix the tags
    """
        sys.exit(0)

    if '-noverbose' in sys.argv:
        m=MontyTagger(0)

        while 1:
            sentence=sys.stdin.readline()
            print '\n'+string.strip(m.tag(sentence))
            print '--\n\n'

    if '-trace' in sys.argv:
        trace_p=1
    else :
        trace_p=0

    if '-allpos' in sys.argv:
        all_pos_p=1
    else :
        all_pos_p=0

    if '-repair' in sys.argv:
        repair_p=1
    else :
        repair_p=0
    print '\n***** INITIALIZING ******'

    if trace_p:print 'TRACE is on!'

    if all_pos_p:print 'ALL POS is on!'

    if repair_p:print 'REPAIR MODE is on!'
    m=MontyTagger(trace_p)
    print '*************************\n'
    print 'MontyTagger v1.2'
    print '--send bug reports to hugo@media.mit.edu--'
    print '\n'

    try :

        while 1:
            sentence=''

            try :
                sentence=raw_input('> ')
            except :
                raise
            time1=time.time()

            if repair_p:
                print '\nREPAIRED: '+m.verify_and_repair(sentence)
            else :
                print '\n'+m.tag(sentence,0,all_pos_p)
            time2=time.time()
            print "-- monty took",str(round(time2-time1,2)),'seconds. --\n'
    except KeyboardInterrupt:
        print "\n-- monty says goodbye! --"
        sys.exit(0)