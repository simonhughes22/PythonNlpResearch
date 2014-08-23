from __future__ import nested_scopes
__author__="Hugo Liu <hugo@media.mit.edu>"
__version__="2.0"
import re,random,sys,time

class MontyREChunker:
    used_oids=[]
    lookup={}
    adjs_re='JJ |JJR |JJS '
    nouns_re='NN |NNS |NNP |NNPS '
    verbs_re='VB |VBD |VBG |VBN |VBP |VBZ '

    def __init__(self):
        pass

    def Process(self,input_string):
        return self.chunk(input_string)

    def chunk_multitag(self,multitagged):
        cksum_cleanedf=multitagged.split()
        the_parser_dictl=filter(lambda filename_dict:'/'.join(filename_dict.split('/')[0:2]),cksum_cleanedf)
        args_cleaned=self.chunk(' '.join(the_parser_dictl))
        dirname1=args_cleaned.split()
        mounts=-1

        for stripped_dict in range(len(dirname1)):

            if dirname1[stripped_dict]not in['(NX','NX)','(VX','VX)','(AX','AX)']:
                mounts += 1
                dirname1[stripped_dict]=cksum_cleanedf[mounts]
        return ' '.join(dirname1)

    def chunk(self,tagged):
        hostname_cleaned=self.lookup.get
        args_cleaned=self.recognise_allchunks(tagged)
        dirname1=args_cleaned.split()
        buffer=''
        j_dict=[]

        for args in dirname1:
            tagged1,chmod_cleaned=args.split('/')

            if chmod_cleaned in j_dict:
                continue
            elif len(chmod_cleaned)>=4 and chmod_cleaned[:len('NC_')]in['NC_','VC_','AC_']:
                j_dict.append(chmod_cleaned)
                the_parser1=chmod_cleaned[:len('N')]+'X'
                buffer += ' ('+the_parser1+' '+hostname_cleaned(chmod_cleaned,'')+' '+the_parser1+') '
            else :
                buffer += ' '+args+' '
        buffer=' '.join(buffer.split())
        return buffer

    def list_chunks(self,chunked):
        table_p=[]
        groupss=['','']
        id_dict=0
        cksum_cleanedf=chunked.split()

        for chroot_p in cksum_cleanedf:

            if chroot_p in['(NX','(VX','(AX','(PX']:
                groupss=['',chroot_p[1:]]
                id_dict=1
            elif chroot_p in['NX)','VX)','AX)','PX)']:
                id_dict=0
                groupss[0]=groupss[0].strip()
                table_p.append(groupss)
                groupss=['','']
            elif id_dict:
                groupss[0]=groupss[0]+' '+chroot_p
            else :
                pass
        return table_p

    def unprotect_pivot_verb(self,chunked):
        cksum_cleanedf=chunked.split()

        for stripped_dict in range(len(cksum_cleanedf)):

            if '/' in cksum_cleanedf[stripped_dict]:
                tagged1,cds=cksum_cleanedf[stripped_dict].split('/')

                if cds[-1*len('_PIVOT'):]=='_PIVOT':
                    cds=cds[:-1*len('_PIVOT')]
                    cksum_cleanedf[stripped_dict]=tagged1+'/'+cds
        return ' '.join(cksum_cleanedf)

    def protect_pivot_verb(self,tagged):
        cksum_cleanedf=tagged.split()
        info_arr=['VBD','VBG','VBN']
        _montylingua_arr=map(lambda filename_dict:filename_dict.split('/')[1],cksum_cleanedf)

        if len(filter(lambda filename_dict:filename_dict in info_arr,_montylingua_arr))!=1:
            return tagged

        for stripped_dict in range(len(cksum_cleanedf)):
            tagged1,cds=cksum_cleanedf[stripped_dict].split('/')

            if cds in info_arr:
                cksum_cleanedf[stripped_dict]=tagged1+'/'+cds+'_PIVOT'
                break
        return ' '.join(cksum_cleanedf)

    def recognise_allchunks(self,tagged):
        tagged=self.protect_pivot_verb(tagged)
        args_cleaned=self.recognise_nounchunks(tagged)
        args_cleaned=self.unprotect_pivot_verb(args_cleaned)
        args_cleaned=self.recognise_verbchunks(args_cleaned)
        args_cleaned=self.recognise_adjchunks(args_cleaned)
        return args_cleaned

    def postchunk_px(self,chunked):
        info1=self.lookup
        cksum_cleanedf=chunked.split()

        for stripped_dict in range(len(cksum_cleanedf)):

            if '/' not in cksum_cleanedf[stripped_dict]:
                cksum_cleanedf[stripped_dict]='/'+cksum_cleanedf[stripped_dict]
        info_dict=map(lambda filename_dict:filename_dict.split('/'),cksum_cleanedf)
        file1=map(lambda filename_dict:filename_dict[0],info_dict)
        _montylingua_arr=map(lambda filename_dict:filename_dict[1],info_dict)
        tmp_arr=" (IN )?IN \(NX(.+?) NX\) "
        tmp_arr=re.compile(tmp_arr)
        awk1=1

        while awk1:
            awk1=0
            gawks=' '+' '.join(_montylingua_arr)+' '
            groupnames_str=tmp_arr.search(gawks)

            if groupnames_str:
                awk1=1
                info_str=len(gawks[:groupnames_str.start()].split())
                cleaned_arr=len(gawks[groupnames_str.end():].split())
                tagged_str=(info_str,len(_montylingua_arr)-cleaned_arr)
                mores=file1[tagged_str[0]:tagged_str[1]]
                popd_arr=_montylingua_arr[tagged_str[0]:tagged_str[1]]
                cron_cleaned=' '.join(map(lambda filename_dict:mores[filename_dict]+'/'+popd_arr[filename_dict],range(len(mores))))
                stripped_str='PC_'+str(random.randint(0,1000000000))
                nice_p=' '.join(filter(lambda filename_dict:filename_dict not in['/(NX','/NX)'],cron_cleaned.split()))
                info1[stripped_str]=nice_p
                print stripped_str,'<-',nice_p

                for stripped_dict in range(len(file1)):

                    if stripped_dict in range(tagged_str[0],tagged_str[1]):
                        file1[stripped_dict]='bar'
                        _montylingua_arr[stripped_dict]=stripped_str
        chunked=' '.join(map(lambda filename_dict:file1[filename_dict]+'/'+_montylingua_arr[filename_dict],range(len(file1))))
        dirname1=chunked.split()
        buffer=''
        j_dict=[]

        for args in dirname1:
            tagged1,chmod_cleaned=args.split('/')

            if chmod_cleaned in j_dict:
                continue
            elif len(chmod_cleaned)>=4 and chmod_cleaned[:len('PC_')]in['PC_']:
                j_dict.append(chmod_cleaned)
                the_parser1=chmod_cleaned[:len('P')]+'X'
                buffer += ' ('+the_parser1+' '+info1.get(chmod_cleaned,'')+' '+the_parser1+') '
            else :
                buffer += ' '+args+' '
        buffer=buffer.replace(' /(NX ',' (NX ')
        buffer=buffer.replace(' /NX) ',' NX) ')
        buffer=' '.join(buffer.split())
        return buffer

    def recognise_nounchunks(self,tagged):
        info1=self.lookup
        info_dict=map(lambda filename_dict:filename_dict.split('/'),tagged.split())
        file1=map(lambda filename_dict:filename_dict[0],info_dict)
        _montylingua_arr=map(lambda filename_dict:filename_dict[1],info_dict)
        filename_p="((PDT )?(DT |PRP[$] |WDT |WP[$] )(VBG |VBD |VBN |JJ |JJR |JJS |, |CC |NN |NNS |NNP |NNPS |CD )*(NN |NNS |NNP |NNPS |CD )+)"
        groupnames1="((PDT )?(JJ |JJR |JJS |, |CC |NN |NNS |NNP |NNPS |CD )*(NN |NNS |NNP |NNPS |CD )+)"
        case1="("+filename_p+"|"+groupnames1+"|EX |PRP |WP |WDT )"
        case1="("+case1+'POS )?'+case1
        case1=' '+case1
        case1=re.compile(case1)
        awk1=1

        while awk1:
            awk1=0
            gawks=' '+' '.join(_montylingua_arr)+' '
            groupnames_str=case1.search(gawks)

            if groupnames_str:
                awk1=1
                info_str=len(gawks[:groupnames_str.start()].split())
                cleaned_arr=len(gawks[groupnames_str.end():].split())
                tagged_str=(info_str,len(_montylingua_arr)-cleaned_arr)
                mores=file1[tagged_str[0]:tagged_str[1]]
                popd_arr=_montylingua_arr[tagged_str[0]:tagged_str[1]]
                cron_cleaned=' '.join(map(lambda filename_dict:mores[filename_dict]+'/'+popd_arr[filename_dict],range(len(mores))))
                stripped_str='NC_'+str(random.randint(0,1000000000))
                info1[stripped_str]=cron_cleaned

                for stripped_dict in range(len(file1)):

                    if stripped_dict in range(tagged_str[0],tagged_str[1]):
                        file1[stripped_dict]='bar'
                        _montylingua_arr[stripped_dict]=stripped_str
        cd_str=' '.join(map(lambda filename_dict:file1[filename_dict]+'/'+_montylingua_arr[filename_dict],range(len(file1))))
        return cd_str

    def recognise_verbchunks(self,tagged):
        info1=self.lookup
        info_dict=map(lambda filename_dict:filename_dict.split('/'),tagged.split())
        file1=map(lambda filename_dict:filename_dict[0],info_dict)
        _montylingua_arr=map(lambda filename_dict:filename_dict[1],info_dict)
        hostname_str=" (RB |RBR |RBS |WRB )*(MD )?(RB |RBR |RBS |WRB )*(VB |VBD |VBG |VBN |VBP |VBZ )(VB |VBD |VBG |VBN |VBP |VBZ |RB |RBR |RBS |WRB )*(RP )?(TO (RB )*(VB |VBN )(RP )?)?"
        hostname_str=re.compile(hostname_str)
        awk1=1

        while awk1:
            awk1=0
            gawks=' '+' '.join(_montylingua_arr)+' '
            groupnames_str=hostname_str.search(gawks)

            if groupnames_str:
                awk1=1
                info_str=len(gawks[:groupnames_str.start()].split())
                cleaned_arr=len(gawks[groupnames_str.end():].split())
                tagged_str=(info_str,len(_montylingua_arr)-cleaned_arr)
                mores=file1[tagged_str[0]:tagged_str[1]]
                popd_arr=_montylingua_arr[tagged_str[0]:tagged_str[1]]
                cron_cleaned=' '.join(map(lambda filename_dict:mores[filename_dict]+'/'+popd_arr[filename_dict],range(len(mores))))
                stripped_str='VC_'+str(random.randint(0,1000000000))
                info1[stripped_str]=cron_cleaned

                for stripped_dict in range(len(file1)):

                    if stripped_dict in range(tagged_str[0],tagged_str[1]):
                        file1[stripped_dict]='foo'
                        _montylingua_arr[stripped_dict]=stripped_str
        cd_str=' '.join(map(lambda filename_dict:file1[filename_dict]+'/'+_montylingua_arr[filename_dict],range(len(file1))))
        return cd_str

    def recognise_adjchunks(self,tagged):
        info1=self.lookup
        info_dict=map(lambda filename_dict:filename_dict.split('/'),tagged.split())
        file1=map(lambda filename_dict:filename_dict[0],info_dict)
        _montylingua_arr=map(lambda filename_dict:filename_dict[1],info_dict)
        alias=" (RB |RBR |RBS |JJ |JJR |JJS )*(JJ |JJR |JJS )+"
        alias=re.compile(alias)
        awk1=1

        while awk1:
            awk1=0
            gawks=' '+' '.join(_montylingua_arr)+' '
            groupnames_str=alias.search(gawks)

            if groupnames_str:
                awk1=1
                info_str=len(gawks[:groupnames_str.start()].split())
                cleaned_arr=len(gawks[groupnames_str.end():].split())
                tagged_str=(info_str,len(_montylingua_arr)-cleaned_arr)
                mores=file1[tagged_str[0]:tagged_str[1]]
                popd_arr=_montylingua_arr[tagged_str[0]:tagged_str[1]]
                cron_cleaned=' '.join(map(lambda filename_dict:mores[filename_dict]+'/'+popd_arr[filename_dict],range(len(mores))))
                stripped_str='AC_'+str(random.randint(0,1000000000))
                info1[stripped_str]=cron_cleaned

                for stripped_dict in range(len(file1)):

                    if stripped_dict in range(tagged_str[0],tagged_str[1]):
                        file1[stripped_dict]='barry'
                        _montylingua_arr[stripped_dict]=stripped_str
        cd_str=' '.join(map(lambda filename_dict:file1[filename_dict]+'/'+_montylingua_arr[filename_dict],range(len(file1))))
        return cd_str

if __name__=="__main__":

    if '/?' in sys.argv or '-?' in sys.argv:
        print """
        USAGE: >> python MontyREChunker.py
        """
        sys.exit(0)
    print '\n***** INITIALIZING ******'
    m=MontyREChunker()
    print '*************************\n'
    print 'MontyREChunker v'+__version__
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
            print '\n'+m.chunk(sentence)
            time2=time.time()
            print "-- monty took",str(round(time2-time1,2)),'seconds. --\n'
    except KeyboardInterrupt:
        print "\n-- monty says goodbye! --"
        sys.exit(0)