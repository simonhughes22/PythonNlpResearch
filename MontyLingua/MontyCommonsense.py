from __future__ import nested_scopes
__author__="Hugo Liu <hugo@media.mit.edu>"
__version__="2.0"
import zlib
import MontyUtils,MontyLemmatiser,MontyTagger

class MontyCommonsense:
    cssdb_filename='CSSDB.MDF'

    def __init__(self,MontyLemmatiser_handle=None,MontyTagger_handle=None):

        if not MontyLemmatiser_handle:
            MontyLemmatiser_handle=MontyLemmatiser.MontyLemmatiser()
        self.theMontyLemmatiser=MontyLemmatiser_handle

        if not MontyTagger_handle:
            MontyTagger_handle=MontyTagger.MontyTagger()
        self.theMontyTagger=MontyTagger_handle
        self.tag_tokenized=self.theMontyTagger.tag_tokenized
        self.lemmatise_word=self.theMontyLemmatiser.lemmatise_word
        ps1={}
        _montylingua_cleaned=MontyUtils.MontyUtils().find_file(self.cssdb_filename)

        if not _montylingua_cleaned:
            self.build_cs_selection_db()
            _montylingua_cleaned=MontyUtils.MontyUtils().find_file(self.cssdb_filename)
        hash1=open(_montylingua_cleaned,'rb')
        cat_p=self.setitem
        map(lambda stripped:cat_p(ps1,stripped[0],(stripped[1].split(),stripped[2].split())),map(lambda tmps:tmps.split('|'),filter(lambda tagged_cleaned:tagged_cleaned.strip()!='',zlib.decompress(hash1.read()).split('\n'))))
        self.cssdb=ps1
        print "Commonsense OK!"
        return

    def cs_verify_tagged(self,tagged):
        pathname=self.lemmatise_word
        ps1=self.cssdb
        cmp1=tagged.split()
        hostnames_dict=['VBD','VBG','VBN','VBZ','VB','VBP']
        table_arr=map(lambda tagged_cleaned:tagged_cleaned.split('/')[1],cmp1)

        if len(filter(lambda tagged_cleaned:tagged_cleaned in hostnames_dict,table_arr))>0:
            return tagged
        groupnames_cleaned=' '.join(map(lambda tagged_cleaned:tagged_cleaned.split('/')[0],cmp1))
        id_p=self.tag_tokenized(groupnames_cleaned,all_pos_p=1,commonsense_p=0)
        cmp1=id_p.split()
        argss=len(cmp1)
        csplit_cleaned=0

        for input_p in range(len(cmp1)):

            if csplit_cleaned:
                break
            hostnames_p=cmp1[input_p].split('/')
            chmod_str=hostnames_p[0]
            table_arr=hostnames_p[1:]

            for buffer_dict in table_arr:

                if buffer_dict in hostnames_dict:
                    cron=pathname(chmod_str,'verb').lower()

                    if ps1.has_key(cron):
                        buffer_arr,ress=ps1[cron]

                        if input_p-1>=0 and buffer_arr:
                            line_p=cmp1[input_p-1].split('/')[0]
                            c_arr=pathname(line_p,'noun').lower()

                            if c_arr in buffer_arr:
                                csplit_cleaned=1

                        if input_p+1<argss and ress:
                            nice1m=cmp1[input_p+1].split('/')[0]

                            if nice1m.lower()in['the','a','an','some','every','each','most']:

                                if input_p+2<argss:
                                    nice1m=cmp1[input_p+2].split('/')[0]
                            b_dict=pathname(nice1m,'noun').lower()

                            if b_dict in ress:
                                csplit_cleaned=1

                    if csplit_cleaned:
                        cmp1[input_p]=chmod_str+'/'+buffer_dict
                        break
        info1=' '.join(map(lambda tagged_cleaned:tagged_cleaned[0]+'/'+tagged_cleaned[1],map(lambda tmps:tmps.split('/'),cmp1)))

        if info1!=tagged:
            print "Common sense violated! Correcting..."
        return info1

    def unpp(self,pp):
        cmp1=pp.strip(' ()\n').split()
        args_p=cmp1[0]
        chgrp_p=' '.join(cmp1[1:])[1:-1].split('" "')
        hash1,input_p=map(lambda tagged_cleaned:int(tagged_cleaned.split('=')[1]),chgrp_p.pop().split(';')[:2])
        return args_p,chgrp_p[0],chgrp_p[1],hash1,input_p

    def build_cs_selection_db(self):
        history1=self.unpp
        ps1={}
        input='generalised_predicates.huge.txt'
        hash1=open(input,'r')
        more_dict=hash1.readline()
        aliass=0

        while more_dict:
            aliass += 1

            if aliass % 100000==0:
                print aliass

            if more_dict[:len('(CapableOf')]!='(CapableOf' and more_dict[:len('(CapableOfReceivingAction')]!='(CapableOfReceivingAction':
                more_dict=hash1.readline()
                continue
            args_p,pathname_str,alias1,input_arr,hashs=history1(more_dict)

            if args_p=='CapableOf':
                pairs_cleaned=alias1.split()[0]
                cron_arr=pathname_str.split()[-1]

                if len(pairs_cleaned.split())==1 and len(cron_arr.split())>0:

                    if not ps1.has_key(pairs_cleaned):
                        ps1[pairs_cleaned]=([],[])
                    ps1[pairs_cleaned][0].append(cron_arr)
            elif args_p=='CapableOfReceivingAction':
                pairs_cleaned=alias1.split()[0]
                hostnames_pq=pathname_str.split()[-1]

                if not ps1.has_key(pairs_cleaned):
                    ps1[pairs_cleaned]=([],[])
                ps1[pairs_cleaned][1].append(hostnames_pq)
            more_dict=hash1.readline()
        hash1.close()
        cat_p=self.setitem

        for buf_cleaned in ps1.keys():
            buffer_arr,ress=ps1[buf_cleaned]
            groups_cleaned,built_in_arr={},{}
            map(lambda tagged_cleaned:cat_p(groups_cleaned,tagged_cleaned,1),buffer_arr)
            map(lambda tagged_cleaned:cat_p(built_in_arr,tagged_cleaned,1),ress)
            buffer_arr=groups_cleaned.keys()
            ress=built_in_arr.keys()
            ps1[buf_cleaned]=(buffer_arr,ress)
        cp_dict=open(self.cssdb_filename,'wb')
        buf_arr=[]

        for more in ps1.items():
            buf_cleaned,crons=more
            buffer_arr,ress=crons
            buffer_arr=' '.join(buffer_arr)
            ress=' '.join(ress)
            more_dict='|'.join((buf_cleaned,buffer_arr,ress))
            buf_arr.append(more_dict)
        info1='\n'.join(buf_arr)
        mounts=zlib.compress(info1,1)
        cp_dict.write(mounts)
        cp_dict.close()
        return

    def setitem(self,dict,key,value):
        dict[key]=value

if __name__=='__main__':
    m=MontyCommonsense()