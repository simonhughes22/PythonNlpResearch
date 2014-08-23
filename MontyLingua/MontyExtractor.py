from __future__ import nested_scopes
__author__="Hugo Liu <hugo@media.mit.edu>"
__version__="2.0"
import sys,string,os,re

class MontyExtractor:

    def __init__(self):
        print "Semantic Interpreter OK!"
        return

    def extract_info(self,chunked_text,lemmatise_function_handle=None):
        cp_cleaned=self.strip_tags
        factor1=self.extract_prep_phrases
        pathname_arr=self.make_concise_verb_arg_structure
        input_str=self.make_parameterized_predicates
        enabled_dict=lemmatise_function_handle
        dict={}
        id_p=self.extract_phrases(chunked_text)
        the_parser_cleaned=map(lambda output_p:output_p[0],filter(lambda alias_str:alias_str[1]=='AX',id_p))
        groupnames_p=map(lambda output_p:output_p[0],filter(lambda alias_str:alias_str[1]=='NX',id_p))
        env_cleaned=chunked_text.split()

        for cp_arr in range(len(env_cleaned)):

            if env_cleaned[cp_arr]=='(AX':
                env_cleaned[cp_arr]='(NX'
            elif env_cleaned[cp_arr]=='AX)':
                env_cleaned[cp_arr]='NX)'
        chunked_text=' '.join(env_cleaned)
        j_str=map(lambda output_p:output_p[0],filter(lambda alias_str:alias_str[1]=='VX',id_p))
        groups_arr=map(lambda alias_str:' '.join(alias_str),factor1(chunked_text))
        print chunked_text
        res_arr=self.extract_pos(chunked_text,['JJ','JJS','JJR','RB','RBR','RBS'])
        cmp_str=self.find_verb_arg_structures(chunked_text)
        cleaned1=map(lambda alias_str:pathname_arr(alias_str,enabled_dict),cmp_str)
        dirname1=map(lambda alias_str:input_str(alias_str,enabled_dict),self.find_verb_arg_structures(chunked_text))
        dict['noun_phrases_tagged']=groupnames_p
        dict['noun_phrases']=map(cp_cleaned,groupnames_p)
        dict['verb_phrases_tagged']=j_str
        dict['verb_phrases']=map(cp_cleaned,j_str)
        dict['adj_phrases_tagged']=the_parser_cleaned
        dict['adj_phrases']=map(cp_cleaned,the_parser_cleaned)
        dict['prep_phrases_tagged']=groups_arr
        dict['prep_phrases']=map(cp_cleaned,groups_arr)
        dict['modifiers_tagged']=res_arr
        dict['modifiers']=map(cp_cleaned,res_arr)
        dict['verb_arg_structures']=cmp_str
        dict['verb_arg_structures_concise']=cleaned1
        dict['parameterized_predicates']=dirname1
        return dict

    def make_parameterized_predicates(self,verbose_verb_arg_structure,lemmatise_function_handle=None):
        hostnames_cleanedt=self.filter_by_tag
        cp_cleaned=self.strip_tags
        hostnamess=self.strip_tags_lemmatised
        cron_cleaned,names_arr,output_pu=verbose_verb_arg_structure
        cal=[]
        inputs=[]
        user_cleanedh=map(lambda alias_str:[],output_pu)
        alias1=map(lambda output_p:output_p.split('/')[0],filter(lambda alias_str:alias_str.split('/')[1]in['IN','TO'],names_arr.split()))

        if len(alias1)>0:
            inputs.append('prep='+alias1[0])
        cksum_p=map(lambda output_p:output_p.split('/')[0],filter(lambda alias_str:alias_str.split('/')[1]in['DT','CD','PRP$'],names_arr.split()))

        if len(cksum_p)>0:
            inputs.append('determiner='+cksum_p[0])
        buf=['DT','CD','PRP$']

        if len(names_arr.split())>1:
            names_arr=hostnames_cleanedt(names_arr,buf)

        for cp_arr in range(len(output_pu)):
            alias1=map(lambda output_p:output_p.split('/')[0],filter(lambda alias_str:alias_str.split('/')[1]in['IN','TO'],output_pu[cp_arr].split()))

            if len(alias1)>0:
                user_cleanedh[cp_arr].append('prep='+alias1[0])
            cksum_p=map(lambda output_p:output_p.split('/')[0],filter(lambda alias_str:alias_str.split('/')[1]in['DT','CD','PRP$'],output_pu[cp_arr].split()))

            if len(cksum_p)>0:
                user_cleanedh[cp_arr].append('determiner='+cksum_p[0])

            if len(output_pu[cp_arr].split())>1:
                output_pu[cp_arr]=hostnames_cleanedt(output_pu[cp_arr],buf)

        if 'not/rb' in map(lambda alias_str:alias_str.lower(),cron_cleaned.split()):
            cal.append('negation')
        chroot=self.jist_verb_chunk(cron_cleaned)

        if lemmatise_function_handle==None:
            chroot=cp_cleaned(chroot)
            names_arr=cp_cleaned(names_arr)
            output_pu=map(cp_cleaned,output_pu)
        else :
            table_arr=chroot
            a=names_arr
            hostnamessx=map(lambda alias_str:alias_str,output_pu)
            chroot=lemmatise_function_handle(chroot)
            names_arr=lemmatise_function_handle(names_arr)
            output_pu=map(lemmatise_function_handle,output_pu)
            chroot=hostnamess(chroot)
            names_arr=hostnamess(names_arr)
            output_pu=map(hostnamess,output_pu)

            if cp_cleaned(a).lower()!=names_arr.lower():
                inputs.append('plural')

            for cp_arr in range(len(output_pu)):

                if cp_cleaned(hostnamessx[cp_arr]).lower()!=output_pu[cp_arr].lower():
                    user_cleanedh[cp_arr].append('plural')
            history1=['was','were','had','did']
            _hugo_p=0

            if cp_cleaned(table_arr).lower()in history1:
                _hugo_p=1

            if chroot.lower()!=cp_cleaned(table_arr).lower()and chroot.lower()not in['be','have','do']:

                if len(chroot)>3 and len(table_arr)>3 and table_arr.lower().split('/')[0][-1]=='s' and chroot.lower()[-1]!='s':
                    _hugo_p=0
                    cal.append('perfect_tense')
                else :
                    _hugo_p=1

            if _hugo_p:
                cal.append('past_tense')

                for cron in map(lambda alias_str:alias_str.lower().split('/')[0],cron_cleaned.split()):

                    if cron in['be','been','is','are','was','were']:
                        cal.append('passive_voice')
                        break
        tmp1=[]
        tmp1.append([chroot,cal])
        tmp1.append([names_arr,inputs])

        for cp_arr in range(len(output_pu)):
            tmp1.append([output_pu[cp_arr],user_cleanedh[cp_arr]])
        return tmp1

    def make_concise_verb_arg_structure(self,verbose_verb_arg_structure,lemmatise_function_handle=None):
        hostnames_cleanedt=self.filter_by_tag
        cp_cleaned=self.strip_tags
        hostnamess=self.strip_tags_lemmatised
        chroot,names_arr,output_pu=verbose_verb_arg_structure
        buf=['DT',',']

        if len(names_arr.split())>1:
            names_arr=hostnames_cleanedt(names_arr,buf)

        for cp_arr in range(len(output_pu)):

            if len(output_pu[cp_arr].split())>1:
                output_pu[cp_arr]=hostnames_cleanedt(output_pu[cp_arr],buf)
        chroot=self.jist_verb_chunk(chroot)

        if lemmatise_function_handle==None:
            chroot=cp_cleaned(chroot)
            names_arr=cp_cleaned(names_arr)
            output_pu=map(cp_cleaned,output_pu)
        else :
            chroot=lemmatise_function_handle(chroot)
            names_arr=lemmatise_function_handle(names_arr)
            output_pu=map(lemmatise_function_handle,output_pu)
            chroot=hostnamess(chroot)
            names_arr=hostnamess(names_arr)
            output_pu=map(hostnamess,output_pu)
        tmp1='("'+chroot+'" "'+names_arr+'" '+' '.join(map(lambda alias_str:'"'+alias_str+'"',output_pu))+')'
        return tmp1

    def jist_verb_chunk(self,verbchunk):
        env_cleaned=verbchunk.split()
        env_cleaned=filter(lambda alias_str:alias_str not in['(VX','VX)'],env_cleaned)
        env_cleaned=map(lambda alias_str:alias_str.split('/'),env_cleaned)
        env_cleaned=filter(lambda alias_str:alias_str[0]=='not' or alias_str[1]not in['MD','RB','TO'],env_cleaned)
        case_cleaned=range(len(env_cleaned))
        case_cleaned.reverse()
        inputsa=0

        for cp_arr in case_cleaned:

            if inputsa:

                if env_cleaned[cp_arr][1]in['VB','VBD','VBG','VBN','VBP','VBZ']:
                    env_cleaned[cp_arr][1]='DELETE'
                    continue
                continue

            if env_cleaned[cp_arr][1]in['VB','VBD','VBG','VBN','VBP','VBZ']:
                inputsa=1
                continue
        env_cleaned=filter(lambda alias_str:alias_str[1]!='DELETE',env_cleaned)
        env_cleaned=map(lambda alias_str:alias_str[0]+'/'+alias_str[1],env_cleaned)

        if len(env_cleaned)>=2 and env_cleaned[-1]=='not/RB':
            env_cleaned=[env_cleaned[-1]]+env_cleaned[:-1]
        tmp1=' '.join(env_cleaned)
        return tmp1

    def _find_linked_subject(self,toks,vc_start_index):
        case_cleaned=range(0,vc_start_index)
        case_cleaned.reverse()
        names_arr=[]

        if len(case_cleaned)>=2 and toks[case_cleaned[0]]=='NX)':

            for cp_arr in case_cleaned[1:]:

                if toks[cp_arr]=='(NX':
                    break
                names_arr.insert(0,toks[cp_arr])
            return ' '.join(names_arr).strip()
        else :
            return ''

    def _find_linked_objects(self,toks,vc_end_index,prev_obj=''):
        output_pu=[]
        iter=range(vc_end_index+1,len(toks))

        if prev_obj in['','NP']and len(iter)>=2 and toks[iter[0]]=='(NX':
            factor_arr=[]
            filename_arr=len(toks)

            for cp_arr in iter[1:]:

                if toks[cp_arr]=='NX)':
                    filename_arr=cp_arr
                    break
                factor_arr.append(toks[cp_arr])
            output_pu.append(' '.join(factor_arr))
            output_pu += self._find_linked_objects(toks,filename_arr,'NP')
        elif prev_obj in['']and len(iter)>=1 and('/' in toks[iter[0]])and toks[iter[0]].split('/')[1]not in['IN','TO']:
            factor_arr=[]

            for cp_arr in iter:

                if toks[cp_arr]in['(NX','(VX']:
                    break
                elif '/' in toks[cp_arr]and toks[cp_arr].split('/')[1]in['IN','TO']:
                    break
                factor_arr.append(toks[cp_arr])
            output_pu.append(' '.join(factor_arr))
        elif prev_obj in['','NP','PP']and len(iter)>=3 and('/' in toks[iter[0]])and toks[iter[0]].split('/')[1]in['IN','TO']and toks[iter[1]]=='(NX':
            factor_arr=[]
            factor_arr.append(toks[iter[0]])
            filename_arr=len(toks)

            for cp_arr in iter[2:]:

                if toks[cp_arr]=='NX)':
                    filename_arr=cp_arr
                    break
                factor_arr.append(toks[cp_arr])
            output_pu.append(' '.join(factor_arr))
            output_pu += self._find_linked_objects(toks,filename_arr,'PP')
        elif len(iter)>=4 and('/' in toks[iter[0]])and toks[iter[0]].split('/')[1]in['IN','TO']and('/' in toks[iter[1]])and toks[iter[1]].split('/')[1]in['IN','TO']and toks[iter[2]]=='(NX':
            factor_arr=[]
            factor_arr.append(toks[iter[0]])
            factor_arr.append(toks[iter[1]])
            filename_arr=len(toks)

            for cp_arr in iter[3:]:

                if toks[cp_arr]=='NX)':
                    filename_arr=cp_arr
                    break
                factor_arr.append(toks[cp_arr])
            output_pu.append(' '.join(factor_arr))
            output_pu += self._find_linked_objects(toks,filename_arr,'PP')
        else :
            return[]

        for cp_arr in range(len(output_pu)):
            gawk_dict=['JJ','JJS','JJR','NN','NNS','NNP','NNPS','VBG','CD','PRP','PRP$','EX','SYM','WP','WP$','WDT']
            tagged_dict=self.extract_pos(output_pu[cp_arr],gawk_dict)

            if len(tagged_dict)==0:
                output_pu[cp_arr]=''
            else :
                output_pu[cp_arr]=output_pu[cp_arr].strip()
        output_pu=filter(lambda alias_str:alias_str!='',output_pu)
        return output_pu

    def find_verb_arg_structures(self,chunked):
        contents_dict=self._find_linked_subject
        awk=self._find_linked_objects
        env_cleaned=chunked.split()
        chgrp1=[]
        filename_str=''
        tmps=0
        hash1=-1

        for cp_arr in range(len(env_cleaned)):

            if env_cleaned[cp_arr]=='(VX':
                tmps=1
                hash1=cp_arr
                continue

            if tmps and env_cleaned[cp_arr]=='VX)':
                tmps=0
                names_arr=contents_dict(env_cleaned,hash1)
                pairs1=awk(env_cleaned,cp_arr)
                filename_str=filename_str.strip()
                chgrp1.append([filename_str,names_arr,pairs1])
                filename_str=''
                hash1=-1
            elif tmps:
                filename_str += ' '+env_cleaned[cp_arr]
        return chgrp1

    def extract_pos(self,tagged_text,pos_whitelist,white_or_blacklist='white'):
        env_cleaned=tagged_text.split()
        env_cleaned=filter(lambda alias_str:alias_str not in['(NX','NX)','(VX','VX)'],env_cleaned)
        env_cleaned=map(lambda alias_str:alias_str.split('/'),env_cleaned)
        dirname_arr=pos_whitelist

        if white_or_blacklist.lower()=='black':
            env_cleaned=filter(lambda alias_str:alias_str[1]not in dirname_arr,env_cleaned)
        else :
            env_cleaned=filter(lambda alias_str:alias_str[1]in dirname_arr,env_cleaned)
        env_cleaned=map(lambda alias_str:'/'.join(alias_str),env_cleaned)
        return env_cleaned

    def extract_prep_phrases(self,chunked_text):
        tmp1=[]
        env_cleaned=chunked_text.split()

        for cp_arr in range(len(env_cleaned)):

            if ((env_cleaned[cp_arr]=='(NX')and cp_arr>0 and cp_arr!=len(env_cleaned)-1):
                built_in_cleaned=env_cleaned[cp_arr-1]

                if '/' not in built_in_cleaned:
                    continue
                cd_dict=built_in_cleaned.split('/')[1]

                if cd_dict in['IN','TO']:
                    cleaned_arr=built_in_cleaned.split('/')[0]+'/'+cd_dict
                    groups_dict=' '.join(env_cleaned[cp_arr+1:])+' '
                    groups_dict=groups_dict[:groups_dict.find('NX)')].strip()
                    tmp1.append([cleaned_arr,groups_dict])
        return tmp1

    def extract_phrases(self,chunked_text,only_salient_words_p=1):
        tmp1=[]
        groupnamesm=['','']
        buf1=chunked_text
        mount=0
        env_cleaned=buf1.split()

        for c_dict in env_cleaned:

            if c_dict=='(NX':
                groupnamesm=['','NX']
                mount=1
            elif c_dict=='NX)':
                mount=0
                groupnamesm[0]=groupnamesm[0].strip()
                tmp1.append(groupnamesm)
                groupnamesm=['','']
            elif c_dict=='(AX':
                groupnamesm=['','AX']
                mount=1
            elif c_dict=='AX)':
                mount=0
                groupnamesm[0]=groupnamesm[0].strip()
                tmp1.append(groupnamesm)
                groupnamesm=['','']
            elif c_dict=='(VX':
                groupnamesm=['','VX']
                mount=1
            elif c_dict=='VX)':
                mount=0
                groupnamesm[0]=groupnamesm[0].strip()
                tmp1.append(groupnamesm)
                groupnamesm=['','']
            elif mount:
                groupnamesm[0]=groupnamesm[0]+' '+c_dict
            else :
                pass

        if not only_salient_words_p:
            return tmp1
        table_str=[]
        the_tokenizer1=tmp1
        cd=[]

        for more in range(len(the_tokenizer1)):
            hostname_cleaned=the_tokenizer1[more]
            hostname_arrd=map(lambda alias_str:alias_str.split('/'),hostname_cleaned[0].split())
            hostname_arrd=filter(lambda alias_str:alias_str[1]not in['DT','MD'],hostname_arrd)
            hostname_arrd=filter(lambda alias_str:alias_str[0].lower()not in table_str,hostname_arrd)
            env_str=map(lambda alias_str:alias_str[0],hostname_arrd)
            popd=' '.join(env_str).lower().strip()

            if popd=='':
                continue
            hostname_arrd=map(lambda alias_str:alias_str[0]+'/'+alias_str[1],hostname_arrd)
            cd.append([' '.join(hostname_arrd),hostname_cleaned[1]])
        return cd

    def filter_by_tag(self,chunked_text,blacklist):
        env_cleaned=chunked_text.split()

        for cp_arr in range(len(env_cleaned)):

            if '/' not in env_cleaned[cp_arr]:
                continue

            if env_cleaned[cp_arr].split('/')[1]in blacklist:
                env_cleaned[cp_arr]=''
        env_cleaned=filter(lambda alias_str:alias_str!='',env_cleaned)
        return ' '.join(env_cleaned)

    def strip_tags(self,chunked_text):
        env_cleaned=chunked_text.split()
        env_cleaned=filter(lambda alias_str:'/' in alias_str,env_cleaned)
        env_cleaned=map(lambda alias_str:alias_str.split('/')[0],env_cleaned)
        return ' '.join(env_cleaned)

    def strip_tags_lemmatised(self,lemmatised_text):
        env_cleaned=lemmatised_text.split()
        env_cleaned=filter(lambda alias_str:'/' in alias_str,env_cleaned)
        env_cleaned=map(lambda alias_str:alias_str.split('/')[2],env_cleaned)
        return ' '.join(env_cleaned)