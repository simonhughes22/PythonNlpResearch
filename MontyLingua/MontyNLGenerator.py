from __future__ import nested_scopes
__author__="Hugo Liu <hugo@media.mit.edu>"
__version__="2.0"
import zlib
import MontyUtils

class MontyNLGenerator:
    xtag_morph_filename="./xtag_morph_english.txt"
    morph_dict_filename="MONTYMORPH.MDF"

    def __init__(self):
        print "Loading Morph Dictionary!"
        tmp_str={}
        self.features_dict=tmp_str
        hostnames=self.get_features
        alias_cleaned={}
        self.morph_dict=alias_cleaned
        cmp_p=MontyUtils.MontyUtils().find_file(self.morph_dict_filename)

        if not cmp_p:
            print "Morph Dictionary not found...Now Building!"
            self.build_morph_dict()
            cmp_p=MontyUtils.MontyUtils().find_file(self.morph_dict_filename)
        case_p=open(cmp_p,'rb')
        groups_arr=self.setitem
        cats=zlib.decompress(case_p.read()).split('\n')

        for names in cats:
            output,domain_arr=names.split('=')
            domain_arr=tuple(map(lambda arg_cleaned:tuple(arg_cleaned.split(',')),domain_arr.split(';')))
            domain_arr=map(lambda arg_cleaned:(arg_cleaned[0],hostnames(arg_cleaned[1])),domain_arr)
            alias_cleaned[output]=domain_arr

    def generate_summary(self,vsoos):
        info_p=map(self.generate_sentence,vsoos)
        groups_dict=''
        buf_cleaned=0

        while buf_cleaned<len(info_p):

            if buf_cleaned>0:
                csplit_str=info_p[buf_cleaned-1].strip()

                if csplit_str[-1]in['!','.','?']:
                    csplit_str=csplit_str[:-1]
                groups_dicta=csplit_str.split()

                if len(groups_dicta)>1:
                    groups_dict=groups_dicta[-1]
                else :
                    groups_dict=''
            groups_dicta=info_p[buf_cleaned].split()

            if len(groups_dicta)>1:
                taggeds=groups_dicta[0]
            else :
                taggeds=''

            if taggeds.lower()==groups_dict.lower():
                mount=info_p[buf_cleaned-1].strip()

                if mount[-1]in['!','.','?']:
                    mount=mount[:-1]
                env_p=mount+' '+' '.join(groups_dicta[1:])
                info_p[buf_cleaned-1]=env_p
                del info_p[buf_cleaned]
                buf_cleaned -= 1
            buf_cleaned += 1
        return '  '.join(info_p)

    def generate_sentence(self,vsoo,sentence_type='declaration',tense='past',s_dtnum=('',1),o1_dtnum=('',1),o2_dtnum=('',1),o3_dtnum=('',1)):
        chroot_arr=self.conjugate_verb
        chownsq,outputf=vsoo[0:2]
        cron_p=vsoo[2:]
        file_dict=['aboard','about','above','across','after','against','along','amid','among','anti','around','as','at','before','behind','below','beneath','beside','besides','between','beyond','but','by','concerning','considering','despite','down','during','except','excepting','excluding','following','for','from','in','inside','into','like','minus','near','of','off','on','onto','opposite','outside','over','past','per','plus','regarding','round','save','since','than','through','to','toward','towards','under','underneath','unlike','until','up','upon','versus','via','with','within','without']

        if outputf.strip().lower()=='i':
            awk_dict=1
        else :
            awk_dict=0
        cron1=self.determine_verb(chownsq,tense,subject_number=1,ego_p=awk_dict)
        buffer_dict=self.determine_nounphrase(outputf,det=s_dtnum[0],number=s_dtnum[1])
        gawk_str=[]

        for buf_cleaned in range(len(cron_p)):
            object=cron_p[buf_cleaned]
            buf=''
            args=1

            if buf_cleaned==0:
                buf,args=o1_dtnum
            elif buf_cleaned==1:
                buf,args=o2_dtnum
            elif buf_cleaned==2:
                buf,args=o3_dtnum
            groups_dicta=object.split()

            if len(groups_dicta)>1 and groups_dicta[0]in file_dict:
                gawk_str.append(self.determine_prepphrase(object,det=buf,number=args))
            else :
                gawk_str.append(self.determine_nounphrase(object,det=buf,number=args))
        case_pa=''

        if sentence_type=='imperative':
            case_pa=' '.join([cron1,' '.join(gawk_str)])+'!'
        elif sentence_type in('can','may','would','should','could'):
            cron1=self.determine_verb(chownsq,tense='infinitive',subject_number=1)
            case_pa=' '.join([sentence_type,buffer_dict,cron1,' '.join(gawk_str)])+'?'
        elif sentence_type=='question':

            if tense in['progressive','past_progressive','future']:
                case_pa=' '.join([cron1.split()[0],buffer_dict,' '.join(cron1.split()[1:]),' '.join(gawk_str)])+'?'
            else :
                case_pa=' '.join([buffer_dict,cron1,' '.join(gawk_str)])+'?'
        elif sentence_type in('who','what','when','where','why','how'):

            if tense in['progressive','past_progressive','future']:
                case_pa=' '.join([sentence_type,cron1.split()[0],buffer_dict,' '.join(cron1.split()[1:]),' '.join(gawk_str)])+'?'
            else :
                case_pa=' '.join([sentence_type,buffer_dict,cron1,' '.join(gawk_str)])+'?'
        else :
            case_pa=' '.join([buffer_dict,cron1,' '.join(gawk_str)])+'.'
        case_pa=case_pa.strip()

        if len(case_pa)>1:
            case_pa=case_pa[0].upper()+case_pa[1:]
        return case_pa

    def determine_prepphrase(self,prepphrase,det='',number=1):
        hash1=['from','of','to','in','out']
        groups_dicta=prepphrase.split()

        if len(groups_dicta)<2:
            return prepphrase
        cp_cleanedk=groups_dicta[0]
        case=groups_dicta[1]
        dirname_cleaned=groups_dicta[2:]

        if case in hash1:
            cp_cleanedk=' '.join(groups_dicta[0:2])
        else :
            dirname_cleaned=groups_dicta[1:]
        dirname_cleaned=' '.join(dirname_cleaned)
        cmp_str=self.determine_nounphrase(dirname_cleaned,det=det,number=number)
        return ' '.join([cp_cleanedk,cmp_str])

    def determine_nounphrase(self,nounphrase,det='',number=1):
        gawk_cleaned=['who','what','when','whom','it','its','his','her','hers','they','their','us','you','me','them','those','these','he','she','we','mine','yours','ours','theirs','myself','yourself','himself','herself','itself','ourselves','yourselves','themselves','oneself','my']

        if not nounphrase:
            return ''
        groups_dicta=nounphrase.split()

        if len(groups_dicta)>0 and groups_dicta[0]in gawk_cleaned:
            det=''

        if len(groups_dicta)==1:
            filename1=''
            nice_str=nounphrase
        else :
            filename1=' '.join(groups_dicta[:-1])
            nice_str=groups_dicta[-1]
        chmod_p=self.morph_noun(nice_str,number=number)

        if det in['a','an']:
            output_dict=nounphrase[0]

            if output_dict in['a','e','i','o','u']:
                det='an'
            else :
                det='a'
        return ' '.join(((' '.join([det,filename1,chmod_p])).split()))

    def determine_verb(self,verb,tense,subject_number=1,ego_p=0):
        chroot_arr=self.conjugate_verb
        arg_cleanedj='VBZ'

        if ego_p:
            arg_cleanedj='VBP'
        chmods=0

        if len(verb.split())>1 and verb.split()[0]=='not':
            chmods=1
            verb=verb.split()[1]

        if tense=='present' and not chmods:
            cron1=chroot_arr(verb,arg_cleanedj)
        elif tense=='present' and chmods:

            if verb in['have','be']:
                cron1=chroot_arr(verb,arg_cleanedj)+' not'
            else :
                cron1="does not "+chroot_arr(verb,'VB')
        elif tense=='past' and not chmods:
            cron1=chroot_arr(verb,'VBD')
        elif tense=='past' and chmods:

            if verb in['have','be']:
                cron1=chroot_arr(verb,'VBD')+' not'
            else :
                cron1="did not "+chroot_arr(verb,'VB')
        elif tense=='progressive' and not chmods:
            cron1=chroot_arr('be',arg_cleanedj)+' '+chroot_arr(verb,'VBG')
        elif tense=='progressive' and chmods:
            cron1=chroot_arr('be',arg_cleanedj)+' not '+chroot_arr(verb,'VBG')
        elif tense=='past_progressive' and not chmods:
            cron1="was "+chroot_arr(verb,'VBG')
        elif tense=='past_progressive' and chmods:
            cron1="was not "+chroot_arr(verb,'VBG')
        elif tense=='future' and not chmods:
            cron1="will "+chroot_arr(verb,'VB')
        else :

            if chmods:
                cron1=chroot_arr(verb,'VB')+' not'
            else :
                cron1=chroot_arr(verb,'VB')
        return cron1

    def get_features(self,feature_string):
        tmp_str=self.features_dict

        if not tmp_str.has_key(feature_string):
            tmp_str[feature_string]=tuple(feature_string.split('|'))
        return tmp_str[feature_string]

    def build_morph_dict(self):
        self.load_xtag_morph()
        self.output_morph_dict()

    def reformulate_lifenet(self):
        case_p=open('action-items.txt','r')
        cats=case_p.read().split('\n')
        case_p.close()
        file_arr=[]

        for names in cats:
            file_arr += self.all_egocentric_declarations(names)
            file_arr +=['']
        b_dict=open('paraphrased_actions.txt','w')
        b_dict.write('\n'.join(file_arr))
        b_dict.close()
        case_p=open('thing-items.txt','r')
        cats=case_p.read().split('\n')
        cats=map(lambda arg_cleaned:'see '+arg_cleaned,cats)
        case_p.close()
        file_arr=[]

        for names in cats:
            file_arr += self.all_egocentric_declarations(names)
            file_arr +=['']
        b_dict=open('paraphrased_things.txt','w')
        b_dict.write('\n'.join(file_arr))
        b_dict.close()
        case_p=open('place-items.txt','r')
        cats=case_p.read().split('\n')
        cats=map(lambda arg_cleaned:'am '+arg_cleaned,cats)
        case_p.close()
        file_arr=[]

        for names in cats:
            file_arr += self.all_egocentric_declarations(names)
            file_arr +=['']
        b_dict=open('paraphrased_places.txt','w')
        b_dict.write('\n'.join(file_arr))
        b_dict.close()
        return

    def all_egocentric_declarations(self,simple_vp):
        names=simple_vp
        file_arr=[]
        groups_dicta=names.split()

        if len(groups_dicta)==0:
            return[]
        chownsq=groups_dicta[0]

        if chownsq=='am':
            chownsq='be'
        dirname_cleaned=' '.join(groups_dicta[1:])
        file_arr.append(('I '+self.conjugate_verb(chownsq,'VBP')+' '+dirname_cleaned).strip())
        file_arr.append(('I '+self.conjugate_verb(chownsq,'VBD')+' '+dirname_cleaned).strip())
        file_arr.append(('I '+self.conjugate_verb(chownsq,'VBD')+' not '+dirname_cleaned).strip())
        file_arr.append(('I am '+self.conjugate_verb(chownsq,'VBG')+' '+dirname_cleaned).strip())
        file_arr.append(('I am not '+self.conjugate_verb(chownsq,'VBG')+' '+dirname_cleaned).strip())
        file_arr.append(('I had '+self.conjugate_verb(chownsq,'VBN')+' '+dirname_cleaned).strip())
        file_arr.append(('I hadn\'t '+self.conjugate_verb(chownsq,'VBN')+' '+dirname_cleaned).strip())
        file_arr.append(('I have '+self.conjugate_verb(chownsq,'VBN')+' '+dirname_cleaned).strip())
        file_arr.append(('I haven\'t '+self.conjugate_verb(chownsq,'VBN')+' '+dirname_cleaned).strip())
        file_arr.append(('I want to '+self.conjugate_verb(chownsq,'VB')+' '+dirname_cleaned).strip())
        file_arr.append(('I don\'t want to '+self.conjugate_verb(chownsq,'VB')+' '+dirname_cleaned).strip())
        file_arr.append(('I wanted to '+self.conjugate_verb(chownsq,'VB')+' '+dirname_cleaned).strip())
        file_arr.append(('I didn\'t want to '+self.conjugate_verb(chownsq,'VB')+' '+dirname_cleaned).strip())
        file_arr.append(('I would '+self.conjugate_verb(chownsq,'VB')+' '+dirname_cleaned).strip())
        file_arr.append(('I would\'nt  '+self.conjugate_verb(chownsq,'VB')+' '+dirname_cleaned).strip())
        file_arr.append(('I can '+self.conjugate_verb(chownsq,'VB')+' '+dirname_cleaned).strip())
        file_arr.append(('I can\'t '+self.conjugate_verb(chownsq,'VB')+' '+dirname_cleaned).strip())
        file_arr.append(('I could '+self.conjugate_verb(chownsq,'VB')+' '+dirname_cleaned).strip())
        file_arr.append(('I couldn\'t '+self.conjugate_verb(chownsq,'VB')+' '+dirname_cleaned).strip())
        file_arr.append(('I may '+self.conjugate_verb(chownsq,'VB')+' '+dirname_cleaned).strip())
        file_arr.append(('I may not '+self.conjugate_verb(chownsq,'VB')+' '+dirname_cleaned).strip())
        file_arr.append(('I might '+self.conjugate_verb(chownsq,'VB')+' '+dirname_cleaned).strip())
        file_arr.append(('I might not '+self.conjugate_verb(chownsq,'VB')+' '+dirname_cleaned).strip())
        file_arr.append(('I must '+self.conjugate_verb(chownsq,'VB')+' '+dirname_cleaned).strip())
        file_arr.append(('I must not '+self.conjugate_verb(chownsq,'VB')+' '+dirname_cleaned).strip())
        file_arr.append(('I ought to '+self.conjugate_verb(chownsq,'VB')+' '+dirname_cleaned).strip())
        file_arr.append(('I ought not to '+self.conjugate_verb(chownsq,'VB')+' '+dirname_cleaned).strip())
        file_arr.append(('I shall '+self.conjugate_verb(chownsq,'VB')+' '+dirname_cleaned).strip())
        file_arr.append(('I shall not '+self.conjugate_verb(chownsq,'VB')+' '+dirname_cleaned).strip())
        file_arr.append(('I should '+self.conjugate_verb(chownsq,'VB')+' '+dirname_cleaned).strip())
        file_arr.append(('I shouldn\'t '+self.conjugate_verb(chownsq,'VB')+' '+dirname_cleaned).strip())
        file_arr.append(('I will '+self.conjugate_verb(chownsq,'VB')+' '+dirname_cleaned).strip())
        file_arr.append(('I won\'t '+self.conjugate_verb(chownsq,'VB')+' '+dirname_cleaned).strip())
        file_arr.append(('I would have '+self.conjugate_verb(chownsq,'VBN')+' '+dirname_cleaned).strip())
        file_arr.append(('I wouldn\'t have '+self.conjugate_verb(chownsq,'VBN')+' '+dirname_cleaned).strip())
        return file_arr

    def morph_noun(self,noun_lemma,number=1):
        tables=''

        if number>1:
            tables=self.get_morph_complex(noun_lemma+'/N',['3pl'])

        if not tables:
            tables=noun_lemma
        return tables

    def conjugate_verb(self,verb_lemma,mode):
        tables=''

        if mode=='VB':
            tables=self.get_morph_complex(verb_lemma+'/V',['INF'])
        elif mode=='VBD':
            tables=self.get_morph_complex(verb_lemma+'/V',['PAST'])
        elif mode=='VBG':
            tables=self.get_morph_complex(verb_lemma+'/V',['PROG'])
        elif mode=='VBN':
            tables=self.get_morph_complex(verb_lemma+'/V',['PPART'])
        elif mode=='VBP':
            tables=self.get_morph_complex(verb_lemma+'/V',['1sg','PRES'])
        elif mode=='VBZ':
            tables=self.get_morph_complex(verb_lemma+'/V',['3sg','PRES'])

        if not tables:
            tables=verb_lemma

        if tables=='wert':
            tables='was'
        return tables

    def get_morph_complex(self,tagged_lemma,desired_features):
        cleaned=self.morph_dict.get(tagged_lemma,[])

        if len(cleaned)==0:
            return ''
        popd_arr=[]

        for outputs in cleaned:
            stripped_p,command_str=outputs
            chmod=1

            for the_tokenizer_cleaned in desired_features:

                if not(the_tokenizer_cleaned in command_str):
                    chmod=0

            if chmod:
                popd_arr.append(stripped_p)

        if not popd_arr:
            return ''
        else :
            return popd_arr[0]

    def output_morph_dict(self):
        alias_cleaned=self.morph_dict
        argsv=alias_cleaned.keys()
        argsv.sort()
        cats=[]

        for output in argsv:
            domain_arr=alias_cleaned[output]
            domain_arr=';'.join(map(lambda arg_cleaned:arg_cleaned[0]+','+'|'.join(arg_cleaned[1]),domain_arr))
            names=output+'='+domain_arr
            cats.append(names)
        line_cleaned='\n'.join(cats)
        info_dict=zlib.compress(line_cleaned,1)
        b_dict=open(self.morph_dict_filename,'wb')
        b_dict.write(info_dict)
        b_dict.close()
        return

    def load_xtag_morph(self):
        case_p=open(self.xtag_morph_filename,'r')
        info_dictv=case_p.readline()

        while info_dictv:

            if info_dictv[0:1]==';':
                info_dictv=case_p.readline()
                continue
            groups_dicta=info_dictv.split()

            if len(groups_dicta)<3:
                info_dictv=case_p.readline()
                continue
            csplit1=groups_dicta[0]
            dirnames=' '.join(groups_dicta[1:]).split('#')
            dirnames=map(lambda arg_cleaned:arg_cleaned.split(),dirnames)

            for gawk_cleanedk in dirnames:
                output=gawk_cleanedk[0]
                hostname_str=gawk_cleanedk[1]
                chmod_dict=gawk_cleanedk[2:]
                dirname_arr=output+'/'+hostname_str
                args_str=self.morph_dict.get(dirname_arr,[])
                args_str.append([csplit1,chmod_dict])
                self.morph_dict[dirname_arr]=args_str
            info_dictv=case_p.readline()
        case_p.close()

    def setitem(self,dict,key,value):
        dict[key]=value

if __name__=="__main__":
    m=MontyNLGenerator()
    m.reformulate_lifenet()