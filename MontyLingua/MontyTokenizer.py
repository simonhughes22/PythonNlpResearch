__author__="Hugo Liu <hugo@media.mit.edu>"
__version__="2.0"
import string,re

class MontyTokenizer:

    def __init__(self):
        pass

    def split_paragraphs(self,text):
        info1=re.split('[\n]+',text)
        return info1

    def split_sentences(self,text):
        argss=self.common_abbrev_and_acro
        input=text.split()
        dirname=0
        contents_arr=re.compile('([?!]+|[.][.]+)$')
        more1=re.compile('([.])$')

        while dirname<len(input):
            info_cleaned=contents_arr.search(input[dirname])

            if info_cleaned:
                input.insert(dirname+1,'<sentence_break/>')
                dirname += 1
                continue
            info_cleaned=more1.search(input[dirname])

            if info_cleaned:

                if input[dirname].lower()not in argss:
                    input.insert(dirname+1,'<sentence_break/>')
                dirname += 1
                continue
            dirname += 1
        text=' '.join(input)
        text=re.sub('(\<sentence_break\/\> ?){2,20}','<sentence_break/> ',text)
        text=re.sub('(\<(sentence|paragraph)_break\/\> *)+$','',text)
        return text.split('<sentence_break/>')

    def tokenize(self,sentence,expand_contractions_p=0):
        built_in_str=self.common_abbrev_and_acro
        cd=re.search
        the_tokenizer_p=string.uppercase
        sentence=' '+sentence+' '
        j_arr=['`','^','*','=','+','|','\\','[',']','}','{',',','!','?','#','&','(',')','"','>','<','~',';']
        b_dict=['.','@','/',':']
        sentence=sentence.replace('/',':')

        for hostnamess in j_arr:
            sentence=sentence.replace(hostnamess,' '+hostnamess+' ')
        buffer_str=sentence.split()

        for dirname in range(len(buffer_str)):
            bs=buffer_str[dirname]

            if '/' in bs and bs[bs.index('/'):].upper()==bs[bs.index('/'):]:
                continue

            if bs.lower()in built_in_str:
                continue
            info_cleaned=cd('^([A-Z][.])+$',bs)

            if info_cleaned:
                continue
            info_cleaned=cd('^[$][0-9]{1,3}[.][0-9][0-9](?P<period>[.]?)$',bs)

            if info_cleaned:

                if info_cleaned.group('period')=='.':
                    buffer_str[dirname]=buffer_str[dirname][:-1]+' '+'.'
                continue

            for hostnamess in b_dict:
                buffer_str[dirname]=buffer_str[dirname].replace(hostnamess,' '+hostnamess+' ')
            buffer_str[dirname]=buffer_str[dirname].strip()
        sentence=' '.join(buffer_str)

        if expand_contractions_p:
            filename_str=self.contractions_unwound
        else :
            filename_str=self.contractions_separated
        domain_cleaned=' (?P<begin>)(?P<word>'

        for _montylingua in filename_str.keys():
            domain_cleaned += _montylingua+'|'
        domain_cleaned=domain_cleaned[:-1]
        domain_cleaned += ')(?P<end>) '
        _montylingua_arr=1

        while _montylingua_arr:
            _montylingua_arr=0
            info_cleaned=cd(domain_cleaned,sentence.lower())

            if info_cleaned:
                id_arr=filename_str[info_cleaned.group('word')]

                if sentence[info_cleaned.start('begin')]in the_tokenizer_p:
                    id_arr=id_arr[0].upper()+id_arr[1:]
                else :
                    id_arr=id_arr[0].lower()+id_arr[1:]
                sentence=sentence[:info_cleaned.start('begin')]+id_arr+sentence[info_cleaned.end('end'):]
                _montylingua_arr=1
        sentence=sentence.replace("'s "," 's ")
        sentence=sentence.replace("'d "," 'd ")

        if expand_contractions_p:
            sentence=sentence.replace("'ll "," will ")
        else :
            sentence=sentence.replace("'ll "," 'll ")

        if expand_contractions_p:
            sentence=sentence.replace(" i "," I ")
        return sentence
    contractions_separated={
"ain't":"ai n't",
"aren't":"are n't",
"isn't":"is n't",
"wasn't":"was n't",
"weren't":"were n't",
"didn't":"did n't",
"doesn't":"does n't",
"don't":"do n't",
"hadn't":"had n't",
"hasn't":"has n't",
"haven't":"have n't",
"can't":"ca n't",
"couldn't":"could n't",
"needn't":"need n't",
"shouldn't":"should n't",
"shan't":"sha n't",
"won't":"wo n't",
"wouldn't":"would n't",
"i'm":"i 'm",
"you're":"you 're",
"he's":"he 's",
"she's":"she 's",
"it's":"it 's",
"we're":"we 're",
"they're":"they 're",
"i've":"i 've",
"you've":"you 've",
"we've":"we 've",
"they've":"they 've",
"who've":"who 've",
"what've":"what 've",
"when've":"when 've",
"where've":"where 've",
"why've":"why 've",
"how've":"how 've",
"i'd":"i 'd",
"you'd":"you 'd",
"he'd":"he 'd",
"she'd":"she 'd",
"we'd":"we 'd",
"they'd":"they 'd",
"i'll":"i 'll",
"you'll":"you 'll",
"he'll":"he 'll",
"she'll":"she 'll",
"we'll":"we 'll",
"they'll":"they 'll",
}
    contractions_unwound={
"ain't":"ai not",    }
    common_abbrev_and_acro=[
'mr.',
'mrs.',
'ms.',
'sr.',
'esq.',
'jr.',
'dr.',
's.b.',
'ph.d.',
'm.d.',
'm.eng.',
'm.f.a.',
'd.d.s.',
'sc.d.',
'b.s.',
'b.sc.',
'b.a.',
'a.b.',
'm.a.',
'c.p.a.',
'prof.',
'capt.',
'col.',
'gen.',
'sgt.',
'lt.',
'priv.',
'ft.',
'nav.',
'a.f.',
'u.s.a.f.',
'a.f.b.'
 'i.e.',
'etc.',
'e.g.',
'c.f.',
'p.s.',
'q.e.d.',
'i.',
'ii.',
'iii.',
'iv.',
'v.',
'vi.',
'vii.',
'viii.',
'ix.',
'x.',
'a.m.',
'p.m.',
'morn.',
'eve.',
'corp.',
'inc.',
'co.',
'ltd.',
'reg.',
'u.p.s.',
'u.s.p.s.',
'fedex.',
'i.b.m.',
'a.o.l.',
'jan.',
'feb.',
'febr.',
'mar.',
'apr.',
'may.',
'jun.',
'jul.',
'aug.',
'sep.',
'sept.',
'oct.',
'nov.',
'dec.',
'ala.',
'ariz.',
'ark.',
'calif.',
'colo.',
'conn.',
'del.',
'd.c.',
'fla.',
'ga.',
'ill.',
'ind.',
'kans.',
'ky.',
'la.',
'md.',
'mass.',
'mich.',
'minn.',
'miss.',
'mo.',
'nebr.',
'nev.',
'n.h.',
'n.j.',
'n.m.',
'n.y.',
'n.c.',
'n.d.',
'okla',
'ore.',
'pa.',
'p.r.',
'r.i.',
's.c.',
's.d.',
'tenn.',
'tex.',
'vt.',
'va.',
'v.i.',
'wash.',
'w.va.',
'wis.',
'wyo.',
'v.c.r.',
'v.h.s.',
'd.v.d.',
'v.c.d.',
'c.d.',
'tele.',
'tv.',
't.v.',
'p.c.',
'd.s.l.',
'a.s.a.p.',
'r.s.v.p.',
'n.y.c.',
'c.o.d.',
's.u.v.']