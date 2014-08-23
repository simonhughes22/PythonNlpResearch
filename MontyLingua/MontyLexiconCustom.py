__author__="Hugo Liu <hugo@media.mit.edu>"
__version__="2.0"
import zlib
import MontySettings,MontyUtils

class MontyLexiconCustom:
    custom_lexicon_filename='CUSTOMLEXICON.MDF'

    def __init__(self):
        self.word_pos_table={}

        if MontyUtils.MontyUtils().find_file(self.custom_lexicon_filename)!='':
            print "Custom Lexicon Found! Now Loading!"
            self.load_customlexicon()

    def get(self,word,default):
        return self.word_pos_table.get(word,default)

    def set_word(self,word,poses):
        self.word_pos_table[word]=poses
        return

    def load_customlexicon(self):
        awk1=MontyUtils.MontyUtils()
        groupnames_p=awk1.find_file(self.custom_lexicon_filename)
        contents_cleaned=open(groupnames_p,'r')
        cmp_cleaned=contents_cleaned.read()
        chmods=cmp_cleaned.split('\n')
        chmods=map(lambda case_cleaned:case_cleaned.strip(),chmods)
        chmods=map(lambda case_cleaned:case_cleaned.split(),chmods)
        tagged_str=map(lambda chroot_cleaned:[chroot_cleaned[0],chroot_cleaned[1:]],
filter(lambda case_cleaned:len(case_cleaned)>=2,chmods))

        for pairss in tagged_str:
            file_p,chown=pairss
            self.word_pos_table[file_p]=chown
        return 