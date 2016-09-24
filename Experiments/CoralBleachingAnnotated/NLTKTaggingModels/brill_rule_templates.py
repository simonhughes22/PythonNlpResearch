# coding=utf-8
from nltk.tag.brill import Word
from nltk.tbl import Template


def brill_rules_pos_wd_feats_offset_4():
    """
    Return 24 templates of the seminal TBL paper, Brill (1995)
    """
    return [
        Template(Word([-1])),
        Template(Word([-2])),
        Template(Word([-3])),
        Template(Word([-4])),

        Template(Word([0])),

        Template(Word([1])),
        Template(Word([2])),
        Template(Word([3])),
        Template(Word([4])),
    ]

def brill_rules_pos_bigram_feats_offset_4():
    """
    Return 24 templates of the seminal TBL paper, Brill (1995)
    """
    return [
        Template(Word([-1, 0])),
        Template(Word([-2, -1])),
        Template(Word([-3, -2])),
        Template(Word([-4, -3])),

        Template(Word([1, 0])),
        Template(Word([2, 1])),
        Template(Word([3, 2])),
        Template(Word([4, 3]))
    ]