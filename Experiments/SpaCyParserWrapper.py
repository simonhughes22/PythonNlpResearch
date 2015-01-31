__author__ = 'simon.hughes'

from spacy.en import English
from Decorators import memoize
from collections import defaultdict

class BinaryRelation(object):
    def __init__(self, head, relation, child):
        self.relation = relation
        self.head = head
        self.child = "None" if (child is None or child.strip() == "") else child

    def __repr__(self):
        return "[%s]%s -> %s" % (self.relation, self.head, self.child)

class Relation(object):

    def __init__(self, head, relation, children):
        self.relation = relation
        self.head = head
        self.children = children

    def __repr__(self):
        skids = ",".join(self.children)
        return "[%s]%s -> %s" % (self.relation, self.head, skids)

    def binary_relations(self):
        rels = []
        if len(self.children) == 0:
            rels.append(BinaryRelation(self.head, self.relation, None))
        else:
            for ch in self.children:
                rels.append(BinaryRelation(self.head, self.relation, ch))


class Parser(object):

    def __init__(self):
        self.nlp = English()

    def parse(self, tokens):
        stokens = unicode(" ".join(tokens))

        tokens = self.__tokenize_(stokens)
        children_for_head = defaultdict(set)
        for token in tokens:
            children_for_head[token.head.i].add(token.string.strip())

        relations = []
        for token in tokens:
            kids = children_for_head[token.i]
            relations.append(Relation(token.head.string, token.dep_, list(kids)))

        assert len(relations) == len(tokens), "There are a different number of tokens to relations"
        return relations

    def pos_tag(self, tokens):
        stokens = unicode(" ".join(tokens))
        tokens = self.__tokenize_(stokens)
        return map(lambda t: t.pos_, tokens)

    @memoize
    def __tokenize_(self, sentence):
        return list(self.nlp(sentence, tag=True, parse=True))

if __name__ == "__main__":

    parser = Parser()
    split = "The increasing levels of carbon dioxide caused coral bleaching".split(" ")
    parsed = parser.parse(split)
    print parsed

    tags = parser.pos_tag(split)
    print tags
    pass