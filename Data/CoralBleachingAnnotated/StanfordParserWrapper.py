from stanford_parser import parser

class Dependency(object):
    def __init__(self, relation, gov_tuple, dep_tuple):
        self.relation = relation
        self.gov_txt, self.gov_start, self.gov_end = gov_tuple
        self.dep_txt, self.dep_start, self.dep_end = dep_tuple

        self.gov_tuple = gov_tuple
        self.dep_tuple = dep_tuple

    def __repr__(self):
        return "%s %s=> %s" % (self.gov_txt.ljust(10), ("[" + self.relation + "]").ljust(10), self.dep_txt)

class SentenceProcessorMixin(object):
    def get_sentence_txt(self, tokens):
        if type(tokens) == str:
            sentence_txt = tokens
        else:
            sentence_txt = " ".join(t for t in tokens if t.isalnum())
        return sentence_txt

class DependencyParser(SentenceProcessorMixin):

    def __init__(self):
        self.__parser_    = parser.Parser()


    def parse(self, tokens):
        sentence_txt = self.get_sentence_txt(tokens)
        dependencies = self.__parser_.parseToStanfordDependencies(sentence_txt)
        depResult  = [ Dependency(rel, (gov.text, gov.start, gov.end), (dep.text, dep.start, dep.end))
                            for rel, gov, dep in dependencies.dependencies
                        ]
        return depResult

class ConstituencyParser(SentenceProcessorMixin):
    def __init__(self):
        self.__parser_ = parser.Parser()

    def parse(self, tokens):
        sentence_txt = self.get_sentence_txt(tokens)
        # return the root nodes' toString - returns the flattened parse tree
        return list(self.__parser_.parse(sentence_txt)[1])[0].toString()

if __name__ == "__main__":

    txt = "The cat was chased by the dog , who was carrying a bone , up the tree"

    dep_parser = DependencyParser()
    tuples = dep_parser.parse(txt.split())

    print "\n".join(txt.split())
    for dep in sorted(tuples, key=lambda dep: min(dep.gov_start, dep.dep_start)):
        print dep



    con_parser = ConstituencyParser()
    print con_parser.parse(txt.split())