# coding=utf-8


class GWConceptCodes(object):

    def __init__(self):
        self.CONCEPT_CODES = set("0,1,3,12,20,22,38,40,42,50,P14,P21,P28,P33,P34,P4,P40,P49".split(","))

        tmp = []
        for a in self.CONCEPT_CODES:
            tmp.append("Causer:{0}".format(a))
            tmp.append("Result:{0}".format(a))
            for b in self.CONCEPT_CODES:
                if a == b:
                    continue
                tmp.append("Causer:{0}->Result:{1}".format(a,b))
        self.CAUSAL_CODES = set(tmp)

    def is_valid_code(self, code):
        code = code.strip();
        return code in self.CONCEPT_CODES or code in self.CAUSAL_CODES

