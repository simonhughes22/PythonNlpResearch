from shift_reduce_helper import *
from collections import defaultdict

class Oracle(object):
    def __init__(self, crels, parser):
        self.parser = parser
        self.raw_crels = crels
        self.crels = norm_arcs(crels)  # type: Set[Tuple[str,str]]
        self.mapping = self.build_mappings(crels)

    def build_mappings(self, pairs):
        mapping = defaultdict(set)
        for c, res in pairs:
            mapping[c].add(res)
            mapping[res].add(c)
        return mapping

    def should_continue(self, action):
        # continue parsing if REDUCE or LARC
        return action in (REDUCE, LARC)

    def remove_relation(self, a, b):
        self.mapping[a].remove(b)
        if len(self.mapping[a]) == 0:
            del self.mapping[a]
        self.mapping[b].remove(a)
        if len(self.mapping[b]) == 0:
            del self.mapping[b]

    def consult(self, tos, buffer):
        """
        Performs optimal decision for parser
        If true, continue processing, else Consume Buffer
        """
        parser = self.parser
        a, b = norm_arc((tos, buffer))
        if (a, b) in self.crels:
            # TOS has arcs remaining? If so, we need RARC, else LARC
            if len(self.mapping[tos]) == 1:
                return LARC
            else:
                return RARC
        else:
            if buffer not in self.mapping:
                return SKIP
            # If the buffer has relations further down in the stack, we need to POP the TOS
            for item in self.mapping[buffer]:
                if item == tos:
                    continue
                if parser.in_stack(item):
                    return REDUCE
            # end for
            # ELSE
            return SHIFT

    def execute(self, action, tos, buffer):
        """
        Performs optimal decision for parser
        If true, continue processing, else Consume Buffer
        """
        parser = self.parser
        if action == LARC:
            parser.left_arc(buffer)
            self.remove_relation(tos, buffer)
        elif action == RARC:
            parser.right_arc(buffer)
            self.remove_relation(tos, buffer)
        elif action == REDUCE:
            parser.reduce()
        elif action == SHIFT:
            parser.shift(buffer)
        elif action == SKIP:
            pass
        else:
            raise Exception("Unknown parsing action %s" % action)
        return self.should_continue(action)

    def tos(self):
        return self.parser.stack.tos()

    def is_stack_empty(self):
        return self.parser.stack.len() == 0

    def clone(self):
        cloney = Oracle(set(self.raw_crels), self.parser.clone())
        cloney.mapping = dict(self.mapping.items())
        return cloney