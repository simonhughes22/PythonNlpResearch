from shift_reduce_helper import *

class Parser(object):
    def __init__(self, stack):
        self.stack = stack
        self.arcs = []
        self.normed_arcs = set()
        # nodes with heads
        self.children = set()
        self.actions = []

    def get_dependencies(self):
        return [(l, r) for (l, r) in self.arcs if r != ROOT and l != ROOT]

    def left_arc(self, buffer):
        tos = self.stack.pop()
        # Pre-condition
        # assert self.has_head(tos) == False
        arc = (tos, buffer)
        n_arc = norm_arc(arc)
        assert n_arc not in self.normed_arcs, "Arc already processed %s" % (n_arc)
        self.arcs.append(arc)
        self.normed_arcs.add(arc)
        self.children.add(tos)
        self.actions.append("L ARC   : " + tos + "->" + buffer)

    def right_arc(self, buffer):
        tos = self.stack.tos()
        # normalize arc
        arc = (buffer, tos)
        n_arc = norm_arc(arc)
        assert n_arc not in self.normed_arcs, "Arc already processed %s" % (n_arc)
        self.arcs.append(arc)
        self.normed_arcs.add(n_arc)
        self.actions.append("R ARC   : " + tos + "<-" + buffer)
        self.children.add(buffer)
        self.stack.push(buffer)

    def reduce(self):
        tos = self.stack.pop()
        # assert self.has_head(tos) == True
        self.actions.append("REDUCE  : Pop  %s" % tos)

    def shift(self, buffer):
        self.stack.push(buffer)
        self.actions.append("SHIFT   : Push %s" % buffer)

    def skip(self, buffer):
        self.actions.append("SKIP    : item %s" % buffer)

    def has_head(self, item):
        return item in self.children

    def in_stack(self, item):
        return self.stack.contains(item)

    def clone(self):
        cloney = Parser(self.stack.clone())
        cloney.arcs = list(self.arcs)
        cloney.normed_arcs = set(self.normed_arcs)
        # nodes with heads
        cloney.children = set(self.children)
        cloney.actions = list(self.actions)
        return cloney