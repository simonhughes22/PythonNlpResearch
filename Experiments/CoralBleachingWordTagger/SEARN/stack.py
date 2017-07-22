class Stack(object):
    def __init__(self, verbose=False):
        self.stack = []
        self.verbose = verbose

    def tos(self):
        if self.len() == 0:
            return None
        # assert self.len() > 0, "Can't peek when stack is empty"
        return self.stack[-1]

    def pop(self):
        assert self.len() > 0, "Can't pop when stack is empty"
        item = self.stack.pop()
        if self.verbose:
            print("POPPING: %s" % item)
            print("LEN:     %i" % len(self.stack))
        return item

    def push(self, item):
        self.stack.append(item)
        if self.verbose:
            print("PUSHING: %s" % item)
            print("LEN:     %i" % len(self.stack))

    def len(self):
        return len(self.stack)

    def contains(self, item):
        return item in self.stack

    def __repr__(self):
        return "|".join(self.stack)

    def clone(self):
        cloney = Stack(self.verbose)
        cloney.stack = list(self.stack)
        return cloney