class Stack(object):
    def __init__(self, verbose=False):
        self.__stack_ = []
        self.__verbose_ = verbose

    def tos(self):
        if self.len() == 0:
            return None
        # assert self.len() > 0, "Can't peek when stack is empty"
        return self.__stack_[-1]

    def pop(self):
        assert self.len() > 0, "Can't pop when stack is empty"
        item = self.__stack_.pop()
        if self.__verbose_:
            print("POPPING: %s" % str(item))
            print("LEN:     %i" % len(self.__stack_))
        return item

    def push(self, item):
        self.__stack_.append(item)
        if self.__verbose_:
            print("PUSHING: %s" % str(item))
            print("LEN:     %i" % len(self.__stack_))

    def len(self):
        return len(self.__stack_)

    def contains(self, item):
        return item in self.__stack_

    def __repr__(self):
        return "|".join(map(str, self.__stack_))

    def clone(self):
        cloney = Stack(self.__verbose_)
        cloney.__stack_ = list(self.__stack_)
        return cloney

    def contents(self):
        return list(self.__stack_)