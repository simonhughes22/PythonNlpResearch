# coding=utf-8

class A(object):
    def __add_(self, a,b):
        return a+b

    def add(self, a, b):
        return self.__add_(a,b)

    def nested_function_add(self, a,b):
        def nested_add(a,b):
            return a + b
        return nested_add(a,b)

a = A()
#print "Add using class reference (allowed)"
#print a._A__add_(1,3)

print "Add using public method"
print a.add(1,2)

print "Add using nested add"
print a.nested_function_add(1,2)

print "Add using instance reference (not allowed)"
print a.__add_(1,2)
