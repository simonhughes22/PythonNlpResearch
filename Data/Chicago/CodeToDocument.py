'''
Created on Jan 27, 2013

@author: Simon
'''

class CodeToDocument(object):
    '''
    Computes a category for an sm code
    Categories are 
        Claim (CL),
        Sub-Claim (SCL),
        Evidence (ESCL), 
        Causal Relation (RC),
        Inferred Relation (IR)
    '''
    def Map(self, code):
        number = self.__first_number__(code)
        if number == None:
            return "None"
        
        if number == 1:
            return "IU"
        if number == 2:
            return "BLU"
        if number == 3:
            return "TU"
        
        raise Exception("CodeToDocument - Unknown document number: " + str(number))
        
    def __first_number__(self, str):
        for ch in str:
            if self.__is_number__(ch):
                return int(ch)
        return None
            
    def __is_number__(self, ch):
        ordVal = ord(ch)
        return ordVal >= 48 and ordVal <= 57