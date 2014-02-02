'''
Created on Jan 27, 2013

@author: Simon
'''

class CodeToCategory(object):
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
        
        if code.startswith("CL"):
            return "CL"
        if code.startswith("SCL"):
            return "SCL"
        if code.startswith("ESCL"):
            return "ESCL"
        if code.startswith("IR"):
            return "IR"
        if code.startswith("RC"):
            return "RC"
        #Not sure how to categorize the remainder
        return code
        
        
    