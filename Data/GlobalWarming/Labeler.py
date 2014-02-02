'''
Created on Mar 30, 2013

@author: Simon
'''
def is_causal(code):
        if code.replace(" ","").find(">") > -1:
            return 1.0
        return 0.0

def to_parent_code(code):
    if code.count(".") != 1 or code.startswith("."):
        return code

    index = code.find(".")
    return code[0:index]


if __name__ == "__main__":

  code = "50.1"
  to_parent_code(code)
  print code

