
def collapse_num(num):
    if not num.isdigit():
        return num
    return "[" + ("1" * len(num)) + "]"

def collapse_dates(num):
    if len(num) == 0:
        return num
    
    #if num[-1] == "%" and num[:-1].isdigit():
     #   return "[99%]"
    
    if num[:-1].isdigit() and not num[-1].isdigit():
        return collapse_dates(num[:-1]) + num[-1]
    
    if not num.isdigit():
        return num
    
    val = float(num)
    if len(num) == 4 and val > 1000 and val < 2050:
        return "[DATE]"
    return "[" + ("1" * len(num)) + "]"

