import GwData
import os

def write_to_file(parent_dir, dir, jobs_file, sentences):
    
    full_dir_path = parent_dir + dir + "\\"
    ensure_dir(full_dir_path)
    handle = open(full_dir_path + jobs_file, "w+")
    for s in sentences:
        handle.write(s.lower().strip() + "\n")
    handle.close()

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

# Code exists to segment the data set into folders, one per code, and within those folders spit out
# 4 files : Test (positive and negative) and VD (positive and negative)
# Used for RAE INPUT primarily
def split_by_class(dir, pctTest = 10):
    
    if not dir.endswith("\\"):
        dir = dir + "\\"
    
    data = GwData.GwData()
    pctTrain = (1.0 - (pctTest / 100.0))

    for code in data.sm_codes:
        s = data.sentences_for_code(code)
        not_s = data.sentences_not_for_code(code)
        
        train_s_cnt = int(len(s) * pctTrain) 
        train_s = s[:train_s_cnt] 
        test_s  = s[train_s_cnt:]
    
        train_not_s_cnt = int(len(not_s) * pctTrain)
        train_not_s = not_s[:train_not_s_cnt]
        test_not_s  = not_s[train_not_s_cnt:]
        
        friendly_code = code.replace(".", "_")
        write_to_file(dir, friendly_code, friendly_code + ".txt", train_s)
        write_to_file(dir, friendly_code, "test_" + friendly_code + ".txt", test_s)

        write_to_file(dir, friendly_code, "not_" + friendly_code + ".txt", train_not_s)
        write_to_file(dir, friendly_code, "test_not_" + friendly_code + ".txt", test_not_s)
    
    print "Done"
    
if __name__ == "__main__":
    split_by_class("C:\Users\Simon\Dropbox\PhD\Code\NLP Library\NlpLibrary\PyDevNLPLibrary\PyDevNLPLibrary\src\Data\GlobalWarming\SentencesByCode", 10.0)