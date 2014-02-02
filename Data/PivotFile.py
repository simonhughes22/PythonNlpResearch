from collections import defaultdict

class PivotFile(object):

    def __init__(self, map_file, delim = "\t",  directory = None):
        if directory == None:
            directory = "C:\Users\simon\Dropbox\PhD\Code\NLP Library\NlpLibrary\PyDevNLPLibrary\PyDevNLPLibrary\src\Data\Dice"

        if not directory.endswith("\\"):
            directory += "\\"
        
        file = open(directory + map_file)
        lines = file.readlines()
        file.close()
        
        # Skip header
        key = ""
        
        key_to_tokens = defaultdict(list)
        values = set()
        
        for i,line in enumerate(lines[1:]):
            
            if i % 100000 == 0:
                print "Index:" + str(i)
            
            if len(line.strip()) == 0:
                continue
            
            splt = [s.strip() for s in line.lower().split(delim)]
            assert len(splt) == 2

            key = splt[0]
        
            key_to_tokens[key].append(splt[1])
            values.add(splt[1])
        
        self.keys = []
        self.key_to_ix = dict()
        self.ix_to_key = dict()
        self.tokens = []
        
        i = 0
        for d,l in key_to_tokens.items():
            self.keys.append(d)
            ix = len(self.keys) - 1
            self.key_to_ix[d] = ix
            self.ix_to_key[ix] = d
            self.tokens.append(l)
            
            i += 1
        
        print str(len(self.keys)) + " keys processed"
        print str(len(values)) + " values processed"

def load_job_skills():
    js = PivotFile("job_skill.txt", "\t" )
    return js

def load_employee_skills():
    es = PivotFile("employee_skill.txt", "\t" )
    return es

if __name__ == "__main__":
    
    es = load_employee_skills()
    js = load_job_skills()
    i= 10
    pass