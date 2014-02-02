import numpy as np
import logging
import sys

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
def vectors_to_file(vectors, jobs_file, delim = ","):
    
    f = open(jobs_file, "w+")
    for v in vectors:
        v_str = delim.join([str(x) for x in v])
        f.write(v_str + "\n")
    f.close()
    
def vectors_from_file(jobs_file, delim = ","):
    f = open(jobs_file, "r+")
    vectors = []
    for line in f.readlines():
        splt = line.split(delim)
        vectors.append([float(x.strip()) for x in splt])
        
    f.close()
    return np.array(vectors)

def file_apply(file_names, func, skip_header = False, delim = ",", progress_every = -1, stop_on_error = True):    
    """ 
        Read in a file, applying the func to each line
        and return result (or None if func does not return a value)
    """
    rslt = []
    if type(file_names) == type(""):
        file_names = [file_names]
    
    for jobs_file in file_names:
        row = 0
        print("Opening: " + jobs_file)
        with open(jobs_file, "r+") as f:
            line = f.readline().strip()
            if skip_header:
                line = f.readline().strip()
            while line != "":
                row +=1
                if progress_every > 0 and row % progress_every == 0:
                    print row
                
                if line.strip() != "":
                    splt = line.split(delim)
                    
                    try:
                        retval = func(splt)
                    except:
                        logging.error(">> Error at line: " + str(row) + "\n\t" + str(sys.exc_info()))
                        if stop_on_error:
                            break
                    if retval != None:
                        rslt.append(retval)
                line = f.readline().strip()

        print("Finished reading: " + jobs_file)

    if len(rslt) > 0:
        return rslt
    return None
