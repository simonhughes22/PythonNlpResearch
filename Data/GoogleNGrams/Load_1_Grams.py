import DirUtils
import CsvUtils
from collections import defaultdict

cut_off_year = 2000
tally = defaultdict(int)

def process_lines(tokens):
    word = tokens[0].strip().lowe()
    year = int(tokens[1])
    if year > cut_off_year:
        cnt = int(tokens[2])
        tally[word] += cnt
    return None

file_path = "c:\NLPData\google-1-grams"
files = map(lambda f: file_path +"\\" + f,  DirUtils.list_files(file_path))

CsvUtils.file_apply(files, process_lines, skip_header = False, delim = "\t", progress_every = 1000000)
total_count = sum(tally.values())

output_file = "c:\NLPData\\raw_unigram_counts.txt"
with open(output_file, "w+") as fout:
    fout.write(str(total_count) + "\n")

    for k,v in tally.items():
        fout.write(str(k) + "," + str(v) + "\n")

""" Filter words to alpha only, and sort """
fskeys = sorted(filter(lambda k: k.isaplha(), tally.keys()))

output_file2 = "c:\NLPData\\sorted_alpha_unigram_counts.txt"
with open(output_file2, "w+") as fout:
    for key in fskeys:
        val = tally[key]
        fout.write(str(key) + "," + str(val) + "\n")
pass


