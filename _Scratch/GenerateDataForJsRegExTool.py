
from GwData import GwData
from WordTokenizer import tokenize

import nltk.tag.crf

""" Generates a dataset for the html and js based reg ex exploration tool for exploring patterns over the dataset """

def map_punctuation(token):
    token = token.strip()
    if token == ".":
        return ":period"
    elif token == ",":
        return ":comma"
    elif token == "?":
        return ":question-mark"
    elif token == "!":
        return ":exclpoint"
    elif token == ",":
        return ":comma"
    elif token == ":":
        return ":colon"
    elif token == ";":
        return ":semicolon"
    elif token == "-":
        return ":dash"
    elif token == "(":
        return ":lparen"
    elif token == ")":
        return ":rparen"
    else:
        return token

output_dir = "C:\Dump\\"

data = GwData()
tokenized_docs = tokenize(data.documents,min_word_count=0, stem=False, lemmatize=False, remove_stop_words=False, spelling_correct=True)

scodes = sorted(data.sm_codes)
filtered_codes = set()

from collections import defaultdict
sentence2codeset = defaultdict(list)
sentencecount = defaultdict(int)

dupes = set()

with open(output_dir + "Data.txt", "w+") as f_out:
    LINE_DELIM = "$$"

    processed_sentences = set()
    assert len(data.codes_per_document) == len(tokenized_docs)

    for i, (codes, tokens) in enumerate(zip(data.codes_per_document, tokenized_docs)):

        if len(tokens) == 0:
            continue
        if len(codes) == 0:
            print "Missing codes"

        filtered_codes.update(codes)
        mapped = map(map_punctuation, tokens)
        sentence = " ".join(mapped)
        str_codes = ",".join(codes)

        sentence2codeset[sentence].append(list(codes))
        sentencecount[sentence] += 1
        if sentence in processed_sentences:
            dupes.add(sentence)
            print "Sentence at line " + str(i + 1) + " has a duplicate sentence:\n\t" + sentence
        else:
            processed_sentences.add(sentence)

        f_out.write(sentence + "$" + str_codes + LINE_DELIM)

print "\nInconsistently Coded Dupes"
for dupe in dupes:
    code_sets = sentence2codeset[dupe]
    first = code_sets[0]
    if any(map(lambda cs: cs != first, code_sets)):
        print dupe #, sentencecount[dupe]
        for cs in code_sets:
            print str(cs)

# Dump codes
sorted_codes = ["all"] + sorted(filtered_codes)
with open(output_dir + "Codes.txt", "w+") as f_out:
    f_out.write(",".join(sorted_codes))

print "Done"