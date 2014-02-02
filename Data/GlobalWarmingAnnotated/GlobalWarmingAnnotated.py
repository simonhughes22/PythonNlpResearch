__author__ = 'simon.hughes'

import Settings

s = Settings.Settings()
directory = s.data_directory + "GlobalWarmingAnnotated\\"

fname = directory + "global-warm-deps-sents.tsv"

data = []
with open(fname, "r+") as f:
    lines = f.readlines()

for line in lines:
    splt = line.strip().split("\t")
    if len(splt) < 3:
        break

    essay_field, text, codes, dependencies = splt
    essay, num = essay_field.replace("\"", "").replace("'", "").split(",")
    data.append([essay, int(num), text, codes, dependencies])
