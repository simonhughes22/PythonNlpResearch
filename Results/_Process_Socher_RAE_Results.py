__author__ = 'simon.hughes'

from ExperimentBase import *

fname = "C:\Users\simon.hughes\Dropbox\PhD\Code\PyCharmProjects\NlpResearch\Results\GlobalWarming\_Socher_RAE_Results.csv"
lines = []
with open(fname) as f:
    lines = map(lambda l: l.strip(), f.readlines())

def to_rpfa(line):
    splt = line.split(",")
    #Skip the code
    reals = map(float, splt[1:])
    return rpfa(reals[0], reals[1], reals[2], -1.0, nc = int(reals[3]))

results = map(to_rpfa, lines[1:])
mean_metrics = mean_rpfa(results)
weighted_mean_metrics = weighted_mean_rpfa(results)

print "Mean: " + mean_metrics.to_str(True)
print "Weighted Mean: " + weighted_mean_metrics.to_str(True)