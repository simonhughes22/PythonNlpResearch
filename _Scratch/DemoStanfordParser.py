__author__ = 'simon.hughes'

from stanford_parser import parser

txt = "Pick up the tire pallet."
p = parser.Parser()
dependencies = p.parseToStanfordDependencies(txt)
tupleResult= [(rel, (gov.text, gov.start, gov.end), (dep.text, dep.start, dep.end))
                    for rel, gov, dep in dependencies.dependencies]

tokens, tree = p.parse(txt)
kids = tree.children


for tuple in tupleResult:
    print tuple
print ""
print "\n".join(map(str,dependencies.dependencies[0]))

def extract_dependencies(txt):
    dependencies = p.parseToStanfordDependencies(txt)
    return [(rel, (gov.text, gov.start, gov.end), (dep.text, dep.start, dep.end))
                for rel, gov, dep in dependencies.dependencies]

deps = extract_dependencies(txt)

