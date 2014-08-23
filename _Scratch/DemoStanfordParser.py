__author__ = 'simon.hughes'

from stanford_parser import parser

p = parser.Parser()


dependencies = p.parseToStanfordDependencies("Pick up the tire pallet.")
tupleResult = [(rel, gov.text, dep.text) for rel, gov, dep in dependencies.dependencies]

for tuple in tupleResult:
    print tuple

print ""
print "\n".join(map(str,dependencies.dependencies[0]))
