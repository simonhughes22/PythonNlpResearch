__author__ = 'simon.hughes'

from stanford_parser import parser

parser = parser.Parser()

dependencies = parser.parseToStanfordDependencies("Pick up the tire pallet.")
tupleResult = [(rel, gov.text, dep.text) for rel, gov, dep in dependencies.dependencies]

for tuple in tupleResult:
    print tuple

#assertEqual(tupleResult, [('prt', 'Pick', 'up'),
#                               ('det', 'pallet', 'the'),
#                               ('nn', 'pallet', 'tire'),
 #                              ('dobj', 'Pick', 'pallet')