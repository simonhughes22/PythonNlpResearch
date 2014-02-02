__author__ = 'simon.hughes'

from Rule import Rule, DisjointRule, PositiveNotNegativeRule
from DictionaryHelper import tally_items

def merge_rules(rules):
    """
        Takes a set of rules, extracts their patterns,
        and attempts to merge
        Returns  - list of merged rules. Every original rule is covered by merged list

        rules   :   [[str], [str]] or [Rule, Rule] or an iterable of either

    """
    if len(rules) == 0:
        return []

    first = rules[0]

    pass

if __name__ == "__main__":
    rules = [
        ['affect', 'temperatur'],
        ['believ'],
        ['carbon', 'global'],
        ['cold', 'phase'],
        ['due'],
        ['earth', 'increas'],
        ['earth', 'warmer'],
        ['fossil', 'global'],
        ['global', 'car'],
        ['global', 'differ', 'past'],
        ['global', 'earth'],
        ['global', 'fossil'],
        ['global', 'increas'],
        ['global', 'pattern'],
        ['global', 'rise'],
        ['global', 'warm'],
        ['greenhous', 'warmer'],
        ['higher', 'gase'],
        ['hotter'],
        ['humankind', 'caus'],
        ['increas', 'global'],
        ['increas', 'temperatur'],
        ['last', '000'],
        ['longer', 'period'],
        ['mani', 'global'],
        ['pattern', 'global', 'differ'],
        ['pattern', 'global', 'temperatur'],
        ['pattern', 'observ'],
        ['pattern', 'reason'],
        ['pattern'],
        ['possibl', 'respons'],
        ['reason', 'global', 'temperatur'],
        ['reason', 'temperatur', 'chang'],
        ['rise', 'temperatur'],
        ['sinc', 'earth'],
        ['temperatur', 'centuri'],
        ['temperatur', 'differ'],
        ['temperatur', 'higher'],
        ['temperatur', 'rise'],
        ['temperatur', 'risen'],
        ['warm', 'last'],
        ['warmer']
    ]

