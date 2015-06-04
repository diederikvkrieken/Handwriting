__author__ = 'diederik'

import compareSegments

csc_columns = [2,5,10,20]
wordsXML = [[0,4, "a"], [4,8, "a"], [18,14, "c"], [14,20, "k"]]
comparator = compareSegments.Comparator()
print comparator.compare(csc_columns,wordsXML)
