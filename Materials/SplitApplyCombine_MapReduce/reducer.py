#!/usr/bin/env python
import sys

# dictionary, keys will be words, values will be counts
wordcountdict = {}

# read key-value pair "word\tcount\n" from standard input
for line in sys.stdin:

    word, value = line.split('\t', 1)
    value = int(value)

    try:
        wordcountdict[word] = wordcountdict[word]+value
    except:
        wordcountdict[word] = value #new word

for word in wordcountdict.keys():
    print(word, wordcountdict[word], sep="\t")
