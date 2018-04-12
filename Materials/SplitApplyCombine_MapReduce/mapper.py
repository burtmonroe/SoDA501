#!/usr/bin/env python
import sys, string

# Read lines from the "standard input" (sys.stdin).
#    sys.stdin is treated like a file object would be.
#    Unless you change it, this means "the console."
#    This could be from your keyboard or the output of
#    a command like "cat."

for line in sys.stdin:

    # lower-case, remove punctuation, strip extra whitespace
    line = line.lower()
    line = line.translate(str.maketrans('','',string.punctuation))
    line = line.strip()

    # split line into words (on spaces)
    words = line.split()

    # print (to standard output) key value pairs: "word 1"
    for word in words:
        print(word,1,sep="\t")
