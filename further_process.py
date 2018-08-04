import csv
import itertools
import re
from collections import defaultdict

columns = defaultdict(list)


with open("movies.csv", "r") as infile :
    reader = csv.reader((infile), skipinitialspace=True)
    writer = csv.writer(open("pre_processed.csv", "w"),quoting = csv.QUOTE_NONE , escapechar=' ')
  
    conversion = set('"[]\' ')
    for row in reader:
        newrow = [''.join(' ' if c in conversion else c for c in entry) for entry in row]
        writer.writerow(newrow)

        