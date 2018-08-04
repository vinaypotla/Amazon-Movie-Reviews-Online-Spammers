import csv
import itertools
import re

with open('movies.txt','r') as movies_file:
    edited_line = (line.strip() for line in movies_file)
    lines = (((re.sub(r"[,]","",line)).split(':')[-1:]) for line in edited_line if line)
    grouped_lines = zip(*[lines] * 8)
    with open('movies.csv', 'w') as outputfile:
        writer = csv.writer(outputfile)
        writer.writerow(('productId', 'userId', 'profileName', 'helpfulness', 'review_score', 'review_time', 'review_summary', 'text' ))
        writer.writerows(grouped_lines)