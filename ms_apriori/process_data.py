import re

with open("result.txt") as f:
    for line in f:
         str= re.findall(r"[\w']+", line)
         if(len(str)==2):
             