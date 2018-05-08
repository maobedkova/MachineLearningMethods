#!/usr/bin/env python3
# usage: ./check-classification-results.py < CLASSIFICATION_RESULTS.txt

import sys
import re


waserror=False

def eprint(lineno, *args, **kwargs):
  # report error on some input line
  sys.stderr.write(str(lineno)+":")
  print(*args, file=sys.stderr, **kwargs)
  global waserror
  waserror = True

valid_datasets_list="""car-evaluation cinlp-twitter connect-4-interpreted
  connect-4-raw credit-card-fraud motion-capture-hand mushrooms
  music-genre-classification poker poker-with-extra-features spectf-heart
  wine-quality pamap-easy""".split()
valid_datasets_dict=dict([(k,1) for k in valid_datasets_list])

floatre = re.compile("^[0-9.]*[0-9]+$")
comment_feats_re_str = "ORIGFEATS|ONEHOT"
comment_feats_re = re.compile(comment_feats_re_str)

authorname=None
for lineno, line in enumerate(sys.stdin, 1):
  line = line.rstrip("\n\r") # chomp
  if line.lstrip() == "": continue # empty lines OK
  if line.lstrip()[0] == "#": continue # comments OK
  cols=[col.strip() for col in line.split("\t")] # get stripped cols
  if len(cols) != 4 and len(cols) != 5:
    eprint(lineno, "Expected 4 or 5 columns, got", len(cols))
  if not valid_datasets_dict.get(cols[0]):
    eprint(lineno, "Unknown dataset:", cols[0])

  if cols[1] == "":
    eprint(lineno, "Empty method name.")

  has_result = False
  if cols[2] != "":
    try:
      f = float(cols[2])
      if f < 0 or f > 100:
        eprint(lineno, "Bad range, should be between 0 and 100:", f)
      has_result = True
    except:
      eprint(lineno, "Not a float:", cols[2])

  if authorname:
    if cols[3] != authorname:
      eprint(lineno,
        "Unexpected author name '"+cols[3]+"; expected: ", authorname)
  else:
    authorname = cols[3]

  if has_result:
    # must also mention the ORIGFEATS or ONEHOT
    if not comment_feats_re.match(cols[4]):
      eprint(lineno, "Comment must mention", comment_feats_re)

if waserror:
  sys.exit(1)

