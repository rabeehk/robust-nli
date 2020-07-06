import csv
import sys
from os.path import join
import os
import argparse

def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        return lines


def split_hans_dataset(input_file, outputdir):
    lines = read_tsv(input_file)
    types = ["lexical_overlap", "constituent", "subsequence"]
    lines_types = [[], [], []]
    for i, line in enumerate(lines):
        if i == 0:
            first_line = "\t".join(line)
        if line[8] == types[0]:
            lines_types[0].append("\t".join(line))
        elif line[8] == types[1]:
            lines_types[1].append("\t".join(line))
        elif line[8] == types[2]:
            lines_types[2].append("\t".join(line))


    # Write the splitted files.
    for i, heuristic in enumerate(types):
        datadir = join(outputdir, heuristic)
        if not os.path.exists(datadir):
           os.makedirs(datadir)
        filepath = join(datadir, "heuristics_evaluation_set.txt")
        lines = [first_line]+lines_types[i]
        with open(filepath, "w") as f:
            for line in lines:
               f.write(line+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("process HANS dataset.")
    parser.add_argument("--inputfile", type=str)
    parser.add_argument("--outputdir", type=str)
    args = parser.parse_args()
    split_hans_dataset(args.inputfile, args.outputdir)
