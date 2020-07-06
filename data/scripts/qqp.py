import csv
import random
import os
import argparse
import numpy as np
import csv

class QQPDataset(object):
    def __init__(self, datadir, outputdir):
        self.datadir = datadir
        self.outputdir = outputdir
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        self.label_dict = {'1': "entailment", '0': "neutral"}

    def writeData(self, lines, fpath):
        """Writes the given data in the format of each
        data in one line.
        """
        with open(fpath, 'w') as f:
            for line in lines:
                print(line, file=f)

    def loadFile(self, datapath):
        sentences1 = []
        sentences2= []
        labels = []
        with open(datapath) as tsvfile:
           reader = csv.reader(tsvfile, delimiter='\t')
           for row in reader:
              sentences1.append(row[1])
              sentences2.append(row[2])
              labels.append(row[0])

        labels = [self.label_dict[label] for label in labels]
        data = {}
        data['s1'] = sentences1 
        data['s2'] = sentences2 
        data['labels'] = labels
        return data

    def process(self):
        train = self.loadFile(os.path.join(self.datadir, 'train.tsv'))
        dev = self.loadFile(os.path.join(self.datadir, 'dev.tsv'))
        test = self.loadFile(os.path.join(self.datadir, 'test.tsv'))
        mpe_data = {'train':train, 'dev':dev, 'test':test}
        for name, data in mpe_data.items():
            self.writeData(data['s1'], os.path.join(self.outputdir, 's1.'+name))
            self.writeData(data['s2'], os.path.join(self.outputdir, 's2.'+name))
            self.writeData(data['labels'], os.path.join(self.outputdir, 'labels.'+name))

if __name__ == "__main__":
      parser = argparse.ArgumentParser("Processing QQP datasets.")
      parser.add_argument("--datadir", type=str, default="Quora_question_pair_partition", \
         help="Defines the path to the qqp datasets")
      parser.add_argument("--outputpath", type=str, \
         help="Path to the output folder for the processed dataset")
      params, unknowns = parser.parse_known_args()
      QQPDataset(params.datadir, params.outputpath).process()


