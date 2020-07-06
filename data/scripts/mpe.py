import csv
import random
import os
import argparse
import pandas as pd
import numpy as np

class MPEDataset(object):
    def __init__(self, datadir, outputdir):
        self.datadir = datadir
        self.outputdir = outputdir
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

    def writeData(self, lines, fpath):
        """Writes the given data in the format of each
        data in one line.
        """
        with open(fpath, 'w') as f:
            for line in lines:
                print(line, file=f)

    def loadFile(self, datapath):
        df = pd.read_csv(datapath, sep="\t")
        premise1 = df['premise1'].tolist()
        premise2 = df['premise2'].tolist()
        premise3 = df['premise3'].tolist()
        premise4 = df['premise4'].tolist()

        premise1 = [s.split('/')[1] for s in premise1]
        premise2 = [s.split('/')[1] for s in premise2]
        premise3 = [s.split('/')[1] for s in premise3]
        premise4 = [s.split('/')[1] for s in premise4]

        sentences1 = [" ".join([s1, s2, s3, s4]) for s1, s2, s3, s4 in zip(premise1, premise2, premise3, premise4)]
        sentences2 = df['hypothesis'].tolist()
        labels = df['gold_label'].tolist()

        indices = [i for i, x in enumerate(labels) if x is not np.nan]
        data = {}
        data['s1'] = np.array(sentences1)[indices]
        data['s2'] = np.array(sentences2)[indices]
        data['labels'] = np.array(labels)[indices]
        return data

    def process(self):
        train = self.loadFile(os.path.join(self.datadir, 'mpe_train.txt'))
        dev = self.loadFile(os.path.join(self.datadir, 'mpe_dev.txt'))
        test = self.loadFile(os.path.join(self.datadir, 'mpe_test.txt'))
        mpe_data = {'train':train, 'dev':dev, 'test':test}
        for name, data in mpe_data.items():
            self.writeData(data['s1'], os.path.join(self.outputdir, 's1.'+name))
            self.writeData(data['s2'], os.path.join(self.outputdir, 's2.'+name))
            self.writeData(data['labels'], os.path.join(self.outputdir, 'labels.'+name))

if __name__ == "__main__":
      parser = argparse.ArgumentParser("Processing MPE datasets.")
      parser.add_argument("--datadir", type=str, default="/idiap/temp/rkarimi/datasets/mpe/", \
                          help="Defines the path to the mpe datasets")
      parser.add_argument("--outputpath", type=str, default="/idiap/temp/rkarimi/datasets/MPE", \
                          help="Defines the path to the output folder for the processed dataset")
      params, unknowns = parser.parse_known_args()
      MPEDataset(params.datadir, params.outputpath).process()


