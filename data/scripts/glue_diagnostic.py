# This scripts process the SICK-E dataset and
# writes it in the format of SNLI dataset.
import os
import argparse
import pandas as pd
import numpy as np

class GlueDiagnosticDataset(object):
    def __init__(self, testpath, outputdir):
        self.testpath = testpath
        self.outputdir = outputdir
        # Creates the output directory if does not exist.
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
        df = pd.read_csv(datapath, sep='\t')
        labels = df['Label'].values.tolist()
        # Filters all nan labels.
        indices = [i for i, x in enumerate(labels) if x is not np.nan]
        data = {}
        data['s1'] = np.array(df['Premise'].values.tolist())[indices]
        data['s2'] = np.array(df['Hypothesis'].values.tolist())[indices]
        data['labels'] = np.array(df['Label'].values.tolist())[indices]
        assert (len(data['s1']) == len(data['s2']) == len(data['labels']))
        return data

    def process(self):
        test = self.loadFile(os.path.join(self.testpath))
        self.writeData(test['s1'], os.path.join(self.outputdir, 's1.test'))
        self.writeData(test['s2'], os.path.join(self.outputdir, 's2.test'))
        self.writeData(test['labels'], os.path.join(self.outputdir, 'labels.test'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Process Glue Diagnostic test set.")
    parser.add_argument("--testpath", type=str, #default="/idiap/temp/rkarimi/datasets/diagnostic-full.tsv?dl=1", \
       help="Defines the path to GLUE test set")
    parser.add_argument("--outputpath", type=str, #default="/idiap/temp/rkarimi/datasets/GlueDiagnostic/", \
       help="Defines the path to the output folder for the processed dataset")
    params, unknowns = parser.parse_known_args()
    GlueDiagnosticDataset(params.testpath, params.outputpath).process()






