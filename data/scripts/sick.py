# This scripts process the SICK-E dataset and
# writes it in the format of SNLI dataset.
import os 
import pandas as pd
import argparse


class SickDataset(object):
    def __init__(self, datadir, outputdir):
        self.datadir = datadir
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
        label_dict = {"NEUTRAL": "neutral", "CONTRADICTION":"contradiction",\
           "ENTAILMENT":"entailment"}
        df = pd.read_csv(datapath, sep="\t")
        sentences1 = df['sentence_A'].tolist()
        sentences2 = df['sentence_B'].tolist()
        labels = df['entailment_judgment'].tolist()
        labels = [label_dict[label] for label in labels]
        data = {}
        data['s1'] = sentences1
        data['s2'] = sentences2
        data['labels'] = labels
        return data

    def process(self):
        train = self.loadFile(os.path.join(self.datadir, 'SICK_train.txt'))
        dev = self.loadFile(os.path.join(self.datadir, 'SICK_trial.txt'))
        test = self.loadFile(os.path.join(self.datadir, 'SICK_test_annotated.txt'))
        scitail_data = {'train':train, 'dev':dev, 'test':test}
        for name, data in scitail_data.items():
            self.writeData(data['s1'], os.path.join(self.outputdir, 's1.'+name))
            self.writeData(data['s2'], os.path.join(self.outputdir, 's2.'+name))
            self.writeData(data['labels'], os.path.join(self.outputdir, 'labels.'+name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Process Sick dataset.")
    parser.add_argument("--datadir", type=str, default="/idiap/temp/rkarimi/datasets/sick/", \
       help="Defines the path to the nli_format folder of Sick dataset")
    parser.add_argument("--outputpath", type=str, default="/idiap/temp/rkarimi/datasets/SICK/", \
       help="Defines the path to the output folder for the processed dataset")
    params, unknowns = parser.parse_known_args()
    SickDataset(params.datadir, params.outputpath).process()

