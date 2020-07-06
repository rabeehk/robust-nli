# This scripts process the Scitail dataset and writes it in the
# format of SNLI dataset.
import os 
import json_lines
import argparse


class SciTailDataset(object):
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
        sentences2 = []
        sentences1 = []
        labels = []
        with open(datapath, 'rb') as f:
            for item in json_lines.reader(f):
                sentences2.append(item['sentence2'])
                sentences1.append(item['sentence1'])
                labels.append(item['gold_label'])

        data = {}
        data['s1'] = sentences1
        data['s2'] = sentences2
        data['labels'] = labels
        return data

    def process(self):
        train = self.loadFile(os.path.join(self.datadir, 'scitail_1.0_train.txt'))
        dev = self.loadFile(os.path.join(self.datadir, 'scitail_1.0_dev.txt'))
        test = self.loadFile(os.path.join(self.datadir, 'scitail_1.0_test.txt'))
        scitail_data = {'train':train, 'dev':dev, 'test':test}
        for name, data in scitail_data.items():
            self.writeData(data['s1'], os.path.join(self.outputdir, 's1.'+name))
            self.writeData(data['s2'], os.path.join(self.outputdir, 's2.'+name))
            self.writeData(data['labels'], os.path.join(self.outputdir, 'labels.'+name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Process SciTail dataset.")
    parser.add_argument("--datadir", type=str,
       help="Defines the path to the nli_format folder of SciTail dataset")
    parser.add_argument("--outputpath", type=str,
       help="Defines the path to the output folder for the processed dataset")
    params, unknowns = parser.parse_known_args()
    SciTailDataset(params.datadir, params.outputpath).process()

