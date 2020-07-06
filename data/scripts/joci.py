import csv
import random
import os
import argparse

class JOCIDataset(object):
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

    def convert_label(self, num):
        if num == 1:
            return 'contradiction'
        if num == 5:
            return 'entailment'
        return 'neutral'


    def loadFile(self, split):
        sentences1 = []
        sentences2 = []
        labels = []

        with open(os.path.join(params.datadir, 'joci.csv'), 'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            line_num = -1
            for row in csv_reader:
                line_num += 1
                if line_num == 0:
                    continue
                hyp_src = row[4]

                # TODO: check why this is in the codes.
                '''
                # This is only for processing B.
                if "AGCI" not in hyp_src:
                    continue
                '''
                premise, hypothesis, label = row[0], row[1], self.convert_label(int(row[2]))
                sentences1.append(premise.strip())
                sentences2.append(hypothesis.strip())
                labels.append(label)

        # Now we have all the data in both section A and B.
        combined = list(zip(sentences1, sentences2, labels))
        random.shuffle(combined)
        sentences1[:], sentences2[:], labels[:] = zip(*combined)

        data = {}
        data['s1'] = sentences1
        data['s2'] = sentences2
        data['labels'] = labels
        return data

    def process(self):
        train = self.loadFile('train')
        dev = self.loadFile('dev')
        test = self.loadFile('test')
        joci_data = {'train':train, 'dev':dev, 'test':test}
        for name, data in joci_data.items():
            self.writeData(data['s1'], os.path.join(self.outputdir, 's1.'+name))
            self.writeData(data['s2'], os.path.join(self.outputdir, 's2.'+name))
            self.writeData(data['labels'], os.path.join(self.outputdir, 'labels.'+name))

if __name__ == "__main__":
      parser = argparse.ArgumentParser("Processing Joci  datasets.")
      parser.add_argument("--datadir", type=str,
                          help="Defines the path to the joci datasets")
      parser.add_argument("--outputpath", type=str,
                          help="Defines the path to the output folder for the processed dataset")
      params, unknowns = parser.parse_known_args()
      JOCIDataset(params.datadir, params.outputpath).process()


