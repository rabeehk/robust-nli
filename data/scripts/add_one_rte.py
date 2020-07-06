import os
import argparse

class AddOneRTEDataset(object):
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

    def convert_label(self, score, is_test):
        """ Converts not_entailed to contradiction, since we convert
        contradiction and neutral to one label, it does not matter to
        which label we convert the not_entailed labels.
        """
        score = float(score)
        if is_test:
            if score <= 3:
                return "contradiction"
            elif score >= 4:
                return "entailment"
            return

        if score < 3.5:
            return "contradiction"
        return "entailment"

    def loadFile(self, type):
        sentences1 = []
        sentences2 = []
        labels = []

        line_count = -1
        for line in open(os.path.join(self.datadir,"addone-entailment/splits/data.%s" % (type))):
            line_count += 1
            line = line.split("\t")
            assert (len(line) == 7)  # "add one rte %s file has a bad line" % (f))
            label = self.convert_label(line[0], type == "test")
            if not label:
                continue
            labels.append(label)
            hypothesis = line[-1].replace("<b><u>", "").replace("</u></b>", "").strip()
            premise =  line[-2].replace("<b><u>", "").replace("</u></b>", "").strip()
            sentences2.append(hypothesis)
            sentences1.append(premise)

        assert (len(labels) == len(sentences2) == len(sentences1))

        data = {}
        data['s1'] = sentences1
        data['s2'] = sentences2
        data['labels'] = labels
        return data

    def process(self):
        train = self.loadFile('train')
        dev = self.loadFile('dev')
        test = self.loadFile('test')
        add_one_rte_data = {'train':train, 'dev':dev, 'test':test}
        for name, data in add_one_rte_data.items():
            self.writeData(data['s1'], os.path.join(self.outputdir, 's1.'+name))
            self.writeData(data['s2'], os.path.join(self.outputdir, 's2.'+name))
            self.writeData(data['labels'], os.path.join(self.outputdir, 'labels.'+name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Processing ADD-ONE-RTE dataset.")
    parser.add_argument("--datadir", type=str, #default="/idiap/temp/rkarimi/datasets/AN-composition/", \
       help="Defines the path to the nli_format folder of Add-One-RTE dataset")
    parser.add_argument("--outputpath", type=str, #default="/idiap/temp/rkarimi/datasets/AddOneRTE", \
       help="Defines the path to the output folder for the processed dataset")
    params, unknowns = parser.parse_known_args()
    AddOneRTEDataset(params.datadir, params.outputpath).process()

