import os
import argparse
import glob

class RecastWhiteDataset(object):
    def __init__(self, datadir, outputdir):
        self.datadir = datadir
        self.outputdir = outputdir

    def writeData(self, lines, fpath):
        """Writes the given data in the format of each
        data in one line.
        """
        with open(fpath, 'w') as f:
            for line in lines:
                print(line, file=f)

    def process_file(self, datapath):
        data = {}
        for type in ['train', 'dev', 'test']:
            data[type] = {}
            data[type]['s1'] = []
            data[type]['s2'] = []
            data[type]['labels'] = []

        dataset_name = (datapath.split("/")[-1].split("_")[0]).upper()
        orig_sent, hyp_sent, data_split, src, label = None, None, None, None, None
        for line in open(datapath):
            if line.startswith("entailed: "):
                label = "entailment"
                if "not-entailed" in line:
                    label = "contradiction"
            elif line.startswith("text: "):
                orig_sent = " ".join(line.split("text: ")[1:]).strip()
            elif line.startswith("hypothesis: "):
                hyp_sent = " ".join(line.split("hypothesis: ")[1:]).strip()
            elif line.startswith("partof: "):
                data_split = line.split("partof: ")[-1].strip()
            elif line.startswith("provenance: "):
                src = line.split("provenance: ")[-1].strip()
            elif not line.strip():
                assert orig_sent != None
                assert hyp_sent != None
                assert data_split != None
                assert src != None
                assert label != None
                data[data_split]['labels'].append(label)
                data[data_split]['s1'].append(orig_sent)
                data[data_split]['s2'].append(hyp_sent)

                orig_sent, hyp_sent, data_split, src, label = None, None, None, None, None

        # Creates the output directory if does not exist.
        if not os.path.exists(os.path.join(self.outputdir, dataset_name)):
            os.makedirs(os.path.join(self.outputdir, dataset_name))

        # Writes the dataset.
        for name, data in data.items():
            self.writeData(data['s1'], os.path.join(self.outputdir, dataset_name, 's1.' + name))
            self.writeData(data['s2'], os.path.join(self.outputdir, dataset_name, 's2.' + name))
            self.writeData(data['labels'], os.path.join(self.outputdir, dataset_name, 'labels.' + name))


    def process(self):
        input_files = glob.glob(os.path.join(self.datadir,"*_data.txt"))
        for file in input_files:
            self.process_file(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Recast White datasets.")
    parser.add_argument("--datadir", type=str, default="/idiap/temp/rkarimi/datasets/rte/", \
       help="Defines the path to the datasets recats by White et al")
    parser.add_argument("--outputpath", type=str, default="/idiap/temp/rkarimi/datasets/", \
       help="Defines the path to the output folder for the processed dataset")
    params, unknowns = parser.parse_known_args()
    RecastWhiteDataset(params.datadir, params.outputpath).process()

