import json_lines
import os
import argparse

def process_nli_hardset(datapth, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Writes data in the file.
    sentences2 = []
    sentences1 = []
    labels = []
    pair_ids = []
    with open(datapth, 'rb') as f:
        for item in json_lines.reader(f):
            sentences2.append(item['sentence2'])
            sentences1.append(item['sentence1'])
            labels.append(item['gold_label'])
            pair_ids.append(item['pairID'])
 
    with open(os.path.join(outdir, 'labels.test'), 'w') as f:
        f.write('\n'.join(labels))

    with open(os.path.join(outdir, 's1.test'), 'w') as f:
        f.write('\n'.join(sentences1))

    with open(os.path.join(outdir, 's2.test'), 'w') as f:
        f.write('\n'.join(sentences2))

    with open(os.path.join(outdir, 'ids.test'), 'w') as f:
        f.write('\n'.join(pair_ids))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Processing Hard NLI dataset.")
    parser.add_argument("--datapath", type=str,  
                                help="Defines the path to the hardset of NLI dataset")
    parser.add_argument("--outputpath", type=str, 
                                help="Defines the path to the output folder for the processed dataset")
    params, unknowns = parser.parse_known_args()
    process_nli_hardset(params.datapath, params.outputpath)
