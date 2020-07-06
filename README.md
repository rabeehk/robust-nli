# End-to-End Bias Mitigation by Modelling Biases in Corpora
This repo contains the PyTorch implementation of the ACL, 2020 paper
[End-to-End Bias Mitigation by Modelling Biases in Corpora](https://arxiv.org/pdf/1909.06321.pdf).

## Datasets

To get all datasets used in this work, run the following commands.
```
cd data
bash get_datasets.sh 
```
Downloads the [GLOVE (v1)](https://nlp.stanford.edu/projects/glove/)
```
cd data
bash get_glove.sh
```

## Implementations
- The product of experts (PoE), Debiased Focal Loss (DFL), and RuBi loss 
implemnentations are provided in **src/losses.py** 
- The codes for BERT baseline are provided in **src/BERT/** and the scripts 
to reproduce the results are provided in **src/BERT/scripts/** 
- The codes for InferSent baseline are provided in **src/InferSent/** and 
the scripts to reproduce the results are provided in **src/InferSent/scripts**


## Tested envrionment
pytorch-transformers 1.1.0, transformers 2.5.0, pytorch 1.2.0, pytorch-pretrained-bert 0.6.2   


## Datasets
- To download the MNLI Mismatched/Matched development set from ACL 2020 paper 
[End-to-End Bias Mitigation by Modelling Biases in Corpora](https://arxiv.org/pdf/1909.06321.pdf)
use these links [mismatched](https://www.dropbox.com/s/bidxvrd8s2msyan/MNLIMismatchedHardWithHardTest.zip?dl=1), [matched](
https://www.dropbox.com/s/3aktzl4bhmqti9n/MNLIMatchedHardWithHardTest.zip?dl=1)

- By running the get_datasets.sh scripts, the generated files will be downloaded 
under the names of **MNLIMismatchedHardWithHardTest** and **MNLIMatchedHardWithHardTest**.

### Datasets format
Each dataset has three files:
- **s1.test**   each lines shows a premise
- **s2.test**   each line shows a hypothesis
- **labels.test**   each line shows a label.
 

## Bibliography
If you find this repo useful, please cite our paper.

```
@inproceedings{karimi2020endtoend,
  title={End-to-End Bias Mitigation by Modelling Biases in Corpora},
  author={Karimi Mahabadi, Rabeeh and Belinkov, Yonatan and Henderson, James},
  booktitle={Annual Meeting of the Association for Computational Linguistics},
  year={2020}
}
```

## Final words
Hope this repo is useful for your research. For any questions, please create an issue or
email rabeeh.karimi@idiap.ch, and we will get back to you as soon as possible.

