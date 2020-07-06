dataset_dir="datasets"
scripts_dir="scripts"
mkdir -p $dataset_dir
preprocess_exec="sed -f $scripts_dir/tokenizer.sed"

function tokenize {
   fpath=$1
   for split in train dev test
   do
     if [ -e $fpath/s1.$split ]; then
        cut -f1 $fpath/s1.$split | $preprocess_exec | sponge $fpath/s1.$split
     else
        echo "File "$fpath/s1.$split" does not exist."
     fi
     if [ -e $fpath/s2.$split ]; then
        cut -f1 $fpath/s2.$split | $preprocess_exec | sponge $fpath/s2.$split
     else
        echo "File "$fpath/s2.$split" does not exist."
     fi
   done
}

function swap {
tmp=`mktemp`
mv $1 $tmp
mv $2 $1
mv $tmp $2
}

ZIPTOOL="unzip"

#if [ "$OSTYPE" == "darwin"* ]; then
#    # unzip can't handle large files on some MacOS versions
#    ZIPTOOL="7za x"
#fi


# Downloads SNLI
SNLI='https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
mkdir $dataset_dir/SNLI
curl -Lo $dataset_dir/SNLI/snli_1.0.zip $SNLI
$ZIPTOOL $dataset_dir/SNLI/snli_1.0.zip -d $dataset_dir/SNLI
rm $dataset_dir/SNLI/snli_1.0.zip
rm -r $dataset_dir/SNLI/__MACOSX

for split in train dev test
do
    fpath=$dataset_dir/SNLI/$split.snli.txt
    awk '{ if ( $1 != "-" ) { print $0; } }'  $dataset_dir/SNLI/snli_1.0/snli_1.0_$split.txt | cut -f 1,6,7 | sed '1d' > $fpath
    cut -f1 $fpath > $dataset_dir/SNLI/labels.$split
    cut -f2 $fpath | $preprocess_exec > $dataset_dir/SNLI/s1.$split
    cut -f3 $fpath | $preprocess_exec > $dataset_dir/SNLI/s2.$split
    rm $fpath
done
rm -r $dataset_dir/SNLI/snli_1.0
# Downloads the SNLI tsv files.
mkdir -p $dataset_dir/SNLI/original/
wget -O $dataset_dir/SNLI/original/SNLI.zip  "https://www.dropbox.com/s/l3iqo9t1xmt7snc/SNLI.zip"
unzip -j $dataset_dir/SNLI/original/SNLI.zip -d $dataset_dir/SNLI/original/
rm $dataset_dir/SNLI/original/SNLI.zip


# Downloads MultiNLI and tokenize it.
MultiNLI='https://www.nyu.edu/projects/bowman/multinli/multinli_0.9.zip'
# Test set not available yet : we define dev set as the "matched" set and the test set as the "mismatched"
mkdir $dataset_dir/MultiNLI
curl -Lo $dataset_dir/MultiNLI/multinli_0.9.zip $MultiNLI
$ZIPTOOL $dataset_dir/MultiNLI/multinli_0.9.zip -d $dataset_dir/MultiNLI
rm $dataset_dir/MultiNLI/multinli_0.9.zip
rm -r $dataset_dir/MultiNLI/__MACOSX
mv $dataset_dir/MultiNLI/multinli_0.9/multinli_0.9_train.txt  $dataset_dir/MultiNLI/train.multinli.txt
mv $dataset_dir/MultiNLI/multinli_0.9/multinli_0.9_dev_matched.txt   $dataset_dir/MultiNLI/dev.matched.multinli.txt
mv $dataset_dir/MultiNLI/multinli_0.9/multinli_0.9_dev_mismatched.txt $dataset_dir/MultiNLI/dev.mismatched.multinli.txt
rm -r $dataset_dir/MultiNLI/multinli_0.9
for split in train dev.matched dev.mismatched
do
    fpath=$dataset_dir/MultiNLI/$split.multinli.txt
    awk '{ if ( $1 != "-" ) { print $0; } }' $fpath | cut -f 1,6,7 | sed '1d' > $fpath.tok
    cut -f1 $fpath.tok > $dataset_dir/MultiNLI/labels.$split
    cut -f2 $fpath.tok | $preprocess_exec > $dataset_dir/MultiNLI/s1.$split
    cut -f3 $fpath.tok | $preprocess_exec > $dataset_dir/MultiNLI/s2.$split
    rm $fpath $fpath.tok 
done


echo "Creating dataset files for MNLI-matched set, where  the test set is MNLI matched,
and the development set is MNLI mismatched set"
mkdir -p $dataset_dir/MNLIMatched
cp $dataset_dir/MultiNLI/* $dataset_dir/MNLIMatched
mv $dataset_dir/MNLIMatched/labels.dev.matched   $dataset_dir/MNLIMatched/labels.test 
mv $dataset_dir/MNLIMatched/labels.dev.mismatched  $dataset_dir/MNLIMatched/labels.dev 
mv $dataset_dir/MNLIMatched/s1.dev.matched $dataset_dir/MNLIMatched/s1.test 
mv $dataset_dir/MNLIMatched/s1.dev.mismatched $dataset_dir/MNLIMatched/s1.dev 
mv $dataset_dir/MNLIMatched/s2.dev.matched  $dataset_dir/MNLIMatched/s2.test
mv $dataset_dir/MNLIMatched/s2.dev.mismatched $dataset_dir/MNLIMatched/s2.dev



echo "Creating dataset files for MNLI-mismatched set, where the test set is MNLI mismatched,
and the development set is the MNLI Matched set"
mkdir -p $dataset_dir/MNLIMismatched
cp   $dataset_dir/MNLIMatched/* $dataset_dir/MNLIMismatched
swap $dataset_dir/MNLIMismatched/s1.test      $dataset_dir/MNLIMismatched/s1.dev
swap $dataset_dir/MNLIMismatched/s2.test      $dataset_dir/MNLIMismatched/s2.dev
swap $dataset_dir/MNLIMismatched/labels.test  $dataset_dir/MNLIMismatched/labels.dev


echo "Processing GLUE diagnostic test set"
wget --directory-prefix=$dataset_dir  https://www.dropbox.com/s/ju7d95ifb072q9f/diagnostic-full.tsv?dl=1
python $scripts_dir/glue_diagnostic.py --testpath $dataset_dir/diagnostic-full.tsv?dl=1  --outputpath $dataset_dir/GLUEDiagnostic
tokenize $dataset_dir/GLUEDiagnostic
rm $dataset_dir/diagnostic-full.tsv?dl=1
# dev set is MNLI matched and train set is MNLI train set.
for filename in s1.train s2.train labels.train s1.dev s2.dev labels.dev
do
    cp $dataset_dir/MNLIMismatched/$filename $dataset_dir/GLUEDiagnostic/
done

echo "Downloading and Processing SciTail"
wget --directory-prefix=$dataset_dir http://data.allenai.org.s3.amazonaws.com/downloads/SciTailV1.zip
unzip $dataset_dir/SciTailV1.zip -d $dataset_dir
rm $dataset_dir/SciTailV1.zip
rm -r $dataset_dir/__MACOSX/
python $scripts_dir/scitail.py --datadir $dataset_dir/SciTailV1/snli_format/ --outputpath $dataset_dir/SciTail
tokenize $dataset_dir/SciTail/
rm -r $dataset_dir/SciTailV1/


echo "Downloading and Processing add-1 RTE"
wget --directory-prefix=$dataset_dir http://www.seas.upenn.edu/~nlp/resources/AN-composition.tgz
tar -zxvf $dataset_dir/AN-composition.tgz -C $dataset_dir 
rm $dataset_dir/AN-composition.tgz 
python $scripts_dir/add_one_rte.py --datadir $dataset_dir/AN-composition/ --outputpath $dataset_dir/AddOneRTE
tokenize $dataset_dir/AddOneRTE
rm -r $dataset_dir/AN-composition/
rm -r $dataset_dir/._AN-composition


echo "Downloading and Processing FN+, DRP, and SPR"
mkdir $dataset_dir/rte
wget --directory-prefix=$dataset_dir/rte  https://github.com/decompositional-semantics-initiative/DNC/raw/master/inference_is_everything.zip
unzip $dataset_dir/rte/inference_is_everything.zip -d $dataset_dir/rte
rm $dataset_dir/rte/inference_is_everything.zip
python $scripts_dir/recast_white.py --datadir $dataset_dir/rte --outputpath $dataset_dir
for dataset_name in DPR  FNPLUS SPRL
do
    tokenize $dataset_dir/$dataset_name
done
rm -r $dataset_dir/rte


echo "Downloading and Processing JOCI dataset"
git clone https://github.com/sheng-z/JOCI
unzip JOCI/data/joci.csv.zip -d $dataset_dir/joci
mkdir -p $dataset_dir/joci
rm -rf JOCI
python $scripts_dir/joci.py --datadir $dataset_dir/joci --outputpath $dataset_dir/JOCI
tokenize $dataset_dir/JOCI
rm -r $dataset_dir/joci


echo "Downloading and Processing SNLI-hard set"
mkdir $dataset_dir/snli_hard
wget --directory-prefix=$dataset_dir/snli_hard  https://nlp.stanford.edu/projects/snli/snli_1.0_test_hard.jsonl
python $scripts_dir/nli_hardset.py --datapath $dataset_dir/snli_hard/snli_1.0_test_hard.jsonl --outputpath $dataset_dir/SNLIHard
tokenize $dataset_dir/SNLIHard
rm -r $dataset_dir/snli_hard
# Manually copy paste train and dev from SNLI dataset.
for filename in s1.train s2.train labels.train s1.dev s2.dev labels.dev
do
    cp $dataset_dir/SNLI/$filename $dataset_dir/SNLIHard
done


echo "Downloading and processing SICK dataset."
SICK='http://alt.qcri.org/semeval2014/task1/data/uploads'
mkdir $dataset_dir/sick

for split in train trial test_annotated
do
    urlname=$SICK/sick_$split.zip
    curl -Lo $dataset_dir/sick/sick_$split.zip $urlname
    unzip $dataset_dir/sick/sick_$split.zip -d $dataset_dir/sick/
    rm $dataset_dir/sick/readme.txt
    rm $dataset_dir/sick/sick_$split.zip
done
python $scripts_dir/sick.py --datadir $dataset_dir/sick/ --outputpath $dataset_dir/SICK
rm -r $dataset_dir/sick
tokenize $dataset_dir/SICK


echo "Downloading MPE"
mkdir $dataset_dir/mpe
curl https://raw.githubusercontent.com/aylai/MultiPremiseEntailment/master/data/MPE/mpe_train.txt -o  $dataset_dir/mpe/mpe_train.txt
curl https://raw.githubusercontent.com/aylai/MultiPremiseEntailment/master/data/MPE/mpe_dev.txt -o    $dataset_dir/mpe/mpe_dev.txt
curl https://raw.githubusercontent.com/aylai/MultiPremiseEntailment/master/data/MPE/mpe_test.txt -o   $dataset_dir/mpe/mpe_test.txt
python $scripts_dir/mpe.py --datadir $dataset_dir/mpe/ --outputpath $dataset_dir/MPE
rm -r $dataset_dir/mpe
tokenize $dataset_dir/MPE


# We downloaded the data from here.
# https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view 
# https://github.com/zhiguowang/BiMPM
mkdir -p $dataset_dir/QQP/original
wget -O $dataset_dir/QQP/QQP.zip  "https://www.dropbox.com/s/f07ph9epx3jdbgn/Quora_question_pair_partition.zip"
unzip -j $dataset_dir/QQP/QQP.zip -d $dataset_dir/QQP/original
python $scripts_dir/qqp.py --outputpath $dataset_dir/QQP --datadir $dataset_dir/QQP/original
rm -r $dataset_dir/QQP/original
rm  $dataset_dir/QQP/QQP.zip
tokenize $dataset_dir/QQP


# Process MNLI Mismatched Hardset.
mkdir $dataset_dir/MNLIMismatchedHard
python $scripts_dir/nli_hardset.py --datapath $scripts_dir/mnli_raw/multinli_0.9_test_mismatched_unlabeled_hard.jsonl --outputpath $dataset_dir/MNLIMismatchedHard
tokenize $dataset_dir/MNLIMismatchedHard
# Manually copy paste train and dev from MNLI dataset.
# dev set needs to be the MisMatched one.
for filename in s1.train s2.train labels.train s1.dev s2.dev labels.dev
do
    cp $dataset_dir/MNLIMatched/$filename $dataset_dir/MNLIMismatchedHard
done

# Process MNLI Matched Hardset.
mkdir $dataset_dir/MNLIMatchedHard
python $scripts_dir/nli_hardset.py --datapath $scripts_dir/mnli_raw/multinli_0.9_test_matched_unlabeled_hard.jsonl --outputpath $dataset_dir/MNLIMatchedHard
tokenize $dataset_dir/MNLIMatchedHard
# Manually copy paste train and dev from MNLI dataset.
# dev set needs to be the Matched one.
for filename in s1.train s2.train labels.train s1.dev s2.dev labels.dev
do
    cp $dataset_dir/MNLIMismatched/$filename $dataset_dir/MNLIMatchedHard
done


# Process the MNLI Mismatched with the original test set.
mkdir $dataset_dir/MNLITrueMismatched
python $scripts_dir/nli_hardset.py --datapath $scripts_dir/mnli_raw/multinli_0.9_test_mismatched_unlabeled.jsonl --outputpath $dataset_dir/MNLITrueMismatched
tokenize $dataset_dir/MNLITrueMismatched
# Manually copy paste train and dev from MNLI dataset.
# dev set needs to be the MisMatched one.
for filename in s1.train s2.train labels.train s1.dev s2.dev labels.dev
do
    cp $dataset_dir/MNLIMatched/$filename $dataset_dir/MNLITrueMismatched
done


# Process MNLI Matched with the original test set.
mkdir $dataset_dir/MNLITrueMatched
python $scripts_dir/nli_hardset.py --datapath $scripts_dir/mnli_raw/multinli_0.9_test_matched_unlabeled.jsonl --outputpath $dataset_dir/MNLITrueMatched
tokenize $dataset_dir/MNLITrueMatched
# Manually copy paste train and dev from MNLI dataset.
# dev set needs to be the Matched one.
for filename in s1.train s2.train labels.train s1.dev s2.dev labels.dev
do
    cp $dataset_dir/MNLIMismatched/$filename $dataset_dir/MNLITrueMatched
done


# Process the HANS dataset.
mkdir $dataset_dir/HANS
wget -O  $dataset_dir/HANS/heuristics_evaluation_set.txt \
    "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt"
python $scripts_dir/hans.py --inputfile $dataset_dir/HANS/heuristics_evaluation_set.txt   --outputdir $dataset_dir/HANS


# Process the FEVER dataset.
mkdir $dataset_dir/FEVER
# downloads the training set
wget -O  $dataset_dir/FEVER/nli.train.jsonl\
   "https://www.dropbox.com/s/v1a0depfg7jp90f/fever.train.jsonl"
wget -O  $dataset_dir/FEVER/nli.dev.jsonl\
   "https://www.dropbox.com/s/bdwf46sa2gcuf6j/fever.dev.jsonl"

# downloads the test set
mkdir $dataset_dir/FEVER-symmetric-generated
wget -O $dataset_dir/FEVER-symmetric-generated/nli.dev.jsonl \
  "https://raw.githubusercontent.com/TalSchuster/FeverSymmetric/master/symmetric_v0.1/fever_symmetric_generated.jsonl"


# Downloads the MNLIMatchedHard devlopment set.
mkdir  $dataset_dir/MNLIMatchedHardWithHardTest
wget -O $dataset_dir/MNLIMatchedHardWithHardTest/MNLIMatchedHardWithHardTest.zip  "https://www.dropbox.com/s/3aktzl4bhmqti9n/MNLIMatchedHardWithHardTest.zip"
unzip -j $dataset_dir/MNLIMatchedHardWithHardTest/MNLIMatchedHardWithHardTest.zip -d $dataset_dir/MNLIMatchedHardWithHardTest
rm $dataset_dir/MNLIMatchedHardWithHardTest/MNLIMatchedHardWithHardTest.zip
# we copy the train/dev set from the MNLIMatched dataset.
for filename in s1.train s2.train labels.train s1.dev s2.dev labels.dev
do
    cp $dataset_dir/MNLITrueMatched/$filename $dataset_dir/MNLIMatchedHardWithHardTest
done


# Downloads the MNLIMismatchedHard development set.
mkdir -p $dataset_dir/MNLIMismatchedHardWithHardTest
wget -O $dataset_dir/MNLIMismatchedHardWithHardTest/MNLIMismatchedHardWithHardTest.zip  "https://www.dropbox.com/s/bidxvrd8s2msyan/MNLIMismatchedHardWithHardTest.zip"
unzip -j $dataset_dir/MNLIMismatchedHardWithHardTest/MNLIMismatchedHardWithHardTest.zip -d $dataset_dir/MNLIMismatchedHardWithHardTest
rm $dataset_dir/MNLIMismatchedHardWithHardTest/MNLIMismatchedHardWithHardTest.zip 
# we copy the train/dev set from the MNLIMismatched dataset.
for filename in s1.train s2.train labels.train s1.dev s2.dev labels.dev
do
    cp $dataset_dir/MNLITrueMismatched/$filename $dataset_dir/MNLIMismatchedHardWithHardTest
done


# Downloads the not tokenized version of MNLI dataset.
python $scripts_dir/download_glue.py  --tasks MNLI --data_dir $dataset_dir






