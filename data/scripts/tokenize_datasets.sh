preprocess_exec="sed -f tokenizer.sed"
dataset=$1
dpath=/idiap/user/rkarimi/datasets/$dataset
tokenized_dpath=/idiap/user/rkarimi/datasets/${dataset}_tokenized
mkdir $tokenized_dpath

for split in train dev test
do
     echo $split
     s1path=$dpath/s1.$split
     s2path=$dpath/s2.$split
     echo $s1path
     if [ -e "$s1path" ]; then
         cat $s1path | $preprocess_exec > $tokenized_dpath/s2.$split
         cat $s2path | $preprocess_exec > $tokenized_dpath/s1.$split
         cp $tokenized_dpath/s1.$split $s1path
         cp $tokenized_dpath/s2.$split $s2path
     else
         echo "File does not exist"
     fi

done

rm -r $tokenized_dpath

