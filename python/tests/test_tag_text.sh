#!/usr/bin/env bash
CONLL_FILE=test_data/eng.testb.small.conll
TEXT_FILE=test_data/eng.testb.small.txt
TAGGER_DRIVER=../../api-examples/tag-text.py
MODEL_LINK=https://www.dropbox.com/s/ftv4amoq7dzxnxo/conll-6419.zip?dl=1
MODEL_FILE=conll-6419.zip

ERROR_COLOR='\033[0;31m'
MSG_COLOR='\033[0;32m'
END='\033[0m'

if [ ! -f $MODEL_FILE ];
    then
         wget $MODEL_LINK -O $MODEL_FILE
fi

function check_diff {
   DIFF=$(diff $1 $2)
   if [ "$DIFF" != "" ]
       then
           printf "${ERROR_COLOR}Tagger outputs $1 and $2 dont match, test failed \n${END}"
           exit 1
   fi
}   

python ${TAGGER_DRIVER} --model ${MODEL_FILE} --text ${TEXT_FILE} > tmp1
python ${TAGGER_DRIVER} --model ${MODEL_FILE} --text ${CONLL_FILE} --conll true > tmp2
check_diff tmp1 tmp2
python ${TAGGER_DRIVER} --model ${MODEL_FILE} --text ${CONLL_FILE} --feature text --conll true > tmp3
check_diff tmp2 tmp3
python ${TAGGER_DRIVER} --model ${MODEL_FILE} --text ${CONLL_FILE} --feature text:0 --conll true > tmp4
check_diff tmp3 tmp4

printf "${MSG_COLOR}Tagger outputs match, test passed\n${END}"
rm tmp1
rm tmp2
rm tmp3
rm tmp4
