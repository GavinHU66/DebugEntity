#!/usr/bin/env bash

#echo $(date +%Y-%m-%d\ %H:%M:%S)
cfg=${1}
gpu=${2}
fold_num=${3}

for ((i=1;i<=$fold_num;i++));
do
    python train.py example $cfg $gpu $i
done
#echo $(date +%Y-%m-%d\ %H:%M:%S)
