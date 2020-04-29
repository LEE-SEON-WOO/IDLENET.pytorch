#!/bin/bash
data="khnp"
backbone="resnet101"
model_name="/model-90000"
f_name="518-PP04"
blur_check=True

python infer_result_save.py -s=${data} -b=${backbone} -c=${model_name}.pth /data/saewool1/color/${f_name}.jpg /result/${f_name}.jpg /data/saewool1/csv/${f_name}.csv /data/saewool1/annotation/${f_name}.xml /result/${f_name}.json ${blur_check}
