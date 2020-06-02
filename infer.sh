#!/bin/bash
data="khnp"
backbone="resnet101"
model_name="/model/model-90000"
blur_check=True
data_path="/data/saewool1"

python infer_result_save.py -s=${data} -b=${backbone} -c=${model_name}.pth $data_path ${blur_check}
