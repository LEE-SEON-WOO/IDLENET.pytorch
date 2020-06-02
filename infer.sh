#!/bin/bash
data="khnp"
backbone="resnet101"
model_name="/model/model-90000"
img_path="/workspace/easy-faster-rcnn.pytorch/data/saewool1"
csv_path="/workspace/easy-faster-rcnn.pytorch/data/saewool1"
xml_path="/workspace/easy-faster-rcnn.pytorch/data/saewool1"
out_path="/workspace/easy-faster-rcnn.pytorch/result/saewool1"

python infer_result_save.py -s=${data} -b=${backbone} -c=${model_name}.pth "${img_path}" "${csv_path}" "${xml_path}" "${out_path}" ${blur_check}

