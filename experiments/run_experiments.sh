#!/bin/bash

##################################################
# Angles = 60
##################################################
# FlowDPS 
python ./src/test.py --config ./configs/finetune_config.yaml --weights ./outputs/mayo_fm_finetuning/weights/latest.pth --img_path ../data/Mayo/test/C081/35.png --method flowdps --steps 500 --scale 5.0 --angles 60
# PnP
python ./src/test.py --config ./configs/finetune_config.yaml --weights ./outputs/mayo_fm_finetuning/weights/latest.pth --img_path ../data/Mayo/test/C081/35.png --method pnp --steps 200 --scale 0.4 --angles 60
# FLOWERS
python ./src/test.py --config ./configs/finetune_config.yaml --weights ./outputs/mayo_fm_finetuning/weights/latest.pth --img_path ../data/Mayo/test/C081/35.png --method flowers --steps 300 --scale 50.0 --angles 60
# ICTM
python ./src/test.py --config ./configs/finetune_config.yaml --weights ./outputs/mayo_fm_finetuning/weights/latest.pth --img_path ../data/Mayo/test/C081/35.png --method ictm --steps 100 --scale 0.9 --angles 60

##################################################
# Angles = 90
##################################################
# FlowDPS 
python ./src/test.py --config ./configs/finetune_config.yaml --weights ./outputs/mayo_fm_finetuning/weights/latest.pth --img_path ../data/Mayo/test/C081/35.png --method flowdps --steps 500 --scale 5.0 --angles 90
# PnP
python ./src/test.py --config ./configs/finetune_config.yaml --weights ./outputs/mayo_fm_finetuning/weights/latest.pth --img_path ../data/Mayo/test/C081/35.png --method pnp --steps 200 --scale 0.4 --angles 90 
# FLOWERS
python ./src/test.py --config ./configs/finetune_config.yaml --weights ./outputs/mayo_fm_finetuning/weights/latest.pth --img_path ../data/Mayo/test/C081/35.png --method flowers --steps 300 --scale 50.0 --angles 90
# ICTM
python ./src/test.py --config ./configs/finetune_config.yaml --weights ./outputs/mayo_fm_finetuning/weights/latest.pth --img_path ../data/Mayo/test/C081/35.png --method ictm --steps 100 --scale 0.9 --angles 90

##################################################
# Angles =120
##################################################
# FlowDPS 
python ./src/test.py --config ./configs/finetune_config.yaml --weights ./outputs/mayo_fm_finetuning/weights/latest.pth --img_path ../data/Mayo/test/C081/35.png --method flowdps --steps 500 --scale 1.0 --angles 120
# PnP
python ./src/test.py --config ./configs/finetune_config.yaml --weights ./outputs/mayo_fm_finetuning/weights/latest.pth --img_path ../data/Mayo/test/C081/35.png --method pnp --steps 200 --scale 0.3 --angles 120 
# FLOWERS
python ./src/test.py --config ./configs/finetune_config.yaml --weights ./outputs/mayo_fm_finetuning/weights/latest.pth --img_path ../data/Mayo/test/C081/35.png --method flowers --steps 300 --scale 25.0 --angles 120
# ICTM
python ./src/test.py --config ./configs/finetune_config.yaml --weights ./outputs/mayo_fm_finetuning/weights/latest.pth --img_path ../data/Mayo/test/C081/35.png --method ictm --steps 100 --scale 0.8 --angles 120