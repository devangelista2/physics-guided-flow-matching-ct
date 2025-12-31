#!/bin/bash

##################################################
# Angles = 60
##################################################
python ./src/run_diffusion_comparison.py --config ./configs/monai_config.yaml --weights ./outputs/mayo_monai_finetuning/weights/latest.pth --img_path ../data/Mayo/test/C081/35.png --img_size 256 --angles 60 --dps_steps 100 --dps_zeta 0.01 --ddrm_steps 50 --cgls_steps 2 --seed 1

##################################################
# Angles = 90
##################################################
python ./src/run_diffusion_comparison.py --config ./configs/monai_config.yaml --weights ./outputs/mayo_monai_finetuning/weights/latest.pth --img_path ../data/Mayo/test/C081/35.png --img_size 256 --angles 90 --dps_steps 150 --dps_zeta 0.015 --ddrm_steps 50 --cgls_steps 2 --seed 1


##################################################
# Angles = 120
##################################################
python ./src/run_diffusion_comparison.py --config ./configs/monai_config.yaml --weights ./outputs/mayo_monai_finetuning/weights/latest.pth --img_path ../data/Mayo/test/C081/35.png --img_size 256 --angles 120 --dps_steps 150 --dps_zeta 0.02 --ddrm_steps 50 --cgls_steps 2 --seed 1