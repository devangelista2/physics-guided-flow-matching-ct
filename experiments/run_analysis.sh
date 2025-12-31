#!/bin/bash

# Point this to your specific experiment name folder
EXP_DIR="outputs/mayo_fm_finetuning" # "mayo_fm_training"

python utils/analyze_results.py --exp_dir "$EXP_DIR" --angles 60