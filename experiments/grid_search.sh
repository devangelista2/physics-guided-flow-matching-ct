#!/bin/bash

# --- CONFIGURATION ---
WEIGHTS=".\outputs\mayo_fm_finetuning\weights\latest.pth"
IMG="..\data\Mayo\test\C081\35.png"
ANGLES=60

# --- GRID ---
METHODS=("flowdps" "pnp" "ictm" "flowers")
SCALES=(0.01 0.05 0.1 0.3 0.5 1.0 2.0 5.0)
STEPS=(20 50 100)

# --- EXECUTION ---
echo "Starting Grid Search..."

for method in "${METHODS[@]}"; do
    for step in "${STEPS[@]}"; do
        for scale in "${SCALES[@]}"; do
            
            echo "------------------------------------------------"
            echo "Running: Method=$method | Steps=$step | Scale=$scale"
            echo "------------------------------------------------"
            
            python test.py \
                --config ".\configs\finetune_config.yaml" \
                --weights "$WEIGHTS" \
                --img_path "$IMG" \
                --method "$method" \
                --steps "$step" \
                --scale "$scale" \
                --angles "$ANGLES"
                
        done
    done
done

echo "Grid Search Complete!"