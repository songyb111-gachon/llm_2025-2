#!/bin/bash

# Model Extraction Attack - Linux/Mac Shell Script
# Run extraction experiments with different data sizes

echo "========================================"
echo "Model Extraction Attack Experiments"
echo "========================================"
echo ""

# Experiment 1: 1000 training samples
echo "[1/2] Running experiment with 1000 training samples..."
python extraction_attack.py \
    --victim_model gpt2 \
    --adversary_model distilgpt2 \
    --dataset wikitext-2-raw-v1 \
    --train_samples 1000 \
    --test_samples 500 \
    --batch_size 32 \
    --epochs 5 \
    --learning_rate 5e-5 \
    --temperature 2.0 \
    --alpha 0.7 \
    --output_dir extraction_results \
    --save_model

echo ""
echo "----------------------------------------"
echo ""

# Experiment 2: 10000 training samples
echo "[2/2] Running experiment with 10000 training samples..."
python extraction_attack.py \
    --victim_model gpt2 \
    --adversary_model distilgpt2 \
    --dataset wikitext-2-raw-v1 \
    --train_samples 10000 \
    --test_samples 500 \
    --batch_size 32 \
    --epochs 5 \
    --learning_rate 5e-5 \
    --temperature 2.0 \
    --alpha 0.7 \
    --output_dir extraction_results \
    --save_model

echo ""
echo "========================================"
echo "All experiments completed!"
echo "========================================"
echo ""
echo "Results saved in: extraction_results/"
echo ""


