#!/bin/bash
set -e

DATA_BASE="../WSS-Interclass-Sep"
MODEL_OUT="MCT_Plus_BraTS_saved_models"
CAM_OUT="MCTformer_Plus_BraTS_results"

######### 1. Train MCTformerPlus on BraTS ##########
echo "Starting BraTS Training (DeiT-Base)..."
python main_brats.py \
    --model deit_base_MCTformerPlus \
    --batch-size 16 \
    --data-set BRATS \
    --data-path ${DATA_BASE} \
    --output_dir ${MODEL_OUT} \
    --finetune https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth

######### 2. Generate Train Seed Maps ##########
echo "Generating Attention Maps for Train Set..."
python main_brats.py \
    --model deit_base_MCTformerPlus \
    --batch-size 16 \
    --data-set BRATSMS \
    --data-path ${DATA_BASE} \
    --split train \
    --gen_attention_maps \
    --resume ${MODEL_OUT}/checkpoint.pth \
    --cam-npy-dir ${CAM_OUT}/cam-npy-train \
    --attention-dir ${CAM_OUT}/cam-png-train


######### 3. Generate Val and Test Seed Maps ##########
echo "Generating Attention Maps for Val Set..."
python main_brats.py --model deit_base_MCTformerPlus --batch-size 16 --data-set BRATSMS --data-path ${DATA_BASE} --split val --gen_attention_maps --resume ${MODEL_OUT}/checkpoint.pth --cam-npy-dir ${CAM_OUT}/cam-npy-val --attention-dir ${CAM_OUT}/cam-png-val

echo "Generating Attention Maps for Test Set..."
python main_brats.py --model deit_base_MCTformerPlus --batch-size 16 --data-set BRATSMS --data-path ${DATA_BASE} --split test --gen_attention_maps --resume ${MODEL_OUT}/checkpoint.pth --cam-npy-dir ${CAM_OUT}/cam-npy-test --attention-dir ${CAM_OUT}/cam-png-test


######### 4. Evaluate ##########
echo "Running curve on VALIDATION set to find the best threshold..."
python eval_brats.py \
    --csv_path ${DATA_BASE}/val.csv \
    --base_dir ${DATA_BASE} \
    --predict_dir ${CAM_OUT}/cam-npy-val \
    --type npy \
    --curve \
    --logfile ${MODEL_OUT}/evallog_brats.txt \
    --comment "DeiT-Base Validation Curve"

echo "Running final, locked evaluation on TEST set..."
python eval_brats.py \
    --csv_path ${DATA_BASE}/test.csv \
    --base_dir ${DATA_BASE} \
    --predict_dir ${CAM_OUT}/cam-npy-test \
    --type npy \
    --t 0.59 \
    --csv_output base_tumor_result_test_cams.csv \
    --logfile ${MODEL_OUT}/evallog_brats.txt \
    --comment "FINAL DeiT-Base Test Set Eval"