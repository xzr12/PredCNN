#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
cd ..
nohup python3 -u train.py \
    --is_training True \
    --dataset_name mnist \
    --train_data_paths data/moving-mnist-example/moving-mnist-train.npz \
    --valid_data_paths data/moving-mnist-example/moving-mnist-valid.npz \
    --save_dir checkpoints/mnist_predcnn \
    --gen_frm_dir results/mnist_predcnn \
    --model_name predcnn \
    --input_length 10 \
    --seq_length 11 \
    --filter_size 3 \
    --patch_size 4 \
    --num_hidden 64 \
    --encoder_length 4 \
    --decoder_length 6 \
    --lr 0.0003 \
    --batch_size 32 \
    --max_iterations 100000 \
    --display_interval 100 \
    --test_interval 5000 \
    --snapshot_interval 5000 > logs/predcnn_train.log &