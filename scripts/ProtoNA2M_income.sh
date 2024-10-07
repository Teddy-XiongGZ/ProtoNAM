for fold in 0 1 2 3 4
do
    python src/run.py \
        --data income \
        --model ProtoNAM \
        --lr 2e-4 \
        --max_epoch 1000 \
        --batch_size 2048 \
        --exp_str 'Optimal_'$fold \
        --device cuda:0 \
        --n_layers 4 \
        --h_dim 64 \
        --n_proto 32 \
        --dropout 0.0 \
        --dropout_output 0.0 \
        --weight_decay 8e-3 \
        --fold $fold \
        --seed 0 \
        --tau 16 \
        --output_penalty 1e-3 \
        --p 2
done