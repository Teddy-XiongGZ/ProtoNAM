for fold in 0 1 2 3 4
do
    python src/run.py \
        --data mimic3 \
        --model ProtoNAM \
        --lr 1e-2 \
        --max_epoch 500 \
        --batch_size 2048 \
        --exp_str 'Optimal_'$fold \
        --device cuda:0 \
        --n_layers 4 \
        --h_dim 64 \
        --n_proto 16 \
        --dropout 0.4 \
        --dropout_output 0.0 \
        --weight_decay 5e-2 \
        --fold $fold \
        --seed 0 \
        --tau 16 \
        --output_penalty 1e-2 \
        --p 2
done
