for fold in 0 1 2 3 4
do
    python src/run.py \
        --data mimic2 \
        --model ProtoNAM \
        --lr 4e-3 \
        --max_epoch 1000 \
        --batch_size 2048 \
        --exp_str 'Optimal_'$fold \
        --device cuda \
        --n_layers 4 \
        --h_dim 64 \
        --n_proto 64 \
        --dropout 0.5 \
        --dropout_output 0.0 \
        --weight_decay 5e-3 \
        --fold $fold \
        --seed 0 \
        --tau 16 \
        --output_penalty 5e-2 \
        --p 2
done