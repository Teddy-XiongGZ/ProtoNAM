for seed in `seq 0 9`
do
    python src/run.py \
        --data housing \
        --model ProtoNAM \
        --lr 2e-3 \
        --max_epoch 1000 \
        --batch_size 2048 \
        --exp_str 'Optimal_'$seed \
        --device cuda:0 \
        --n_layers 4 \
        --h_dim 64 \
        --n_proto 32 \
        --dropout 0.0 \
        --dropout_output 0.0 \
        --weight_decay 1e-3 \
        --seed $seed \
        --tau 16 \
        --output_penalty 1e-2 \
        --p 2
done