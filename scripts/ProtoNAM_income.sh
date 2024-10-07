for fold in 0 1 2 3 4
do
    for seed in 0
    do
        python src/run.py \
            --data income \
            --model ProtoNAM \
            --lr 1e-3 \
            --max_epoch 1000 \
            --batch_size 2048 \
            --exp_str 'Optimal_'$seed'_'$fold \
            --device cuda:0 \
            --n_layers 4 \
            --h_dim 64 \
            --n_proto 32 \
            --dropout 0.0 \
            --dropout_output 0.0 \
            --weight_decay 1e-2 \
            --fold $fold \
            --seed $seed \
            --tau 16 \
            --output_penalty 1e-2
    done
done
