for fold in 0 1 2 3 4
do
    for seed in 0
    do
        python src/run.py \
            --data mimic3 \
            --model ProtoNAM \
            --lr 5e-2 \
            --max_epoch 200 \
            --batch_size 2048 \
            --exp_str 'Optimal_'$seed'_'$fold \
            --device cuda:0 \
            --n_layers 1 \
            --h_dim 64 \
            --n_proto 1 \
            --dropout 0 \
            --dropout_output 0 \
            --weight_decay 1e-8 \
            --fold $fold \
            --seed $seed \
            --tau 16 \
            --output_penalty 1e-5 \
            --n_layers_pred 1 \
            --batch_norm
    done
done
