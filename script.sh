device=1
for batch_size in 4
do
    for dropout in 0.7
    do
        for hidden_dim in 250 500 1000 1250 1500 1750
        do
            for num_layers in 3
            do
                for lr in 1e-5
                do
                    for weight_decay in 1e-5
                    do
                        for non_lin in relu
                        do
                            for seed in 0
                            do
                                python train.py --dataset "diatoms_others" --seed "$seed" --device "$device" --lr "$lr" --dropout "$dropout" --hidden_dim "$hidden_dim"  --num_layers "$num_layers" --weight_decay "$weight_decay" --non_lin "$non_lin" --batch_size "$batch_size" &
                            done
                        done
                    done
                done
            done
        done
    done
done




