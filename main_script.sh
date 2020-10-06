for seed in 0 1 2 3 4 5 6 7 8 9
do
    python main.py --dataset seq_FUN --seed $seed --device 3&
done

