for seed in 0 1 2 3 4 5 6 7 8 9
do
    python main.py --dataset imclef07a_others --hidden_dim 1000 --lr 1e-5 --epochs 593 --seed $seed --device 3&
done

