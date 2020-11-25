# C-HMCNN

Code and data for the paper "[Coherent Hierarchical Multi-label Classification Networks](https://proceedings.neurips.cc//paper/2020/file/6dd4e10e3296fa63738371ec0d5df818-Paper.pdf)". 

## Evaluate C-HMCNN

In order to evaluate the model for a single seed run:
```
  python main.py --dataset <dataset_name> --seed <seed_num> --device <device_num>
```
Example:
```
  python main.py --dataset cellcycle_FUN --seed 0 --device 0
```

**Note:** the parameter passed to "dataset" must end with: '_FUN', '_GO', or '_others'.

If you want to execute the model for 10 seeds you can modify the script ```main_script.sh``` and execute it.

The results will be written in the folder ```results/``` in the file ```<dataset_name>.csv```.

## Hyperparameters search

If you want to execute again the hyperparameters search you can modify the script ```script.sh```according to your necessity and execute it. 


## Architecture

The code was run on a Titan Xp with 12GB memory. A description of the environment used and its dependencies is given in ```c-hmcnn_enc.yml```.

By running the script ```main_script.sh``` we obtain the following results (average over the 10 runs):

| Dataset       | Result |
| ---           | ----   |
| Cellcycle_FUN | 0.255  |
| Derisi_FUN    | 0.195  |
| Eisen_FUN     | 0.306  |
| Expr_FUN      | 0.302  |
| Gasch1_FUN    | 0.286  |
| Gasch2_FUN    | 0.258  |
| Seq_FUN       | 0.292  |
| Spo_FUN       | 0.215  |
| Cellcycle_GO  | 0.413  |
| Derisi_GO     | 0.370  |
| Eisen_GO      | 0.455  |
| Expr_GO       | 0.447  |
| Gasch1_GO     | 0.436  |
| Gasch2_GO     | 0.414  |
| Seq_GO        | 0.446  |
| Spo_GO        | 0.382  |
| Diatoms_others| 0.758  |
| Enron_others  | 0.756  |
| Imclef07a_others | 0.956 |
| Imclef07d_others | 0.927 |



## Reference
```
@inproceedings{giunchiglia2020neurips,
    title     = {Coherent Hierarchical Multi-label Classification Networks},
    author    = {Eleonora Giunchiglia and
               Thomas Lukasiewicz},
    booktitle = {34th Conference on Neural Information Processing Systems (NeurIPS 2020)},
    address = {Vancouver, Canada},
    month = {December},
    year = {2020}
}
```
