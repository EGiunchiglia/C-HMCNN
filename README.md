# C-HMCNN

Code and data for the paper "Coherent Hierarchical Multi-label Classification Networks" (link here). 

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

If you want to execute the model for multiple seeds you can modify the script ```main_script.sh``` and execute it.

The results will be written in the folder ```results/```in the file ```<dataset_name>.csv```.

## Hyperparameters search

If you want to execute again the hyperparameters search you can modify the script ```script.sh```according to your necessity and execute it. 
