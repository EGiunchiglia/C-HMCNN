import matplotlib.pyplot as plt
import numpy as np


# ---------------
# Visual The Loss
# ---------------
def draw_loss_acc(model_name, dataset_name, train_list, validation_list, mode='Loss'):
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    # 设置间隔
    data_len = len(train_list)
    x_ticks = np.arange(1, data_len+1)
    x_ticks_labels = np.arange(1, data_len+1) if data_len <= 30 else np.arange(1, data_len+1, data_len//20)
    if mode == 'Loss':
        plt.xticks(x_ticks_labels)
        plt.plot(x_ticks, train_list, label='Training Loss')
        plt.plot(x_ticks, validation_list, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('visualization/fig/{}_{}_epoch_loss.jpg'.format(model_name, dataset_name))
    elif mode == 'Accuracy':
        plt.xticks(x_ticks_labels)
        plt.plot(x_ticks, train_list, label='Training Accuracy')
        plt.plot(x_ticks, validation_list, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('visualization/fig/{}_{}_epoch_accuracy.jpg'.format(model_name, dataset_name))
    elif mode == 'Score':
        plt.xticks(x_ticks_labels)
        plt.plot(x_ticks, train_list, label='Training Score')
        plt.plot(x_ticks, validation_list, label='Validation Score')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig('visualization/fig/{}_{}_epoch_score.jpg'.format(model_name, dataset_name))
