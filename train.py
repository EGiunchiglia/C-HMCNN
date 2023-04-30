import os
import importlib
os.environ["DATA_FOLDER"] = "./"

import argparse

import torch
import torch.utils.data
import torch.nn as nn

from utils.parser import *
from utils import datasets
import random
import sys

from sklearn.impute import SimpleImputer

from sklearn import preprocessing
from sklearn.model_selection import train_test_split


from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve, roc_auc_score, auc

from models.Transformer import TransformerModel
from models.ConstrainedFFNNModel import ConstrainedFFNNModel, get_constr_out
from models.RNN import LSTMModel
from visualization.viz import draw_loss_acc


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Train neural network')

    # Required  parameters
    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, required=True,
                        help='learning rate')
    parser.add_argument('--dropout', type=float, required=True,
                        help='dropout probability')
    parser.add_argument('--hidden_dim', type=int, required=True,
                        help='size of the hidden layers')
    parser.add_argument('--num_layers', type=int, required=True,
                        help='number of hidden layers')
    parser.add_argument('--weight_decay', type=float, required=True,
                        help='weight decay')
    parser.add_argument('--non_lin', type=str, required=True,
                        help='non linearity function to be used in the hidden layers')

    # Other parameters
    parser.add_argument('--device', type=int, default=0,
                        help='device (default:0)')
    parser.add_argument('--num_epochs', type=int, default=2000,
                        help='Max number of epochs to train (default:2000)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default:0)')
    parser.add_argument('--model', type=str, default='fc',
                        help='model')
    # for transformer only
    parser.add_argument('--num_heads', type=int, default=5, help='number of heads for multi-head self-attention')

    args = parser.parse_args()
    hyperparams = {'num_heads': args.num_heads, 'batch_size':args.batch_size, 'num_layers':args.num_layers, 'dropout':args.dropout, 'non_lin':args.non_lin, 'hidden_dim':args.hidden_dim, 'lr':args.lr, 'weight_decay':args.weight_decay}

    assert('_' in args.dataset)
    assert('FUN' in args.dataset or 'GO' in args.dataset or 'others' in args.dataset)


    # Load train, val and test set
    dataset_name = args.dataset
    data = dataset_name.split('_')[0]
    ontology = dataset_name.split('_')[1]

    # Dictionaries with number of features and number of labels for each dataset
    # note: enrontext input dimension is the corpus size (30522 is corpus used by BERT tokenizer)
    #################### corpus size for seq language models ############################
    corpus_size = {'BERT': 30522}
    ########################################################################
    input_dims = {'amz':512, 'enrontext':512, 'diatoms':371, 'enron':1001,'imclef07a': 80, 'imclef07d': 80,'cellcycle':77, 'church':27, 'derisi':63, 'eisen':79, 'expr':561, 'gasch1':173, 'gasch2':52, 'hom':47034, 'seq':529, 'spo':86}
    output_dims_FUN = {'cellcycle':499, 'church':499, 'derisi':499, 'eisen':461, 'expr':499, 'gasch1':499, 'gasch2':499, 'hom':499, 'seq':499, 'spo':499}
    output_dims_GO = {'cellcycle':4122, 'church':4122, 'derisi':4116, 'eisen':3570, 'expr':4128, 'gasch1':4122, 'gasch2':4128, 'hom':4128, 'seq':4130, 'spo':4116}
    output_dims_others = {'amz':554, 'enrontext':56, 'diatoms':398,'enron':56, 'imclef07a': 96, 'imclef07d': 46, 'reuters':102}
    output_dims = {'FUN':output_dims_FUN, 'GO':output_dims_GO, 'others':output_dims_others}


    # Set seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Pick device
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")

    # Load the datasets
    if ('others' in args.dataset):
        train, test = initialize_other_dataset(dataset_name, datasets)
        train.to_eval, test.to_eval = torch.tensor(train.to_eval, dtype=torch.uint8),  torch.tensor(test.to_eval, dtype=torch.uint8)
        train.X, valX, train.Y, valY = train_test_split(train.X, train.Y, test_size=0.30, random_state=seed)
    else:
        train, val, test = initialize_dataset(dataset_name, datasets)
        train.to_eval, val.to_eval, test.to_eval = torch.tensor(train.to_eval, dtype=torch.uint8), torch.tensor(val.to_eval, dtype=torch.uint8), torch.tensor(test.to_eval, dtype=torch.uint8)

    different_from_0 = torch.tensor(np.array((test.Y.sum(0)!=0), dtype = np.uint8), dtype=torch.uint8)

    # Compute matrix of ancestors R
    # Given n classes, R is an (n x n) matrix where R_ij = 1 if class i is ancestor of class j
    R = np.zeros(train.A.shape)
    np.fill_diagonal(R, 1)
    g = nx.DiGraph(train.A)

    for i in range(len(train.A)):
        descendants = list(nx.descendants(g, i))
        if descendants:
            R[i, descendants] = 1

    R = torch.tensor(R)
    #Transpose to get the ancestors for each node 
    R = R.transpose(1, 0)
    R = R.unsqueeze(0).to(device)


    # Rescale dataset and impute missing data
    if ('others' in args.dataset):
        ############################### for new datasets #################################
        if args.dataset == 'enrontext_others':
            train.X, train.Y = torch.tensor(train.X, dtype=torch.int64).to(device), torch.tensor(train.Y, dtype=torch.int64).to(device)
            valX, valY = torch.tensor(valX, dtype=torch.int64).to(device), torch.tensor(valY, dtype=torch.int64).to(device)
        ##############################################################################
        else:
            scaler = preprocessing.StandardScaler().fit((train.X))
            imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit((train.X))
            train.X, train.Y = torch.tensor(scaler.transform(imp_mean.transform(train.X))).to(device), torch.tensor(train.Y).to(device)
            valX, valY = torch.tensor(scaler.transform(imp_mean.transform(valX))).to(device), torch.tensor(valY).to(device)
    else:
        scaler = preprocessing.StandardScaler().fit(np.concatenate((train.X, val.X)))
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit(np.concatenate((train.X, val.X)))
        val.X, val.Y = torch.tensor(scaler.transform(imp_mean.transform(val.X))).to(device), torch.tensor(val.Y).to(device)
        train.X, train.Y = torch.tensor(scaler.transform(imp_mean.transform(train.X))).to(device), torch.tensor(train.Y).to(device)        

    # Create loaders 
    train_dataset = [(x, y) for (x, y) in zip(train.X, train.Y)]
    if ('others' in args.dataset):
        val_dataset = [(x, y) for (x, y) in zip(valX, valY)]
    else:
        val_dataset = [(x, y) for (x, y) in zip(val.X, val.Y)]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=args.batch_size, 
                                            shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=args.batch_size, 
                                            shuffle=False)


    num_epochs = args.num_epochs
    if 'GO' in dataset_name:
        num_to_skip = 4
    else:
        num_to_skip = 1

    # Create the model
    if 'fc' in args.model:
        model = ConstrainedFFNNModel(input_dims[data], args.hidden_dim, output_dims[ontology][data]+num_to_skip, hyperparams, R)
    elif 'transformer' in args.model:
        ############################# Transformer ###########################################
        input_size = corpus_size['BERT']
        output_size = output_dims[ontology][data]+num_to_skip
        hidden_size = args.hidden_dim
        num_layers = hyperparams['num_layers']
        num_heads = hyperparams['num_heads']
        dropout = hyperparams['dropout']
        model = TransformerModel(input_size, output_size, hidden_size, num_layers, num_heads, dropout, R)
        ########################################################################
    elif 'lstm' in args.model:
        input_size = input_dims[data]
        output_size = output_dims[ontology][data]+num_to_skip
        hidden_size = args.hidden_dim
        num_layers = hyperparams['num_layers']
        model = LSTMModel(input_size, output_size, hidden_size, num_layers, R)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    criterion = nn.BCELoss()

    # Set patience 
    patience, max_patience = 20, 20
    max_score = 0.0

    # Create folder for the dataset (if it does not exist)
    if not os.path.exists('logs/'+str(dataset_name)+'/'):
         os.makedirs('logs/'+str(dataset_name)+'/')

    ############################## Visualization ##################################
    train_loss_list = []
    val_loss_list = []

    train_acc_list = []
    val_acc_list = []

    train_score_list = []
    val_score_list = []
    ################################################################

    for epoch in range(num_epochs):
        total_train = 0.0
        correct_train = 0.0
        model.train()

        train_score = 0

        for i, (x, labels) in enumerate(train_loader):
            # import pdb
            # pdb.set_trace()
            x = x.to(device)
            labels = labels.to(device)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            if 'fc' in args.model or ('lstm' in args.model):
                ########################### FC #############################################
                output = model(x.float())
            elif 'transformer' in args.model:
                ########################### Transformer #############################################
                output = model(x)
                ########################################################################
            constr_output = get_constr_out(output, R)
            train_output = labels * output.double()
            train_output = get_constr_out(train_output, R)
            train_output = (1-labels) * constr_output.double() + labels * train_output

            if 'fc' in args.model or ('lstm' in args.model):
                ########################### FC #############################################
                loss = criterion(train_output[:, train.to_eval], labels[:, train.to_eval].double())
            elif 'transformer' in args.model:
                ########################### Transformer #############################################
                loss = criterion(train_output[:, train.to_eval], labels[:, train.to_eval].double())
            ############ accuracy is thresholded at 0.5 ###############
            predicted = constr_output.data > 0.5
            # Total number of labels
            total_train += labels.size(0) * labels.size(1)
            # Total correct predictions
            correct_train += (predicted == labels.byte()).sum()
  
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()

        model.eval()
        constr_output = constr_output.to('cpu')
        labels = labels.to('cpu')
        train_score = average_precision_score(labels[:, train.to_eval], constr_output.data[:, train.to_eval], average='micro')

        for i, (x,y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)
            if ('fc' in args.model) or ('lstm' in args.model):
                ######################## fc #############################
                constrained_output = model(x.float())
            elif 'transformer' in args.model:
                ######################## transformer #############################
                constrained_output = model(x)

            predicted = constrained_output.data > 0.5
            # Total number of labels
            total = y.size(0) * y.size(1)
            # Total correct predictions
            correct = (predicted == y.byte()).sum()

            # Move output and label back to cpu to be processed by sklearn
            cpu_constrained_output = constrained_output.to('cpu')
            y = y.to('cpu')

            if i == 0:
                constr_val = cpu_constrained_output
                y_val = y
            else:
                constr_val = torch.cat((constr_val, cpu_constrained_output), dim=0)
                y_val = torch.cat((y_val, y), dim=0)

        score = average_precision_score(y_val[:, train.to_eval], constr_val.data[:, train.to_eval], average='micro')

        if score >= max_score:
            patience = max_patience
            max_score = score
        else:
            patience = patience - 1

        ############################## Visualization ##########################
        train_acc_list.append(float(correct_train)/float(total_train))
        val_acc_list.append(float(correct)/float(total))

        train_score_list.append(train_score)
        val_score_list.append(score)
        ################################################################
        floss = open('logs/'+'/'+str(dataset_name)+'/' + args.model + '_measures_batch_size_'+str(args.batch_size)+'_lr_'+str(args.lr)+'_weight_decay_'+str(args.weight_decay)+'_seed_'+str(args.seed)+'_num_layers_'+str(args.num_layers)+'._hidden_dim_'+str(args.hidden_dim)+'_dropout_'+str(args.dropout)+'_'+args.non_lin, 'a')
        floss.write('\nEpoch: {} - Loss: {:.4f}, Accuracy train: {:.5f}, Accuracy: {:.5f}, Precision score: ({:.5f})\n'.format(epoch,
                    loss, float(correct_train)/float(total_train), float(correct)/float(total), score))
        floss.close()

        if patience == 0:
            break
    ############################## Visualization #############################
    draw_loss_acc(args.model, args.dataset, train_acc_list, val_acc_list, 'Accuracy')
    draw_loss_acc(args.model, args.dataset, train_score_list, val_score_list, 'Score')
    ################################################################

if __name__ == "__main__":
    main()
