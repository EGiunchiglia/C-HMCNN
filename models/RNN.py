import torch
import torch.nn as nn
 
class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, R):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.activation = nn.Sigmoid()
        self.R = R
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out)
        out = self.activation(out)
        # print("out after linear: ", out)
        
        # Add MCM constraints
        if self.training:
            constrained_out = out
        else:
            constrained_out = get_constr_out(out, self.R)
        return constrained_out



def get_constr_out(x, R):
    """ Given the output of the neural network x returns the output of MCM given the hierarchy constraint expressed in the matrix R """
    c_out = x.double()
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x), R.shape[1], R.shape[1])
    R_batch = R.expand(len(x), R.shape[1], R.shape[1])
    final_out, _ = torch.max(R_batch*c_out.double(), dim=2)
    return final_out