import torch
import torch.nn as nn


# Define the Transformer model architecture
class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, num_heads, dropout, R):
        super(TransformerModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # Embedding layer
        self.embedding = nn.Embedding(input_size, hidden_size)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Classification layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Activation function
        self.activation = nn.Sigmoid()

        # MCM contraint metrics
        self.R = R


    def forward(self, x):
        # Embedding layer
        embedded = self.embedding(x)

        # Transformer layers
        encoded = self.transformer_encoder(embedded)

        # Classification layer
        logits = self.fc(encoded[:, 0, :])
        outputs = self.activation(logits)

        # Add MCM constraints
        if self.training:
            constrained_out = outputs
        else:
            constrained_out = get_constr_out(outputs, self.R)
        return constrained_out


def get_constr_out(x, R):
    """ Given the output of the neural network x returns the output of MCM given the hierarchy constraint expressed in the matrix R """
    c_out = x.double()
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x), R.shape[1], R.shape[1])
    R_batch = R.expand(len(x), R.shape[1], R.shape[1])
    final_out, _ = torch.max(R_batch*c_out.double(), dim=2)
    return final_out
