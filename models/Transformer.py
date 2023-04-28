import torch
import torch.nn as nn


# Define the Transformer model architecture
class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, num_heads, dropout):
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

    def forward(self, x):
        # Embedding layer
        embedded = self.embedding(x)

        # Transformer layers
        encoded = self.transformer_encoder(embedded)

        # Classification layer
        logits = self.fc(encoded[:, 0, :])
        outputs = self.activation(logits)
        return outputs
