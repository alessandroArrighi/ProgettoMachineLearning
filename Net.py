import torch
import torch.nn as nn

class Net(nn.Module):
    
    def __init__(self, classes: list[int], vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int) -> None:
        super(Net, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout = 0.5, batch_first = True)
        self.dropOut = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_size, len(classes))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        x = self.embed(x)
        x, _ = self.lstm(x)

        # Rissegnate gli elementi in celle di memoria contigui
        x = x.contiguous().view(-1, self.hidden_size)

        x = self.dropOut(x)
        x = self.linear(x)
        x = self.sigmoid(x)

        # Modifica della forma del tensore
        x = x.view(batch_size, -1)
        # Si prendono gli ultimi elementi, quelli più significativi
        x = x[:, -1]

        return x