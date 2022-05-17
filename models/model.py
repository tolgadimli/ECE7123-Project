import torch
import torch.nn as nn

class SimpleMLP1(nn.Module):

    def __init__(self, input_size, hid1, hid2, out): 
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hid1),
            nn.ReLU(),
            nn.Linear(hid1, hid1),
            nn.ReLU(),
            nn.Linear(hid1, hid2),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(hid2, hid2),
            nn.ReLU(),
            nn.Linear(hid2, out)
        )

    def forward(self, x):
        out = self.layers(x)
        return out



class SimpleMLP2(nn.Module):

    def __init__(self, input_size, hid1, hid2, out): 
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hid1),
            nn.ReLU(),
            nn.Linear(hid1, hid2),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(hid2, out)
        )

    def forward(self, x):
        out = self.layers(x)
        return out




# sequence length is 1 in our case!
class RecurrentModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device, type = 'RNN'):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.type = type

        if type == 'RNN':
            self.recurrent = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
        elif type == 'LSTM':
            self.recurrent = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        elif type == 'GRU':
            self.recurrent = nn.GRU(input_size, hidden_size, num_layers, batch_first = True)
        else:
            raise ValueError("No such Recurrent model, choose from: RNN | LSTM | GRU")

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_size, 1024*4)
        self.fc2 = nn.Linear(1024*4, num_classes)

        self.device = device

    def forward(self,x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        if self.type == 'LSTM':
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) 
            out, _ = self.recurrent(x, (h0, c0))
        else:
            #print(x.shape)
            out, _ = self.recurrent(x, h0)
        out = out[:, -1, :]

        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out

