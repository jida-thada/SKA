import torch
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim, n_layers):    
        super().__init__()
        self.input_size =input_size
        self.output_size=output_size
        self.n_layers=n_layers
        self.hidden_dim=hidden_dim 
        self.lstm=nn.LSTM(input_size, hidden_dim, n_layers,  batch_first=True)
        self.fc=nn.Linear(self.hidden_dim,self.output_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, hidden):        
        self.lstm_out, self.hidden=self.lstm(x, hidden)         
        self.logits=self.fc(self.lstm_out) # logits
        self.softmax_out= self.softmax(self.logits)
        self.out= self.softmax_out[:, -1]
        return self.lstm_out, self.hidden, self.logits, self.softmax_out

    def init_hidden(self, batch_size):
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim),
                      torch.zeros(self.n_layers, batch_size, self.hidden_dim))
        return hidden
    

class LSTM_multilabel(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim, n_layers):    
        super().__init__()
        self.input_size =input_size
        self.output_size=output_size
        self.n_layers=n_layers
        self.hidden_dim=hidden_dim 
        self.lstm=nn.LSTM(input_size, hidden_dim, n_layers,  batch_first=True)
        self.fc=nn.Linear(self.hidden_dim,self.output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden):   
        self.lstm_out, self.hidden=self.lstm(x, hidden)         
        self.logits=self.fc(self.lstm_out) 
        self.sigmoid_outs= self.sigmoid(self.logits)
        self.out= self.sigmoid_outs[:, -1]
        return self.lstm_out, self.hidden, self.logits, self.sigmoid_outs

    def init_hidden(self, batch_size):
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim),
                      torch.zeros(self.n_layers, batch_size, self.hidden_dim))
        return hidden















