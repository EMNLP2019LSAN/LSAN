import torch
import torch.nn.functional as F


class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path=None):
        if path is None:
            raise ValueError('Please specify the saving road!!!')
        torch.save(self.state_dict(), path)
        return path


class StructuredSelfAttention(BasicModule):

    def __init__(self, batch_size, lstm_hid_dim, d_a, n_classes, label_embed, embeddings):
        super(StructuredSelfAttention, self).__init__()
        self.n_classes = n_classes
        self.embeddings = self._load_embeddings(embeddings)
        self.label_embed = self.load_labelembedd(label_embed)
        self.lstm = torch.nn.LSTM(300, hidden_size=lstm_hid_dim, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.linear_first = torch.nn.Linear(lstm_hid_dim * 2, d_a)
        self.linear_second = torch.nn.Linear(d_a, n_classes)

        self.weight1 = torch.nn.Linear(lstm_hid_dim * 2, 1)
        self.weight2 = torch.nn.Linear(lstm_hid_dim * 2, 1)

        self.output_layer = torch.nn.Linear(lstm_hid_dim*2, n_classes)
        self.embedding_dropout = torch.nn.Dropout(p=0.3)
        self.batch_size = batch_size
        self.lstm_hid_dim = lstm_hid_dim

    def _load_embeddings(self, embeddings):
        """Load the embeddings based on flag"""
        word_embeddings = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
        word_embeddings.weight = torch.nn.Parameter(embeddings)
        return word_embeddings
    def load_labelembedd(self, label_embed):
        """Load the embeddings based on flag"""
        embed = torch.nn.Embedding(label_embed.size(0), label_embed.size(1))
        embed.weight = torch.nn.Parameter(label_embed)
        return embed

    def init_hidden(self):
        return (torch.randn(2,self.batch_size,self.lstm_hid_dim).cuda(),torch.randn(2,self.batch_size,self.lstm_hid_dim).cuda())

    def forward(self,x):
        embeddings = self.embeddings(x)
        embeddings = self.embedding_dropout(embeddings)
        hidden_state = self.init_hidden()
        #step1 get LSTM outputs
        outputs, hidden_state = self.lstm(embeddings, hidden_state)
        #step2 get self-attention document representation
        #step3 get label-attention document representation
        #step4 get the weights for self-sttention and label-attention 
        #step5 adaptive fusion of two part document representations for classification 
        return pred



