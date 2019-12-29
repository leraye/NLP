import torch.nn as nn
import torch.nn.functional as F

class NNTagger(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, window_size, tagset_size, nonlinear, pretrained=None):
        super(NNTagger, self).__init__()
        if pretrained is None:
            self.emd = embedding_dim
            self.word_embeddings = nn.Embedding(vocab_size, self.emd)
            self.init_weights()
        else:
            self.emd = pretrained.size(1)
            self.word_embeddings = nn.Embedding(vocab_size, self.emd)

            # Set pretrained embeddings
            self.word_embeddings.weight.data.copy_(pretrained)
            self.word_embeddings.weight.requires_grad = False
        
        self.hidden = nn.Linear((2*window_size+1) * self.emd, hidden_dim)
        if nonlinear in ['relu','tanh','sigmoid']:
            self.nonlinear = getattr(F, nonlinear)
        else:
            raise ValueError( """An invalid option for `--nonlinear` was supplied,
                                 options are ['RELU', 'TANH', 'SIGMOID']""")
        self.linear = nn.Linear(hidden_dim, tagset_size)
        
    def init_weights(self):
        initrange = 0.1
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        emb = self.word_embeddings(input)
        out = self.hidden(emb.reshape(emb.size(0), -1).unsqueeze(1))
        out = self.linear(out)
        return out
