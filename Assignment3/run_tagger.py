import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from random import shuffle
import dataset
import tagger

parser = argparse.ArgumentParser(description='Baseline Neural Network Tagger: Main Function')
parser.add_argument('--data', type=str, default='./tweet-pos/',
                    help='location of the data corpus')
parser.add_argument('--nonlinear', type=str, default='relu',
                    help='type of nonlinearity (TANH, RELU, SIGMOID)')
parser.add_argument('--wsize', type=int, default=1,
                    help='size of context window')
parser.add_argument('--emsize', type=int, default=50,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units per layer')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit')
parser.add_argument('--pretrained', action='store_true',
                    help='use pretrained embeddings')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

torch.manual_seed(args.seed)

corpus = dataset.Corpus(args.data, args.wsize)

if args.pretrained:
    emd_path = 'embeddings-twitter.txt'
    pre_tr = corpus.load_pretrained(emd_path)
else:
    pre_tr = None

ntokens = corpus.dictionary.vocab_len()
ntags = corpus.dictionary.tag_len()
model = tagger.NNTagger(args.emsize, args.nhid, ntokens, args.wsize, ntags, args.nonlinear,pre_tr)
criterion = nn.CrossEntropyLoss()
if args.pretrained:
    parameters = filter(lambda p: p.requires_grad, model.parameters())
else:
    parameters = model.parameters()
optimizer = optim.Adagrad(parameters, lr=0.1, lr_decay=1e-5, weight_decay=1e-5)

def evaluate(src):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for seq, lable in src:
            output = model(seq)
            output_flat = output.view(-1, ntags)
            yhat = output_flat.argmax(1)
            total += len(yhat)
            correct += torch.sum(torch.eq(lable, yhat)).item()
    return correct / total

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    shuffle(corpus.train)
    for seq, lable in corpus.train:
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
#        model.zero_grad()
        optimizer.zero_grad()
        output = model(seq)
        loss = criterion(output.view(-1, ntags), lable)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()
    return total_loss

best_val_acc = None

for epoch in range(1, args.epochs+1):
    trloss = train()
    val_acc = evaluate(corpus.valid)
    print('-' * 80)
    print('| end of epoch {:3d} | valid accuracy {:5.3f} | training loss {:8.4f}'.format(epoch, val_acc, trloss))
    print('-' * 80)
    # Save the model if the validation loss is the best we've seen so far.
    if not best_val_acc or val_acc > best_val_acc:
        torch.save({'state_dict': model.state_dict()}, 'checkpoint.pth.tar')
#        with open(args.save, 'wb') as f:
#            torch.save(model, f)
        best_val_acc = val_acc

model = tagger.NNTagger(args.emsize, args.nhid, ntokens, args.wsize, ntags, args.nonlinear, pre_tr)
checkpoint = torch.load('checkpoint.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
# Load the best saved model.
#with open(args.save, 'rb') as f:
#    model = torch.load(f)
#    model.rnn.flatten_parameters()

# Run on test data.
test_acc = evaluate(corpus.test)
print('=' * 89)
print('End of training | test accuracy {:5.2f}'.format(test_acc))
print('=' * 89)