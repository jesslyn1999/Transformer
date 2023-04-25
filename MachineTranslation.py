import torch
torch.cuda.empty_cache()
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.functional import pad
import spacy 
from Transformer import Transformer
import torch.optim as optim
from optparse import OptionParser
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.metrics import bleu_score




'''
Apply our implemented transformer to the task of English-to-German machine translation on the Multi30K dataset. 
'''
class MachineTranslation():

    def __init__(self, device, batch_size, num_epochs, num_encoder_layers, 
                 num_decoder_layers, model_dim, num_heads, hidden_dim, 
                 dropout_rate, learning_rate, weight_decay):
        self.device = device
        self.batch_size = batch_size 
        self.num_epochs = num_epochs
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
    

    def load_tokenizers(self):
        '''
        We use Spacy tokenizers to tokenize texts.
        Spacy is an open-source software library for advanced natural language processing, 
        for detailed introduction please visit: https://spacy.io/models
        '''
        self.en_tokenizer = spacy.load('en_core_web_sm')  # English tokenizer
        self.de_tokenizer = spacy.load('de_core_news_sm') # German tokenizer
    

    def tokenize(self, text, tokenizer):
        return [tok.text for tok in tokenizer.tokenizer(text)]
    

    def yield_tokens(self, data_iter, tokenizer, index):
        for from_to_tuple in data_iter:
            yield tokenizer(from_to_tuple[index])
    

    def tokenize_en(self, text):
        return self.tokenize(text, self.en_tokenizer)
        

    def tokenize_de(self, text):
        return self.tokenize(text, self.de_tokenizer)
    

    def build_vocabulary(self):
        '''
        Build English and German vocabulary using torchtext.vocab.build_vocab_from_iterator.
        special tokens are: 
        "<sos>": start of a sentence;
        "<eos>": end of a sentence;
        "<pad>": padded token;
        "<unk>": unknown(out-of-vocabulary) token.
        '''
        self.load_tokenizers()

        print('Building English vocabulary...')
        train_data, valid_data, test_data = torchtext.datasets.Multi30k(language_pair = ('en', 'de'))
        self.src_vocab = build_vocab_from_iterator(
        self.yield_tokens(train_data + valid_data + test_data, self.tokenize_en, index = 0),
        min_freq = 1,
        specials=["<sos>", "<eos>", "<pad>", "<unk>"])

        print('Building German vocabulary...')
        train_data, valid_data, test_data = torchtext.datasets.Multi30k(language_pair = ('en', 'de'))
        self.tgt_vocab = build_vocab_from_iterator(
        self.yield_tokens(train_data + valid_data + test_data, self.tokenize_de, index = 1),
        min_freq = 1,
        specials=["<sos>", "<eos>", "<pad>", "<unk>"])

        self.src_vocab.set_default_index(self.src_vocab["<unk>"]) 
        self.tgt_vocab.set_default_index(self.tgt_vocab["<unk>"])

        self.src_vocab_size = len(self.src_vocab)
        self.tgt_vocab_size = len(self.tgt_vocab)
        self.sos_idx, self.eos_idx, self.pad_idx = 0, 1, 2
        print("source English vocabulary size:", len(self.src_vocab))
        print("target German vocabulary size:", len(self.tgt_vocab))
    

    def collate_batch(self, batch, max_padding = 64, pad_id = 2):
        '''
        Convert tokenized texts into a list of integers and pad all sequences to
        have the same length using 'pad_id', which corresponds to the special token "<pad>".
        '''
        sos_id = torch.LongTensor([0]).to(self.device)  # <sos> token id
        eos_id = torch.LongTensor([1]).to(self.device)  # <eos> token id

        src_list, tgt_list = [], []

        for (src, tgt) in batch:
            processed_src = torch.cat([
                sos_id, 
                torch.tensor(self.src_vocab(self.tokenize_en(src)), dtype = torch.long, device = self.device),
                eos_id], dim = 0)
            processed_tgt = torch.cat([
                sos_id,
                torch.tensor(self.tgt_vocab(self.tokenize_de(tgt)), dtype = torch.long, device = self.device),
                eos_id], dim = 0)
            src_list.append(pad(processed_src, (0, max_padding - len(processed_src)), value = pad_id))
            tgt_list.append(pad(processed_tgt, (0, max_padding - len(processed_tgt)), value = pad_id))

        src = torch.stack(src_list)
        tgt = torch.stack(tgt_list)

        return (src, tgt)
    

    def create_dataloaders(self):
        '''
        Convert Iterable-style data into map-style datasets and then build dataloaders.
        (Directly build dataloaders frm Iterable-style data will raise the StopIteration error.)
        '''
        self.build_vocabulary()

        train_data, valid_data, test_data = torchtext.datasets.Multi30k(language_pair = ('en', 'de'))
        train_iter = to_map_style_dataset(train_data)
        valid_iter = to_map_style_dataset(valid_data)
        test_iter = to_map_style_dataset(test_data)
        self.train_dataloader = DataLoader(train_iter, batch_size = self.batch_size, shuffle = True, collate_fn = self.collate_batch)
        self.valid_dataloader = DataLoader(valid_iter, batch_size = self.batch_size, shuffle = False, collate_fn = self.collate_batch)
        self.test_dataloader = DataLoader(test_iter, batch_size = self.batch_size, shuffle = False, collate_fn = self.collate_batch)
    

    def load_transformer(self):
        '''
        load our implemented transformer.
        '''
        self.model = Transformer(
            src_vocab_size = self.src_vocab_size, 
            tgt_vocab_size = self.tgt_vocab_size,
            num_encoder_layers = self.num_encoder_layers,
            num_decoder_layers = self.num_decoder_layers,
            model_dim = self.model_dim,
            num_heads = self.num_heads,
            hidden_dim = self.hidden_dim,
            dropout_rate = self.dropout_rate,
            max_len = 10000,
            device = self.device).to(self.device)

    
    def get_optimizers(self):
        '''
        We use Adam as our optimizer with a ReduceLROnPlateau scheduler.
        '''
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor = 0.95, patience = 5, threshold = 0.00001, verbose = True)

    
    def get_loss_fn(self):
        '''
        We use Cross-Entropy loss as the loss function. Ignore the padded value when calculating the loss. 
        '''
        self.loss_fn = nn.CrossEntropyLoss(ignore_index = self.pad_idx).to(self.device)

    
    def train_transformer_per_epoch(self, dataloader, train = True):
        '''
        Train and evaluate transformer.
        '''
        if train:
            self.model.train()
        else:
            self.model.eval()
        total_loss = 0.0
        for _, data in enumerate(dataloader):
            if train:
                self.optimizer.zero_grad()
            src, tgt = [d.to(self.device) for d in data]
            output = self.model(src, tgt)
            output = output[:, 1:, :].reshape(-1, output.shape[-1]) # remove the sos_id
            tgt = tgt[:, 1:].long().reshape(-1) # remove the sos_id
            loss = self.loss_fn(output, tgt)
            total_loss += loss.item()
            if train:
                loss.backward()
                self.optimizer.step()

        return total_loss
    

    def train_transformer(self):
        self.create_dataloaders()
        self.load_transformer()
        self.get_optimizers()
        self.get_loss_fn()

        print("NUM EPOCH: ", self.num_epochs)

        for e in range(self.num_epochs):
            print("epoch-", e)
            train_loss = self.train_transformer_per_epoch(self.train_dataloader, True)
            with torch.no_grad():
                valid_loss = self.train_transformer_per_epoch(self.valid_dataloader, False)
            print('Epoch {}: train loss {} valid loss {}'.format(e + 1, round(train_loss, 4), round(valid_loss, 4)))
            self.scheduler.step(valid_loss)
        
        torch.save(self.model.state_dict(), './checkpoints/model.pt') # change this to your own path
        self.evaluate_transformer()
    

    def evaluate_transformer(self):
        '''
        We use the BLEU score to evaluate model performance.
        BLEU is a widely-used metric for machine translation, which measures the similarity of the 
        machine-translated text to a set of high quality reference translations.
        The current SOTA BLEU score on this EN-DE task is 49.3 by ERNIE-UniX2.
        '''
        self.model.eval()
        preds, targets = [], []
        for _, data in enumerate(self.test_dataloader):
            src, tgt = [d.to(self.device) for d in data]
            output = self.model(src, tgt)
            pred = torch.argmax(output, dim = -1)
            for i in range(pred.shape[0]):
                pred_tokens = [self.tgt_vocab.get_itos()[id] for id in pred[i] if id != self.pad_idx or id != self.sos_idx or id != self.eos_idx]
                tgt_tokens = [self.tgt_vocab.get_itos()[id] for id in tgt[i] if id != self.pad_idx or id != self.sos_idx or id != self.eos_idx]
                preds.append(pred_tokens)
                targets.append([tgt_tokens])
        
        print('Transformer performance on the test set: ', bleu_score(preds, targets) * 100) 
        


'''
Hyperparameters.
'''
def get_args():
    parser = OptionParser()

    parser.add_option('--batch_size', dest = 'batch_size', default = 64, type = 'int', help = 'batch size')
    parser.add_option('--num_epochs', dest = 'num_epochs', default = 200, type = 'int', help = 'number of epochs')
    parser.add_option('--num_encoder_layers', dest = 'num_encoder_layers', default = 6, type = 'int', help = 'number of encoder layers')
    parser.add_option('--num_decoder_layers', dest = 'num_decoder_layers', default = 6, type = 'int', help = 'number of decoder layers')
    parser.add_option('--model_dim', dest = 'model_dim', default = 512, type = 'int', help = 'embedding dimension for each token')
    parser.add_option('--num_heads', dest = 'num_heads', default = 8, type = 'int', help = 'number of heads in multi-head self attention')
    parser.add_option('--hidden_dim', dest = 'hidden_dim', default = 2048, type = 'int', help = 'hidden dimension in position-wise feedforward layer')
    parser.add_option('--dropout_rate', dest = 'dropout_rate', default = 0.1, type = 'float', help = 'dropout rate')
    parser.add_option('--learning_rate', dest = 'learning_rate', default = 0.0001, type = 'float', help = 'learning rate')
    parser.add_option('--weight_decay', dest = 'weight_decay', default = 0.00001, type = 'float', help = 'weight decay parameter')
    (options, _) = parser.parse_args()

    return options

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    num_encoder_layers = args.num_encoder_layers
    num_decoder_layers = args.num_decoder_layers
    model_dim = args.model_dim
    num_heads = args.num_heads
    hidden_dim = args.hidden_dim
    dropout_rate = args.dropout_rate
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay

    machine_translation = MachineTranslation(device, batch_size, num_epochs, num_encoder_layers,
                                             num_decoder_layers, model_dim, num_heads, hidden_dim,
                                             dropout_rate, learning_rate, weight_decay)
    machine_translation.train_transformer()



    









