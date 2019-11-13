from transformers import *

from utils.pixelcnnpp_utils import *
import pdb
from torch.nn.utils import weight_norm as wn
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from models.infersent import InferSent


class Embedder(nn.Module):
    def __init__(self, embed_size):
        super(Embedder, self).__init__()
        self.embed_size = embed_size

    def forward(self, captions):
        raise NotImplementedError


class BERTEncoder(Embedder):
    '''
    pretrained model used to embed text to a 768 dimensional vector
    '''

    def __init__(self, device):
        super(BERTEncoder, self).__init__(embed_size=768)
        self.pretrained_weights = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_weights)
        self.model = BertModel.from_pretrained(self.pretrained_weights)
        self.max_len = 50
        self.device = device

    def tokenize(self, text_batch):
        text_token_ids = [
            torch.tensor(self.tokenizer.encode(string_, add_special_tokens=False, max_length=self.max_len)) for
            string_ in text_batch]
        padded_input = pad_sequence(text_token_ids, batch_first=True, padding_value=0)
        return padded_input

    def forward(self, captions):
        '''
        :param list captions: list of strings, sentences to embed
        :return: torch.tensor embeddings: embeddings of shape (batch_size,embed_size)
        '''

        padded_input = self.tokenize(captions).to(self.device)
        # takes the mean of the last hidden states computed by the pre-trained BERT encoder and return it
        return self.model(padded_input)[0].mean(dim=1)


class UnconditionalClassEmbedding(Embedder):
    def __init__(self, device):
        super(UnconditionalClassEmbedding, self).__init__(embed_size=1)
        self.device = device

    def forward(self, captions):
        '''
        :param list captions: list of strings, sentences to embed
        :return: torch.tensor embeddings: embeddings of shape (batch_size,embed_size)
        '''
        zero = torch.zeros(len(captions), 1).to(self.device)
        return zero


class InferSentEmbedding(Embedder):
    '''
    pretrained model from Facebook to embed text to a 4096 dimensional vector
    '''

    def __init__(self, device):
        super(InferSentEmbedding, self).__init__(embed_size=4096)
        model_version = 2
        MODEL_PATH = "encoder/infersent%s.pkl" % model_version
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
        self.device = device
        self.model = InferSent(params_model)
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model = self.model.to(device)
        W2V_PATH = 'GloVe/glove.840B.300d.txt' if model_version == 1 else 'fastText/crawl-300d-2M.vec'
        self.model.set_w2v_path(W2V_PATH)
        # Load embeddings of K most frequent words
        self.model.build_vocab_k_words(K=100000)
        print('Loaded InferSent model')

    def forward(self, captions):
        '''
        :param list captions: list of strings, sentences to embed
        :return: torch.tensor embeddings: embeddings of shape (batch_size,embed_size)
        '''
        embeddings = self.model.encode(captions, bsize=128)
        result = torch.as_tensor(embeddings, device=self.device)
        return result
