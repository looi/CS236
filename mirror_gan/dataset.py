import os
import sys

from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import torch
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms

from config import cfg
from pytorch_pretrained_bert import BertTokenizer
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')

def get_imgs(img_path, imsize, bbox=None, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []

    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:
        for i in range(cfg.TREE.BRANCH_NUM):
            # print(imsize[i])
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Resize(imsize[i])(img)
            else:
                re_img = img
            re_img = normalize(re_img)
            ##Move channel last
            ##convert to numpy
            re_img = torch.Tensor.numpy(re_img).transpose(1, 2, 0)
            ret.append(re_img)

    return ret


class TextDataset(data.Dataset):
    def __init__(self,
                 data_dir,
                 split='train',
                 base_size=64,
                 transform=None,
                 target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(
            bbox_path, delim_whitespace=True, header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r", encoding="utf-8") as f:
                captions = f.read().split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    # tokenizer = RegexpTokenizer(r'\w+')
                    cap = ''.join(ch for ch in cap if ch.isalnum())
                    tokens = TOKENIZER.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d' %
                          (filenames[i], cnt))
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        # word_counts = defaultdict(float)
        # captions = train_captions + test_captions
        # for sent in captions:
        #     for word in sent:
        #         word_counts[word] += 1

        # vocab = [w for w in word_counts if word_counts[w] >= 0]

        # ixtoword = {}
        # ixtoword[0] = '<end>'
        # wordtoix = {}
        # wordtoix['<end>'] = 0
        # ix = 1
        # for w in vocab:
        #     wordtoix[w] = ix
        #     ixtoword[ix] = w
        #     ix += 1

        train_captions_new = []
        for t in train_captions:
            # rev = []
            # for w in t:
            #     if w in wordtoix:
            #         rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            indexed_tokens = TOKENIZER.convert_tokens_to_ids(t)
            train_captions_new.append(indexed_tokens)

        test_captions_new = []
        for t in test_captions:
            # rev = []
            # for w in t:
            #     if w in wordtoix:
            #         rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            indexed_tokens = TOKENIZER.convert_tokens_to_ids(t)
            test_captions_new.append(indexed_tokens)

        ixtoword, wordtoix = None, None
        n_words = 30522

        return [
            train_captions_new, test_captions_new, ixtoword, wordtoix,
            n_words
        ]

    def load_text_data(self, data_dir, split, load=False):
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath) or not load:
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            # with open(filepath, 'wb') as f:
            #     pickle.dump(
            #         [train_captions, test_captions, ixtoword, wordtoix],
            #         f,
            #         protocol=2)
            #     print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        return filenames, captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding='latin1')
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def __getitem__(self, index):
        #
        key = self.filenames[index]
        cls_id = self.class_id[index]
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        #
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(
            img_name, self.imsize, bbox, self.transform, normalize=self.norm)
        # random select a sentence
        sent_ix = np.random.randint(0, self.embeddings_num)
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(new_sent_ix)
        """
        imgs : numpy.array shape(h, w, c)のリスト　サイズ違い3枚を出力（64, 128, 256）
        caps: 二次元のarray shape(seq, 1) 
        """
        return imgs, caps, cap_len, cls_id, key

    def __len__(self):
        return len(self.filenames)
