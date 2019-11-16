from nltk.tokenize import RegexpTokenizer
from collections import defaultdict

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
import pickle

def get_img(img_path, imsize, bbox=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        x,y,w,h = bbox
        img = img.crop([x,y,x+w,y+h])

    #transforms.Compose([transforms.Scale(int(imsize * 76 / 64)), transforms.RandomCrop(imsize), transforms.RandomHorizontalFlip()])
    img = img.resize((imsize, imsize))

    norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = norm(img)
    
    return img


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 imsize=64,
                 captions_per_image=10,
                 transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.captions_per_image = captions_per_image

        self.imsize = imsize

        self.data = []
        self.data_dir = data_dir
        self.bbox = self.load_bbox() # only for birds
        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions = self.load_text_data(data_dir, split)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
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
            with open(cap_path, "r") as f:
                captions = f.read().split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    all_captions.append(cap)
                    cnt += 1
                    if cnt == self.captions_per_image:
                        break
                if cnt < self.captions_per_image:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'bird_captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        train_captions = self.load_captions(data_dir, train_names)
        test_captions = self.load_captions(data_dir, test_names)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        return filenames, captions

    def load_class_id(self, data_dir, total_num):
        with open(data_dir + '/class_info.pickle', 'rb') as f:
            class_id = pickle.load(f, encoding='bytes')
        # The class IDs are from 1 to 200 inclusive, convert to 0 to 199
        class_id = [x-1 for x in class_id]
        assert(all(0 <= x <= 199 for x in class_id))
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
        imgs = get_img(img_name, self.imsize, bbox)
        # random select a sentence
        sent_ix = random.randint(0, self.captions_per_image)
        new_sent_ix = index * self.captions_per_image + sent_ix
        cap = self.captions[new_sent_ix]
        return imgs, cap, cls_id, key


    def __len__(self):
        return len(self.filenames)
