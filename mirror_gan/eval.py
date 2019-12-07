import numpy as np
from config import cfg
from dataset import TextDataset
from generator import DataGenerator
from model import *
from model_load import model_create
import torchvision.transforms as transforms
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_test_dataset_generator():
    imsize = cfg.TREE.BASE_SIZE * (2**(cfg.TREE.BRANCH_NUM - 1))  #64, 3
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()
    ])

    dataset = TextDataset(
        cfg.DATA_DIR,
        "train",
        base_size=cfg.TREE.BASE_SIZE,
        transform=image_transform)
    return dataset, DataGenerator(dataset, batchsize=cfg.TRAIN.BATCH_SIZE)

def eval(dataset, test_generator, num_epochs=100):
    G_model, D_model, GRD_model, CR_model, RNN_model = model_create(dataset)

    counter = 0
    for i in range(num_epochs):
        image_list, captions_ar, captions_ar_prezeropad, \
            z_code, eps_code, mask, keys_list, captions_label, \
                real_label, fake_label = next(test_generator)
        if cfg.TREE.BRANCH_NUM == 1:
            gen_imgs = G_model.predict([captions_ar_prezeropad, eps_code, z_code])
        else:
            gen_imgs = G_model.predict([captions_ar_prezeropad, eps_code, z_code, mask])

        gen_imgs = (gen_imgs * 127.5 + 127.5).astype("int")

        for j in range(len(gen_imgs)):
            plt.axis('off')
            plt.imshow(gen_imgs[j])
            plt.savefig("test_gen_img_v3/%d.png" % counter, bbox_inches='tight')
            plt.close()
            counter += 1


if __name__ == "__main__":
    dataset, generator = load_test_dataset_generator()
    eval(dataset, generator, num_epochs=20)