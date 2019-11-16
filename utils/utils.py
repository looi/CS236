import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import torch
import textwrap

##############
## PixelCNN ##
##############

def sample_image(model, encoder, output_image_dir, n_row, epoch, dataloader, device, conditioning, imsize):
    """Saves a grid of generated pictures with captions"""
    target_dir = os.path.join(output_image_dir, "samples/")
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    captions = []
    gen_imgs = []
    # get sample captions
    remaining = n_row ** 2
    for (imgs, caps, cls_ids, keys) in dataloader:
        if len(caps) > remaining:
            caps = caps[:remaining]
            imgs = imgs[:remaining]
            cls_ids = cls_ids[:remaining]
            keys = keys[:remaining]
        print('%d samples remaining, generating %d' % (remaining, len(caps)))
        captions += caps
        conditional_embeddings = encoder(captions)
        imgs = model.sample(conditional_embeddings).cpu()
        gen_imgs.append(imgs)
        #gen_imgs.append(torch.zeros(imgs.shape[0], 3, 64, 64))
        remaining -= len(caps)
        if remaining <= 0: break

    gen_imgs = torch.cat(gen_imgs).numpy()
    gen_imgs = (gen_imgs+1)/2 # from [-1,1] to [0,1]
    gen_imgs = np.clip(gen_imgs, 0, 1)

    fig = plt.figure(figsize=((8, 8)))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_row, n_row), axes_pad=1.0)#0.2)

    for i in range(n_row ** 2):
        grid[i].imshow(gen_imgs[i].transpose([1, 2, 0]))
        grid[i].set_title('\n'.join(textwrap.wrap(captions[i], width=30)))
        grid[i].title.set_fontsize(7)
        grid[i].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=True)

    save_file = os.path.join(target_dir, "{}_{}_epoch_{}.png".format(conditioning, imsize, epoch))
    plt.savefig(save_file)
    print("saved  {}".format(save_file))
    plt.close()


def load_model(file_path, generative_model):
    dict = torch.load(file_path)
    generative_model.load_state_dict(dict)

###########
## ACGAN ##
###########

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# compute the current classification accuracy
def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc
