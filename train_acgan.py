"""
Code modified from PyTorch DCGAN examples: https://github.com/pytorch/examples/tree/master/dcgan
"""
from __future__ import print_function
import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from utils.utils import weights_init, compute_acc
from models.acgan import _netG, _netD, _netD_CIFAR10, _netG_CIFAR10
from models.embedders import BERTEncoder, InferSentEmbedding, UnconditionalClassEmbedding
import datasets

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="birds", choices=["birds"])
parser.add_argument("--conditioning", type=str, default="unconditional", choices=["unconditional", "infersent", "bert"])
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--imsize", type=int, default=32, help="Image size in pixels")
parser.add_argument('--nz', type=int, default=200, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument("--use_cuda", type=int, default=1, help="use cuda if available")
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='outputs/acgan', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--embed_size', default=100, type=int, help='embed size')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for AC-GAN')
parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')
parser.add_argument("--infersent_path", type=str, default='encoder', help="Path to pre-trained InferSent model")

def main(args=None):
    opt = parser.parse_args(args)
    print(opt)

    # specify the gpu id if using only 1 gpu
    if opt.ngpu == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    use_cuda = torch.cuda.is_available() and opt.use_cuda
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    if use_cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    print("loading dataset")
    if opt.dataset == 'birds':
        train_dataset = datasets.TextDataset('data/birds', 'train', imsize=opt.imsize)
        val_dataset = datasets.TextDataset('data/birds', 'test', imsize=opt.imsize)
    else:
        raise NotImplementedError("No such dataset {}".format(opt.dataset))

    print("creating dataloaders")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
    )

    print("Len train : {}, val : {}".format(len(train_dataloader), len(val_dataloader)))

    # some hyper parameters
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    num_classes = int(opt.num_classes)
    nc = 3

    # Define the generator and initialize the weights
    netG = _netG_CIFAR10(ngpu, nz)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    # Define the discriminator and initialize the weights
    netD = _netD_CIFAR10(ngpu, num_classes)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    # loss functions
    dis_criterion = nn.BCELoss()
    aux_criterion = nn.NLLLoss()

    # tensor placeholders
    input = torch.FloatTensor(opt.batch_size, 3, opt.imsize, opt.imsize)
    noise = torch.FloatTensor(opt.batch_size, nz, 1, 1)
    eval_noise = torch.FloatTensor(opt.batch_size, nz, 1, 1).normal_(0, 1)
    dis_label = torch.FloatTensor(opt.batch_size)
    aux_label = torch.LongTensor(opt.batch_size)
    real_label = 1
    fake_label = 0

    # if using cuda
    if use_cuda:
        netD.cuda()
        netG.cuda()
        dis_criterion.cuda()
        aux_criterion.cuda()
        input, dis_label, aux_label = input.cuda(), dis_label.cuda(), aux_label.cuda()
        noise, eval_noise = noise.cuda(), eval_noise.cuda()

    # define variables
    input = Variable(input)
    noise = Variable(noise)
    eval_noise = Variable(eval_noise)
    dis_label = Variable(dis_label)
    aux_label = Variable(aux_label)

    if opt.conditioning == 'unconditional':
        encoder = UnconditionalClassEmbedding(device)
    elif opt.conditioning == "bert":
        encoder = BERTEncoder(device)
    else:
        assert opt.conditioning == "infersent"
        encoder = InferSentEmbedding(device, opt.infersent_path)

    # noise for evaluation
    eval_noise_ = np.random.normal(0, 1, (opt.batch_size, nz))
    eval_label = np.zeros(opt.batch_size)#np.random.randint(0, num_classes, opt.batch_size)
    #if opt.dataset == 'cifar10':
    #            captions = [cifar_text_labels[per_label] for per_label in eval_label]
    #            embedding = encoder(eval_label, captions)
    #            embedding = embedding.detach().numpy()
    #eval_noise_[np.arange(opt.batch_size), :opt.embed_size] = embedding[:, :opt.embed_size]
    eval_noise_ = (torch.from_numpy(eval_noise_))
    eval_noise.data.copy_(eval_noise_.view(opt.batch_size, nz, 1, 1))

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    avg_loss_D = 0.0
    avg_loss_G = 0.0
    avg_loss_A = 0.0
    for epoch in range(opt.niter):
        for i, (imgs, captions, cls_ids, keys) in enumerate(train_dataloader):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = imgs
            batch_size = real_cpu.size(0)
            # Dummy label for now
            label = torch.zeros(batch_size)
            label[0] = 1
            if use_cuda:
                real_cpu = real_cpu.cuda()
            with torch.no_grad():
                input.resize_as_(real_cpu).copy_(real_cpu)
                dis_label.resize_(batch_size).fill_(real_label)
                aux_label.resize_(batch_size).copy_(label)
            dis_output, aux_output = netD(input)

            dis_errD_real = dis_criterion(dis_output, dis_label)
            aux_errD_real = aux_criterion(aux_output, aux_label)
            errD_real = dis_errD_real + aux_errD_real
            errD_real.backward()
            D_x = dis_output.data.mean()

            # compute the current classification accuracy
            accuracy = compute_acc(aux_output, aux_label)

            # train with fake
            with torch.no_grad():
                noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
            label = np.random.randint(0, num_classes, batch_size)
            #if opt.dataset == 'cifar10':
            #    captions = [cifar_text_labels[per_label] for per_label in label]
            #    embedding = encoder(label, captions)
            #    embedding = embedding.detach().numpy()
            noise_ = np.random.normal(0, 1, (batch_size, nz))
            
            #noise_[np.arange(batch_size), :opt.embed_size] = embedding[:, :opt.embed_size]
            noise_ = (torch.from_numpy(noise_))
            with torch.no_grad():
                noise.copy_(noise_.view(batch_size, nz, 1, 1))
                aux_label.resize_(batch_size).copy_(torch.from_numpy(label))

            fake = netG(noise)
            dis_label.data.fill_(fake_label)
            dis_output, aux_output = netD(fake.detach())
            dis_errD_fake = dis_criterion(dis_output, dis_label)
            aux_errD_fake = aux_criterion(aux_output, aux_label)
            errD_fake = dis_errD_fake + aux_errD_fake
            errD_fake.backward()
            D_G_z1 = dis_output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            dis_label.data.fill_(real_label)  # fake labels are real for generator cost
            dis_output, aux_output = netD(fake)
            dis_errG = dis_criterion(dis_output, dis_label)
            aux_errG = aux_criterion(aux_output, aux_label)
            errG = dis_errG + aux_errG
            errG.backward()
            D_G_z2 = dis_output.data.mean()
            optimizerG.step()

            # compute the average loss
            curr_iter = epoch * len(train_dataloader) + i
            all_loss_G = avg_loss_G * curr_iter
            all_loss_D = avg_loss_D * curr_iter
            all_loss_A = avg_loss_A * curr_iter
            all_loss_G += errG.item()
            all_loss_D += errD.item()
            all_loss_A += accuracy
            avg_loss_G = all_loss_G / (curr_iter + 1)
            avg_loss_D = all_loss_D / (curr_iter + 1)
            avg_loss_A = all_loss_A / (curr_iter + 1)

            print('[%d/%d][%d/%d] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f) D(x): %.4f D(G(z)): %.4f / %.4f Acc: %.4f (%.4f)'
                  % (epoch, opt.niter, i, len(train_dataloader),
                     errD.item(), avg_loss_D, errG.item(), avg_loss_G, D_x, D_G_z1, D_G_z2, accuracy, avg_loss_A))
            if i % 100 == 0:
                vutils.save_image(
                    real_cpu, '%s/real_samples.png' % opt.outf, range=(-1,1), normalize=True)
                #print('Label for eval = {}'.format(eval_label))
                #fake = netG(eval_noise)
                vutils.save_image(
                    fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch), range=(-1,1), normalize=True
                )

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))

if __name__ == "__main__":
    main()
