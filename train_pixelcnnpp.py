import argparse
import os
import datasets
from models.embedders import BERTEncoder, InferSentEmbedding, UnconditionalClassEmbedding
import torch
from models.pixelcnnpp import ConditionalPixelCNNpp
from utils.utils import sample_image, load_model
from torch.optim import lr_scheduler
import time
from tqdm import tqdm
#from tensorboardX import SummaryWriter
import numpy as np

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=150, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument('--lr_decay', type=float, default=1.0,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_resnet", type=int, default=5, help="number of layers for the pixelcnn model")
parser.add_argument("--n_filters", type=int, default=80, help="dimensionality of the latent space")
parser.add_argument("--use_cuda", type=int, default=1, help="use cuda if available")
parser.add_argument("--output_dir", type=str, default="outputs/pixelcnn", help="directory to store the sampled outputs")
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--train_on_val", type=int, default=0, help="train on val set, useful for debugging")
parser.add_argument("--train", type=int, default=1, help="0 = eval, 1=train")
parser.add_argument("--model_checkpoint_epoch", type=int, default=None,
                    help="load model from checkpoint with this epoch number")
parser.add_argument("--print_every", type=int, default=10)
parser.add_argument("--dataset", type=str, default="birds", choices=["birds"])
parser.add_argument("--conditioning", type=str, default="unconditional", choices=["unconditional", "infersent", "bert"])
parser.add_argument("--imsize", type=int, default=64, help="Image size in pixels")
parser.add_argument("--samples_n_row", type=int, default=2, help="Number of rows for samples")
parser.add_argument("--infersent_path", type=str, default='encoder', help="Path to pre-trained InferSent model")


def train(device, writer, model, embedder, optimizer, scheduler,
          train_loader, val_loader, opt):
    print("TRAINING STARTS")
    start_epoch = opt.model_checkpoint_epoch+1 if opt.model_checkpoint_epoch is not None else 0
    for epoch in range(start_epoch, opt.n_epochs):
        model = model.train()
        loss_to_log = 0.0
        for i, (imgs, captions, cls_ids, keys) in enumerate(train_loader):
            start_batch = time.time()
            imgs = imgs.to(device)

            with torch.no_grad():
                condition_embd = embedder(captions)

            optimizer.zero_grad()
            outputs = model.forward(imgs, condition_embd)
            loss = outputs['loss'].mean()
            loss.backward()
            optimizer.step()
            batches_done = epoch * len(train_loader) + i
            if writer is not None:
                writer.add_scalar('train/bpd', loss / np.log(2), batches_done)
            loss_to_log += loss.item()
            if (i + 1) % opt.print_every == 0:
                loss_to_log = loss_to_log / (np.log(2) * opt.print_every)
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [bpd: %f] [Time/batch %.3f]"
                    % (epoch + 1, opt.n_epochs, i + 1, len(train_loader), loss_to_log, time.time() - start_batch)
                )
                loss_to_log = 0.0

        torch.save(model.state_dict(),
                   os.path.join(opt.output_dir, 'models', '{}_{}_epoch_{}.pt'.format(opt.conditioning, opt.imsize, epoch)))
        print("sampling_images")
        model = model.eval()
        sample_image(model, embedder, opt.output_dir, n_row=opt.samples_n_row,
                     epoch=epoch,
                     dataloader=val_loader, device=device,
                     conditioning=opt.conditioning, imsize=opt.imsize)
        val_bpd = eval(device, model, embedder, val_loader)
        if writer is not None:
            writer.add_scalar("val/bpd", val_bpd, (epoch + 1) * len(train_loader))

        scheduler.step()


def eval(device, model, embedder, test_loader):
    print("EVALUATING ON VAL")
    model = model.eval()
    bpd = 0.0
    for i, (imgs, captions, cls_ids, keys) in tqdm(enumerate(test_loader)):
        imgs = imgs.to(device)

        with torch.no_grad():
            condition_embd = embedder(captions)
            outputs = model.forward(imgs, condition_embd)
            loss = outputs['loss'].mean()
            bpd += loss / np.log(2)
    bpd /= len(test_loader)
    print("VAL bpd : {}".format(bpd))
    return bpd


def main(args=None):
    opt = parser.parse_args(args)
    print(opt)

    print("loading dataset")
    if opt.dataset == "birds":
        train_dataset = datasets.TextDataset('data/birds', 'train', imsize=opt.imsize)
        val_dataset = datasets.TextDataset('data/birds', 'test', imsize=opt.imsize)

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

    device = torch.device("cuda") if (torch.cuda.is_available() and opt.use_cuda) else torch.device("cpu")
    print("Device is {}".format(device))

    print("Loading models on device...")

    # Initialize embedder
    if opt.conditioning == 'unconditional':
        encoder = UnconditionalClassEmbedding(device)
    elif opt.conditioning == "bert":
        encoder = BERTEncoder(device)
    else:
        assert opt.conditioning == "infersent"
        encoder = InferSentEmbedding(device, opt.infersent_path)

    generative_model = ConditionalPixelCNNpp(embd_size=encoder.embed_size, img_shape=(3, opt.imsize, opt.imsize),
                                             nr_resnet=opt.n_resnet, nr_filters=opt.n_filters,
                                             nr_logistic_mix=10)

    generative_model = generative_model.to(device)
    encoder = encoder.to(device)
    print("Models loaded on device")

    # Configure data loader

    print("dataloaders loaded")
    # Optimizers
    optimizer = torch.optim.Adam(generative_model.parameters(), lr=opt.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=opt.lr_decay)
    if opt.model_checkpoint_epoch is not None:
        for i in range(opt.model_checkpoint_epoch+1):
            scheduler.step()
    # create output directory

    os.makedirs(os.path.join(opt.output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, "tensorboard"), exist_ok=True)
    #writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, "tensorboard"))
    writer = None

    if opt.model_checkpoint_epoch is not None:
        print("Loading model from state dict...")
        model_path = os.path.join(opt.output_dir, 'models', '{}_{}_epoch_{}.pt'.format(opt.conditioning, opt.imsize, opt.model_checkpoint_epoch))
        load_model(model_path, generative_model)
        print("Model loaded.")

    # ----------
    #  Training
    # ----------
    if opt.train:
        train(device=device, writer=writer, model=generative_model, embedder=encoder, optimizer=optimizer, scheduler=scheduler,
              train_loader=train_dataloader, val_loader=val_dataloader, opt=opt)
    else:
        assert opt.model_checkpoint_epoch is not None, 'no model checkpoint epoch specified'
        generative_model = generative_model.eval()
        sample_image(generative_model, encoder, opt.output_dir, n_row=opt.samples_n_row,
                     epoch=opt.model_checkpoint_epoch,
                     dataloader=val_dataloader, device=device,
                     conditioning=opt.conditioning, imsize=opt.imsize)
        eval(device=device, model=generative_model, embedder=encoder, test_loader=val_dataloader)

if __name__ == "__main__":
    main()
