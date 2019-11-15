import numpy as np
from config import cfg
from dataset import TextDataset
from generator import DataGenerator_encode
from model_load import model_create_pretrain
import torchvision.transforms as transforms
from copy import deepcopy
from keras import callbacks

#########Run at　config.TREE.BRANCH_NUM = 3 ########

def main():
    #DataGenerator
    imsize = cfg.TREE.BASE_SIZE * (2**(cfg.TREE.BRANCH_NUM - 1))  #64, 3
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()
    ])
    #cfg.DATA_DIR = "data/birds"
    dataset = TextDataset(
        cfg.DATA_DIR,
        "train",
        base_size=cfg.TREE.BASE_SIZE,
        transform=image_transform)
    assert dataset

    dataset_val = TextDataset(
        cfg.DATA_DIR,
        "test",
        base_size=cfg.TREE.BASE_SIZE,
        transform=image_transform)
    assert dataset_val

    traingenerator = DataGenerator_encode(
        dataset, batchsize=cfg.TRAIN.BATCH_SIZE)

    val_generator = DataGenerator_encode(
        dataset_val, batchsize=cfg.TRAIN.BATCH_SIZE)

    #Create model
    CR_model = model_create_pretrain(dataset)
    print(CR_model.summary())

    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=cfg.TRAIN.DEC_SAVE_PATH,
            monitor="val_loss",
            save_weights_only=True,
            save_best_only=True)]

    step_epoch = int(len(dataset) / cfg.TRAIN.BATCH_SIZE)
    step_epoch_val = int(len(dataset_val) / cfg.TRAIN.BATCH_SIZE)

    hist = CR_model.fit_generator(
        traingenerator,
        steps_per_epoch=step_epoch,
        epochs=cfg.TRAIN.DEC_MAX_EPOCH,
        validation_data=val_generator,
        validation_steps=step_epoch,
        callbacks=callbacks_list)

    print('End of learning')


if __name__ == '__main__':
    main()