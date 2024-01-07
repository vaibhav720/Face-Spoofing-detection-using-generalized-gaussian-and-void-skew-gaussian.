import os
import shutil
import time

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader

from configs import *
from dataset_helpers import get_datasets

torch.manual_seed(0)
np.random.seed(0)


def get_model(freeze_till: int):
    model = torchvision.models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(1280, 2)
    i: int = 0
    for name, param in model.named_parameters():
        # print(i, name)
        if i < freeze_till:
            param.requires_grad = False
        else:
            break
        i += 1
    return model


def train_one_epoch(model, data_loader, criterion, optimizer):
    start = time.time()

    model.train()
    avg_loss = 0
    nb_steps = 0
    for image, target in data_loader:
        optimizer.zero_grad()

        image, target = image.to(DEVICE), target.to(DEVICE)
        output = model(image)
        loss = criterion(output, target)

        avg_loss += loss.item()
        nb_steps += 1
        loss.backward()
        optimizer.step()
    avg_loss /= nb_steps

    print(f'Train loss: {avg_loss} | time: {time.time() - start}s')

    return avg_loss


def evaluate(model, data_loader, criterion):
    start = time.time()

    model.eval()
    avg_loss = 0
    nb_steps = 0
    with torch.no_grad():
        for image, target in data_loader:
            image, target = image.to(DEVICE), target.to(DEVICE)
            output = model(image)
            loss = criterion(output, target)
            avg_loss += loss.item()
            nb_steps += 1
    avg_loss /= nb_steps

    print(f'Test loss: {avg_loss} | time: {time.time() - start}s')

    return avg_loss


def train(dataset_path, csv_fp, usecols):
    CKPT_DIR_PATH = 'models'

    if os.path.isdir(CKPT_DIR_PATH):
        shutil.rmtree(CKPT_DIR_PATH)
    os.makedirs(CKPT_DIR_PATH)

    dataset = get_datasets(dataset_path, csv_fp, usecols)
    train_ds = dataset['train']
    test_ds = dataset['test']

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=RandomSampler(train_ds),
                              num_workers=NB_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, sampler=SequentialSampler(test_ds),
                             num_workers=NB_WORKERS, pin_memory=True)

    model = get_model(KEEP_FROZEN)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    min_test_loss = 500
    best_epoch: int = -1
    for epoch in range(EPOCHS):
        start = time.time()
        print(f'Epoch {epoch}/{EPOCHS - 1}:')
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss = evaluate(model, test_loader, criterion)

        lr_scheduler.step()

        if min_test_loss > test_loss:
            min_test_loss = test_loss
            best_epoch = epoch
            checkpoint = {
                'model': model,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'epoch': epoch
            }
            torch.save(checkpoint, os.path.join(CKPT_DIR_PATH, f'model_{epoch}.pth'))

        print(f'Epoch {epoch} total time: {time.time() - start}s')


def main():
    print(f'Experiment Start: {time.strftime("%Y-%m-%d %H:%M")}')
    dataset_path = '../../data/CASIA_faceAntisp/'
    csv_fp = '../../data/merged_data.csv'
    usecols = ['Frame Name',
               'Video Name',
               'Folder Name',
               'Subset Path',
               'Liveness']

    train(dataset_path, csv_fp, usecols)


if __name__ == '__main__':
    main()
