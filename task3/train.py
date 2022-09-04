import torch
import logging
import numpy as np
from torch import nn
from tqdm import tqdm
import config


def train_epoch(train_loader, model, optimizer, scheduler, epoch):
    model.train()
    train_loss = 0.0
    for idx, batch_sample in enumerate(tqdm(train_loader)):
        batch_data, batch_label = batch_sample
        batch_mask = batch_data.gt(0)

        outputs = model(batch_data, batch_mask, labels=batch_label)
        loss = outputs['loss']
        train_loss += loss
        model.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
        optimizer.step()
        scheduler.step()
    train_loss = float(train_loss) / len(train_loader)
    return train_loss


def train(train_loader, dev_loader, model, optimizer, scheduler, model_dir):
    best_val_acc = 0.0
    patience_counter = 0
    for epoch in range(1, config.epoch + 1):
        train_loss = train_epoch(train_loader, model, optimizer, scheduler, epoch)
        val_metrics = evaluate(dev_loader, model)

def evaluate(dev_loader, model):
    model.eval()
    dev_loss = 0.0
    pred, gold = [], []
    with torch.no_grad():
        for idx, batch_sample in enumerate(dev_loader):
            batch_data, batch_label = batch_sample
            batch_mask = batch_data.gt(0)
            outputs = model(batch_data, batch_mask, labels=batch_label)


if __name__ == '__main__':
    x = torch.zeros(2, 2)
    # logging.info('ok ok')
