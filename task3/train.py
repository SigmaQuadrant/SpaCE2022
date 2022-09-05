import torch
import logging
import numpy as np
from torch import nn
from tqdm import tqdm
import config
from transformers import BertTokenizer
from data_loader import special_token_dicts


def train_epoch(train_loader, model, optimizer, scheduler, epoch):
    model.train()
    train_loss = 0.0

    for idx, batch_sample in enumerate(tqdm(train_loader)):
        batch_data, batch_label, coref = batch_sample
        batch_mask = batch_data.gt(0)
        outputs = model(input_ids=batch_data,
                        attention_mask=batch_mask,
                        labels=batch_label)
        loss = outputs['loss']
        train_loss += loss
        model.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
        optimizer.step()
        scheduler.step()
    train_loss = float(train_loss) / len(train_loader)
    return train_loss


def train(train_loader, dev_loader, model, optimizer, scheduler, model_dir, tokenizer):
    best_f1 = 0.0
    patience_counter = 0
    for epoch in range(1, config.epoch + 1):
        train_loss = train_epoch(train_loader, model, optimizer, scheduler, epoch)
        logging.info('Epoch: {}, Train_loss: {:.6f}'.format(epoch, train_loss))
        evaluate(dev_loader, model, tokenizer)
        '''
        val_metrics = evaluate(dev_loader, model)
        f1 = val_metrics['f1']
        logging.info('Epoch: {}, Train_loss: {:.6f} f1: {:.6f}'.format(epoch, train_loss, f1))

        if f1 > best_f1:
            model.save_pretrained(model_dir)
            logging.info('Save best model!')
            if f1 - best_f1 < config.patience:
                patience_counter += 1
            else:
                patience_counter = 0
            best_f1 = f1
        else:
            patience_counter += 1
            if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch:
                logging.info('Best val accuracy: {}'.format(best_f1))
                break
            logging.info('Training Finished!')
        '''

def evaluate(dev_loader, model, tokenizer):
    model.eval()
    dev_loss = 0.0

    with torch.no_grad():
        for idx, batch_sample in enumerate(dev_loader):
            if idx == 1:
                break
            batch_data, batch_label, coref = batch_sample
            ids = model.generate(batch_data, num_beams=1, do_sample=False, max_length=200)
            decoder_output = tokenizer.batch_decode(ids)
            # batch_data batch_label
            print(decoder_output)
            print(type(decoder_output))


