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
        val_acc = val_metrics['accuracy']
        logging.info('Epoch: {}, Train_loss: {:.6f} Accuracy: {:.6f}'.format(epoch, train_loss, val_acc))

        if val_acc > best_val_acc:
            model.save_pretrained(model_dir)
            logging.info('Save best model!')
            if val_acc - best_val_acc < config.patience:
                patience_counter += 1
            else:
                patience_counter = 0
            best_val_acc = val_acc
        else:
            patience_counter += 1
        if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch:
            logging.info('Best val accuracy: {}'.format(best_val_acc))
            break
        logging.info('Training Finished!')
        # logging.info('patience_counter: {}'.format(patience_counter))


def evaluate(dev_loader, model):
    model.eval()
    dev_loss = 0.0
    pred, gold = [], []
    with torch.no_grad():
        for idx, batch_sample in enumerate(dev_loader):
            batch_data, batch_label = batch_sample
            batch_mask = batch_data.gt(0)
            outputs = model(batch_data, batch_mask)
            # dev_loss += outputs['loss'].item()
            batch_output = outputs['logits']
            batch_output = batch_output.detach().cpu().numpy()
            batch_gold = batch_label.to('cpu').numpy()
            pred.extend(np.argmax(batch_output, axis=-1))
            gold.extend(batch_gold)

    assert len(pred) == len(gold)
    correct, total = 0, len(pred)

    for idx in range(0, total):
        if pred[idx] == gold[idx]:
            correct += 1

    final_result = {
        'accuracy': correct / total,
        'dev_loss': dev_loss
    }
    return final_result