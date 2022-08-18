import config
import logging

from utils import set_logger
from data_loader import SpaceDataset
from torch.utils.data import DataLoader
from model import DebertaReader
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from train import train


def run():
    set_logger()
    logging.info('device: {}'.format(config.device))
    train_dataset = SpaceDataset(config.train_dir, config)
    dev_dataset = SpaceDataset(config.dev_dir, config)
    logging.info('Dataset Bulid!')

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=config.shuffle, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=config.shuffle, collate_fn=dev_dataset.collate_fn)
    logging.info('Get DataLoader!')
    model = DebertaReader.from_pretrained(config.bert_model)
    model.to(config.device)
    logging.info('Load Model Form {}'.format(config.bert_model))

    train_steps_per_epoch = len(train_dataset) // config.batch_size
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                            num_warmup_steps=train_steps_per_epoch,
                                            num_training_steps=train_steps_per_epoch * config.epoch)
    logging.info('Starting Training')
    train(train_loader, dev_loader, model, optimizer, scheduler, config.model_dir)


if __name__ == '__main__':
    run()
