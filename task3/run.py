import config
import logging

from utils import set_logger, set_seed
from data_loader import SpaceDataset, special_token_dicts
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BertTokenizer
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from train import train


def run():
    set_logger()
    set_seed(config.seed)
    logging.info('device: {}'.format(config.device))
    train_dataset = SpaceDataset(config.train_dir, config, mode='train')
    dev_dataset = SpaceDataset(config.dev_dir, config, mode='dev')
    logging.info('Dataset Build!')

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=config.shuffle, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=dev_dataset.collate_fn)
    logging.info('Get DataLoader!')

    tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=config.bert_cased)
    tokenizer.add_special_tokens(special_tokens_dict=special_token_dicts)
    model = BartForConditionalGeneration.from_pretrained(config.bert_model)
    model.resize_token_embeddings(len(tokenizer))
    model.to(config.device)
    logging.info('Load Model From {}'.format(config.bert_model))

    train_steps_per_epoch = len(train_dataset) // config.batch_size
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=train_steps_per_epoch, num_training_steps=train_steps_per_epoch * config.epoch)
    logging.info('Starting Training')
    train(train_loader, dev_loader, model, optimizer, scheduler, config.model_dir, tokenizer)


if __name__ == '__main__':
    run()
