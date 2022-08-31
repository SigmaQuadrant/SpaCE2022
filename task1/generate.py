import jsonlines
import numpy as np
import torch

import config
import logging
from data_loader import SpaceDataset
from torch.utils.data import DataLoader
from transformers import ElectraForSequenceClassification


def generate():

    dataset = SpaceDataset(config.test_dir, config)
    logging.info('Dataset build!')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)
    logging.info('Dataloader build!')
    if config.model_dir is not None:
        model = ElectraForSequenceClassification.from_pretrained(config.model_dir, num_labels=2)
        model.to(config.device)
    # print(model.device)
    logging.info('Load the model {}'.format(config.bert_model + str(config.learning_rate) + str(config.seed)))
    logging.info('Test Beginning!')

    model.eval()
    pred_tags = []
    with torch.no_grad():
        for idx, batch_sample in enumerate(dataloader):
            batch_data = batch_sample[0]
            batch_mask = batch_data.gt(0)

            outputs = model(batch_data, batch_mask)
            batch_label = outputs.logits.detach().cpu().numpy()
            pred_tags.extend(np.argmax(batch_label, axis=-1))

    return pred_tags


if __name__ == '__main__':

    pred_tags = generate()
    with open(config.test_dir, 'r') as fr:
        items = []
        for idx, item in enumerate(jsonlines.Reader(fr)):
            context, judge = item['context'], item['judge']
            items.append({'context': context, 'judge': int(pred_tags[idx])})

    # print(items)
    with jsonlines.open(config.prediction_dir, 'w') as fw:
        fw.write_all(items)

