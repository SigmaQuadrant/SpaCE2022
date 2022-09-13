import jsonlines
import numpy as np
import torch

import config
import logging
from data_loader import SpaceDataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from modeling_cpt import CPTForConditionalGeneration
from utils import set_logger
from decode import list_to_tuple, text2idxes
from data_loader import special_token_dicts
from copy import deepcopy



def generate():
    set_logger()
    dataset = SpaceDataset(config.dev_dir, config, mode='test')
    logging.info('Dataset build!')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)
    tokenizer = BertTokenizer.from_pretrained(config.bert_model)
    tokenizer.add_special_tokens(special_tokens_dict=special_token_dicts)
    model = CPTForConditionalGeneration.from_pretrained(config.model_dir)
    model.resize_token_embeddings(len(tokenizer))
    model.to(config.device)
    logging.info('Load the model from {}'.format(config.model_dir))
    logging.info('Testing Beginning')
    model.eval()

    text, idxes = [], []

    with torch.no_grad():
        for idx, batch_sample in enumerate(dataloader):
            batch_data, batch_coref = batch_sample
            ids = model.generate(batch_data, num_beams=config.num_beams, do_sample=False, max_length=300)
            decoder_output = tokenizer.batch_decode(ids)
            batch_decoder_tuples = list_to_tuple(decoder_output)
            batch_text = [tokenizer.convert_ids_to_tokens(data) for data in batch_data]
            text.append(deepcopy(batch_decoder_tuples[0]))
            batch_idxes = text2idxes(batch_text, batch_decoder_tuples)
            idxes.append(batch_idxes[0])

    return text, idxes


if __name__ == '__main__':
    list_text, list_idxes = generate()
    with open(config.dev_dir, 'r') as fr:
        items = []
        for idx, item in enumerate(jsonlines.Reader(fr)):
            qid, context, coref = item['qid'], item['context'], item['corefs']
            text, idxes, outputs = list_text[idx], list_idxes[idx], []
            for text_triple, idx_triple in zip(text, idxes):
                answer_triple = []
                for element1, element2 in zip(text_triple, idx_triple):
                    if element1 is None or element2 is None:
                        answer_triple.append(None)
                    else:
                        answer_triple.append({'text': ''.join(element1), 'idxes': element2})
                outputs.append(answer_triple)
            items.append({'qid': qid, 'context': context, 'corefs': coref, 'outputs': outputs})

    # print(len(items))
    # print(items[:5])
    with jsonlines.open(config.prediction_dir, 'w') as fw:
        fw.write_all(items)
