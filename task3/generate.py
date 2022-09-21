import jsonlines
import hanlp
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


def search(context, main_body, element):
    if not element['idxes']:
        return None
    cur_position = element['idxes'][-1]
    distance = float('inf')
    idxes = None
    for span in main_body:
        if span[1] < cur_position and cur_position - span[1] < distance:
            distance = cur_position - span[1]
            idxes = [_ for _ in range(span[0], span[1] + 1)]
        if span[1] > cur_position and span[1] - cur_position < distance:
            distance = span[1] - cur_position
            idxes = [_ for _ in range(span[0], span[1] + 1)]

    return_value = None
    if idxes is not None:
        text = context[idxes[0]: idxes[-1] + 1]
        return_value = {'text': text, 'idxes': idxes}

    return return_value


if __name__ == '__main__':
    list_text, list_idxes = generate()

    HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
    with open(config.dev_dir, 'r') as fr:
        items = []
        for idx, item in enumerate(jsonlines.Reader(fr)):
            qid, context, coref = item['qid'], item['context'], item['corefs']
            srls = HanLP(context, tasks='srl')['srl']
            # find the main_body by Hanlp
            main_body = []
            for srl in srls:
                for role in srl:
                    if role[1] in ['ARG0', 'ARG1']:
                        start = context.find(role[0])
                        main_body.append((start, start + len(role[0]) - 1))

            text, idxes, outputs = list_text[idx], list_idxes[idx], []
            for text_triple, idx_triple in zip(text, idxes):
                answer_triple = []
                for element1, element2 in zip(text_triple, idx_triple):
                    if element1 is None or element2 is None:
                        answer_triple.append(None)
                    else:
                        answer_triple.append({'text': ''.join(element1), 'idxes': element2})
                if answer_triple[0] is None:
                    continue
                if answer_triple[0]:
                    outputs.append(answer_triple)

            items.append({'qid': qid, 'context': context, 'corefs': coref, 'outputs': outputs})

    with jsonlines.open(config.prediction_dir, 'w') as fw:
        fw.write_all(items)
