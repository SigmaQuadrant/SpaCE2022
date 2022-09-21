import torch
import logging
import numpy as np
from torch import nn
from tqdm import tqdm
import config
from decode import list_to_tuple, text2idxes
from scipy import optimize


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
        # evaluate(dev_loader, model, tokenizer)

        val_metrics = evaluate(dev_loader, model, tokenizer)
        f1 = val_metrics['micro_f1']
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


def evaluate(dev_loader, model, tokenizer):
    model.eval()

    with torch.no_grad():
        precision, recall, f1 = [], [], []
        for idx, batch_sample in enumerate(dev_loader):
            batch_data, batch_label, coref = batch_sample
            ids = model.generate(batch_data, num_beams=config.num_beams, do_sample=False, max_length=300)
            decoder_output = tokenizer.batch_decode(ids)
            # print('len of raw: {}, len_of_tuples: {}'.format(len(decoder_output), len(batch_label)))
            batch_decoder_tuples = list_to_tuple(decoder_output)
            batch_text = [tokenizer.convert_ids_to_tokens(data) for data in batch_data]
            # print('len_of_text: {}, len_of_tuples: {}'.format(len(batch_decoder_tuples), len(batch_label)))
            assert(len(decoder_output) == len(batch_decoder_tuples))

            batch_idxes = text2idxes(batch_text, batch_decoder_tuples)
            p, r, f = get_metric(batch_idxes, batch_label, coref)
            precision.extend(p)
            recall.extend(r)
            f1.extend(f)

        total = 207
        # print('len of precision :{} type: {}'.format(len(precision), type(precision)))
        # print('len of recall :{}'.format(len(recall)))
        # print('len of f1 :{}'.format(len(f1)))
        # print('len of total :{}'.format(total))
        ave_precision = sum(precision) / total
        ave_recall = sum(recall) / total
        macro_f1 = sum(f1) / total
        micro_f1 = 2 * (ave_precision * ave_recall) / (ave_precision + ave_recall) if ave_precision + ave_recall != 0.0 else 0.0
        return {
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'ave_precision': ave_precision,
            'ave_recall': ave_recall
        }


def intersection_and_union(input, target):
    _input, _target = set(input), set(target)
    intersection = _input & _target
    union = _input | _target

    return len(intersection), len(union)


def cal_similarity(golden_tuple, predicted_tuple, corefs):
    if len(golden_tuple) != len(predicted_tuple):
        return 0
    non_null_pair = 0
    total_score = 0.0
    for i, (g_element, p_element) in enumerate(zip(golden_tuple, predicted_tuple)):
        if (g_element is None) and (p_element is None):
            continue
        non_null_pair += 1
        if (g_element is None) or (p_element is None):
            element_sim_score = 0
        else:
            if isinstance(g_element, str):
                if not isinstance(p_element, str):
                    element_sim_score = 0.0
                elif g_element != p_element:
                    element_sim_score = 0.0
                else:
                    element_sim_score = 1.0
            else:
                n_inter, n_union = intersection_and_union(p_element, g_element)
                element_sim_score = n_inter / n_union
                if ((i == 0) or (i == 1)):
                    n_inter, n_union = intersection_and_union(p_element, g_element)
                    element_sim_score = n_inter / n_union
                    g_idx_set = set(g_element)
                    for key in corefs:
                        key_idx_set = set(eval(key))
                        if (key_idx_set.issubset(g_idx_set)):
                            diff_set = g_idx_set - key_idx_set
                            for c in corefs[key]:
                                corefed_g_idx = set(c['idxes']) | diff_set
                                n_inter, n_union = intersection_and_union(p_element, corefed_g_idx)
                                element_sim_score = max(element_sim_score, n_inter / n_union)

                    print('Golden entity: ', g_element)
                    print('Predicted entity: ', p_element)
                    print('Score: ', element_sim_score)

        if ((i == 0) or (i == 1)) and (element_sim_score == 0):  # 关键实体（空间实体）不能完全错误
              return 0

        total_score += element_sim_score
    return total_score / non_null_pair


def KM_algorithm(pair_scores):
    row_ind, col_ind = optimize.linear_sum_assignment(-pair_scores)  # 求负将最大和转变为最小和
    max_score = pair_scores[row_ind, col_ind].sum()
    return max_score


def get_metric(batch_prediction, batch_label, corefs):
    precision, recall, f1 = [], [], []
    for prediction, label, coref in zip(batch_prediction, batch_label, corefs):
        # print(prediction, label, coref)
        N, M = len(prediction), len(label)
        pair_scores = np.zeros((M, N))
        coref_dict = {}
        for coref_set in coref:
            for coref_element in coref_set:
                idx_str = str(coref_element['idxes'])
                if idx_str not in coref_dict:
                    coref_dict[idx_str] = coref_set
        if N > 100:
            continue
        for i in range(M):
            for j in range(N):
                pair_scores[i][j] = cal_similarity(label[i], prediction[j], coref_dict)
        max_bipartite_score = KM_algorithm(pair_scores)
        _precision = max_bipartite_score / N
        _recall = max_bipartite_score / M
        if _precision + _recall == 0:
            _f1 = 0
        else:
            _f1 = 2 * (_precision * _recall) / (_precision + _recall)
        precision.append(_precision)
        recall.append(_recall)
        f1.append(_f1)
    return precision, recall, f1


