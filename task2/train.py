import torch
import logging
import numpy as np
from torch import nn
from tqdm import tqdm
import config
import copy


def train_epoch(train_loader, model, optimizer, scheduler, epoch):

    model.train()
    train_loss = 0.0
    for idx, batch_sample in enumerate(tqdm(train_loader)):
        batch_data, batch_label = batch_sample
        batch_mask = batch_data.gt(0)

        # pred = model(batch_data, batch_mask)
        outputs = model(batch_data, batch_mask, labels=batch_label)
        loss = outputs['loss']
        train_loss += loss
        model.zero_grad()
        loss.backward()
        # is it necessary to use the clip_grad_norm ?
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
        optimizer.step()
        scheduler.step()
    train_loss = float(train_loss) / len(train_loader)
    return train_loss


def train(train_loader, dev_loader, model, optimizer, scheduler, model_dir):
    best_f1 = 0.0
    patience_counter = 0
    for epoch in range(1, config.epoch + 1):

        train_loss = train_epoch(train_loader, model, optimizer, scheduler, epoch)
        val_metrics = evaluate(dev_loader, model)
        val_f1 = val_metrics['F1']
        logging.info('Epoch: {}, Train_loss: {:6f}, F1: {:.6f}'.format(epoch, train_loss, val_f1))

        if val_f1 > best_f1:
            model.save_pretrained(model_dir)
            logging.info('Save best model!')
            if val_f1 - best_f1 < config.patience:
                patience_counter += 1
            else:
                patience_counter = 0
            best_f1 = val_f1
        else:
            patience_counter += 1
        if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch:
            logging.info('Best f1: {}'.format(best_f1))
            break

    logging.info('Training Finished!')
     # logging.info('patience_counter: {}'.format(patience_counter))


def score_f1(A, B):
    if A[1] < A[0] or A[3] < A[2]:
        return 0.0, 0.0, 0.0
    _A = set([i for i in range(A[0], A[1] + 1)] + [i for i in range(A[2], A[3] + 1)])
    _B = set([i for i in range(B[0], B[1] + 1)] + [i for i in range(B[2], B[3] + 1)])
    _intersection = _A & _B
    precision = len(_intersection) / len(_A)
    recall = len(_intersection) / len(_B)
    if precision == 0.0 or recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate(dev_loader, model):
    model.eval()
    dev_loss = 0.0
    P, R, F1 = 0.0, 0.0, 0.0
    with torch.no_grad():
        for idx, batch_sample in enumerate(tqdm(dev_loader)):

            batch_data, batch_label = batch_sample
            batch_mask = batch_data.gt(0)

            outputs = model(batch_data, batch_mask, labels=batch_label[:, :4])
            dev_loss += outputs['loss'].item()

            # how to decode
            A_start = torch.argmax(outputs['A_start_logits'], dim=-1, keepdim=True)
            A_end = torch.argmax(outputs['A_end_logits'], dim=-1, keepdim=True)
            B_start = torch.argmax(outputs['B_start_logits'], dim=-1, keepdim=True)
            B_end = torch.argmax(outputs['B_end_logits'], dim=-1, keepdim=True)
            postion = torch.cat((A_start, A_end, B_start, B_end), dim=-1)
            sum_P, sum_R, sum_F1 = 0.0, 0.0, 0.0
            for B in range(batch_data.size(0)):
                if len(batch_label[B]) == 4 or sum(batch_label[B][4:8] == -1).item() == 4:
                    p, r, f1 = score_f1(postion[B], batch_label[B][0:4])
                    sum_P , sum_R, sum_F1 = sum_P + p, sum_R + r, sum_F1 + f1
                else:
                    # choose the best span
                    p1, r1, f1 = score_f1(postion[B], batch_label[B][0:4])
                    p2, r2, f2 = score_f1(postion[B], batch_label[B][4:8])
                    if f1 > f2:
                        sum_P, sum_R, sum_F1 = sum_P + p1, sum_R + r1, sum_F1 + f1
                    else:
                        sum_P, sum_R, sum_F1 = sum_P + p2, sum_R + r2, sum_F1 + f2

            sum_P = sum_P / batch_data.size(0)
            sum_R = sum_R / batch_data.size(0)
            sum_F1 = sum_F1 / batch_data.size(0)
            P, R, F1 = P + sum_P, R + sum_R, F1 + sum_F1

            '''
            for B in range(batch_data.size(0)):

                A_start = outputs['A_start_logits'][B]
                A_end = outputs['A_end_logits'][B]
                B_start = outputs['B_start_logits'][B]
                B_end = outputs['B_end_logits'][B]

                # max P(A,B) + P(C,D) ?
                # max P(A,B) * P(C,D) ?
                P_max = 0.0
                Best_postion = []
                len_sentence = batch_data.size(1)

                for A_l in range(len_sentence):
                    for A_r in range(A_l, min(len_sentence, A_l + 6)):
                        for B_l in range(A_r, len_sentence):
                            for B_r in range(B_l, min(len_sentence, B_l + 6)):
                                if A_start[A_l] * A_end[A_r] + B_start[B_l] * B_end[B_r] > P_max:
                                    P_max = A_start[A_l] * A_end[A_r] + B_start[B_l] * B_end[B_r]
                                    Best_postion = copy.deepcopy([A_l, A_r, B_l, B_r])

                p, r, f = score_f1(Best_postion, batch_label[B])
                sum_p, sum_r, sum_f1 = sum_p + p, sum_r + r, sum_f1 + f
            ave_p, ave_r, ave_f1 = sum_p / batch_data.size(0), sum_r / batch_data.size(0), sum_f1 / batch_data.size(0)
            P, R, F = P + ave_p, R + ave_r, F + ave_f1
        P, R, F = P / len(dev_loader), R / len(dev_loader), F / len(dev_loader)
        '''
    return {
        # fix the length of dev dataset
        'precision': P / len(dev_loader),
        'recall': R / len(dev_loader),
        'F1': F1 / len(dev_loader),
        'dev_loss': dev_loss
    }


if __name__ == '__main__':
    A = [1, 3, 5, 8]
    B = [1, 4, 5, 6]

    print(score_f1(A,B))






