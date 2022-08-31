import torch
import json
import argparse
from config import dev_dir, prediction_dir

'''
input: prediction and answer
output: result
'''
def get_metric(answer_path, prediction_path):

    answers, predictions = [], []

    with open(answer_path, 'r') as fin:
        for line in fin:
            answers.append(json.loads(line))
    with open(prediction_path, 'r') as fin:
        for line in fin:
            predictions.append(json.loads(line))

    if len(answers) != len(predictions):
        correct, total = 0, 0
        status, score = 'The length of predictions and answers not match!', 0
    else:
        correct, total = 0, len(answers)
        for x, y in zip(answers, predictions):
            if x['judge'] == y['judge']:
                correct += 1

        status, score = 'Accepted', correct/total

    print(status)
    print('Accuracy %d/%d = %f' %(correct, total, score))

    final_result = {
        'correct': correct,
        'total': total,
        'accuracy': score,
    }

    return status, final_result


if __name__ == '__main__':

    answer_path = dev_dir
    prediction_path = './ensemble_5.jsonl'
    get_metric(answer_path, prediction_path)

