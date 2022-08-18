import torch
import json
import argparse
from config import test_dir


def get_metric(params):

    answers, predictions = [], []

    with open(params['answer_path'], 'r') as fin:
        for line in fin:
            answers.append(json.loads(line))
    with open(params['prediction_path'], 'r') as fin:
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--answer_path', type=str, default=test_dir)
    parser.add_argument('--prediction_path', type=str, default=test_dir)
    # should be prediction_dir

    args = parser.parse_args().__dict__
    get_metric(args)

    ## to be continued
