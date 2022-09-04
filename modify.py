import os.path

import jsonlines

def check(span: list) -> bool:
    if span[-1] - span[0] + 1 != len(span):
        return False
    else:
        return True

def cal_f1(A: set, B: set):
    intersection = A & B
    precision = len(intersection) / len(A)
    recall = len(intersection) / len(B)
    if precision == 0 or recall == 0:
        return 0.0
    else:
        return 2 * precision * recall / (precision + recall) 

def change(span: list) -> list:
    maxf1 = float('-inf')
    L, R = None, None
    for l in range(span[0], span[-1] + 1):
        for r in range(l, span[-1] + 1):
            pred = set(i for i in range(l, r + 1))
            gold = set(span)
            cur_f1 = cal_f1(pred, gold)
            if cur_f1 > maxf1:
                maxf1 = cur_f1
                L, R = l, r
    return [i for i in range(L, R + 1)]
    
for task in ['subtask1', 'subtask2', 'subtask3']:
    route_path = os.path.join('dataset', 'jsonl', task)
    file_path = os.path.join(route_path, 'task2_train.jsonl')
    output_path = os.path.join(route_path, 'task2_train_modified.jsonl')

    with open(file_path, 'r') as fr:
        items = []
        for item in jsonlines.Reader(fr):
            qid, context, reasons = item['qid'], item['context'], item['reasons']
            for i, reason in enumerate(reasons):
                fragment, type = reason['fragments'], reason['type']
                for j, element in enumerate(fragment):
                    role, text, _ = element['role'], element['text'], element['idxes']
                    element['idxes'].sort()
                    # print(element['idxes'])
                    if not check(element['idxes']):
                        # print('exist!')
                        reasons[i]['fragments'][j]['idxes'] = change(element['idxes'])
            items.append({'qid': qid, 'context': context, 'reasons': reasons})

    with jsonlines.open(output_path, 'w') as fw:
        fw.write_all(items)


