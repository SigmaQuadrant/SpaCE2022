import os.path
import config
import jsonlines

# ensemble 3/5/7
seed = [42, 40, 48, 41, 49, 43, 46]
ensemble_3 = seed[:3]
ensemble_5 = seed[:5]
ensemble_7 = seed[:7]

path = './experiments/electra-large_lr_1.2e-05_bsz_16sd'
ensemble_result = [0] * 3152
for seed_idx in ensemble_5:
    cur_path = path + str(seed_idx) + '/prediction.jsonl'
    with open(cur_path, 'r') as fr:
        for idx, item in enumerate(jsonlines.Reader(fr)):
            _, judge = item['context'], item['judge']
            if judge == 1:
                ensemble_result[idx] = ensemble_result[idx] + 1

ensemble_result = [1 if v >= 3 else 0 for v in ensemble_result]

with open(config.test_dir, 'r') as fr:
    items = []
    for idx, item in enumerate(jsonlines.Reader(fr)):
        qid, context = item['qid'], item['context']
        items.append({'qid': qid, 'context': context, 'judge': int(ensemble_result[idx])})

with jsonlines.open('./ensemble_5.jsonl', 'w') as fw:
    fw.write_all(items)


