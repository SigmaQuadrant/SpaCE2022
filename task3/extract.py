import hanlp
import jsonlines

dev_dir = '../dataset/jsonl/task3_dev.jsonl'
HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)

recall = []
num_pred = []
with open(dev_dir, 'r') as f:
    for ind, item in enumerate(jsonlines.Reader(f)):

        srls = HanLP(item['context'], tasks='srl')['srl']
        prediction_set, golden_set = set(), set()
        srl_idxes = []
        for srl in srls:
            for role in srl:
                if role[1] in ['ARG0', 'ARG1']:
                    prediction_set.add(role[0])
                    start = item['context'].find(role[0])
                    srl_idxes.append((start, start + len(role[0]) - 1))
        for triple in item['outputs']:
            golden_set.add(triple[0]['text'])

        len_predict = len(prediction_set)
        len_golden = len(golden_set)
        intersection = 0
        for gold in golden_set:
            if any(set(gold) & set(predict) for predict in prediction_set):
                intersection += 1
        recall.append(intersection / len_golden)
        num_pred.append(len_predict)

    print('avg recall: {}'.format(sum(recall) / len(recall)))
    print('avg num_pred: {}'.format(sum(num_pred) / len(num_pred)))




