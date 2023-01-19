from collections import defaultdict


def parse_gold(gold):
    all_mentions = set()
    all_relations = defaultdict(set)
    for k in gold:
        if k != 'doc_key' and k != 'sentences':
            for pair in gold[k]:
                mention1 = (pair[0][0][0], pair[0][0][1])
                mention2 = (pair[1][0][0], pair[1][0][1])
                all_mentions.add(mention1)
                all_mentions.add(mention2)
                all_relations[k].add((mention1, mention2))
    return all_mentions, all_relations


def parse_pred(pred):
    all_mentions = set()
    all_relations = defaultdict(set)
    mention1_starts, mention1_ends, mention2_starts, mention2_ends, labels = pred
    for i, mention1_start in enumerate(mention1_starts):
        mention1 = (mention1_start, mention1_ends[i])
        mention2 = (mention2_starts[i], mention2_ends[i])
        all_mentions.add(mention1)
        all_mentions.add(mention2)
        all_relations[labels[i]].add((mention1, mention2))
    return all_mentions, all_relations


def eval_output(gold_path, preds):
    with open(gold_path) as fp:
        temp_gold = fp.readlines()
    sum_mention_p = 0
    sum_mention_r = 0
    sum_mention_f1 = 0
    sum_relation_p = 0
    sum_relation_r = 0
    sum_relation_f1 = 0
    all_gold = {}
    for line in temp_gold:
        cur_gold = eval(line)
        doc_key = cur_gold['doc_key']
        all_gold[doc_key] = parse_gold(cur_gold)
    for k in preds:
        mention_tp = 0
        relation_tp = 0
        gold_mentions, gold_relations = all_gold[k]
        pred_mentions, pred_relations = parse_pred(preds[k])
        for m in pred_mentions:
            if m in gold_mentions:
                mention_tp += 1
        for rel in pred_relations:
            for r1 in pred_relations[rel]:
                for r2 in gold_relations[rel]:
                    if r1[0] in r2 and r1[1] in r2:
                        relation_tp += 1
        mention_p = mention_tp / len(pred_mentions) if len(pred_mentions) else .0
        mention_r = mention_tp / len(gold_mentions) if len(gold_mentions) else .0
        mention_f1 = (2 * mention_p * mention_r) / (mention_p + mention_r) if (mention_p + mention_r) else .0
        sum_mention_p += mention_p
        sum_mention_r += mention_r
        sum_mention_f1 += mention_f1
        relation_p = relation_tp / (len(pred_relations) * len(pred_relations[list(pred_relations.keys())[0]])) if len(pred_relations) else .0
        relation_r = relation_tp / (len(gold_relations) * len(gold_relations[list(gold_relations.keys())[0]])) if len(gold_relations) else .0
        relation_f1 = (2 * relation_p * relation_r) / (relation_p + relation_r) if (relation_p + relation_r) else .0
        sum_relation_p += relation_p
        sum_relation_r += relation_r
        sum_relation_f1 += relation_f1
    summary_dict = {"Average Mention Precision": sum_mention_p / len(preds),
                    "Average Mention Recall": sum_mention_r / len(preds),
                    "Average Mention F1": sum_mention_f1 / len(preds),
                    "Average Relation Precision": sum_relation_p / len(preds),
                    "Average Relation Recall": sum_relation_r / len(preds),
                    "Average Relation F1": sum_relation_f1 / len(preds)}
    return summary_dict, summary_dict["Average Relation F1"]


def eval_output_overlap(gold_path, preds):
    with open(gold_path) as fp:
        temp_gold = fp.readlines()
    sum_mention_p = 0
    sum_mention_r = 0
    sum_mention_f1 = 0
    sum_relation_p = 0
    sum_relation_r = 0
    sum_relation_f1 = 0
    all_gold = {}
    for line in temp_gold:
        cur_gold = eval(line)
        doc_key = cur_gold['doc_key']
        all_gold[doc_key] = parse_gold(cur_gold)
    for k in preds:
        mention_tp = 0
        relation_tp = 0
        gold_mentions, gold_relations = all_gold[k]
        pred_mentions, pred_relations = parse_pred(preds[k])
        for m in pred_mentions:
            if m in gold_mentions:
                mention_tp += 1
        for rel in pred_relations:
            for r1 in pred_relations[rel]:
                for r2 in gold_relations[rel]:
                    if r1[0][0] in list(range(r2[0][0], r2[0][1])) or r1[0][1] in list(range(r2[0][0], r2[0][1])):
                        if r1[1][0] in list(range(r2[1][0], r2[1][1])) or r1[1][1] in list(range(r2[1][0], r2[1][1])):
                            relation_tp += 1
                    if r1[0][0] in list(range(r2[1][0], r2[1][1])) or r1[0][1] in list(range(r2[1][0], r2[1][1])):
                        if r1[1][0] in list(range(r2[0][0], r2[0][1])) or r1[1][1] in list(range(r2[0][0], r2[0][1])):
                            relation_tp += 1
        mention_p = mention_tp / len(pred_mentions) if len(pred_mentions) else .0
        mention_r = mention_tp / len(gold_mentions) if len(gold_mentions) else .0
        mention_f1 = (2 * mention_p * mention_r) / (mention_p + mention_r) if (mention_p + mention_r) else .0
        sum_mention_p += mention_p
        sum_mention_r += mention_r
        sum_mention_f1 += mention_f1
        relation_p = relation_tp / (len(pred_relations) * len(pred_relations[list(pred_relations.keys())[0]])) if len(pred_relations) else .0
        relation_r = relation_tp / (len(gold_relations) * len(gold_relations[list(gold_relations.keys())[0]])) if len(gold_relations) else .0
        relation_f1 = (2 * relation_p * relation_r) / (relation_p + relation_r) if (relation_p + relation_r) else .0
        sum_relation_p += relation_p
        sum_relation_r += relation_r
        sum_relation_f1 += relation_f1
    summary_dict = {"Average Mention Precision": sum_mention_p / len(preds),
                    "Average Mention Recall": sum_mention_r / len(preds),
                    "Average Mention F1": sum_mention_f1 / len(preds),
                    "Average Relation Precision": sum_relation_p / len(preds),
                    "Average Relation Recall": sum_relation_r / len(preds),
                    "Average Relation F1": sum_relation_f1 / len(preds)}
    return summary_dict, summary_dict["Average Relation F1"]


if __name__ == '__main__':
    with open('/home/bingyang/projects/coref-under-transformation/data/r2vq_cutler_120/coref_resolution_data/train.jsonl') as fp:
        temp_gold = fp.readlines()
    preds_rel = {'r-4130': [[5, 17, 18, 20, 30, 31, 41, 48, 50, 52, 70, 80, 87, 98, 100, 102, 104, 111], [5, 17, 18, 20, 30, 32, 41, 48, 50, 52, 70, 80, 87, 98, 100, 102, 104, 111], [2, 14, 14, 14, 27, 27, 38, 45, 45, 45, 67, 77, 84, 95, 95, 95, 95, 108], [2, 14, 14, 14, 27, 27, 38, 45, 45, 45, 67, 77, 84, 95, 95, 95, 95, 108], ['cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut']], 'f-BMZKXRGQ': [[6, 21, 24, 27, 30, 32, 36, 48, 55, 72, 80, 108, 128, 132, 134], [7, 22, 25, 28, 30, 33, 37, 48, 55, 73, 80, 108, 128, 132, 134], [2, 17, 17, 17, 17, 17, 17, 45, 52, 61, 77, 105, 125, 125, 125], [2, 17, 17, 17, 17, 17, 17, 45, 52, 61, 77, 105, 125, 125, 125], ['cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut']], 'f-P4J28HGK': [[5, 13, 25, 36, 38, 46, 48, 51, 58, 71], [6, 13, 25, 36, 39, 46, 49, 51, 58, 71], [2, 10, 22, 33, 33, 43, 43, 43, 55, 68], [2, 10, 22, 33, 33, 43, 43, 43, 55, 68], ['cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut']], 'r-1671': [[6, 16, 18, 25, 27, 30, 38, 45, 49, 55, 62, 74, 83, 93], [6, 16, 18, 25, 27, 31, 38, 45, 49, 55, 62, 74, 83, 93], [2, 13, 13, 22, 22, 22, 35, 42, 42, 42, 59, 71, 80, 90], [2, 13, 13, 22, 22, 22, 35, 42, 42, 42, 59, 71, 80, 90], ['cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut', 'cut']]}
    preds_coref = {'r-4130': [[], [], [], [], []], 'f-BMZKXRGQ': [[55, 80, 108, 128], [55, 80, 108, 128], [17, 61, 99, 105], [17, 61, 99, 105], ['Coreference', 'Coreference', 'Coreference', 'Coreference']], 'f-P4J28HGK': [[71], [71], [55], [55], ['Coreference']], 'r-1671': [[16, 45, 62, 74, 93], [16, 45, 62, 74, 93], [2, 35, 42, 59, 80], [2, 35, 42, 59, 80], ['Coreference', 'Coreference', 'Coreference', 'Coreference', 'Coreference']]}
    print(eval_output('/home/bingyang/projects/coref-under-transformation/data/r2vq_cutler_120/coref_resolution_data/train.jsonl', preds_rel))
    print(eval_output('/home/bingyang/projects/coref-under-transformation/data/r2vq_cutler_120/coref_resolution_data/train.jsonl', preds_coref))

