import gc
import json
import os
from collections import defaultdict

import torch
from loguru import logger
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score)


class VLMEval:
    def __init__(self, config):
        self.eval_config = config.eval
        self.dataset = self.eval_config['name']
        assert self.dataset in [
            'MME',
        ], 'VLM eval only support MME dataset now.'
        self.eval_dataset_path = self.eval_config['path']
        self.eval_bs = self.eval_config['bs']
        self.output_include_input = True
        if self.dataset == 'MME':
            self.img_qas = self.load_mme()
        self.patch_datasets(config.model.type)
        logger.info('VLMEval load dataset done.')

    def load_mme(self):
        img_qa_json = os.path.join(self.eval_dataset_path, 'img_qa.json')
        fp = open(img_qa_json)
        img_qas = json.load(fp)
        for idx in range(len(img_qas)):
            img_qas[idx]['img'] = os.path.join(
                self.eval_dataset_path, img_qas[idx]['img']
            )
        return img_qas

    def patch_datasets(self, model_type):
        if model_type == 'InternVL2':
            self.output_include_input = False
        elif model_type == 'Llava':
            self.output_include_input = True

    def eval(self, model):
        vlm_model = model.vlm_model
        vlm_tokenizer = model.get_tokenizer()
        vlm_model.cuda()
        results = []
        logger.info(f'len(self.img_qas): {len(self.img_qas)}')
        logger.info(f'eval_bs: {self.eval_bs}')
        for idx in range(0, len(self.img_qas), self.eval_bs):
            logger.info(
                f'index : {(idx + 1) // self.eval_bs}/{len(self.img_qas) // self.eval_bs}'
            )
            start = idx
            end = min(idx + self.eval_bs, len(self.img_qas))
            batch_samples = self.img_qas[start:end]
            inputs = model.batch_process(batch_samples)
            inputs = {
                k: (
                    v.to(next(vlm_model.parameters()).device)
                    if torch.is_tensor(v)
                    else v
                )
                for k, v in inputs.items()
            }
            outputs = vlm_model.generate(**inputs, max_new_tokens=32, do_sample=False)
            if self.output_include_input:
                gen_txts = vlm_tokenizer.batch_decode(
                    outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True
                )
            else:
                gen_txts = vlm_tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
            for n in range(len(batch_samples)):
                result = batch_samples[n].copy()
                result.update({'gen_txt': gen_txts[n]})
                results.append(result)
        if self.dataset == 'MME':
            eval_class = MME()
        vlm_score = eval_class(results)

        vlm_model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        return vlm_score


class MME:
    def __init__(self):
        self.eval_type_dict = {
            'Perception': [
                'existence',
                'count',
                'position',
                'color',
                'posters',
                'celebrity',
                'scene',
                'landmark',
                'artwork',
                'OCR',
            ],
            'Cognition': [
                'commonsense_reasoning',
                'numerical_calculation',
                'text_translation',
                'code_reasoning',
            ],
        }

    def divide_chunks(self, lines, n=2):
        # looping till length lines
        for i in range(0, len(lines), n):
            yield lines[i: i + n]

        return

    def parse_pred_ans(self, pred_ans):
        pred_label = None
        if pred_ans in ['yes', 'no']:
            pred_label = pred_ans
        else:
            prefix_pred_ans = pred_ans[:4]

            if 'yes' in prefix_pred_ans:
                pred_label = 'yes'
            elif 'no' in prefix_pred_ans:
                pred_label = 'no'
            else:
                pred_label = 'other'

        return pred_label

    def compute_metric(self, gts, preds):
        assert len(gts) == len(preds)

        label_map = {
            'yes': 1,
            'no': 0,
            'other': -1,
        }

        gts = [label_map[x] for x in gts]
        preds = [label_map[x] for x in preds]

        acc = accuracy_score(gts, preds)

        clean_gts = []
        clean_preds = []
        other_num = 0
        for gt, pred in zip(gts, preds):
            if pred == -1:
                other_num += 1
                continue
            clean_gts.append(gt)
            clean_preds.append(pred)

        conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1, 0])
        precision = precision_score(clean_gts, clean_preds, average='binary')
        recall = recall_score(clean_gts, clean_preds, average='binary')
        tp, fn = conf_mat[0]
        fp, tn = conf_mat[1]

        metric_dict = dict()
        metric_dict = {
            'TP': tp,
            'FN': fn,
            'TN': tn,
            'FP': fp,
            'precision': precision,
            'recall': recall,
            'other_num': other_num,
            'acc': acc,
        }

        return metric_dict

    def get_lines(self, results):
        lines_dict = defaultdict(list)
        for res in results:
            task_name = res['img'].split('/')[-2]
            assert (
                task_name in self.eval_type_dict['Perception']
                or task_name in self.eval_type_dict['Cognition']
            )
            txt = (
                res['img'].split('/')[-1]
                + '\t'
                + res['question']
                + '\t'
                + res['answer']
                + '\t'
                + res['gen_txt']
                + '\n'
            )
            lines_dict[task_name].append(txt)
        return lines_dict

    def __call__(self, results):
        lines_dict = self.get_lines(results)
        mme_scores = {}
        for eval_type, task_name_list in self.eval_type_dict.items():
            mme_scores[eval_type] = {}

            scores = 0
            task_score_dict = dict()

            for task_name in task_name_list:
                lines = lines_dict[task_name]
                chunk_lines = list(
                    self.divide_chunks(lines)
                )  # one image corresponds to two questions

                img_num = len(chunk_lines)
                task_other_ans_num = 0
                task_score = 0
                acc_plus_correct_num = 0
                gts = []
                preds = []

                for img_items in chunk_lines:
                    assert len(img_items) == 2
                    img_correct_num = 0

                    for img_item in img_items:
                        img_name, question, gt_ans, pred_ans = img_item.split('\t')

                        gt_ans = gt_ans.lower()
                        pred_ans = pred_ans.lower()

                        assert gt_ans in ['yes', 'no']  # gt can only be yes or no.

                        pred_ans = self.parse_pred_ans(pred_ans)
                        assert pred_ans in ['yes', 'no', 'other']

                        gts.append(gt_ans)
                        preds.append(pred_ans)

                        if gt_ans == pred_ans:
                            img_correct_num += 1

                        if pred_ans not in ['yes', 'no']:
                            task_other_ans_num += 1

                    if img_correct_num == 2:
                        acc_plus_correct_num += 1

                # cal TP precision acc, etc.
                metric_dict = self.compute_metric(gts, preds)
                acc_plus = acc_plus_correct_num / img_num
                metric_dict['acc_plus'] = acc_plus

                for k, v in metric_dict.items():
                    if k in ['acc', 'acc_plus']:
                        task_score += v * 100

                task_score_dict[task_name] = task_score

                scores += task_score

            mme_scores[eval_type]['total_score'] = scores
            for task_name, score in task_score_dict.items():
                mme_scores[eval_type][task_name] = score

        return json.dumps(mme_scores, ensure_ascii=False, indent=4)
