import argparse
import json
import os

from loguru import logger


def convert_mme(mme_path):
    img_qa_list = []
    for root, dirs, files in os.walk(mme_path, topdown=False):
        for name in files:
            if name.endswith('.jpg') or name.endswith('.png'):
                img_path = os.path.join(root, name)
                img_path_tmp = img_path.split('/')
                img = os.path.join(img_path_tmp[-3], img_path_tmp[-2], name)
                txt_path = img_path[:-3] + 'txt'
                fp = open(txt_path, 'r')
                lines = fp.readlines()
                for line in lines:
                    question, answer = line.split('\t')
                    img_qa = {
                        'img': img.strip(),
                        'question': question.strip(),
                        'answer': answer.strip()
                    }
                    img_qa_list.append(img_qa)
    fp = open('img_qa.json', 'w')
    json.dump(img_qa_list, fp, indent=4)
    logger.info('img_qa.json is done. You need to move it to MME file folder.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mme_path', type=str, required=True)
    args = parser.parse_args()
    logger.info(f'args : {args}')
    convert_mme(args.mme_path)
