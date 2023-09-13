import numpy as np
import json
import os
import argparse
import text_helper
from pycocotools.coco import COCO
from collections import defaultdict
from PythonHelperTools.vqaTools.vqa import VQA


def extract_answers(q_answers, valid_answer_set):
    all_answers = [answer["answer"] for answer in q_answers]
    valid_answers = [a for a in all_answers if a in valid_answer_set]
    return all_answers, valid_answers


def vqa_processing(image_dir, annotation_file, valid_answer_set, image_set):
    print('building vqa %s dataset' % image_set)

    annFile = annotation_file % image_set
    annImage_dir = image_dir%image_set


    with open(annFile,encoding='UTF8') as f:
        data = json.load(f)
        unk_ans_count = 0
        dataset = [None] * len(data)
        for n_q, q in enumerate(data):
            if (n_q + 1) % 10000 == 0:
                print('processing %d / %d' % (n_q + 1, len(data)))
            image_name = q['image']
            image_path = annImage_dir+image_name
            question_str = q['question']
            question_tokens = text_helper.tokenize(question_str)
            iminfo = dict(image_name=image_name,
                          image_path=image_path,
                          question_str=question_str,
                          question_tokens=question_tokens)

            if image_set!='test':
                all_answers, valid_answers = extract_answers(q['answers'], valid_answer_set)
                if len(valid_answers)==0:
                    valid_answers = ['<unk>']
                    unk_ans_count += 1

                iminfo['all_answers'] = all_answers
                iminfo['valid_answers'] = valid_answers

            dataset[n_q] = iminfo


    print('total %d out of %d answers are <unk>' % (unk_ans_count, len(dataset)))
    return dataset


def main():
    image_dir = 'D:/data/vqa/vizwiz/visual_question_answering' + '/Resize_img/%s/'
    annotation_file = 'D:/data/vqa/vizwiz/visual_question_answering' + '/Annotations/%s.json'
    #question_file = 'D:/data/vqa/vizwiz/visual_question_answering' + '/Questions/v2_OpenEnded_mscoco_%s_questions.json'

    vocab_answer_file = 'D:/data/vqa/vizwiz/visual_question_answering' + '/vocab_answers.txt'
    answer_dict = text_helper.VocabDict(vocab_answer_file)
    valid_answer_set = set(answer_dict.word_list)

    test = vqa_processing(image_dir, annotation_file, valid_answer_set, 'test')
    train = vqa_processing(image_dir, annotation_file, valid_answer_set, 'train')
    val = vqa_processing(image_dir, annotation_file, valid_answer_set, 'val')


    np.save('D:/data/vqa/vizwiz/visual_question_answering' + '/train.npy', np.array(train))
    np.save('D:/data/vqa/vizwiz/visual_question_answering' + '/valid.npy', np.array(val))
    np.save('D:/data/vqa/vizwiz/visual_question_answering' + '/train_valid.npy', np.array(train + val))
    np.save('D:/data/vqa/vizwiz/visual_question_answering' + '/test.npy', np.array(test))


if __name__ == '__main__':

    main()
