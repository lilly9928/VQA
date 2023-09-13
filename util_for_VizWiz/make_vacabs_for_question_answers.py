import os
import argparse
import numpy as np
import json
import re
from collections import defaultdict
from pycocotools.coco import COCO
import string


def make_vocab_questions(input_dir):
    """Make dictionary for questions and save them into text file."""
    vocab_set = set()
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
    question_length = []
    datasets = os.listdir(input_dir+'/Annotations')
    for dataset in datasets:
        with open(input_dir + '/' + dataset, encoding='UTF8') as f:
            questions = json.load(f)
        set_question_length = [None] * len(questions)
        for iquestion, question in enumerate(questions):
            words = SENTENCE_SPLIT_REGEX.split(question['question'].lower())
            words = [w.strip() for w in words if len(w.strip()) > 0]
            vocab_set.update(words)
            set_question_length[iquestion] = len(words)
        question_length += set_question_length

    vocab_list = list(vocab_set)
    vocab_list.sort()
    vocab_list.insert(0, '<pad>')
    vocab_list.insert(1, '<unk>')

    with open(input_dir+'/vocab_questions.txt', 'w') as f:
        f.writelines([w + '\n' for w in vocab_list])

    print('Make vocabulary for questions')
    print('The number of total words of questions: %d' % len(vocab_set))
    print('Maximum length of question: %d' % np.max(question_length))

def make_vocab_answers(input_dir):
    """Make dictionary for questions and save them into text file."""
    vocab_set = set()
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
    question_length = []
    datasets = os.listdir(input_dir)
    for dataset in datasets:
        with open(input_dir + '/' + dataset, encoding='UTF8') as f:
            answers = json.load(f)
            # print(len(answers))
        set_data_length = [None] * len(answers)
        idx = 0
        for i, answers in enumerate(answers):
            # print(answers)
            for j, answer in enumerate(answers['answers']):
                words = SENTENCE_SPLIT_REGEX.split(answer['answer'].lower())
                # print(words)
                words = [w.strip() for w in words if len(w.strip()) > 0]
                vocab_set.update(words)
                # set_data_length[idx] = len(words)
                idx = idx + 1
            # question_length += set_data_length

    vocab_list = list(vocab_set)
    vocab_list.sort()
    vocab_list.insert(0, '<pad>')
    vocab_list.insert(1, '<unk>')

    with open(input_dir+'/vocab_answers.txt', 'w', encoding='UTF8') as f:
        f.writelines([w + '\n' for w in vocab_list])
    #
    # print('Make vocabulary for questions')
    # print('The number of total words of questions: %d' % len(vocab_set))
    # print('Maximum length of question: %d' % np.max(question_length))


# def make_vocab_answers(input_dir, n_answers):
#     """Make dictionary for top n answers and save them into text file."""
#     answers = defaultdict(lambda: 0)
#     datasets = os.listdir(input_dir)
#     for dataset in datasets:
#         with open(input_dir + '/' + dataset) as f:
#             annotations = json.load(f)['annotations']
#         for annotation in annotations:
#             for answer in annotation['answers']:
#                 word = answer['answer']
#                 if re.search(r"[^\w\s]", word):
#                     continue
#                 answers[word] += 1
#
#     answers = sorted(answers, key=answers.get, reverse=True)
#     assert ('<unk>' not in answers)
#     top_answers = ['<unk>'] + answers[:n_answers - 1]  # '-1' is due to '<unk>'
#
#     with open('D:/data/vqa/coco/simple_vqa/vocab_answers.txt', 'w') as f:
#         f.writelines([w + '\n' for w in top_answers])
#
#     print('Make vocabulary for answers')
#     print('The number of total words of answers: %d' % len(answers))
#     print('Keep top %d answers into vocab' % n_answers)


def make_vocab_caption(input_dir):
    """Make dictionary for questions and save them into text file."""

    datasets = ['train2014','val2014']

    for dataset in datasets:
        annFile = input_dir + '/instances_' + dataset+'.json'
        coco = COCO(annFile)
        cats = coco.loadCats(coco.getCatIds())

        subcategories = [cat['name'] for cat in cats]
        catIds = coco.getCatIds(catNms=subcategories)

        subcategories_imageIds = dict()

        for i in range(0, len(catIds)):
            imgIds = coco.getImgIds(catIds=catIds[i])
            img = []
            for j in imgIds:
                img.append(j)
            subcategories_imageIds[subcategories[i]] = img
        train_cat=[]
        for key,value in subcategories_imageIds.items():
            train_cat+=subcategories_imageIds[key]

    imgIdss = coco.getImgIds(imgIds=train_cat)
    print("Total Images: ", len(imgIdss))

    set_caption_length = [None] * len(imgIdss)

    dataset = dict()
    vocab_set = set()
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')

    for dataset in datasets:
        annFile = input_dir + '/caption/captions_' + dataset + '.json'
        coco_caps = COCO(annFile)

        for imgid in imgIdss:
            img = coco.loadImgs(imgid)[0]
            annIds = coco_caps.getAnnIds(imgIds=img['id']);
            anns = coco_caps.loadAnns(annIds)
            imgcaptions = []
            for cap in anns:
                # Remove punctuation
                cap = cap['caption'].translate(str.maketrans('', '', string.punctuation))
                # Replace - to blank
                cap = cap.replace("-", " ")

                # Split string into word list and Convert each word into lower case
                cap = cap.split()

                for word in cap:
                    words = SENTENCE_SPLIT_REGEX.split(word.lower())
                    words = [w.strip() for w in words if len(w.strip()) > 0]
                    vocab_set.update(words)
                   #set_caption_length[cap['caption']] = len(words)

               # cap = [word.lower() for word in cap]

                #imgcaptions.append(cap)

            #print(imgcaptions)
    vocab_list = list(vocab_set)
    vocab_list.sort()
    vocab_list.insert(0, '<pad>')
    vocab_list.insert(1, '<unk>')

    with open('D:/data/vqa/coco/simple_vqa/vocab_caption.txt', 'w') as f:
        f.writelines([w + '\n' for w in vocab_list])

    print('Make vocabulary for questions')
    print('The number of total words of questions: %d' % len(vocab_set))


def make_caption(input_dir):
    """Make dictionary for questions and save them into text file."""

    datasets = ['train2014']

    for dataset in datasets:
        annFile = input_dir + '/instances_' + dataset+'.json'
        coco = COCO(annFile)
        cats = coco.loadCats(coco.getCatIds())

        subcategories = [cat['name'] for cat in cats]
        catIds = coco.getCatIds(catNms=subcategories)

        subcategories_imageIds = dict()

        for i in range(0, len(catIds)):
            imgIds = coco.getImgIds(catIds=catIds[i])
            img = []
            for j in imgIds:
                img.append(j)
            subcategories_imageIds[subcategories[i]] = img
        train_cat=[]
        for key,value in subcategories_imageIds.items():
            train_cat+=subcategories_imageIds[key]

    imgIdss = coco.getImgIds(imgIds=train_cat)
    print("Total Images: ", len(imgIdss))

    set_caption_length = [None] * len(imgIdss)

    dataset = dict()
    vocab_set = set()
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
    img_name=[]
    imgcaptions = []
    for dataset in datasets:
        annFile = input_dir + '/caption/captions_' + dataset + '.json'
        coco_caps = COCO(annFile)

        for imgid in imgIdss:
            img = coco.loadImgs(imgid)[0]
            annIds = coco_caps.getAnnIds(imgIds=img['id']);
            anns = coco_caps.loadAnns(annIds)

            for cap in anns:
                # Remove punctuation
                cap = cap['caption'].translate(str.maketrans('', '', string.punctuation))
                # Replace - to blank
                cap = cap.replace("-", " ")

                img_name.append(img['file_name'])
                imgcaptions.append(cap)


    with open('D:/data/vqa/coco/simple_vqa/captions.txt', 'w') as f:
        f.writelines('image,caption'+'\n')

        for idx in range(len(img_name)):
            if 'jpg' in img_name[idx]:
                text = img_name[idx]
                text+=','
                text+=imgcaptions[idx]
                text += '\n'

                f.writelines(text)

    print('Make vocabulary for questions')
    print('The number of total words of questions: %d' % len(vocab_set))



def main():
    input_dir = 'D:/data/vqa/vizwiz/visual_question_answering'
    make_vocab_questions(input_dir)
    #make_vocab_answers(input_dir + '/Annotations/foranswer/')

if __name__ == '__main__':

    main()