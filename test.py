import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import random
from models import VqaModel
from util import text_helper

def print_examples(model,data_path,vocab):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose(
        [
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406),
                              (0.229, 0.224, 0.225))
        ]
    )
    model.eval()

    for i in range(5):
        testdata = np.load(data_path, allow_pickle=True)
        num = random.randint(0, len(testdata))
        image = testdata[num]['image_path']
        image = Image.open(image).convert('RGB')


        question = testdata[num]['question_str']
        print(image)
        print(question)

        print(
            "Example", i ,"OUTPUT: "
            + " ".join(model.visualization_vqa(image.to(device), vocab))
        )

if __name__ == "__main__":
    test_path = 'D:/data/vqa/coco/simple_vqa/test.npy'
    input_dir = 'D:/data/vqa/coco/simple_vqa'
    ans_vocab = text_helper.VocabDict(input_dir + '/vocab_answers.txt')

    print_examples(VqaModel,test_path,ans_vocab)