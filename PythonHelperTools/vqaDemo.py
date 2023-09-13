# coding: utf-8

from vqaTools.vqa import VQA
import random
import skimage.io as io
import matplotlib.pyplot as plt
import os

dataDir='D:/data/vqa/vizwiz/visual_question_answering'
split = 'train'
annFile='%s/Annotations/%s.json'%(dataDir, split)
imgDir = '%s/Resize_img/' %dataDir

# initialize VQA api for QA annotations
vqa=VQA(annFile)

# load and display QA annotations for given answer types
"""
ansTypes can be one of the following
yes/no
number
other
unanswerable
"""
anns = vqa.getAnns(ansTypes='yes/no');   
randomAnn = random.choice(anns)
vqa.showQA([randomAnn])
imgFilename = randomAnn['image']
if os.path.isfile(imgDir + imgFilename):
	I = io.imread(imgDir + imgFilename)
	#plt.imshow(I)
	#plt.axis('off')
	#plt.show()

# load and display QA annotations for given images
imgs = vqa.getImgs()
anns = vqa.getAnns(imgs=imgs)
randomAnn = random.choice(anns)
vqa.showQA([randomAnn])  
imgFilename = randomAnn['image']
if os.path.isfile(imgDir +split+ '/'+imgFilename):
	I = io.imread(imgDir + split+ '/'+imgFilename)
	plt.imshow(I)
	plt.axis('off')
	plt.show()

