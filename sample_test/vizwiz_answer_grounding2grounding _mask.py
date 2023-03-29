import numpy as np
import json
import cv2
import os
import image_to_numpy
import urllib.request as rq
from PIL import Image

def Toarray(xycoo):
    xycooArray=[[xy["x"],xy["y"]] for xy in xycoo]
    return xycooArray
def StoreMasks(imgpath,XY,width,height,writepath,SaveFormat):
    """
    uncomment to overlap the original image with the mask
    """
    # req = rq.urlopen(imgpath)
    # arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    # image = cv2.resize(cv2.imdecode(arr, -1),(width,height))
    # mask = np.zeros((image.shape), dtype=np.uint8)
    mask  = np.zeros((height,width),dtype = np.uint8) # comment this line to overlap the original image with the mask
    maskarray=np.array(Toarray(XY),np.int32)
    mask=cv2.fillPoly(mask,[maskarray],(255,255,255))
    mask = cv2.resize(mask,(width,height))
    if SaveFormat == "npy":
        np.save(writepath,mask)
    else:
        # result = cv2.bitwise_and(image, mask)
        # result[mask==0] = 255
        # cv2.imwrite(writepath,result)
        cv2.imwrite(writepath,mask)# comment this line to overlap the original image with the mask
def main(task,SaveFormat):
    avg=[]
    avglength=[]
    groundings={}
    path = task+"_grounding.json"
    npypath = "binary_masks_npy/"+task
    pngpath = "binary_masks_png/"+task

    with open(path,'r',encoding="utf-8") as grounding_file:
        if SaveFormat == "npy":
            if not os.path.isdir("binary_masks_npy"):
                os.mkdir("binary_masks_npy")
            if not os.path.isdir(npypath):
                os.mkdir(npypath)
        if SaveFormat == "png":
            if not os.path.isdir("binary_masks_png"):
                os.mkdir("binary_masks_png")
            if not os.path.isdir(pngpath):
                os.mkdir(pngpath)

        datas=json.load(grounding_file)
        for data in datas:
            if SaveFormat == "npy":
                writepath=npypath+"/"+data[:-4]+".npy"
            else:
                writepath=pngpath+"/"+data[:-4]+".png"
            imgpath = datas[data]["vizwiz_url"]
            width = datas[data]["width"]
            height =datas[data]["height"]
            XY=datas[data]["answer_grounding"]
            StoreMasks(imgpath,XY,width,height,writepath,SaveFormat)


if __name__=="__main__":
    task="val"
    SaveFormat = "png"
    main(task,SaveFormat)