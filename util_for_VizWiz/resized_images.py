import os
import argparse
from PIL import Image


def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)


def resize_images(input_dir, output_dir, size):
    """Resize the images in 'input_dir' and save into 'output_dir'."""
    for idir in os.scandir(input_dir):
        if not idir.is_dir():
            continue
        if not os.path.exists(output_dir + '/' + idir.name):
            print(idir.name)
            os.makedirs(output_dir + '/' + idir.name)
        images = os.listdir(idir.path)
        n_images = len(images)
        for iimage, image in enumerate(images):
            try:
                with open(os.path.join(idir.path, image), 'r+b') as f:
                    with Image.open(f) as img:
                        img = resize_image(img, size)
                        img.save(os.path.join(output_dir + '/' + idir.name, image), img.format)
            except(IOError, SyntaxError) as e:
                pass
            if (iimage + 1) % 1000 == 0:
                print("[{}/{}] Resized the images and saved into '{}'."
                      .format(iimage + 1, n_images, output_dir + '/' + idir.name))


def main():
    input_dir_train = 'D:/data/vqa/vizwiz/visual_question_answering/Images/train'
    output_dir_train = 'D:/data/vqa/vizwiz/visual_question_answering/Resize_img'
    input_dir_val = 'D:/data/vqa/vizwiz/visual_question_answering/Images/val'
    output_dir_val = 'D:/data/vqa/vizwiz/visual_question_answering/Resize_img'
    input_dir_test = 'D:/data/vqa/vizwiz/visual_question_answering/Images/test'
    output_dir_test = 'D:/data/vqa/vizwiz/visual_question_answering/Resize_img'

    image_size = [244, 244]
    resize_images(input_dir_train, output_dir_train, image_size)
    resize_images(input_dir_val, output_dir_val, image_size)
    resize_images(input_dir_test, output_dir_test, image_size)



if __name__ == '__main__':


    main()
