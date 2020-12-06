"""
https://blog.paperspace.com/data-augmentation-for-object-detection-rotation-and-shearing/
https://blog.paperspace.com/data-augmentation-for-object-detection-building-i.ut-pipelines/
https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
https://stackoverflow.com/questions/53106780/specify-background-color-when-rotating-an-image-using-opencv-in-python
https://towardsdatascience.com/conversational-ai-design-build-a-contextual-ai-assistant-61c73780d10
https://pythonprogramming.net/custom-objects-tracking-tensorflow-object-detection-api-tutorial/
"""

import os
import cv2
import xmltodict
from utils_augmentation import rotate_image_within_bounds, overlay,\
                               show_image, apply_brightness_contrast,\
                               resize, XML_string

def shear(image):
    pass
logo_name = 'mswipe'
logo_path = 'mswipe.jpg'
xml_path = 'mswipe.xml'
training_data_path = '/home/tanveer/MSwipe/Object_detection/train'
synthetic_data_path = '/home/tanveer/MSwipe/Object_detection/images'

binding_box = []
cropped_logo = None

logo = cv2.imread(logo_path, 1)
with open(xml_path) as xml_file:
    data_dict = xmltodict.parse(xml_file.read())
    box = dict(data_dict['annotation']['object']['bndbox'])
cropped_logo = logo[int(box['ymin']):int(box['ymax']),
                     int(box['xmin']):int(box['xmax'])]

show_image(logo)
show_image(cropped_logo)
show_image(training)
show_image(res)

SYNTHETIC_FACTOR = 1

def generate_synthetic_images(SYNTHETIC_FACTOR):
    images = os.listdir(training_data_path)
    completed = 1
    for f in range(SYNTHETIC_FACTOR):
        for i in range(len(images)):
            try:
                if completed == 0:
                    i = i - 1
                training = cv2.imread(training_data_path+'/'+images[i], 1)
                res = resize(cropped_logo)
                res = apply_brightness_contrast(res)
                show_image(res)
                training, angle = rotate_image_within_bounds(training)
                show_image(training)
                new, box = overlay(training, res, -angle)
                path = synthetic_data_path+'/new{}.png'.format(i)
                cv2.imwrite(path, new)
                with open(synthetic_data_path+"/new{}.xml".format(i), 'w') as f:
                    f.write(XML_string.format(i,
                                              path,
                                              new.shape[1],
                                              new.shape[0],
                                              new.shape[2],
                                              logo_name,
                                              box['xmin'],
                                              box['ymin'],
                                              box['xmax'],
                                              box['ymax'],)
                            )
                completed = 1
                print(i)
            except Exception as e:
                print(str(e))
                completed = 0
                pass

i = 0
while i <= SYNTHETIC_FACTOR:
    try:
        training = cv2.imread('/home/tanveer/MSwipe/Object_detection/Confusion Matrix/All/7', 1)
        res = resize(cropped_logo)
        res = apply_brightness_contrast(res)
        res = rotate_image_within_bounds(res)
        new, box = overlay(training, res)
        i+=1
        path = 'temp/new{}.png'.format(i)
        print(i)
        cv2.imwrite(path, new)
        with open("temp/new{}.xml".format(i), 'w') as f:
            f.write(XML_string.format(i,
                                      i,
                                      new.shape[0],
                                      new.shape[1],
                                      new.shape[2],
                                      logo_name,
                                      box['xmin'],
                                      box['ymin'],
                                      box['xmax'],
                                      box['ymax'],)
                    )
    except Exception as e:
        print(str(e))
        pass
