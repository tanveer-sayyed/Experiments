"""
https://blog.paperspace.com/data-augmentation-for-object-detection-rotation-and-shearing/
https://blog.paperspace.com/data-augmentation-for-object-detection-building-i.ut-pipelines/
https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
https://stackoverflow.com/questions/53106780/specify-background-color-when-rotating-an-image-using-opencv-in-python
https://towardsdatascience.com/conversational-ai-design-build-a-contextual-ai-assistant-61c73780d10
https://pythonprogramming.net/custom-objects-tracking-tensorflow-object-detection-api-tutorial/

https://www.pyimagesearch.com/2016/04/25/watermarking-images-with-opencv-and-python/
"""

import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

def show_image(image):
    plt.imshow(image)
    plt.show()

def rotate_image_within_bounds(image):
    """Rotate the image.
    
    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black. 

    Parameters
    ----------
    image : numpy.ndarray
        numpy image
    angle : float
        angle by which the image is to be rotated
    
    Returns
    -------
    numpy.ndarray
        Rotated Image
        
    float
        angle by which the image is to be rotated
    """
    r = random.randint(1,4)
    if r % 4 == 0:
        angle = random.randint(1,360)
    else:
        angle = random.choice([0,90,180,270])
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    
    image = cv2.warpAffine(image,
                           M,
                           (nW, nH),
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(random.randint(0,255),
                                        random.randint(0,255),
                                        random.randint(0,255)))
    return image, angle

def shear(image, angle):
    rows,cols,ch = image.shape
    src_points = np.float32([[cols//2,rows//2],
                             [cols//2,rows//2+1],
                             [cols//2+1,rows//2+1]])
    rotate = random.randint(-2,2)
    magnify = random.randint(0,2)
    dst_points = np.float32([[cols//2,rows//2],
                             [cols//2 + rotate, rows//2+1 + magnify],
                             [cols//2+1 + rotate,rows//2+1 + magnify]])
    M = cv2.getAffineTransform(src_points, dst_points)
    dst = cv2.warpAffine(image,M,(cols,rows))
    dst = cv2.resize(dst, (cols,rows))
    show_image(dst)
    pass

# def transform(img, bboxes, angle):
#     angle = random.uniform(angle)
#     w,h = img.shape[1], img.shape[0]
#     cx, cy = w//2, h//2
#     img = rotate_image_within_bounds(img, angle)
#     corners = get_corners(bboxes)
#     corners = np.hstack((corners, bboxes[:,4:]))
#     corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)
#     new_bbox = get_enclosing_box(corners)
#     scale_factor_x = img.shape[1] / w
#     scale_factor_y = img.shape[0] / h
#     img = cv2.resize(img, (w,h))
#     new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]
#     bboxes  = new_bbox
#     bboxes = clip_box(bboxes, [0,0,w, h], 0.25)
#     return img, bboxes

def apply_brightness_contrast(input_img):
    brightness = random.randint(-127, 127)
    contrast = random.randint(-64, 64)
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf

def overlay(large, small):
    y_offset = random.randint(int(large.shape[0]*0.25), int(large.shape[0]*0.75))
    x_offset = random.randint(int(large.shape[1]*0.25), int(large.shape[1]*0.75))
    large[y_offset:y_offset + small.shape[0],
          x_offset:x_offset + small.shape[1]] = small
    return large, {'ymin':y_offset, 'ymax':y_offset+small.shape[0],
                   'xmin':x_offset, 'xmax':x_offset+small.shape[1]}

def resize(image):
    f = 0.1 * random.randint(1,5)
    return cv2.resize(image, None, fx=f, fy=f, interpolation = cv2.INTER_CUBIC)

XML_string = """<annotation>
	<folder>Object_detection</folder>
	<filename>new{}.png</filename>
	<path>{}</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>{}</width>
		<height>{}</height>
		<depth>{}</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>{}</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>{}</xmin>
			<ymin>{}</ymin>
			<xmax>{}</xmax>
			<ymax>{}</ymax>
		</bndbox>
	</object>
</annotation>
"""
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
                res, angle = rotate_image_within_bounds(res)
                new, box = overlay(training, res)
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
