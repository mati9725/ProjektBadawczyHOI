import json
from collections import defaultdict
from PIL import Image
from os.path import join
import math
import cv2
import numpy as np
import matplotlib.patches as patches

import skimage as io
import matplotlib.pyplot as plt

def rotate_point(x,y,cx,cy,new_cx,new_cy, angle):
    angle = -angle #temp?
    
    cos = math.cos(math.radians(angle))
    sin = math.sin(math.radians(angle))
    
    temp = ((x-cx)*cos - (y-cy)*sin) + new_cx
    y = ((x-cx)*sin + (y-cy)*cos) + new_cy
    x = temp
    return x,y

def transform_images(instances_all_path,
                      vcoco_test_ids_path,
                      images_dir_path,
                      output_dir_path,
                      transformation_type, 
                      transformation_level, 
                      images_limit=None,
                      scaled_ratio_axis_x = 1,
                      scaled_ratio_axis_y = 1):
    
    with open(instances_all_path, "r") as f:
        instances = json.load(f)
        
    with open(vcoco_test_ids_path, "r") as f:
        test_images_ids = [int(line) for line in f]
        
    if images_limit is not None:
        test_images_ids = test_images_ids[:images_limit]
        
    test_images = {x['id']:x for x in instances['images'] if x['id'] in test_images_ids}

    transformed_images = []
    if transformation_type == 'rotation':
        for image_id in test_images_ids:
            img_data = test_images[image_id]
            file_path = join(images_dir_path, img_data['file_name'])

            #obracanie obrazu
            im = Image.open(file_path)
            roteted_image = im.rotate(transformation_level, expand=True)
            roteted_image = np.array(roteted_image)
            transformed_images.append(roteted_image)

    elif transformation_type == 'scale':
        for image_id in test_images_ids:
            img_data = test_images[image_id]
            file_path = join(images_dir_path, img_data['file_name'])

            # skalowanie obrazu
            img = cv2.imread(file_path)
            scale = (scaled_ratio_axis_x, scaled_ratio_axis_y)
            scaled_img = cv2.resize(img, None, fx = scale[0], fy = scale[1])
            scaled_rgb = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB)
            scaled_rgb = Image.fromarray(scaled_rgb)
            transformed_images.append(scaled_rgb)
    elif transformation_type == 'perspective_transform':
        for image_id in test_images_ids:
            img_data = test_images[image_id]
            file_path = join(images_dir_path, img_data['file_name'])

            img = cv2.imread(file_path)
            w,h,c =img.shape
            # zdefiniuj punkty wejsciowe i wyjsciowe ( górny lewy, górny prawy, dolny prawy, dolny lewy)
            input_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            output_pts = np.float32([[0, 0], [w, 0+ transformation_level/100 * h], [w, (1-transformation_level/100) * h], [0, h]])

            # przekształcenie perspektywiczne
            matrix = cv2.getPerspectiveTransform(input_pts, output_pts)

            # zastosuj przeksztalcenie perspektywiczne
            out = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
            out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            out_rgb = Image.fromarray(out_rgb)
            transformed_images.append(out_rgb)
    else:
        raise Exception(f"Unknown transformation_type: {transformation_type}")

    return transformed_images