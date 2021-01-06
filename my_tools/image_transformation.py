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
                      vcoco_test_json_path,
                      images_dir_path, 
                      output_dir_path, 
                      transformation_type, 
                      transformation_level, 
                      images_limit=None,
                      scaled_ratio_axis_x = 1,
                      scaled_ratio_axis_y = 1):
# for sfadsf in range(1):
#     transformation_level = 45
#     images_dir_path = "images"    
#     instances_all_path = "instances_vcoco_all_2014.json"
#     vcoco_test_ids_path = "vcoco_test.ids"
#     images_limit=10
#     vcoco_test_json_path= "vcoco_test.json"
#     transformation_type = "rotation"
#     output_dir_path = 'output'
    
    with open(instances_all_path, "r") as f:
        instances = json.load(f)
        
    with open(vcoco_test_ids_path, "r") as f:
        test_images_ids = [int(line) for line in f]
        
    if images_limit is not None:
        test_images_ids = test_images_ids[:images_limit]
        
    test_images = {x['id']:x for x in instances['images'] if x['id'] in test_images_ids}
    
    test_annotations = defaultdict(list)
    for annotation in instances['annotations']:
        if annotation['image_id'] in test_images_ids:
            test_annotations[annotation['image_id']].append(annotation)
    
    with open(vcoco_test_json_path, "r") as f:
        vcoco_test_json = json.load(f)
        
    if images_limit is not None:
        for verb_group in vcoco_test_json:
            role_object_id_sublists = []
            images_cnt = len(verb_group['image_id'])
            sublists_cnt = len(verb_group['role_name'])
            for i in range(sublists_cnt):
                sublist = verb_group['role_object_id'][i*images_cnt : (i+1)*images_cnt]
                role_object_id_sublists.append(sublist)
                # print(len(sublist))
            
            new_ann_id = []
            new_image_id = []
            new_label = []
            new_role_object_id_sublists = [[] for i in range(sublists_cnt)]
            for i in range(images_cnt):
                if verb_group['image_id'][i] in test_images_ids:
                    new_image_id.append(verb_group['image_id'][i])
                    new_ann_id.append(verb_group['ann_id'][i])
                    new_label.append(verb_group['label'][i])
                    for j in range(sublists_cnt):
                        new_role_object_id_sublists[j].append(
                            role_object_id_sublists[j][i])
 
            new_role_object_id = []
            for new_sublist in new_role_object_id_sublists:
                new_role_object_id += new_sublist
                
            verb_group['image_id'] = new_image_id
            verb_group['ann_id'] = new_ann_id
            verb_group['label'] = new_label
            verb_group['role_object_id'] = new_role_object_id
                        
    transformed_images = []
    if transformation_type == 'rotation':
        for image_id in test_images_ids:
            img_data = test_images[image_id]
            file_path = join(images_dir_path, img_data['file_name'])
            
            #obracanie obrazu
            im = Image.open(file_path)
            roteted_image = im.rotate(transformation_level, expand=True)
            
            #obracanie adnotacji
            width, height = im.size
            cx = width/2
            cy = height/2
            img_annotations = test_annotations[image_id]
            
            new_width, new_height = roteted_image.size
            new_cx = new_width/2
            new_cy = new_height/2
            img_data["width"] = new_width
            img_data["height"] = new_height
            
            #obracanie wszystkich adnotacji na obrazie 
            for annotation in img_annotations:
                x1, y1, w, h = annotation["bbox"]
                x2 = x1 + w
                y2 = y1 + h
                x3,y3 = x1,y2
                x4,y4 = x2,y1
                
                #plt.imshow(im)#tmp
                #plt.plot([x1], [y1], 'xb', label=annotation['category_id']) #tmp
                #plt.plot([x1,x2,x3,x4], [y1,y2,y3,y4], 'yo', label=annotation['category_id']) #tmp
                #plt.plot([x1,x1,x2,x2,x1], [y1,y2,y2,y1,y1], '-', label=annotation['category_id']) #tmp
                #print(annotation["bbox"], annotation["category_id"]) #tmp
                
                x1r,y1r = rotate_point(x1,y1,cx,cy,new_cx,new_cy, transformation_level)
                x2r,y2r = rotate_point(x2,y2,cx,cy,new_cx,new_cy, transformation_level)
                x3r,y3r = rotate_point(x3,y3,cx,cy,new_cx,new_cy, transformation_level)
                x4r,y4r = rotate_point(x4,y4,cx,cy,new_cx,new_cy, transformation_level)
                # xt,yt = rotate_point(cx,cy,cx,cy,new_cx,new_cy, transformation_level)
                
                new_x1 = min(x1r,x2r,x3r,x4r)
                new_y1 = min(y1r,y2r,y3r,y4r)
                new_x2 = max(x1r,x2r,x3r,x4r)
                new_y2 = max(y1r,y2r,y3r,y4r)
                new_h = new_y2 - new_y1
                new_w = new_x2 - new_x1
                annotation["bbox"] = [new_x1, new_y1, new_w, new_h]
                
                # plt.imshow(roteted_image)#tmp
                # plt.plot([new_x1,new_x1,new_x2,new_x2,new_x1], [new_y1,new_y2,new_y2,new_y1,new_y1], '-', label=annotation['category_id']) #tmp
              
                # x1, y1, x2, y2 = annotation["bbox"] #tmp
                # plt.plot([new_x1,new_x2,new_x3,new_x4], [new_y1,new_y2,new_y3,new_y4], 'xr', label=annotation['category_id']) #tmp
                # plt.plot([xt], [yt], 'yo', label=annotation['category_id']) #tmp
                # return #tmp
            transformed_images.append(roteted_image)
    elif transformation_type == 'scale':
        print("scale")
        for image_id in test_images_ids:
            img_data = test_images[image_id]
            file_path = join(images_dir_path, img_data['file_name'])

            # skalowanie obrazu dla wzdłuż współrzednej x
            img = cv2.imread(file_path)
            # zdefiniuj skale
            scale = (scaled_ratio_axis_x, scaled_ratio_axis_y)
            scaled_img = cv2.resize(img, None, fx = scale[0], fy = scale[1])

            sh, sw = scaled_img.shape[:2]  # sprawdz h, w skalowanego zdjecia
            padded_scaled = np.zeros(img.shape, dtype=np.uint8)
            padded_scaled[0:sh, 0:sw] = scaled_img

            # skalowanie adnotacji
            img_data["width"] = sw
            img_data["height"] = sh

            img_annotations = test_annotations[image_id]
            # obracanie wszystkich adnotacji na obrazie
            for annotation in img_annotations:
                x, y, w, h = annotation["bbox"]
                annotation["bbox"] = [x * scale[0], y * scale[1], w * scale[0], h * scale[1]]
                x, y, w, h = annotation["bbox"]
                #cv2.rectangle(padded_scaled, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
                #plt.imshow(padded_scaled)
            transformed_images.append(padded_scaled)
    elif transformation_type == 'perspective_transform':
        print("PerspectiveTransform")
        for image_id in test_images_ids:
            img_data = test_images[image_id]
            file_path = join(images_dir_path, img_data['file_name'])

            # wczytaj zdjecie
            img = cv2.imread(file_path)
            w,h,c =img.shape
            # punkty ( górny lewy, górny prawy, dolny prawy, dolny lewy)
            input_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            output_pts = np.float32([[0, 0], [w, 0+0.2*h], [w, 0.8 * h], [0, h]])

            # oblicz przekształcenie perspektywiczne
            matrix = cv2.getPerspectiveTransform(input_pts, output_pts)

            # zastosuj przeksztalcenie perspektywiczne
            out = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

            # przeksztalcenie adnotacji
            img_annotations = test_annotations[image_id]
            # przeksztalcanie wszystkich adnotacji na obrazie
            for annotation in img_annotations:
                x, y, w, h = annotation["bbox"]

                p1 = (x, y)  # oryginalny górny-lewy punkt
                px1 = (matrix[0][0] * p1[0] + matrix[0][1] * p1[1] + matrix[0][2]) / (
                (matrix[2][0] * p1[0] + matrix[2][1] * p1[1] + matrix[2][2]))
                py1 = (matrix[1][0] * p1[0] + matrix[1][1] * p1[1] + matrix[1][2]) / (
                (matrix[2][0] * p1[0] + matrix[2][1] * p1[1] + matrix[2][2]))
                p1_after = (int(px1), int(py1))  # po transformacji

                p2 = (x+w, y+h)  # oryginalny dolny prawy
                px2 = (matrix[0][0] * p2[0] + matrix[0][1] * p2[1] + matrix[0][2]) / (
                (matrix[2][0] * p2[0] + matrix[2][1] * p2[1] + matrix[2][2]))
                py2 = (matrix[1][0] * p2[0] + matrix[1][1] * p2[1] + matrix[1][2]) / (
                (matrix[2][0] * p2[0] + matrix[2][1] * p2[1] + matrix[2][2]))
                p2_after = (int(px2), int(py2))  # po transformacji

                # nowa szerokosc i wysokosc ramki
                new_w = p2_after[0] - p1_after[0]
                new_h = p2_after[1] - p1_after[1]

                annotation["bbox"] = [p1_after[0], p1_after[1], new_w, new_h]

                # test
                # x, y, w, h = annotation["bbox"]
                # cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
                # plt.imshow(out)

            transformed_images.append(out)
    else:
        raise Exception(f"Unknown transformation_type: {transformation_type}")
    
    name_suffix = f"first_{images_limit}" if images_limit is not None \
        else "all"
    instances_out_path = join(output_dir_path, f"instances_{transformation_type}_{transformation_level}.json")
    test_json_out_path = join(output_dir_path, f"test_{name_suffix}.json")
    test_ids_out_path = join(output_dir_path, f"test_{name_suffix}.ids")
    with open(instances_out_path, 'w') as f:
        json.dump(instances, f)
    
    with open(test_json_out_path, 'w') as f:
        json.dump(vcoco_test_json, f)
        
    test_images_ids_str = [str(x) for x in test_images_ids]       
    with open(test_ids_out_path, 'w') as f:
        f.write('\n'.join(test_images_ids_str))

    return transformed_images, test_json_out_path, instances_out_path, test_ids_out_path


#EXAMPLE USAGE
#instances_all_path = "/home/wojtek/DRG/Data/v-coco/data/instances_vcoco_all_2014.json"
#vcoco_test_ids_path = "/home/wojtek/DRG/Data/v-coco/data/splits/vcoco_test.ids"
#vcoco_test_json_path = "/home/wojtek/DRG/Data/v-coco/data/vcoco/vcoco_test.json"
#images_dir_path = "/home/wojtek/DRG/Data/v-coco/coco/images/val2014/"
#output_dir_path = "/home/wojtek/DRG/my_output"

#result =  transform_images(instances_all_path,
#                      vcoco_test_ids_path,
#                      vcoco_test_json_path,
#                      images_dir_path,
#                      output_dir_path,
#                      transformation_type = "rotation",
#                      transformation_level= 45,
#                      images_limit=10)

#result_images = result[0]
#plt.imshow(result_images[3])