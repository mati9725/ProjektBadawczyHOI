import json
from collections import defaultdict
from PIL import Image
from os.path import join
import math

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

def get_first_n_from_file(vcoco_test_ids_path, images_limit=None):
    with open(vcoco_test_ids_path, "r") as f:
        test_images_ids = [int(line) for line in f]
        
    if images_limit is not None:
        test_images_ids = test_images_ids[:images_limit]
        
    return test_images_ids

def extract_annotations(instances_all_path,
                   vcoco_test_ids_path,
                   vcoco_test_json_path,
                   images_limit=None):
# for sfadsf in range(1):
#     instances_all_path = "instances_vcoco_all_2014.json"
#     vcoco_test_json_path= "vcoco_test.json"
    # vcoco_test_ids_path = "vcoco_test.json"
    
    test_images_ids = get_first_n_from_file(vcoco_test_ids_path, images_limit)
    
    with open(instances_all_path, "r") as f:
        instances = json.load(f)
    
    with open(vcoco_test_json_path, "r") as f:
        vcoco_test_json = json.load(f)

    all_categories = {c['id']:c["name"] for c in instances['categories']}

    test_images = {x['id']:x for x in instances['images'] if x['id'] in test_images_ids}
    
    annotations_by_id = {x["id"]:x for x in instances['annotations']}
        
    test_annotations = defaultdict(list)
    for annotation in instances['annotations']:
        if annotation['image_id'] in test_images_ids:
            test_annotations[annotation['image_id']].append(annotation)
    
    image_verb_cnt = sum([sum(x['label']) for x in vcoco_test_json])#tmp
    
    simple_annotations_by_img = defaultdict(list)
    verbs_by_img = defaultdict(list)
    for verb_group in vcoco_test_json:
        role_object_id_sublists = []
        images_cnt = len(verb_group['image_id'])
        sublists_cnt = len(verb_group['role_name'])
        for i in range(sublists_cnt):
            sublist = verb_group['role_object_id'][i*images_cnt : (i+1)*images_cnt]
            role_object_id_sublists.append(sublist)
        
        for i in range(images_cnt):
            img_id = verb_group['image_id'][i]
            if img_id in test_images_ids and verb_group['label'][i] == 1:
                verb = verb_group['action_name']
                verbs_by_img[img_id].append(verb)
                
                if sublists_cnt > 1:#dla czasowników z obiektem/-ami
                    for j in range(1, sublists_cnt):#pętla idąca po instr i obj
                        ann_id = role_object_id_sublists[j][i]
                        if ann_id > 0:
                            ann = annotations_by_id[ann_id]
                            simple_annotations_by_img[img_id].append({
                                "verb": verb,
                                "object_category_name":all_categories[ann['category_id']],
                                "object_type": verb_group['role_name'][j],
                                # "object_annotation": ann,#pełna adnotacja jest zbędna?
                                "object_category_id":ann['category_id'],
                                })

    #testowo
    # expected_image_verb_cnt = sum([sum(x['label']) for x in vcoco_test_json])#tmp
    # image_verb_cnt = sum([len(v) for k,v in verbs_by_img.items()])#tmp

    results = dict()
    for img_id in test_images_ids:            
        results[img_id] = {
                "annotations": simple_annotations_by_img[img_id],
                "image_id": img_id,
                "verbs": verbs_by_img[img_id],
                # "image_data": test_images[img_id]#cała informacja o obrazie jest zbędna?
                "file_name": test_images[img_id]["file_name"]
            }
        
    return results
        
    
# # #EXAMPLE USAGE
# results = extract_annotations("instances_vcoco_all_2014.json", "vcoco_test.ids", "vcoco_test.json", 10)
