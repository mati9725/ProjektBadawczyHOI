import pickle
import json
from ExtractAnnotations import extract_annotations
import numpy as np
from collections import Counter

CATEGORIES = ["__background","person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
        "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear",
        "zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
        "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup",
        "fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
        "cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
        "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
    ]

def bbox_iou(boxA, boxB):

    ixmin = np.maximum(boxA[0], boxB[0])
    iymin = np.maximum(boxA[1], boxB[1])
    ixmax = np.minimum(boxA[2], boxB[2])
    iymax = np.minimum(boxA[3], boxB[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((boxB[2] - boxB[0] + 1.) * (boxB[3] - boxB[1] + 1.) +
           (boxA[2] - boxA[0] + 1.) *
           (boxA[3] - boxA[1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps

def get_object_id(bbox, detections):
    max_iou = 0
    idx = -1
    for detect in detections:
        iou = bbox_iou(bbox, detect[2])
        if iou > max_iou:
            max_iou = iou
            idx = detect[4]
    if idx == -1:
        raise NameError('')
    return idx

def unique(annotatons): 
    unique_list = []
    out = {}
    for ann in annotatons:
        x = (ann['verb'], ann['object_category_name'], ann['object_type'])
        if x not in out:
            out[x] = 1
        else:
            out[x] += 1
    return out 


annos_path = "/home/zgorek/Desktop/Projekt Badawczy/DRG/my_tools/test_files/annos.pkl"
annos = pickle.load(open(annos_path, "rb"), encoding='latin1')

threshold = 0.1
verbs = ["stand", "run", "smile", "walk"]
objects = ["catch_obj", "hold_obj", "read_obj", "hit_obj", "look_obj", "throw_obj", "carry_obj", "eat_obj", "kick_obj", "cut_obj"]
instruments = ["skateboard_instr", "point_instr", "snowboard_instr", "hit_instr", "eat_instr", "drink_instr", "lay_instr",
"jump_instr", "sit_instr", "work_on_computer_instr", "talk_on_phone_instr", "ride_instr", "surf_instr", "ski_instr", "cut_instr"]

angles = [-20, -17, -14, -11, -8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8, 11, 14, 17, 20]
angles = [-14]
precissions = []
recalls = []

for key in annos.keys():
    # verbs
    v = annos[key]["verbs"]
    v = dict(Counter(v))
    annos[key]["verbs"] = v

    #annotations
    ann = annos[key]["annotations"]
    new_ann = unique(ann)
    annos[key]["annotations"] = new_ann



for angle in angles:
    predictions_path = "/home/zgorek/Desktop/Projekt Badawczy/DRG/my_tools/test_files/rotations/rotation_{}_1000.pkl".format(angle)
    detections_path = "/home/zgorek/Desktop/Projekt Badawczy/DRG/my_tools/test_files/rotations/detections_rotation_{}_1000.pkl".format(angle)
    detects = pickle.load(open(detections_path, "rb"), encoding='latin1')
    preds =  pickle.load(open(predictions_path, "rb"), encoding='latin1')

    # prediction list to dict
    preds_dict = {}
    for key in annos.keys():
        ids = []
        for i, pred in enumerate(preds):
            if pred["image_id"] == key:
                ids.append(i)
        dict_tmp = {}
        verbs_tmp = []
        anoos_tmp = []
        for i in ids:
            for verb in verbs:
                if preds[i][verb][-1] > threshold:
                    verbs_tmp.append(verb)
            for object in objects:
                if preds[i][object][-1]>threshold:
                    box = preds[i][object][:-1]
                    id = get_object_id(box, detects[key])
                    anoos_tmp.append({'verb': object[:-4], 'object_category_name': CATEGORIES[id], 'object_type': 'obj', 'object_category_id': id})
            for instr in instruments:
                if preds[i][instr][-1]>threshold:
                    box = preds[i][instr][:-1]
                    id = get_object_id(box, detects[key])
                    anoos_tmp.append({'verb': instr[:-6], 'object_category_name': CATEGORIES[id], 'object_type': 'instr', 'object_category_id': id})
        dict_tmp['annotations'] = anoos_tmp
        dict_tmp["verbs"] = verbs_tmp
        preds_dict[key] = dict_tmp

    for key in annos.keys():
        # verbs
        v = preds_dict[key]["verbs"]
        v = dict(Counter(v))
        preds_dict[key]["verbs"] = v

        #annotations
        ann = preds_dict[key]["annotations"]
        new_ann = unique(ann)
        preds_dict[key]["annotations"] = new_ann


    ## Count TP, FP and FN
    TP = 0
    FP = 0
    FN = 0

    for key in annos.keys():
        #verbs TP, FP, FN
        for verb in preds_dict[key]['verbs'].keys():
            if verb in annos[key]['verbs']:
                if preds_dict[key]['verbs'][verb] <= annos[key]['verbs'][verb]:
                    TP += preds_dict[key]['verbs'][verb]
                    FN += annos[key]['verbs'][verb] - preds_dict[key]['verbs'][verb]
                else:
                    TP += annos[key]['verbs'][verb]
                    FP += preds_dict[key]['verbs'][verb] - annos[key]['verbs'][verb]
            else:
                FP += preds_dict[key]['verbs'][verb]

        #verbs FN
        for verb in annos[key]['verbs'].keys():
            if verb not in preds_dict[key]['verbs']:
                FN += annos[key]['verbs'][verb]

        #annotations TP, FP, FN

        for ann in preds_dict[key]['annotations'].keys():
            if ann in annos[key]["annotations"]:
                if preds_dict[key]["annotations"][ann] <= annos[key]['annotations'][ann]:
                    TP += preds_dict[key]['annotations'][ann]
                    FN += annos[key]['annotations'][ann] - preds_dict[key]['annotations'][ann]
                else:
                    TP += annos[key]['annotations'][ann]
                    FP += preds_dict[key]['annotations'][ann] - annos[key]['annotations'][ann]
            else:
                FP += preds_dict[key]['annotations'][ann]
        #annotations FN
        for ann in annos[key]['annotations'].keys():
            if ann not in preds_dict[key]['annotations']:
                FN += annos[key]['annotations'][ann]


    precission = TP/(TP+FP)
    recall = TP/(TP+FN)
    # print("Precission: " + str(precission))
    # print("Recall: " + str(recall))
    precissions.append(precission)
    recalls.append(recall)
    print(TP, FP, FN)

print(precissions)
print(recalls)

#117222, 117424, 117508, 117584, 117676, 117690, 117701, 118069, 118106, 118542, 118544, 118564, 118614, 118740, 118839, 118846, 119365, 119373, 119640, 119876, 119939, 120061, 120070, 120347, 120357, 120441, 120509, 120682, 120771, 120792, 120872, 121014, 121031, 121123, 121154, 121417, 121442, 121570, 121673, 121748, 121827, 121839, 121961, 121989, 122161, 122217, 122281, 122413, 122418, 122476, 122672, 122861, 122934, 122997, 123244, 123555, 123810, 124013, 124279, 124327, 124387, 124416, 124452, 124614, 124832, 124873, 124979, 125129, 125228, 125257, 125632