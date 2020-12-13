# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 20:36:17 2020

@author: mati9
"""
import io
import pickle
import json
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from skimage.draw import line


class Bbox:
    
    def __init__(self, object_data, name, frames = None):
        self.defined = not np.isnan(np.dot(object_data[0:3], object_data[0:3]))
        self.name = name
        self.x1, self.y1, self.x2, self.y2, self.score = object_data
        
        # self.centerx = (self.x1+self.x2)/2
        # self.centery = (self.y1+self.y2)/2
        self.frame = None
        if self.defined and frames is not None:
            for frame in frames:
                frame_bbox = frame[2]
                if np.array_equal(frame_bbox, object_data[0:4]):
                    self.frame = frame
                    # print(self.frame[5])
                    # if(len(np.array(self.frame[5])) != 1):
                    #     print (f"Warning! expected len is 1 but found {len(self.frame[5])}")
                    break
                
        self.obj_category = None
        self.obj_score = None
        if self.frame is not None:
            self.obj_category = CATEGORIES[self.frame[4]] 
            self.obj_score = self.frame[5] 


verbs = ['carry', 'catch', 'cut', 'drink', 'eat', 'hit', 'hold', 'jump', 
         'kick', 'lay', 'look',  'read', 'ride', 'sit', 
         'skateboard', 'ski', 'snowboard', 'surf', 
         'talk_on_phone', 'throw', 'work_on_computer']
# 'point','run', 'smile','stand', 'walk', #Akcje bez obiektów nie opisywane w artykule o DRG

CATEGORIES = ["__background","person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
        "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear",
        "zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
        "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup",
        "fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
        "cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
        "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
    ]
def build_bboxes(prediction, frames=None):
    person_bbox = Bbox(list(prediction['person_box'])+[1], 'human', frames)
    bboxes = [person_bbox]
    for verb in verbs:
        score = verb+'_agent'
        
        object_name = verb+'_obj'
        obj_data = prediction.get(object_name)
        if obj_data is not None:
            obj_bbox = Bbox(obj_data, object_name, frames)
            if obj_bbox.defined:
                bboxes.append(obj_bbox)
                
        instrument_name = verb+'_instr'
        instrument_data = prediction.get(instrument_name)
        if instrument_data is not None:
            instr_bbox = Bbox(instrument_data, instrument_name, frames)
            if instr_bbox.defined:
                bboxes.append(instr_bbox)
                
    return bboxes

def display_prediction(prediction, img, threshold=0.5):
    bboxes = build_bboxes(prediction)
    plt.imshow(img)
    
    for bbox in bboxes:
        if bbox.score < threshold and bbox.name != 'human':
            continue
        
        label = bbox.name + " " + bbox.obj_category if bbox.obj_category is not None \
            else bbox.name
        plt.plot([bbox.x1,bbox.x1,bbox.x2,bbox.x2,bbox.x1], [bbox.y1,bbox.y2,bbox.y2,bbox.y1,bbox.y1], '-', label=bbox.name)
        
    plt.legend()

def save_predictions_to_files(prediction, img, frames, threshold=0.5, output_files_name_without_extension='prediction_result'):
    bboxes = build_bboxes(prediction, frames)
    prediction_description = ''
    img_copy = img.copy()
    for bbox in bboxes:
        if bbox.score < threshold and bbox.name != 'human':
            continue
        
        if bbox.name != 'human':
            verb = bbox.name.replace("_instr", f"({bbox.score*100:.2f}%) using") \
                .replace("_obj", f"({bbox.score*100:.2f}%) ") \
                .replace('_', ' ')
                
            class_text = f"{bbox.obj_category}({bbox.obj_score:.2f}%)" if bbox.obj_category is not None \
                else ''
            prediction_description += f"{verb} {class_text}\n"
        color = np.array(np.random.choice(range(255), size=3))
        xx,yy = line(int(bbox.y1), int(bbox.x1),int(bbox.y2), int(bbox.x1))
        img_copy[xx,yy,:] = color
        xx,yy = line(int(bbox.y2), int(bbox.x1),int(bbox.y2),int(bbox.x2))
        img_copy[xx,yy,:] = color
        xx,yy = line(int(bbox.y2), int(bbox.x2),int(bbox.y1), int(bbox.x2))
        img_copy[xx,yy,:] = color
        xx,yy = line(int(bbox.y1), int(bbox.x2),int(bbox.y1),int(bbox.x1))
        img_copy[xx,yy,:] = color
        
    io.imsave(output_files_name_without_extension+'.png',img_copy)
    with open(output_files_name_without_extension+'.txt', 'w') as f:
        f.write(prediction_description)
    
detection = pickle.load( open( "code_test_data\\detekcje.pkl", "rb" ) )
frames = pickle.load( open( "code_test_data\\ramki.pkl", "rb" ) ) 
#pred = pickle.load( open( "var.pkl", "rb" ) ) 
img = io.imread('code_test_data\\img2.jpg')
display_prediction(detection[0], img, threshold=0.1)
save_predictions_to_files(detection[0], img, frames[0], threshold=0.1)