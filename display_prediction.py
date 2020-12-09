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


class Bbox:
    
    def __init__(self, object_data, name):
        self.defined = not np.isnan(np.dot(object_data[0:3], object_data[0:3]))
        self.name = name
        self.x1, self.y1, self.x2, self.y2, self.score = object_data
        
        self.centerx = (self.x1+self.x2)/2
        self.centery = (self.y1+self.y2)/2


verbs = ['carry', 'catch', 'cut', 'drink', 'eat', 'hit', 'hold', 'jump', 
         'kick', 'lay', 'look',  'read', 'ride', 'sit', 
         'skateboard', 'ski', 'snowboard', 'surf', 
         'talk_on_phone', 'throw', 'work_on_computer']
# 'point','run', 'smile','stand', 'walk', #Akcje bez obiektów nie opisywane w artykule o DRG

def display_prediction(prediction, img, threshold=0.5):
    person_bbox = Bbox(list(prediction['person_box'])+[1], 'human')
    bboxes = [person_bbox]
    for verb in verbs:
        score = verb+'_agent'
        
        object_name = verb+'_obj'
        obj_data = prediction.get(object_name)
        if obj_data is not None:
            obj_bbox = Bbox(obj_data, object_name)
            if obj_bbox.defined:
                bboxes.append(obj_bbox)
            
        instrument_name = verb+'_instr'
        instrument_data = prediction.get(instrument_name)
        if instrument_data is not None:
            instr_bbox = Bbox(instrument_data, instrument_name)
            if instr_bbox.defined:
                bboxes.append(instr_bbox)
    
    plt.imshow(img)
    
    for bbox in bboxes:
        if bbox.score < threshold and bbox.name != 'human':
            continue
        
        plt.plot([bbox.x1,bbox.x1,bbox.x2,bbox.x2,bbox.x1], [bbox.y1,bbox.y2,bbox.y2,bbox.y1,bbox.y1], '-', label=bbox.name)
        #plt.text(bbox.x1, bbox.y2, bbox.name, bbox=dict(fill=True, edgecolor='y', linewidth=2))
        
    plt.legend()

    
# pred = pickle.load( open( "var.pkl", "rb" ) )
# img = io.imread('img.png')
# display_prediction(pred, img)