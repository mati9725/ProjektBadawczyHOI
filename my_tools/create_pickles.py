#%%
from test_on_database import test_on_database
import pandas as pd
import torch
from matplotlib import pyplot as plt

path1 = "/home/zgorek/Desktop/Projekt Badawczy/DRG/Data/v-coco/data/vcoco/vcoco_test.json"
path2 = "/home/zgorek/Desktop/Projekt Badawczy/DRG/Data/v-coco/data/instances_vcoco_all_2014.json"
path3 = "/home/zgorek/Desktop/Projekt Badawczy/DRG/Data/v-coco/data/splits/vcoco_test.ids"
RCNN_weights = "/home/zgorek/Desktop/Projekt Badawczy/DRG/pretrained_model/e2e_faster_rcnn_R_50_FPN_1x.pth"
images_path = "/home/zgorek/Desktop/Projekt Badawczy/DRG/Data/v-coco/coco/images/val2014/"
output_path = "/home/zgorek/Desktop/Projekt Badawczy/DRG/my_tools/test_files/"

df = pd.DataFrame(columns=["images limit", "transformation_type", "transformation_level", "agent AP", "role AP"])

images_limit = 1000
transformation_type = 'rotation'
transformation_levels = [0]

file_name = transformation_type + "_list.txt"
path = "/home/zgorek/Desktop/Projekt Badawczy/DRG/my_tools/test_files/"
file_name = path + file_name

f = open(file_name, "a")

for i, transformation_level in enumerate(transformation_levels):
    name = "{}_{}_{}.pkl".format(transformation_type, transformation_level, images_limit)
    path_detectons = path + "detections_" + name
    test_on_database(path1=path1,
                    path2=path2,
                    path3=path3,
                    RCNN_weights=RCNN_weights,
                    transformation_type=transformation_type, 
                    transformation_level=transformation_level,
                    y = transformation_level,
                    x = transformation_level,
                    images_path=images_path,
                    output_path=output_path,
                    images_limit=images_limit,
                    path=path+name,
                    path_detectons=path_detectons)
    f.write(name + "\n")
# %%
