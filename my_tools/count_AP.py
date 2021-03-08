from evaluation.evaluate import VCOCOeval

import pickle
import numpy as np
import pandas as pd


import os
from os.path import dirname

DRG_path = dirname(os.getcwd())

path1 = DRG_path + "/Data/v-coco/data/vcoco/vcoco_test.json"
path2 = DRG_path + "/Data/v-coco/data/instances_vcoco_all_2014.json"
path3 = DRG_path + "/Data/v-coco/data/splits/vcoco_test.ids"
output_file = DRG_path + "/my_tools/test_files/rotations/rotation_0_1000.pkl"


threshold = 0.3
distance_threshold = 0.8
transformation_levels = [0]
transformation_type = 'rotation'
transformation_type_2 = None

coco_eval = VCOCOeval(path1, path2, path3, threshold)

out = np.zeros((len(transformation_levels), 9), dtype=np.float32)

for i, t_l in enumerate(transformation_levels):
    output_file = DRG_path + "/my_tools/test_files/{}_{}_1000.pkl".format(transformation_type, t_l)
    if transformation_type == 'scale':
        transformation_level = (t_l, t_l)
    else:
        transformation_level = t_l

    agent_mAP, agent_TP, agent_FN, agent_FP, role_mAP, role_TP, role_FN, role_FP = coco_eval._do_eval(output_file, 
                                                ovr_thresh=distance_threshold, 
                                                transformation_type=transformation_type, 
                                                transformation_level=transformation_level)
    out[i, :] = [t_l, agent_mAP, agent_TP, agent_FN, agent_FP, role_mAP, role_TP, role_FN, role_FP]

output_csv = DRG_path + "/my_tools/test_files/{}_{}.csv".format(transformation_type, transformation_type_2)
df = pd.DataFrame(out, columns = ['transformation_level','agent_mAP', 'agent_TP', 
                                    'agent_FN', 'agent_FP', 'role_mAP', 'role_TP', 'role_FN', 'role_FP'])
df.to_csv(output_csv)

