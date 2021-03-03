from evaluation.evaluate import VCOCOeval

import pickle
import numpy as np
import pandas as pd

path1 = "/home/zgorek/Desktop/Projekt Badawczy/DRG/Data/v-coco/data/vcoco/vcoco_test.json"
path2 = "/home/zgorek/Desktop/Projekt Badawczy/DRG/Data/v-coco/data/instances_vcoco_all_2014.json"
path3 = "/home/zgorek/Desktop/Projekt Badawczy/DRG/Data/v-coco/data/splits/vcoco_test.ids"
output_file = "/home/zgorek/Desktop/Projekt Badawczy/DRG/my_tools/test_files/rotations/rotation_0_1000.pkl"


# threshold = 0.3

# coco_eval = VCOCOeval(path1, path2, path3, threshold)
# transformation_type = 'rotation'
# transformation_level = 0
# agent_mAP, agent_TP, agent_FN, agent_FP, role_mAP, role_TP, role_FN, role_FP = coco_eval._do_eval(output_file, 
#                                             ovr_thresh=0.8, 
#                                             transformation_type=transformation_type, 
#                                             transformation_level=transformation_level)

# print(agent_mAP, agent_TP, agent_FN, agent_FP, role_mAP, role_TP, role_FN, role_FP)


threshold = 0.3
distance_threshold = 0.8
transformation_levels = [0.1, 0.15, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 1.75, 2, 3]
# transformation_levels = [2,4, 6, 9, 11, 15, 20, 25, 30, 35, 40]
# transformation_levels = [-20, -17, -14, -11, -8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8, 11, 14,17, 20]
transformation_type = 'scale'
transformation_type_2 = 'xy'

coco_eval = VCOCOeval(path1, path2, path3, threshold)

out = np.zeros((len(transformation_levels), 9), dtype=np.float32)

for i, t_l in enumerate(transformation_levels):
    output_file = "/home/zgorek/Desktop/Projekt Badawczy/DRG/my_tools/test_files/scale_xy/scale_{}_1000.pkl".format(t_l)
    transformation_level = (t_l, t_l)

    agent_mAP, agent_TP, agent_FN, agent_FP, role_mAP, role_TP, role_FN, role_FP = coco_eval._do_eval(output_file, 
                                                ovr_thresh=distance_threshold, 
                                                transformation_type=transformation_type, 
                                                transformation_level=transformation_level)
    out[i, :] = [t_l, agent_mAP, agent_TP, agent_FN, agent_FP, role_mAP, role_TP, role_FN, role_FP]

output_csv = "/home/zgorek/Desktop/Projekt Badawczy/DRG/my_tools/test_files/csv/{}_{}.csv".format(transformation_type, transformation_type_2)
df = pd.DataFrame(out, columns = ['transformation_level','agent_mAP', 'agent_TP', 
                                    'agent_FN', 'agent_FP', 'role_mAP', 'role_TP', 'role_FN', 'role_FP'])
df.to_csv(output_csv)

