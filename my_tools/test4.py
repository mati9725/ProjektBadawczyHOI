import pickle
import numpy as np

from predict_boxes import detect_boxes
from visualisation.display_prediction import display_prediction, save_predictions_to_files
from image_transformation.image_transformation import transform_images

path1 = "/home/zgorek/Desktop/Projekt Badawczy/DRG/Data/v-coco/data/vcoco/vcoco_test.json"
path2 = "/home/zgorek/Desktop/Projekt Badawczy/DRG/Data/v-coco/data/instances_vcoco_all_2014.json"
path3 = "/home/zgorek/Desktop/Projekt Badawczy/DRG/Data/v-coco/data/splits/vcoco_test.ids"
images_path = "/home/zgorek/Desktop/Projekt Badawczy/DRG/Data/v-coco/coco/images/val2014/"
output_path = "/home/zgorek/Desktop/Projekt Badawczy/DRG/my_tools/test_files/"
images_limit = 10
transformation_type = 'rotation'
transformation_level = 20
x=0
y=0
img_number = 328

output_file = "/home/zgorek/Desktop/Projekt Badawczy/DRG/my_tools/test_files/rotations/rotation_{}_1000.pkl".format(transformation_level)

# images = transform_images(instances_all_path=path2,
#                     vcoco_test_ids_path=path3,
#                     images_dir_path=images_path, 
#                     output_dir_path=output_path, 
#                     transformation_type=transformation_type, 
#                     transformation_level=transformation_level, 
#                     images_limit=images_limit,
#                     scaled_ratio_axis_x=x,
#                     scaled_ratio_axis_y=y)

# RCNN_weights_path = "/home/zgorek/Desktop/Projekt Badawczy/DRG/pretrained_model/e2e_faster_rcnn_R_50_FPN_1x.pth"
# preds_dict = detect_boxes([img_number], [images[1]], RCNN_weights_path)


# detects = pickle.load(open(output_file, "rb"), encoding='latin1')
# preds_path = '/home/zgorek/Desktop/Projekt Badawczy/DRG/predictions/preds_pan'

# save_predictions_to_files(detects[1], np.array(images[1]), preds_dict[img_number], threshold=0.1, output_files_name_without_extension=preds_path)
detects = pickle.load(open(output_file, "rb"), encoding='latin1')
for i, key in enumerate(detects[0].keys()):
    print(key)
    print(detects[0][key])