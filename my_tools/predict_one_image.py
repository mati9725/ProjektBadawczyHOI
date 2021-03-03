# %%
from get_one_stream import get_one_stream
from merge_detections import merge_detections
from visualisation.display_prediction import display_prediction, save_predictions_to_files
from PIL import Image
from skimage import io 
from matplotlib import pyplot as plt
from predict_boxes import detect_boxes
import pickle
import numpy as np
from clear_detection_dict import clear_detection_dict

img_name = 'test3'
img_path = '/home/zgorek/Desktop/Projekt Badawczy/DRG/test_images/' + img_name
img_path = "/home/zgorek/Desktop/Projekt Badawczy/DRG/Data/v-coco/coco/images/val2014/COCO_val2014_000000124013.jpg"
RCNN_weights_path = "/home/zgorek/Desktop/Projekt Badawczy/DRG/pretrained_model/e2e_faster_rcnn_R_50_FPN_1x.pth"
img = Image.open(img_path)
img = img.convert('RGB')

img_number = 0

preds_dict = detect_boxes([img_number], [img], RCNN_weights_path)
# preds_dict = clear_detection_dict(preds_dict, [img_number])

(detection_list_app, detection_dict_app) = get_one_stream(0, [img_number], [img], preds_dict)
(detection_list_human, detection_dict_human) = get_one_stream(1, [img_number], [img], preds_dict)
(detection_list_object, detection_dict_object) = get_one_stream(2, [img_number], [img], preds_dict)

detection_list = merge_detections(detection_dict_app, detection_dict_human, detection_dict_object, [img_number])

# display_prediction(detection_list_app[0], img, 0.2)
# display_prediction(detection_list_human[0], img, 0.1)
# display_prediction(detection_list[0], img, 0.2)

preds_path = '/home/zgorek/Desktop/Projekt Badawczy/DRG/predictions/preds_' + img_name

for key in detection_list[0].keys():
    print(key, detection_list[0][key])

save_predictions_to_files(detection_list[1], np.array(img), preds_dict[0], threshold=0.1, output_files_name_without_extension=preds_path)
# %%
