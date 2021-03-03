from get_one_stream import get_one_stream
from merge_detections import merge_detections
from visualisation.display_prediction import display_prediction, save_predictions_to_files
from PIL import Image
from skimage import io 
from matplotlib import pyplot as plt
from predict_boxes import detect_boxes
import pickle
import numpy as np
from maskrcnn_benchmark.data.datasets.evaluation.vcoco.vsrl_eval import VCOCOeval
import sys
from image_transformation.image_transformation import transform_images
import torch

def test_on_database(path1, 
                    path2, 
                    path3, 
                    RCNN_weights, 
                    images_path, 
                    output_path, 
                    transformation_type, 
                    transformation_level,
                    path,
                    path_detectons,
                    images_limit=None,
                    x = 1,
                    y = 1
                    ):

    test_idx_path = "/home/zgorek/Desktop/Projekt Badawczy/DRG/my_tools/test_idx.txt"
    img_numbers = []
    with open(test_idx_path) as f:
        lines = f.readlines()
    for line in lines:
        img_numbers.append(int(line))

    if images_limit is not None:
        img_numbers = img_numbers[:images_limit]

    images = transform_images(instances_all_path=path2,
                     vcoco_test_ids_path=path3,
                     images_dir_path=images_path, 
                     output_dir_path=output_path, 
                     transformation_type=transformation_type, 
                     transformation_level=transformation_level, 
                     images_limit=images_limit,
                     scaled_ratio_axis_x=x,
                     scaled_ratio_axis_y=y)


    print("RCNN")
    preds_dict = detect_boxes(img_numbers, images, RCNN_weights)

    print("Predykcje 1 streamu")
    (detection_list_app, detection_dict_app) = get_one_stream(0, img_numbers, images, preds_dict)
    print("Predykcje 2 streamu")
    (detection_list_human, detection_dict_human) = get_one_stream(1, img_numbers, images, preds_dict)
    print("Predykcje 3 streamu")
    (detection_list_object, detection_dict_object) = get_one_stream(2, img_numbers, images, preds_dict)

    print("Łączenie wyników")
    detection_list = merge_detections(detection_dict_app, detection_dict_human, detection_dict_object, img_numbers)

    pickle.dump(detection_list, open(path , "wb" ) )
    pickle.dump(preds_dict, open(path_detectons , "wb" ) )

    del images
    torch.cuda.empty_cache()




