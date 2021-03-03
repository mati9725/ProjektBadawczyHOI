import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from streams.predictor import COCODemo

import time

def detect_boxes(images_id, images, weight_loading):
    config_file = "my_tools/my_configs/RCNN.yaml"
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_file = os.path.join(ROOT_DIR, config_file)
    confidence_threshold = 0.05
    show_mask_heatmaps = False
    masks_per_dim = 2
    min_image_size = 224

    # load config from file and command-line arguments
    cfg.merge_from_file(config_file)
    cfg.merge_from_list([])
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold = confidence_threshold,
        show_mask_heatmaps = show_mask_heatmaps,
        masks_per_dim = masks_per_dim,
        min_image_size = min_image_size,
        weight_loading = weight_loading
    )

    out_dict = {}
    for image, image_id in zip(images, tqdm(images_id)):
        image = np.array(image)
        preds = coco_demo.compute_prediction(image)
        top_preds = coco_demo.select_top_predictions(preds)

        labels = top_preds.get_field('labels').numpy()
        scores = top_preds.get_field('scores').numpy()
        bbox = top_preds.bbox.numpy()
        
        preds = []

        for label, score, box in zip(labels, scores, bbox):
            tmp = []
            tmp.append(image_id)
            if int(label) is 1:
                tmp.append('Human')
            else:
                tmp.append('Object')
            tmp.append(box)
            tmp.append(np.nan)
            tmp.append(label)
            tmp.append(np.array(score))
            preds.append(tmp)

        out_dict[image_id] = preds
    return out_dict