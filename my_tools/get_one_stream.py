from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import pickle
import json
import logging

import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.timer import Timer, get_time_str
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.data.datasets.evaluation.vcoco.vsrl_eval import VCOCOeval
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
from maskrcnn_benchmark.utils.apply_prior import apply_prior_Graph

from streams.predict_app_func import im_detect as detector0
from streams.predict_sp_func import im_detect as detector1
from streams.predict_object_func import im_detect as detector2

# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')

def get_one_stream(stream, image_id, path, test_detection):
    #Sciezka do pliku konfiguracyjnego
    if stream is 0:
        config_file = "my_tools/my_configs/VCOCO_app_only.yaml"
        fun = detector0
    elif stream is 1:
        config_file = "my_tools/my_configs/VCOCO_sp_human_only.yaml"
        fun = detector1
    elif stream is 2:
        config_file = "my_tools/my_configs/VCOCO_sp_object_only.yaml"
        fun = detector2
    else:
        return 0
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_file = os.path.join(ROOT_DIR, config_file)

    #Zmienna konfiguracyjna
    cfg.merge_from_file(config_file)
    cfg.merge_from_list([])
    cfg.freeze()

    #Budowanie modelu
    model = build_detection_model(cfg)

    #Wykorzystanie GPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    im_dir = path

    #Fast RCNN detecions
    # path_detection = "/home/zgorek/Desktop/Projekt Badawczy/DRG/Data/Test_Faster_RCNN_R-50-PFN_2x_VCOCO.pkl"
    # test_detection_temp = pickle.load(open(path_detection, "rb"), encoding='latin1')
    # test_detection = {}
    # test_detection[image_id] = test_detection_temp[image_id]

    path_embeddings = "/home/zgorek/Desktop/Projekt Badawczy/DRG/Data/fastText_new.pkl"
    word_embeddings = pickle.load(open(path_embeddings, "rb"), encoding='latin1')

    path_mask = "/home/zgorek/Desktop/Projekt Badawczy/DRG/Data/prior_mask_1.pkl"
    prior_mask = pickle.load(open(path_mask, "rb"), encoding='latin1')

    path_action_dict = "/home/zgorek/Desktop/Projekt Badawczy/DRG/Data/action_index.json"
    action_dic = json.load(open(path_action_dict))
    action_dic_inv = {y: x for x, y in action_dic.items()}

    object_thres = 0.1
    human_thres = 0.8
    prior_flag = 1

    detection = []
    detect_dict = {}
    detect_dict[image_id] = []

    if stream is 0:
        fun(model = model, 
            im_dir = im_dir, 
            image_id = image_id, 
            Test_RCNN = test_detection, 
            fastText = word_embeddings, 
            prior_mask = prior_mask, 
            Action_dic_inv = action_dic_inv, 
            object_thres = object_thres, 
            human_thres = human_thres, 
            prior_flag = prior_flag, 
            detection = detection, 
            detect_app_dict = detect_dict, 
            device = device, 
            cfg = cfg)
    elif stream is 1:
        fun(model = model, 
            im_dir = im_dir, 
            image_id = image_id, 
            Test_RCNN = test_detection, 
            fastText = word_embeddings, 
            prior_mask = prior_mask, 
            Action_dic_inv = action_dic_inv, 
            object_thres = object_thres, 
            human_thres = human_thres, 
            prior_flag = prior_flag, 
            detection = detection, 
            detect_human_centric_dict = detect_dict, 
            device = device, 
            cfg = cfg)
    elif stream is 2:
        fun(model = model, 
            im_dir = im_dir, 
            image_id = image_id, 
            Test_RCNN = test_detection, 
            fastText = word_embeddings, 
            prior_mask = prior_mask, 
            Action_dic_inv = action_dic_inv, 
            object_thres = object_thres, 
            human_thres = human_thres, 
            prior_flag = prior_flag, 
            detection = detection, 
            detect_object_centric_dict = detect_dict, 
            device = device, 
            cfg = cfg)

    return (detection, detect_dict)
