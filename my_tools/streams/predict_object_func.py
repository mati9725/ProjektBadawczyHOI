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

# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')

#     apply_prior   prior_mask
# 0        -             -
# 1        Y             -
# 2        -             Y
# 3        Y             Y

def bbox_iou(boxA, boxB):

    ixmin = np.maximum(boxA[0], boxB[0])
    iymin = np.maximum(boxA[1], boxB[1])
    ixmax = np.minimum(boxA[2], boxB[2])
    iymax = np.minimum(boxA[3], boxB[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((boxB[2] - boxB[0] + 1.) * (boxB[3] - boxB[1] + 1.) +
           (boxA[2] - boxA[0] + 1.) *
           (boxA[3] - boxA[1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps


def bbox_trans(human_box_ori, object_box_ori, size=64):
    human_box = human_box_ori.copy()
    object_box = object_box_ori.copy()

    InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                          max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]

    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1

    if height > width:
        ratio = 'height'
    else:
        ratio = 'width'

    # shift the top-left corner to (0,0)

    human_box[0] -= InteractionPattern[0]
    human_box[2] -= InteractionPattern[0]
    human_box[1] -= InteractionPattern[1]
    human_box[3] -= InteractionPattern[1]
    object_box[0] -= InteractionPattern[0]
    object_box[2] -= InteractionPattern[0]
    object_box[1] -= InteractionPattern[1]
    object_box[3] -= InteractionPattern[1]

    if ratio == 'height':  # height is larger than width

        human_box[0] = 0 + size * human_box[0] / height
        human_box[1] = 0 + size * human_box[1] / height
        human_box[2] = (size * width / height - 1) - size * (width - 1 - human_box[2]) / height
        human_box[3] = (size - 1) - size * (height - 1 - human_box[3]) / height

        object_box[0] = 0 + size * object_box[0] / height
        object_box[1] = 0 + size * object_box[1] / height
        object_box[2] = (size * width / height - 1) - size * (width - 1 - object_box[2]) / height
        object_box[3] = (size - 1) - size * (height - 1 - object_box[3]) / height

        # Need to shift horizontally
        InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                              max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
        # assert (InteractionPattern[0] == 0) & (InteractionPattern[1] == 0) & (InteractionPattern[3] == 63) & (InteractionPattern[2] <= 63)
        if human_box[3] > object_box[3]:
            human_box[3] = size - 1
        else:
            object_box[3] = size - 1

        shift = size / 2 - (InteractionPattern[2] + 1) / 2

        human_box += [shift, 0, shift, 0]
        object_box += [shift, 0, shift, 0]

    else:  # width is larger than height

        human_box[0] = 0 + size * human_box[0] / width
        human_box[1] = 0 + size * human_box[1] / width
        human_box[2] = (size - 1) - size * (width - 1 - human_box[2]) / width
        human_box[3] = (size * height / width - 1) - size * (height - 1 - human_box[3]) / width

        object_box[0] = 0 + size * object_box[0] / width
        object_box[1] = 0 + size * object_box[1] / width
        object_box[2] = (size - 1) - size * (width - 1 - object_box[2]) / width
        object_box[3] = (size * height / width - 1) - size * (height - 1 - object_box[3]) / width

        # Need to shift vertically
        InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                              max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]

        # assert (InteractionPattern[0] == 0) & (InteractionPattern[1] == 0) & (InteractionPattern[2] == 63) & (InteractionPattern[3] <= 63)

        if human_box[2] > object_box[2]:
            human_box[2] = size - 1
        else:
            object_box[2] = size - 1

        shift = size / 2 - (InteractionPattern[3] + 1) / 2

        human_box = human_box + [0, shift, 0, shift]
        object_box = object_box + [0, shift, 0, shift]

    return np.round(human_box), np.round(object_box)


def generate_spatial(human_box, object_box):
    H, O = bbox_trans(human_box, object_box)
    Pattern = np.zeros((2, 64, 64))
    Pattern[0, int(H[1]):int(H[3]) + 1, int(H[0]):int(H[2]) + 1] = 1
    Pattern[1, int(O[1]):int(O[3]) + 1, int(O[0]):int(O[2]) + 1] = 1

    return Pattern


def im_detect(model, im_dir, image_id, Test_RCNN, fastText, prior_mask, Action_dic_inv, object_thres, human_thres, prior_flag, detection, detect_object_centric_dict, device, cfg):
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'Data'))
    # if "train" in im_dir:
    #     im_file = os.path.join(DATA_DIR, im_dir, 'COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg')
    # else:
    #     im_file = os.path.join(DATA_DIR, im_dir, 'COCO_val2014_' + (str(image_id)).zfill(12) + '.jpg')
    im_file = im_dir
    img_original = Image.open(im_file)
    img_original = img_original.convert('RGB')
    im_shape = (img_original.height, img_original.width)  # (480, 640)
    transforms = build_transforms(cfg, is_train=False)
    worddim = fastText[1].shape[1]

    for object_out in Test_RCNN[image_id]:
        if (np.max(object_out[5]) > object_thres): # and (object_out[1] == 'Object'): # This is a valid object # it is possible to have human-human interaction

            h_box   = np.empty((0, 4), dtype=np.float32)
            object_word_embedding   = np.empty((0, worddim), dtype=np.float32)
            human_score = np.empty((0, 1), dtype=np.float32)
            object_class = np.empty((0, 1), dtype=np.int32)
            Weight_mask = np.empty((0, 29), dtype=np.float32)

            for human in Test_RCNN[image_id]:
                if (human[1] == 'Human') and (np.max(human[5]) > human_thres) and not (np.all(human[2] == object_out[2])): # This is a valid human
                    h_box_ = np.array([human[2][0], human[2][1], human[2][2], human[2][3]]).reshape(1,4)
                    h_box  = np.concatenate((h_box, h_box_), axis=0)

                    object_word_embedding_ = fastText[object_out[4]]
                    object_word_embedding  = np.concatenate((object_word_embedding, object_word_embedding_), axis=0)

                    # Pattern_ = generate_spatial(human[2], object_out[2]).reshape(1, 2, 64, 64)
                    # Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

                    human_score  = np.concatenate((human_score, np.max(human[5]).reshape(1,1)), axis=0)
                    object_class  = np.concatenate((object_class, np.array(object_out[4]).reshape(1,1)), axis=0)

                    Weight_mask_ = prior_mask[:,object_out[4]].reshape(1,29)
                    Weight_mask  = np.concatenate((Weight_mask, Weight_mask_), axis=0)

            o_box = np.array([object_out[2][0],  object_out[2][1],  object_out[2][2],  object_out[2][3]]).reshape(1,4)

            if len(h_box) == 0:
                continue

            blobs = {}
            pos_num = len(h_box)
            blobs['pos_num'] = pos_num
            human_boxes_cpu = h_box.reshape(pos_num, 4)
            human_boxes = torch.FloatTensor(human_boxes_cpu)
            object_boxes_cpu = np.tile(o_box, [len(h_box), 1]).reshape(pos_num, 4)
            object_boxes = torch.FloatTensor(object_boxes_cpu)

            blobs['object_word_embeddings_object_centric']  = torch.FloatTensor(object_word_embedding).reshape(pos_num, worddim)

            human_boxlist = BoxList(human_boxes, img_original.size, mode="xyxy")  # image_size=(width, height)
            object_boxlist = BoxList(object_boxes, img_original.size, mode="xyxy")  # image_size=(width, height)

            img, human_boxlist, object_boxlist = transforms(img_original, human_boxlist, object_boxlist)

            spatials = []
            for human_box, object_box in zip(human_boxlist.bbox, object_boxlist.bbox):
                ho_spatial = generate_spatial(human_box.numpy(), object_box.numpy()).reshape(1, 2, 64, 64)
                spatials.append(ho_spatial)
            blobs['spatials_object_centric'] = torch.FloatTensor(spatials).reshape(-1, 2, 64, 64)
            blobs['human_boxes'], blobs['object_boxes'] = (human_boxlist,), (object_boxlist,)

            for key in blobs.keys():
                if not isinstance(blobs[key], int) and not isinstance(blobs[key], tuple):
                    blobs[key] = blobs[key].to(device)
                elif isinstance(blobs[key], tuple):
                    blobs[key] = [boxlist.to(device) for boxlist in blobs[key]]

            image_list = to_image_list(img, cfg.DATALOADER.SIZE_DIVISIBILITY)
            image_list = image_list.to(device)

            # compute predictions
            model.eval()
            with torch.no_grad():
                prediction_HO, prediction_H, prediction_O, prediction_sp = model(image_list, blobs)

            #convert to np.array
            prediction_HO = prediction_HO.data.cpu().numpy()
            prediction_H = prediction_H.data.cpu().numpy()
            # prediction_O = prediction_O.data.cpu().numpy()
            prediction_sp = prediction_sp.data.cpu().numpy()

            # test sp branch only
            prediction_HO = prediction_sp

            if prior_flag == 1:
                prediction_HO  = apply_prior_Graph(object_class, prediction_HO)
            if prior_flag == 2:
                prediction_HO  = prediction_HO * Weight_mask
            if prior_flag == 3:
                prediction_HO  = apply_prior_Graph(object_class, prediction_HO)
                prediction_HO  = prediction_HO * Weight_mask

            # save image information
            for idx in range(pos_num):
                human_out = human_boxes_cpu[idx, :]
                dic = {}
                dic['image_id'] = image_id
                dic['person_box'] = human_out
                dic['person_score'] = human_score[idx][0]
                dic['prediction_H'] = prediction_H[idx]
                dic['prediction_sp'] = prediction_sp[idx] # before prior
                dic['object_box'] = object_out[2]
                dic['O_score'] = np.max(object_out[5])
                dic['O_class'] = object_out[4]
                Score_obj = prediction_HO[idx] * np.max(object_out[5])
                Score_obj = np.concatenate((object_out[2], Score_obj), axis=0)
                dic['Score_obj'] = Score_obj

                detect_object_centric_dict[image_id].append(dic)