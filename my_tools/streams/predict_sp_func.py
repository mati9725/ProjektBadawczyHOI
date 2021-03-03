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


def im_detect(model, img_original, image_id, Test_RCNN, fastText, prior_mask, Action_dic_inv, object_thres, human_thres, prior_flag, detection, detect_human_centric_dict, device, cfg):

    # im_orig, im_shape = get_blob(image_id, cfg)

    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'Data'))
    # when using Image.open to read images, img.size= (640, 480), while using cv2.imread, im.shape = (480, 640)
    # to be consistent with previous code, I used img.height, img.width here
    im_shape = (img_original.height, img_original.width)  # (480, 640)
    transforms = build_transforms(cfg, is_train=False)
    worddim = fastText[1].shape[1]

    for Human_out in Test_RCNN[image_id]:
        if (np.max(Human_out[5]) > human_thres) and (Human_out[1] == 'Human'): # This is a valid human

            O_box   = np.empty((0, 4), dtype=np.float32)
            O_vec   = np.empty((0, worddim), dtype=np.float32)
            Pattern = np.empty((0, 2, 64, 64), dtype=np.float32)
            O_score = np.empty((0, 1), dtype=np.float32)
            O_class = np.empty((0, 1), dtype=np.int32)
            Weight_mask = np.empty((0, 29), dtype=np.float32)

            for Object in Test_RCNN[image_id]:
                if (np.max(Object[5]) > object_thres) and not (np.all(Object[2] == Human_out[2])): # This is a valid object
                    O_box_ = np.array([Object[2][0], Object[2][1], Object[2][2], Object[2][3]]).reshape(1,4)
                    O_box  = np.concatenate((O_box, O_box_), axis=0)

                    O_vec_ = fastText[Object[4]]
                    O_vec  = np.concatenate((O_vec, O_vec_), axis=0)

                    Pattern_ = generate_spatial(Human_out[2], Object[2]).reshape(1, 2, 64, 64)
                    Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

                    O_score  = np.concatenate((O_score, np.max(Object[5]).reshape(1,1)), axis=0)
                    O_class  = np.concatenate((O_class, np.array(Object[4]).reshape(1,1)), axis=0)

                    Weight_mask_ = prior_mask[:,Object[4]].reshape(1,29)
                    Weight_mask  = np.concatenate((Weight_mask, Weight_mask_), axis=0)

            H_box = np.array([Human_out[2][0],  Human_out[2][1],  Human_out[2][2],  Human_out[2][3]]).reshape(1,4)

            if len(O_box) == 0:
                continue

            blobs = {}
            pos_num = len(O_box)
            blobs['pos_num'] = pos_num
            # blobs['dropout_is_training'] = False
            human_boxes_cpu = np.tile(H_box, [len(O_box), 1]).reshape(pos_num, 4)
            human_boxes = torch.FloatTensor(human_boxes_cpu)
            object_boxes_cpu = O_box.reshape(pos_num, 4)
            object_boxes = torch.FloatTensor(object_boxes_cpu)

            blobs['object_word_embeddings']  = torch.FloatTensor(O_vec).reshape(pos_num, worddim)

            human_boxlist = BoxList(human_boxes, img_original.size, mode="xyxy")  # image_size=(width, height)
            object_boxlist = BoxList(object_boxes, img_original.size, mode="xyxy")  # image_size=(width, height)

            img, human_boxlist, object_boxlist = transforms(img_original, human_boxlist, object_boxlist)

            spatials = []
            for human_box, object_box in zip(human_boxlist.bbox, object_boxlist.bbox):
                ho_spatial = generate_spatial(human_box.numpy(), object_box.numpy()).reshape(1, 2, 64, 64)
                spatials.append(ho_spatial)

            blobs['spatials'] = torch.FloatTensor(spatials).reshape(-1, 2, 64, 64)
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

            dic_save = {}
            dic_save['image_id'] = image_id
            dic_save['person_box'] = Human_out[2]
            dic_save['person_score'] = np.max(Human_out[5])
            dic_save['prediction_HO'] = prediction_HO
            dic_save['prediction_sp'] = prediction_sp
            dic_save['o_class'] = O_class
            dic_save['object_boxes_cpu'] = object_boxes_cpu
            dic_save['O_score'] = O_score

            detect_human_centric_dict[image_id].append(dic_save)
            
    try:
        del blobs
    except:
        pass

    try:
        del image_list
    except:
        pass

    torch.cuda.empty_cache()

            # if prior_flag == 1:
            #     prediction_HO  = apply_prior_Graph(O_class, prediction_HO)
            # if prior_flag == 2:
            #     prediction_HO  = prediction_HO * Weight_mask
            # if prior_flag == 3:
            #     prediction_HO  = apply_prior_Graph(O_class, prediction_HO)
            #     prediction_HO  = prediction_HO * Weight_mask

            # # save image information
            # dic = {}
            # dic['image_id']   = image_id
            # dic['person_box'] = Human_out[2]

            # Score_obj = prediction_HO * O_score
            # Score_obj = np.concatenate((object_boxes_cpu, Score_obj), axis=1)

            # # # There is only a single human detected in this image. I just ignore it. Might be better to add Nan as object box.
            # # if Score_obj.shape[0] == 0:
            # #     continue

            # # Find out the object box associated with highest action score
            # max_idx = np.argmax(Score_obj,0)[4:]

            # # agent mAP
            # for i in range(29):
            #     #'''
            #     # walk, smile, run, stand
            #     if (i == 3) or (i == 17) or (i == 22) or (i == 27):
            #         agent_name      = Action_dic_inv[i] + '_agent'
            #         dic[agent_name] = np.max(Human_out[5]) * prediction_H[0][i]
            #         continue

            #     # cut
            #     if i == 2:
            #         agent_name = 'cut_agent'
            #         dic[agent_name] = np.max(Human_out[5]) * max(Score_obj[max_idx[2]][4 + 2], Score_obj[max_idx[4]][4 + 4])
            #         continue
            #     if i == 4:
            #         continue

            #     # eat
            #     if i == 9:
            #         agent_name = 'eat_agent'
            #         dic[agent_name] = np.max(Human_out[5]) * max(Score_obj[max_idx[9]][4 + 9], Score_obj[max_idx[16]][4 + 16])
            #         continue
            #     if i == 16:
            #         continue

            #     # hit
            #     if i == 19:
            #         agent_name = 'hit_agent'
            #         dic[agent_name] = np.max(Human_out[5]) * max(Score_obj[max_idx[19]][4 + 19], Score_obj[max_idx[20]][4 + 20])
            #         continue
            #     if i == 20:
            #         continue

            #     # These 2 classes need to save manually because there is '_' in action name
            #     if i == 6:
            #         agent_name = 'talk_on_phone_agent'
            #         dic[agent_name] = np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i]
            #         continue

            #     if i == 8:
            #         agent_name = 'work_on_computer_agent'
            #         dic[agent_name] = np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i]
            #         continue

            #     # all the rest
            #     agent_name =  Action_dic_inv[i].split("_")[0] + '_agent'
            #     dic[agent_name] = np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i]

            # # role mAP
            # for i in range(29):
            #     # walk, smile, run, stand. Won't contribute to role mAP
            #     if (i == 3) or (i == 17) or (i == 22) or (i == 27):
            #         dic[Action_dic_inv[i]] = np.append(np.full(4, np.nan).reshape(1,4), np.max(Human_out[5]) * prediction_H[0][i])
            #         continue

            #     # Impossible to perform this action
            #     if np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i] == 0:
            #        dic[Action_dic_inv[i]] = np.append(np.full(4, np.nan).reshape(1,4), np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i])

            #     # Action with >0 score
            #     else:
            #        dic[Action_dic_inv[i]] = np.append(Score_obj[max_idx[i]][:4], np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i])

            # detection.append(dic)