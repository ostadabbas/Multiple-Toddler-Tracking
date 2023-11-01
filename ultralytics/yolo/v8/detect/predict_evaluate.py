
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import re
import yaml
from pathlib import Path
import random
import hydra
import torch
import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import yaml
import torch.backends.cudnn as cudnn
from numpy import random
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from omegaconf import OmegaConf
import cv2
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np

class HOTA:
    def __init__(self):
        self.array_labels = np.arange(0.05, 0.99, 0.05)
    
    def compute_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1_, y1_, x2_, y2_ = box2
        xi1, yi1, xi2, yi2 = max(x1, x1_), max(y1, y1_), min(x2, x2_), min(y2, y2_)
        inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_ - x1_) * (y2_ - y1_)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area
        return iou

    def visualize_boxes(self, gt_boxes, pred_boxes, matched_indices):
        fig, ax = plt.subplots(1)
        ax.set_xlim(0, 3840)
        ax.set_ylim(0, 2160)
        for i, box in enumerate(gt_boxes):
            color = 'blue'
            for (gt_index, pred_index) in matched_indices:
                if i == gt_index:
                    color = 'green'
                    break
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
        for i, box in enumerate(pred_boxes):
            color = 'red'
            for (gt_index, pred_index) in matched_indices:
                if i == pred_index:
                    color = 'green'
                    break
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
        plt.show()

    def eval_sequence(self, gt_data, pred_data):
        res = {'HOTA_TP': 0, 'HOTA_FN': 0, 'HOTA_FP': 0}
        unique_frames = gt_data['frame_id'].unique()
        for frame in unique_frames:
            gt_boxes = gt_data[gt_data['frame_id'] == frame][['x_tl', 'y_tl', 'x_br', 'y_br']].values
            pred_boxes = pred_data[pred_data['frame_id'] == frame][['x_tl', 'y_tl', 'x_br', 'y_br']].values
            if len(gt_boxes) == 0:
                res['HOTA_FP'] += len(pred_boxes)
                continue
            if len(pred_boxes) == 0:
                res['HOTA_FN'] += len(gt_boxes)
                continue
            iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
            for i, gt_box in enumerate(gt_boxes):
                for j, pred_box in enumerate(pred_boxes):
                    iou_matrix[i, j] = self.compute_iou(gt_box, pred_box)
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_iou = iou_matrix[row_ind, col_ind]
            res['HOTA_TP'] += sum(matched_iou >= 0.3)
            res['HOTA_FN'] += sum(matched_iou < 0.3)
            res['HOTA_FP'] += len(pred_boxes) - sum(matched_iou >= 0.3)
        return res

class ExtendedHOTA(HOTA):
    def __init__(self):
        super().__init__()

    def eval_sequence_extended(self, gt_data, pred_data):
        res = super().eval_sequence(gt_data, pred_data)
        res['ID_Switches'] = 0
        res['Fragmentation'] = 0
        res['Track_Quality'] = 0
        gt_last_id = {}
        pred_last_id = {}
        idtp = 0
        idfp = 0
        idfn = 0
        unique_frames = gt_data['frame_id'].unique()
        for frame in unique_frames:
            gt_boxes = gt_data[gt_data['frame_id'] == frame]
            pred_boxes = pred_data[pred_data['frame_id'] == frame]
            if len(gt_boxes) == 0:
                res['HOTA_FP'] += len(pred_boxes)
                continue
            if len(pred_boxes) == 0:
                res['HOTA_FN'] += len(gt_boxes)
                continue
            iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
            for i, (_, gt_row) in enumerate(gt_boxes.iterrows()):
                for j, (_, pred_row) in enumerate(pred_boxes.iterrows()):
                    gt_box = gt_row[['x_tl', 'y_tl', 'x_br', 'y_br']].values
                    pred_box = pred_row[['x_tl', 'y_tl', 'x_br', 'y_br']].values
                    iou_matrix[i, j] = self.compute_iou(gt_box, pred_box)
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            for i, j in zip(row_ind, col_ind):
                if i < iou_matrix.shape[0] and iou_matrix[i, j] >= 0.5:
                    gt_id = gt_boxes.iloc[i]['obj_id']
                    pred_id = pred_boxes.iloc[j]['obj_id']
                    if gt_id in gt_last_id and pred_id != gt_last_id[gt_id]:
                        res['ID_Switches'] += 1
                    if pred_id in pred_last_id and gt_id != pred_last_id[pred_id]:
                        res['Fragmentation'] += 1
                    gt_last_id[gt_id] = pred_id
                    pred_last_id[pred_id] = gt_id
                    idtp += 1
                else:
                    idfn += 1
            idfp += len(pred_boxes) - sum(iou_matrix[row_ind, col_ind] >= 0.5)
        # res['IDF1_Score'] = idtp / (idtp + 0.5 * (idfp + idfn))
        denominator = idtp + 0.5 * (idfp + idfn)
        res['IDF1_Score'] = idtp / denominator if denominator != 0 else 0
        # res['Track_Quality'] = res['HOTA_TP'] / (res['HOTA_TP'] + res['HOTA_FP'] + res['ID_Switches'])
        denominator = res['HOTA_TP'] + res['HOTA_FP'] + res['ID_Switches']
        res['Track_Quality'] = res['HOTA_TP'] / denominator if denominator != 0 else 0

        return res
    
class MOTA:
    def __init__(self):
        pass
    
    def compute_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1_, y1_, x2_, y2_ = box2
        xi1, yi1, xi2, yi2 = max(x1, x1_), max(y1, y1_), min(x2, x2_), min(y2, y2_)
        inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_ - x1_) * (y2_ - y1_)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area != 0 else 0
        return iou


def load_data(filepath):
    data_df = pd.read_csv(filepath, header=None, names=['frame_id', 'obj_id', 'x_tl', 'y_tl', 'x_br', 'y_br'])
    return data_df


def compute_aggregated_mota_with_id_switches_for_sequence(gt_data, pred_data, frames):
    # Initialize counters for the sequence of frames
    MOTA_FN, MOTA_FP, MOTA_ID_Switches, MOTA_GT = 0, 0, 0, 0
    last_id_mapping = {}  # Dictionary to keep track of the mapping of GT ID to predicted ID in the last frame
    
    mota_evaluator = MOTA()
    
    for frame in frames:
        frame_gt_data = gt_data[gt_data['frame_id'] == frame]
        frame_pred_data = pred_data[pred_data['frame_id'] == frame] if frame in pred_data['frame_id'].unique() else pd.DataFrame()
        
        # Count GT and TP in the current frame
        GT = len(frame_gt_data)
        TP = 0
        FP = len(frame_pred_data)
        id_mapping = {}  # Dictionary to keep track of the mapping of GT ID to predicted ID in the current frame
        iou_matrix = np.zeros((len(frame_gt_data), len(frame_pred_data)))
        for i, (_, gt_row) in enumerate(frame_gt_data.iterrows()):
            for j, (_, pred_row) in enumerate(frame_pred_data.iterrows()):
                gt_box = gt_row[['x_tl', 'y_tl', 'x_br', 'y_br']].values
                pred_box = pred_row[['x_tl', 'y_tl', 'x_br', 'y_br']].values
                iou_matrix[i, j] = mota_evaluator.compute_iou(gt_box, pred_box)
        
        # Count the number of matched boxes (TP) based on IoU threshold and detect ID switches
        while iou_matrix.size > 0 and iou_matrix.max() > 0.5:
            max_iou_idx = np.unravel_index(np.argmax(iou_matrix, axis=None), iou_matrix.shape)
            TP += 1
            FP -= 1  # Reduce the FP count by 1 for every matched box
            gt_id = frame_gt_data.iloc[max_iou_idx[0]]['obj_id']
            pred_id = frame_pred_data.iloc[max_iou_idx[1]]['obj_id']
            id_mapping[gt_id] = pred_id  # Update the ID mapping for the current frame
            
            # Check for ID switches based on the ID mapping from the last frame
            if gt_id in last_id_mapping and last_id_mapping[gt_id] != pred_id:
                MOTA_ID_Switches += 1
            
            iou_matrix[max_iou_idx[0], :] = -1  # Set entire row to -1
            iou_matrix[:, max_iou_idx[1]] = -1  # Set entire column to -1
        
        # Update the MOTA counters for the sequence of frames
        MOTA_FN += GT - TP
        MOTA_FP += FP
        MOTA_GT += GT
        last_id_mapping = id_mapping  # Update the last ID mapping to be the current ID mapping
    
    # Compute final aggregated MOTA with ID switches for the sequence of frames
    mota_score_with_id_switches = 1 - (MOTA_FN + MOTA_FP + MOTA_ID_Switches) / MOTA_GT if MOTA_GT != 0 else 0
    
    return mota_score_with_id_switches, MOTA_FN, MOTA_FP, MOTA_ID_Switches, MOTA_GT

def write_config_to_yaml(config, filename):
    with open(filename, 'w') as file:
        yaml.dump(config, file)


def initialize_population(pop_size, yaml_dir):
    population = []
    for _ in range(pop_size):
        config = {
            'DEEPSORT': {
                'REID_CKPT': "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7",
                'MAX_DIST': random.uniform(0.1, 1.0),
                'MIN_CONFIDENCE': random.uniform(0.2, 0.7),
                'NMS_MAX_OVERLAP': random.uniform(0.2, 0.8),
                'MAX_IOU_DISTANCE': random.uniform(0.2, 0.9),
                'MAX_AGE': random.randint(10, 200),
                'N_INIT': random.randint(2, 15),
                'NN_BUDGET': random.randint(20, 120)
            }
        }
        filename = yaml_dir / f"config_{len(population)}.yaml"
        write_config_to_yaml(config, filename)
        population.append(filename)
    return population

def fitness(metrics):
    return metrics['HOTA_Score'] + metrics['MOTA_Score'] + metrics['idf1']

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}
deepsort = None

def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
    
def init_tracker_with_config(yaml_file: Path):
    # Initialize the tracker with the specified YAML configuration file.
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file(str(yaml_file))  # Convert Path object to string
    
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
##########################################################################################
def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def compute_color_for_id(obj_id):
    """
    Generate a color based on the object ID.
    """
    # Use the object ID to seed the random number generator for consistency
    random.seed(obj_id)
    color = [random.randint(0, 255) for _ in range(3)]
    return tuple(color)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: #person
        color = (85,45,255)
    elif label == 2: # Car
        color = (222,82,175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)



def draw_boxes(img, bbox, names,object_id, identities=None, offset=(0, 0)):
    #cv2.line(img, line[0], line[1], (46,162,112), 3)
    MAX_THICKNESS = 10  # You can set this to a suitable maximum value
    height, width, _ = img.shape
    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
      if key not in identities:
        data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge
        center = (int((x2+x1)/ 2), int((y2+y2)/2))

        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in data_deque:  
          data_deque[id] = deque(maxlen= 512)
        # color = compute_color_for_labels(object_id[i])
        color = compute_color_for_id(id)
        obj_name = names[object_id[i]]
        label = '{}{:d}'.format("", id) + ":"+ '%s' % (obj_name)

        # add center to buffer
        data_deque[id].appendleft(center)
        UI_box(box, img, label=label, color=color, line_thickness=2)
        # draw trail
        for i in range(1, len(data_deque[id])):
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            thickness = max(1, min(MAX_THICKNESS, int(np.sqrt(64 / float(i + 1)) * 1.5)))  # Ensure thickness is in [1, MAX_THICKNESS]
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)
    return img

def compute_hota_score(hota_results):
    tp = hota_results['HOTA_TP']
    fn = hota_results['HOTA_FN']
    fp = hota_results['HOTA_FP']
    idf1 = hota_results['IDF1_Score']
    track_qual = hota_results['Track_Quality']
    
    det_re = tp / (tp + fn) if tp + fn != 0 else 0
    det_pr = tp / (tp + fp) if tp + fp != 0 else 0
    det_a = tp / (tp + fn + fp) if tp + fn + fp != 0 else 0
    
    hota_score = np.sqrt(det_a) if det_a > 0 else 0  # Assuming AssA = 1 for simplicity
    
    return hota_score, det_re, det_pr, det_a, idf1, track_qual


def convert_config_to_yaml_path(config):
    filename = '_'.join([f"{key}_{value}" for key, value in config.items()]) + '.yaml'
    return os.path.join(yaml_dir, filename)


def generate_yaml_from_config(config, yaml_dir):
    """
    Generates a YAML configuration file based on the given configuration dictionary.
    """
    # Ensure the directory exists
    os.makedirs(yaml_dir, exist_ok=True)

    # Add the 'DEEPSORT' root key and 'REID_CKPT' key
    structured_config = {
        "DEEPSORT": {
            **config,
            "REID_CKPT": "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"
        }
    }

    # Convert the config to its corresponding YAML filename
    filename = '_'.join([f"{key}_{value}" for key, value in config.items()]) + '.yaml'
    filepath = os.path.join(yaml_dir, filename)

    # Write the structured_config to a YAML file
    with open(filepath, 'w') as file:
        yaml.dump(structured_config, file)

    print(f"Generating YAML at: {filepath}")  # Debug print statement

    return filepath



class DetectionPredictor(BasePredictor):
    def __init__(self, args, filename):
        super().__init__(args)
        self.frame_id = 0
        self.filename = filename

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds
    def write_results(self, idx, preds, batch):
        f_results = open(self.filename, 'a')
        p, im, im0 = batch
        all_outputs = []
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        xywh_bboxs = []
        confs = []
        oids = []
        person_class_id = 0
        for *xyxy, conf, cls in reversed(det):
            if int(cls) == person_class_id:
                x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                xywh_bboxs.append(xywh_obj)
                confs.append([conf.item()])
                oids.append(int(cls))
        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)
          
        # try:
        outputs = deepsort.update(xywhs, confss, oids, im0)
        # except IndexError as e:

            # return log_string  # Return the log string as is
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            
            draw_boxes(im0, bbox_xyxy, self.model.names, object_id,identities)
            # Write the results to the text file
            for i, box in enumerate(bbox_xyxy):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                label_id = int(identities[i])
                f_results.write(f"{p.stem},{label_id},{x1},{y1},{x2},{y2}\n")
        self.frame_id += 1 
        return log_string

frames_path = '/home/bishoymoussas/Workspace/MOT/R_MTT2/R_MTT2_subscenes/subscene_1/'
yaml_dir = Path('/home/bishoymoussas/Workspace/MOT/YOLOv8-DeepSORT-Object-Tracking/ultralytics/yolo/v8/detect/deep_sort_pytorch/configs_ga_agg/')
output_file_path = frames_path + '{}_results.csv'.format('ga_agg')
@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    for yaml_file in yaml_dir.glob('*.yaml'):
            hota_evaluator = ExtendedHOTA()
            # Load the configuration from the YAML file
            cfg_deep = get_config()
            cfg_deep.merge_from_file(str(yaml_file))
            
            exp_config = cfg_deep['DEEPSORT']
            # Construct a unique filename for preds.txt based on the YAML file name
            filename_parts = [
                f"{key}_{value}" for key, value in exp_config.items() 
                if key != "REID_CKPT" and not callable(value)  # Exclude method objects
            ]
            unique_filename = '_'.join(filename_parts) + '_ga_preds.txt'
            full_file_path = os.path.join(frames_path, unique_filename)
            
            # Remove the file if it already exists
            if os.path.exists(full_file_path):
                os.remove(full_file_path)
            
            # Initialize the tracker and set up the predictor with loaded cfg
            init_tracker_with_config(yaml_file)
            cfg.model = cfg.model or "yolov8n.pt"
            cfg.source = frames_path
            cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
            cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
            
            predictor = DetectionPredictor(cfg, str(full_file_path))
            predictor()
            # Evaluation
            gt_data_df = load_data(frames_path+ 'gt.txt')
            pred_data_df = load_data(full_file_path)
            
            # Evaluate HOTA for the sequence
            hota_results = hota_evaluator.eval_sequence_extended(gt_data_df, pred_data_df)
            hota_score, det_re, det_pr, det_a, idf1, track_qual = compute_hota_score(hota_results)
            
            # Evaluate MOTA for the sequence
            frames = gt_data_df['frame_id'].unique()
            mota_score_with_id_switches_seq, _, _, _, _ = compute_aggregated_mota_with_id_switches_for_sequence(gt_data_df, pred_data_df, frames)
            
            # Prepare and return a dictionary with results and configuration details
            result_dict = {
                'filename': unique_filename,
                'HOTA_Score': hota_score,
                'DetRe': det_re,
                'DetPr': det_pr,
                'DetA': det_a,
                'MOTA_Score': mota_score_with_id_switches_seq,
                'idf1': idf1,
                'track quality': track_qual,
            }
            results_df = pd.DataFrame([result_dict])
            results_df.to_csv(output_file_path, index=False)
            return result_dict

if __name__ == "__main__":
    predict()