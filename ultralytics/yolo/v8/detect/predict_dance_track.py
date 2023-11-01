# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra
import torch
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

import cv2
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}
ground_truth = None
deepsort = None

# Initialize VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for .mp4 format
fps = 20  # frame rate, you may want to set it based on your input video
out = None  # We will initialize it later

def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort_0.yaml")

    deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=None,
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



def draw_boxes(img, bbox, names, object_id, identities=None, ground_truth_boxes=None, offset=(0, 0)):
    height, width, _ = img.shape

    # Remove tracked point from buffer if object is lost
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    # Draw detected boxes
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        center = (int((x2 + x1) / 2), int((y2 + y2) / 2))
        id = int(identities[i]) if identities is not None else 0

        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)

        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = '{}{:d}'.format("", id) + ":" + '%s' % (obj_name)

        data_deque[id].appendleft(center)
        UI_box(box, img, label=label, color=color, line_thickness=2)

        for i in range(1, len(data_deque[id])):
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)

    # Draw ground truth boxes
    if ground_truth_boxes is not None:
        for box in ground_truth_boxes:
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Red color for ground truth

    return img


def read_ground_truth(file_path):
    ground_truth = {}
    with open(file_path, 'r') as f:
        for line in f.readlines():
            elements = line.strip().split(",")
            try:
                frame_id = int(elements[0][3:])  # Removing 'SER' and converting to int
                subject_id = elements[1]
                x1, y1, x2, y2 = map(float, elements[2:])
            except ValueError:
                continue
            if frame_id not in ground_truth:
                ground_truth[frame_id] = []
            ground_truth[frame_id].append([subject_id, int(x1), int(y1), int(x2), int(y2)])
    return ground_truth






class DetectionPredictor(BasePredictor):

    def __init__(self, cfg, ground_truth=None):
        super().__init__(cfg)
        self.ground_truth = ground_truth  # initialize with ground truth if provided
        self.total_TP = 0
        self.total_FP = 0
        self.total_FN = 0
        self.total_IDSW = 0
        self.object_states = {}  # Initialize the object states dictionary
        self.total_gt_objects = 0  # Initialize the total ground truth objects
        self.total_association = 0
        self.track_quality = {} 
        self.MOTA_list = []
        self.HOTA_list = []


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
    

    def iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1_gt, y1_gt, x2_gt, y2_gt = box2

        # Calculate intersection area
        xi1 = max(x1, x1_gt)
        yi1 = max(y1, y1_gt)
        xi2 = min(x2, x2_gt)
        yi2 = min(y2, y2_gt)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # Calculate union area
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
        union_area = box1_area + box2_area - inter_area

        # Avoid divide by zero
        if union_area == 0:
            return 0

        # Calculate IoU
        iou = inter_area / union_area
        return iou

    def calculate_overall_metrics(self):
        overall_precision = self.total_TP / (self.total_TP + self.total_FP) if (self.total_TP + self.total_FP) > 0 else 0
        overall_recall = self.total_TP / (self.total_TP + self.total_FN) if (self.total_TP + self.total_FN) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

        print(f"Overall Metrics: Precision={overall_precision}, Recall={overall_recall}, F1-Score={overall_f1}")

        total_GT = self.total_TP + self.total_FN  # Global GT
        if total_GT > 0:
            overall_MOTA = 1 - (self.total_FP + self.total_FN + self.total_IDSW) / total_GT
        else:
            overall_MOTA = 0  # or some other value to indicate undefined MOTA

        print(f"Overall MOTA: {overall_MOTA}")

        detection_score = self.total_TP / (self.total_TP + 0.5 * (self.total_FP + self.total_FN))
        association_score = self.total_association / self.total_TP  # This is a simplified example
        HOTA_score = (detection_score * association_score) ** 0.5
        print(f"Overall MOTA: {HOTA_score}")

    def plot_metrics(self):
        plt.figure()
        
        plt.subplot(2, 1, 1)
        plt.plot(self.MOTA_list)
        plt.title('MOTA per Frame')
        plt.xlabel('Frame')
        plt.ylabel('MOTA')
        
        plt.subplot(2, 1, 2)
        plt.plot(self.HOTA_list)
        plt.title('HOTA per Frame')
        plt.xlabel('Frame')
        plt.ylabel('HOTA')
        
        plt.tight_layout()
        plt.show()
        plt.savefig('hota_mota_2.png')




    def write_results(self, idx, preds, batch):
        global out  
        p, im, im0 = batch
        image_name = p.name
        # frame_id = int(image_name.split('.')[0])
        frame_id = int(image_name.split('.')[0][3:])
        print('Frame ID', frame_id)
        all_outputs = []
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]
        self.seen += 1
        im0 = im0.copy()
        frame = getattr(self.dataset, 'frame', 0)
        print('Frame',frame)
        self.data_path = p
        save_path = str(self.save_dir / p.name)
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]
        self.annotator = self.get_annotator(im0)
        det = preds[idx]
        all_outputs.append(det)
        if len(det) == 0:
            return log_string
        if out is None:
            height, width, _ = im0.shape
            out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))
        
        

        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        xywh_bboxs = []
        confs = []
        oids = []
        outputs = []
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
        try:
            outputs = deepsort.update(xywhs, confss, oids, im0)
        except IndexError as e:
            print(f"IndexError caught: {e}")
            print("Skipping this frame due to an error.")
            return log_string  # Return the log string as is
        if len(outputs) > 0:

            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            frame_TP = 0  # Reset frame-level counts
            frame_FP = 0
            gt_boxes = ground_truth.get(frame_id, [])
            self.total_gt_objects += len(gt_boxes)
            frame_FN = 0
            frame_IDSW = 0
            frame_FN = len(gt_boxes)
            print(f"Ground truth boxes for frame {frame_id}: {gt_boxes}")  # Debugging line
            
            for i, box in enumerate(bbox_xyxy):
                if i < len(confs):  # Check if index exists in confs
                    conf_value = confs[i][0]  # Assuming confs is a list of lists
                    label = f"{identities[i]}: {self.model.names[object_id[i]]} ({conf_value:.2f})"  # Updated label
                    color = compute_color_for_labels(object_id[i])
                    UI_box(box, im0, label=label, color=color, line_thickness=2)
                else:
                    print(f"Skipping index {i} as it's not in confs list.")
                
                for j, identity in enumerate(identities):
                    unique_object_id = f"{frame_id}_{i}"
                    old_identity = self.object_states.get(unique_object_id, None)
                    
                    if old_identity is not None and old_identity != identity:
                        frame_IDSW += 1
                        
                    self.object_states[unique_object_id] = identity


                gt_boxes_xyxy = [box[1:] for box in gt_boxes]                # Calculate IoU for each detection with each ground truth box
                for det_box in bbox_xyxy:
                    best_iou = 0.0
                    for gt_box in gt_boxes:
                        iou_value = self.iou(det_box, gt_box[1:])
                        best_iou = max(best_iou, iou_value)
                    
                    if best_iou >= 0.5:
                        self.total_association += best_iou
                        frame_TP += 1
                        frame_FN -= 1
                    else:
                        frame_FP += 1
                    
                    # Calculate Precision, Recall, and F1-Score for this frame (Optional)
                    if frame_TP + frame_FP > 0:
                        frame_precision = frame_TP / (frame_TP + frame_FP)
                    else:
                        frame_precision = 0.0

                    if frame_TP + frame_FN > 0:
                        frame_recall = frame_TP / (frame_TP + frame_FN)
                    else:
                        frame_recall = 0.0

                    if frame_precision + frame_recall > 0:
                        frame_f1 = 2 * (frame_precision * frame_recall) / (frame_precision + frame_recall)
                    else:
                        frame_f1 = 0.0
            # 3. Update global counts and calculate MOTA and HOTA for the frame
            self.total_TP += frame_TP
            self.total_FP += frame_FP
            self.total_FN += frame_FN
            self.total_IDSW += frame_IDSW  # Here we add frame-level ID switches to the global count

            # Calculate and append MOTA and HOTA for this frame
            total_GT = frame_TP + frame_FN  # Local GT for this frame
            # Calculate and append MOTA and HOTA for this frame

            total_GT = frame_TP + frame_FN  # Local GT for this frame
            if total_GT > 0:
                frame_MOTA = 1 - (frame_FP + frame_FN + frame_IDSW) / total_GT
            else:
                frame_MOTA = 0  # or some other value to indicate undefined MOTA

            detection_score = self.total_TP / (self.total_TP + 0.5 * (self.total_FP + self.total_FN))
            association_score = self.total_association / self.total_TP  # This is a simplified example
            HOTA_score = (detection_score * association_score) ** 0.5
            
            self.MOTA_list.append(frame_MOTA)
            self.HOTA_list.append(HOTA_score)
            print(f'Frame TP: {frame_TP}, Frame FP: {frame_FP}, Frame FN: {frame_FN}')
            print(f"Frame {frame_id} MOTA: {frame_MOTA}")
            print(f"Frame {frame_id} HOTA: {HOTA_score}")    
            print(f"Frame {frame_id} Metrics: Precision={frame_precision}, Recall={frame_recall}, F1-Score={frame_f1}")



            # Draw the boxes
            # draw_boxes(im0, bbox_xyxy, self.model.names, object_id, identities, gt_boxes_xyxy)
            # Write the frame to the output video
            # out.write(im0)
        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    init_tracker()
    global ground_truth
    ground_truth = read_ground_truth("/home/bishoymoussas/Workspace/MOT/R_MTT1_labelled/gt.txt")    
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = '/home/bishoymoussas/Workspace/MOT/R_MTT1_labelled/'
    print(cfg)
    predictor = DetectionPredictor(cfg)
    predictor()
    predictor.calculate_overall_metrics()
    predictor.plot_metrics()


if __name__ == "__main__":
    predict()
    if out is not None:
        out.release()
