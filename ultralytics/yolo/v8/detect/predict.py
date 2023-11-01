# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra
import torch
import argparse
import time
from pathlib import Path
import json
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import base64
from io import BytesIO
import cv2
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np
from collections import defaultdict, deque
import cv2
import os
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}
GROUND_TRUTH_COLOR = (255, 0, 0)  # Bright red
deepsort = None
color_map = defaultdict(tuple)
gt_tracks = defaultdict(deque)
pred_tracks = defaultdict(deque)
MAX_HISTORY = 20  # Maximum history for the tracks

# Initialize VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for .mp4 format
fps = 20  # frame rate, you may want to set it based on your input video
out = None  # We will initialize it later

def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("/home/bishoymoussas/Workspace/MOT/YOLOv8-DeepSORT-Object-Tracking/ultralytics/yolo/v8/detect/deep_sort_pytorch/configs_ga/deep_sort_gn.yaml")

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

def compute_color_for_id(obj_id):
    """
    Generate a color based on the object ID.
    """
    obj_id = int(obj_id)  # Ensure obj_id is an integer
    if obj_id not in color_map:
        # If the ID is not in the color_map, generate a new color for it
        random.seed(obj_id)
        color_map[obj_id] = tuple([random.randint(0, 255) for _ in range(3)])
    return color_map[obj_id]

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

def draw_boxes_and_tracks(img, data, tracks, is_ground_truth=True):
    """
    Draw boxes and their tracks on the given image.
    """
    for entry in data:
        if is_ground_truth:
            # Extract id and box coordinates for ground truth data
            id, x1, y1, x2, y2 = entry
            id = int(id)  # Convert id to integer
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        else:
            # id and box coordinates for prediction data
            x1, y1, x2, y2, id, *_ = entry
            id = int(id)  # Convert id to integer

        color = compute_color_for_id(id)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # Draw the ID on the bounding box in white color
        text_color = (255, 255, 255)
        text_position = (x1 + 5, y1 + 15)  # Slightly inside the box at the top-left corner
        cv2.putText(img, str(id), text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)

        # Calculate the center of the lower side of the bounding box
        lower_center = (int((x1 + x2) / 2), y2)

        # Draw tracks
        if id in tracks:
            history = tracks[id]
            for i in range(1, len(history)):
                # Draw the track from the lower center of the previous bounding box to the current lower center
                cv2.line(img, history[i - 1], history[i], color, 2)

        # Update the history with the lower center
        tracks[id].appendleft(lower_center)
        if len(tracks[id]) > MAX_HISTORY:
            tracks[id].pop()

    return img





import numpy as np
def process_frame(img, ground_truth_data, prediction_data, gt_output_dir, pred_output_dir, filename):
    # Clone the image for ground truth and predictions
    img_gt = img.copy()
    img_pred = img.copy()

    # Draw ground truth
    draw_boxes_and_tracks(img_gt, ground_truth_data, gt_tracks)
    # Draw predictions
    draw_boxes_and_tracks(img_pred, prediction_data, pred_tracks, False)

    # Calculate the height needed for the canvas to accommodate both sets of tracking paths
    max_y_gt = max([max(track, key=lambda x: x[1])[1] for track in gt_tracks.values()], default=0)
    max_y_pred = max([max(track, key=lambda x: x[1])[1] for track in pred_tracks.values()], default=0)
    canvas_height = max(max_y_gt, max_y_pred, 100)  # Ensure a minimum canvas height of 100 (or adjust as needed)

    # Create separate white canvases with the same width as the frame for ground truth and predictions
    canvas_width = img.shape[1]
    white_canvas_gt = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    white_canvas_pred = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    # Define a scaling factor to adjust the vertical placement of tracking paths on the canvas
    scaling_factor = 3  # Adjust as needed

    # Define a list of unique colors for each track
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Add more colors as needed

    # Draw tracking paths on the white canvas for ground truth
    for idx, (id, track) in enumerate(gt_tracks.items()):
        track = list(track)
        track_color = track_colors[idx % len(track_colors)]  # Cycle through colors
        for i in range(1, len(track)):
            # Map the coordinates to the white canvas
            x1, y1 = track[i - 1]
            x2, y2 = track[i]
            # Offset the y-coordinate to draw the path at the bottom of the canvas
            y1_canvas = canvas_height - 1 - (canvas_height - 1 - y1) * scaling_factor
            y2_canvas = canvas_height - 1 - (canvas_height - 1 - y2) * scaling_factor
            cv2.line(white_canvas_gt, (x1, y1_canvas), (x2, y2_canvas), track_color, 4)  # Increase track width

    # Draw tracking paths on the white canvas for predictions
    for idx, (id, track) in enumerate(pred_tracks.items()):
        track = list(track)
        track_color = track_colors[idx % len(track_colors)]  # Cycle through colors
        for i in range(1, len(track)):
            # Map the coordinates to the white canvas
            x1, y1 = track[i - 1]
            x2, y2 = track[i]
            # Offset the y-coordinate to draw the path at the bottom of the canvas
            y1_canvas = canvas_height - 1 - (canvas_height - 1 - y1) * scaling_factor
            y2_canvas = canvas_height - 1 - (canvas_height - 1 - y2) * scaling_factor
            cv2.line(white_canvas_pred, (x1, y1_canvas), (x2, y2_canvas), track_color, 4)  # Increase track width

    # Stack the white canvas below the frame for ground truth and predictions
    stacked_img_gt = np.vstack((img_gt, white_canvas_gt))
    stacked_img_pred = np.vstack((img_pred, white_canvas_pred))

    # Save the images
    cv2.imwrite(os.path.join(gt_output_dir, filename + '_gt.png'), img_gt)
    cv2.imwrite(os.path.join(pred_output_dir, filename + '_pred.png'), img_pred)
    cv2.imwrite(os.path.join('/home/bishoymoussas/Workspace/MOT/output_video_frames/gt_track/', filename + '_gt_canvas.png'), white_canvas_gt)
    cv2.imwrite(os.path.join('/home/bishoymoussas/Workspace/MOT/output_video_frames/preds_track/', filename + '_pred_canvas.png'), white_canvas_pred)



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
          data_deque[id] = deque(maxlen= 60)
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

def read_ground_truth(file_path):
    """
    Read the ground truth text file and return a dictionary with filename as key and boxes with IDs as values.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    ground_truths = {}
    for line in lines:
        elements = line.strip().split(',')
        filename = elements[0]
        obj_id = int(elements[1])  # Extract object ID
        box = [obj_id] + list(map(float, elements[2:]))  # Include object ID with the box
        if filename not in ground_truths:
            ground_truths[filename] = []
        ground_truths[filename].append(box)

    return ground_truths


def draw_ground_truth_boxes(img, boxes, color=GROUND_TRUTH_COLOR, thickness=1):
    """
    Draw the ground truth boxes on the image.
    """
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

class DetectionPredictor(BasePredictor):

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

        global out  # Declare out as global so it's not local to the function
        p, im, im0 = batch
        _, buffer = cv2.imencode('.jpg', im0)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        height, width, _ = im0.shape
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
        ground_truth_file = "/home/bishoymoussas/Workspace/MOT/R_MTT2/R_MTT2_subscenes/subscene_1/gt.txt"
        ground_truth_data = read_ground_truth(ground_truth_file)
        self.data_path = p
        save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        all_outputs.append(det)
        if len(det) == 0:
            return log_string

        if out is None:
            height, width, _ = im0.shape
            out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        xywh_bboxs = []
        confs = []
        oids = []
        person_class_id = 0  # Replace this with the actual class ID for "person" in your model
        for *xyxy, conf, cls in reversed(det):
            if int(cls) == person_class_id:  # Only process 'person' class
                x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                xywh_bboxs.append(xywh_obj)
                confs.append([conf.item()])
                oids.append(int(cls))

        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)

        # Handle the deepsort.update call
        outputs = []
        try:
            outputs = deepsort.update(xywhs, confss, oids, im0)
        except IndexError as e:
            print(f"IndexError caught: {e}")
            print("Skipping this frame due to an error.")

        # Create the json_data dictionary
        json_data = {
            "version": "4.5.6",
            "flags": {},
            "shapes": [],
            "imagePath": str(p.name),  # Assuming `p.name` is the image filename
            "imageHeight": height,  # Include the image height
            "imageWidth": width,  # Include the image width
            "imageData": image_base64
        }
        image_filename = p.stem  # Extracting image filename from the path

        # Process detections if they exist
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]

            for i, box in enumerate(bbox_xyxy):
                if i < len(confs):  # Check if index exists in confs
                    conf_value = confs[i][0]  # Assuming confs is a list of lists
                    label = f"{identities[i]}: {self.model.names[object_id[i]]} ({conf_value:.2f})"  # Updated label
                    color = compute_color_for_labels(object_id[i])
                    x1, y1, x2, y2 = map(float, box)  # Assuming box is already in float or int format

                    shape_info = {
                        "label": str(identities[i]),  # Use only the subject ID as label
                        "points": [[x1, y1], [x2, y2]],
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {}
                    }

                    json_data["shapes"].append(shape_info)
                else:
                    print(f"Skipping index {i} as it's not in confs list.")
        else:
            print("No detections in this frame.")

        # Save the frame
        process_frame(im0, ground_truth_data.get(image_filename, []), outputs, "/home/bishoymoussas/Workspace/MOT/sup_mat/r_mtt1_1_gt_ga/", "/home/bishoymoussas/Workspace/MOT/sup_mat/r_mtt1_1_preds_ga/", image_filename)

        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    init_tracker()
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    print(cfg)
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict()
    if out is not None:
        out.release()
