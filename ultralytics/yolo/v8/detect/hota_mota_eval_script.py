
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import re
# base_path = '/home/bishoymoussas/Workspace/MOT/R_MTT1_scenes/R_MTT1_0/'
# gt_path = base_path +'gt.txt'
# pred_path = base_path + 'preds.txt'
# Define the paths
algo = 'strongsort'
base_path = '/home/bishoymoussas/Workspace/MOT/R_MTT1_subsc/subscene_4/'
gt_path = base_path + 'gt.txt'
output_file_path = base_path + '{}_results.csv'.format(algo)
# output_file_path = base_path + 'results.csv'
# pred_path = base_path + 'MAX_DIST_0.2_MIN_CONFIDENCE_0.3_NMS_MAX_OVERLAP_0.5_MAX_IOU_DISTANCE_0.7_MAX_AGE_70_N_INIT_3_NN_BUDGET_100_deep_preds.txt'

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
# Initialize HOTA evaluator
hota_evaluator = HOTA()

# # Load data
# gt_data_df = pd.read_csv(gt_path, header=None, names=['frame_id', 'obj_id', 'x_tl', 'y_tl', 'x_br', 'y_br'])
# pred_data_df = pd.read_csv(pred_path, header=None, names=['frame_id', 'obj_id', 'x_tl', 'y_tl', 'x_br', 'y_br'])

# Evaluate HOTA for the sequence
# hota_results = hota_evaluator.eval_sequence(gt_data_df, pred_data_df)

# Compute final HOTA score
# def compute_hota_score(hota_results):
#     tp = hota_results['HOTA_TP']
#     fn = hota_results['HOTA_FN']
#     fp = hota_results['HOTA_FP']
    
#     det_re = tp / (tp + fn)
#     det_pr = tp / (tp + fp)
#     det_a = tp / (tp + fn + fp)
    
#     hota_score = np.sqrt(det_a)  # Assuming AssA = 1 for simplicity
#     return hota_score, det_re, det_pr, det_a
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



# hota_score, det_re, det_pr, det_a = compute_hota_score(hota_results)



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


# Initialize extended HOTA evaluator
extended_hota_evaluator = ExtendedHOTA()

# # Evaluate extended HOTA metrics for the sequence
# extended_hota_results = extended_hota_evaluator.eval_sequence_extended(gt_data_df, pred_data_df)

# # Compute final HOTA and extended metrics
# extended_hota_score, extended_det_re, extended_det_pr, extended_det_a = compute_hota_score(extended_hota_results)

# total_gt_objects = len(gt_data_df)

# # Existing print statement for HOTA and MOTA metrics
# print(f"--- HOTA and MOTA Metrics ---")
# print(f"Detection Recall (DetRe): {det_re:.4f}")
# print(f"Detection Precision (DetPr): {det_pr:.4f}")
# print(f"Detection Accuracy (DetA): {det_a:.4f}")

# print(f"HOTA Score: {extended_hota_score:.4f}")
# print(f"ID Switches: {extended_hota_results['ID_Switches']}")
# print(f"Fragmentation: {extended_hota_results['Fragmentation']}")
# print(f"Track Quality (TQ): {extended_hota_results['Track_Quality']:.4f}")
# print(f"IDF1 Score: {extended_hota_results['IDF1_Score']:.4f}")




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

# Load the data
# gt_data_df = load_data(gt_path)
# pred_data_df = load_data(pred_path)

# Extract unique frame ids from gt data
# frames = gt_data_df['frame_id'].unique()

# Compute aggregated MOTA with ID switches for the sequence of imperfect frames
# mota_score_with_id_switches_seq, MOTA_FN_with_id_switches_seq, MOTA_FP_with_id_switches_seq, MOTA_ID_Switches_seq, MOTA_GT_with_id_switches_seq = compute_aggregated_mota_with_id_switches_for_sequence(gt_data_df, pred_data_df, frames)

# print(f"--- MOTA Metrics for Sequence ---")
# print(f"MOTA Score with ID Switches: {mota_score_with_id_switches_seq:.4f} or {mota_score_with_id_switches_seq * 100:.2f}%")
# print(f"False Negatives (FN): {MOTA_FN_with_id_switches_seq}")
# print(f"False Positives (FP): {MOTA_FP_with_id_switches_seq}")
# print(f"ID Switches: {MOTA_ID_Switches_seq}")
# print(f"Total Ground Truth Annotations (GT): {MOTA_GT_with_id_switches_seq}")

# Initialize the evaluators
hota_evaluator = ExtendedHOTA()
mota_evaluator = MOTA()

# Prepare an empty list to store the results
results_list = []

# Loop through all files in the directory
for filename in os.listdir(base_path):
    
    if algo in filename and 'results' not in filename:
    # if 'MAX_DIST_' in filename:
        # Construct the full path to the preds file
        preds_path = os.path.join(base_path, filename)
        print(preds_path)
        # Load the ground truth and preds data
        gt_data_df = pd.read_csv(gt_path, header=None, names=['frame_id', 'obj_id', 'x_tl', 'y_tl', 'x_br', 'y_br'])
        pred_data_df = pd.read_csv(preds_path, header=None, names=['frame_id', 'obj_id', 'x_tl', 'y_tl', 'x_br', 'y_br'])
        
        # Extract configuration details from the filename
        config_details = re.findall(r'([A-Za-z_]+)_([\d.]+)', filename)
        config_dict = {key: value for key, value in config_details}
        
        # Evaluate HOTA and MOTA for the sequence
        hota_results = hota_evaluator.eval_sequence_extended(gt_data_df, pred_data_df)
        hota_score, det_re, det_pr, det_a, idf1, track_qual = compute_hota_score(hota_results)
        
        frames = gt_data_df['frame_id'].unique()
        mota_score_with_id_switches_seq, _, _, _, _ = compute_aggregated_mota_with_id_switches_for_sequence(gt_data_df, pred_data_df, frames)
        
        # Prepare a dictionary with results and configuration details
        result_dict = {
            'filename': filename,
            'HOTA_Score': hota_score,
            'DetRe': det_re,
            'DetPr': det_pr,
            'DetA': det_a,
            'MOTA_Score': mota_score_with_id_switches_seq,
            'idf1': idf1,
            'track quality': track_qual,
            **config_dict
        }
        
        # Append the results to the list
        results_list.append(result_dict)
for res in results_list:
    print(res['filename'], res['idf1'])
# Convert the results list to a DataFrame and write to a CSV file
results_df = pd.DataFrame(results_list)
# results_df.to_csv(output_file_path, index=False)
