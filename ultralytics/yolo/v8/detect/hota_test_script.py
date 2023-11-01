import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment


class HOTA:
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
    
    hota_evaluator = HOTA()
    
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
                iou_matrix[i, j] = hota_evaluator.compute_iou(gt_box, pred_box)
        
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

gt_path = '/home/bishoymoussas/Workspace/MOT/R_MTT1_scenes/R_MTT1_1/gt.txt'
pred_path = '/home/bishoymoussas/Workspace/MOT/yolo_tracking/runs/track/exp9/mot/MOT17-01-SDP.txt'
# Load the data
gt_data_df = load_data(gt_path)
pred_data_df = load_data(pred_path)

# Extract unique frame ids from gt data
frames = gt_data_df['frame_id'].unique()

# Compute aggregated MOTA with ID switches for the sequence of imperfect frames
mota_score_with_id_switches_seq, MOTA_FN_with_id_switches_seq, MOTA_FP_with_id_switches_seq, MOTA_ID_Switches_seq, MOTA_GT_with_id_switches_seq = compute_aggregated_mota_with_id_switches_for_sequence(gt_data_df, pred_data_df, frames)

print(f"--- MOTA Metrics for Sequence ---")
print(f"MOTA Score with ID Switches: {mota_score_with_id_switches_seq:.4f} or {mota_score_with_id_switches_seq * 100:.2f}%")
print(f"False Negatives (FN): {MOTA_FN_with_id_switches_seq}")
print(f"False Positives (FP): {MOTA_FP_with_id_switches_seq}")
print(f"ID Switches: {MOTA_ID_Switches_seq}")
print(f"Total Ground Truth Annotations (GT): {MOTA_GT_with_id_switches_seq}")

