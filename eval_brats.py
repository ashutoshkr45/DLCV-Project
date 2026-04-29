import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import cv2
from medpy.metric.binary import hd95
from torchvision import transforms

def compute_binary_dice(gt, pred):
    num_gt = np.sum(gt)
    num_pred = np.sum(pred)
    if num_gt == 0:
        if num_pred == 0:
            return 1.0
        else:
            return 0.0
    else:
        intersection = np.logical_and(gt, pred)
        return (2. * np.sum(intersection)) / (num_gt + num_pred)

def compute_binary_mIOU(gt, pred):
    intersection = np.logical_and(gt, pred)
    union = np.logical_or(gt, pred)
    iou_score = (np.sum(intersection) + 1e-5) / (np.sum(union) + 1e-5)
    return iou_score

def compute_binary_HD95(gt, pred):
    num_gt = np.sum(gt)
    num_pred = np.sum(pred)
    if num_gt == 0 and num_pred == 0:
        return 0.0
    if num_gt == 0 or num_pred == 0:
        return 373.12866
    return hd95(pred, gt, voxelspacing=(1, 1))

def compute_seg_metrics(gt, pred):
    result = {}
    gt = gt.astype(np.uint8)
    pred = pred.astype(np.uint8)
    
    result['Dice'] = compute_binary_dice(gt, pred)
    result['IoU'] = compute_binary_mIOU(gt, pred)
    result['HD95'] = compute_binary_HD95(gt, pred)
    return result


def keep_largest_component(mask):
    """Removes scattered noise by keeping only the largest connected blob."""
    if np.sum(mask) == 0:
        return mask
        
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # If only background is found, or only one component exists, return as is
    if num_labels <= 1:
        return mask
        
    # Find the label of the largest connected component (excluding background at index 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8)

def postprocess_masks(pred_core, pred_edema, kernel_size=7):
    clean_core = keep_largest_component(pred_core)
    clean_edema = keep_largest_component(pred_edema)
    
    # Combining into "Whole Tumor" to protect the internal boundary
    whole_tumor = np.logical_or(clean_core, clean_edema).astype(np.uint8)
    
    # Erode only the outer perimeter(False positives) of the Whole Tumor
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed_wt = cv2.morphologyEx(whole_tumor, cv2.MORPH_CLOSE, kernel)
    eroded_wt = cv2.erode(closed_wt, kernel, iterations=2)
    
    # Ensuring perfectly flush borders
    final_core = clean_core
    final_edema = eroded_wt.copy()
    final_edema[final_core == 1] = 0  
    
    return final_core, final_edema

# Evaluation Pipeline
def do_brats_eval(args, df, threshold=1.0, save_csv=False):
    result_metric = {
        'Core Dice': [], 'Core IoU': [], 'Core HD95': [],
        'Edema Dice': [], 'Edema IoU': [], 'Edema HD95': []
    }
    
    if save_csv:
        csv_file = open(args.csv_output, "w+")
        csv_file.writelines("Img Name, Core Dice, Core IoU, Core HD95, Edema Dice, Edema IoU, Edema HD95\n")
        csv_file.close()

    eval_bar = tqdm(df.iterrows(), total=len(df), desc=f"Evaluating (t={threshold:.2f})", leave=False)

    for index, row in eval_bar:
        img_name = os.path.basename(row['image_path']).replace('.png', '')
        gt_path = os.path.join(args.base_dir, row['mask_path'])
        
        # Load Ground Truth 
        gt_pil = Image.open(gt_path).convert("RGB")
        gt_img = np.array(transforms.CenterCrop(224)(gt_pil))
        
        gt_core = ((gt_img[:, :, 0] > 0) | (gt_img[:, :, 1] > 0)).astype(np.uint8)
        gt_edema = (gt_img[:, :, 2] > 0).astype(np.uint8)
        
        # Load Prediction
        if args.type == 'npy':
            predict_file = os.path.join(args.predict_dir, f"{img_name}.npy")
            predict_dict = np.load(predict_file, allow_pickle=True).item()
            
            h, w = gt_img.shape[:2]
            core_pred_prob = predict_dict.get(0, np.zeros((h, w)))
            edema_pred_prob = predict_dict.get(1, np.zeros((h, w)))
            bg_prob = np.full((h, w), threshold)
            
            tensor = np.stack([bg_prob, core_pred_prob, edema_pred_prob], axis=0)
            pred_argmax = np.argmax(tensor, axis=0).astype(np.uint8)
            
            pred_core = (pred_argmax == 1).astype(np.uint8)
            pred_edema = (pred_argmax == 2).astype(np.uint8)
            
        elif args.type == 'png':
            predict_file = os.path.join(args.predict_dir, f"{img_name}.png")
            pred_img = np.array(Image.open(predict_file))
            pred_core = (pred_img == 1).astype(np.uint8)
            pred_edema = (pred_img == 2).astype(np.uint8)

        # APPLY DIP POST-PROCESSING 
        pred_core, pred_edema = postprocess_masks(pred_core, pred_edema)

        # Compute Metrics
        res_core = compute_seg_metrics(gt_core, pred_core)
        res_edema = compute_seg_metrics(gt_edema, pred_edema)
        
        if save_csv:
            with open(args.csv_output, "a") as f:
                f.write(f"{img_name}, {res_core['Dice']:.3f}, {res_core['IoU']:.3f}, {res_core['HD95']:.3f}, "
                        f"{res_edema['Dice']:.3f}, {res_edema['IoU']:.3f}, {res_edema['HD95']:.3f}\n")

        result_metric['Core Dice'].append(res_core['Dice'])
        result_metric['Core IoU'].append(res_core['IoU'])
        result_metric['Core HD95'].append(res_core['HD95'])
        result_metric['Edema Dice'].append(res_edema['Dice'])
        result_metric['Edema IoU'].append(res_edema['IoU'])
        result_metric['Edema HD95'].append(res_edema['HD95'])

    avg_metrics = {k: np.mean(v) for k, v in result_metric.items()}
    std_metrics = {k: np.std(v) for k, v in result_metric.items()}
    
    mean_dice_list = [(c + e) / 2.0 for c, e in zip(result_metric['Core Dice'], result_metric['Edema Dice'])]
    avg_metrics['Mean Dice'] = np.mean(mean_dice_list)
    std_metrics['Mean Dice'] = np.std(mean_dice_list)
    
    return avg_metrics, std_metrics


def writelog(filepath, metric_dict, comment):
    with open(filepath, 'a') as logfile:
        logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        logfile.write(f'\t{comment}\n')
        for k, v in metric_dict.items():
            logfile.write(f"{k}: {v:.3f}  ")
        logfile.write('\n=====================================\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default='../WSS-Interclass-Sep/test.csv', type=str)
    parser.add_argument("--base_dir", default='../WSS-Interclass-Sep', type=str)
    parser.add_argument("--predict_dir", required=True, type=str)
    parser.add_argument("--csv_output", default='tumor_result.csv', type=str)
    parser.add_argument('--logfile', default='./evallog_brats.txt', type=str)
    parser.add_argument('--comment', default='BraTS Eval', type=str)
    
    parser.add_argument('--type', default='npy', choices=['npy', 'png'], type=str)
    parser.add_argument('--t', default=0.59, type=float)
    parser.add_argument('--curve', action='store_true')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=60, type=int)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    if not args.curve:
        print(f"Starting Evaluation for {len(df)} images...")
        avg_metrics, std_metrics = do_brats_eval(args, df, threshold=args.t)
        writelog(args.logfile, avg_metrics, args.comment)
        
        summary_text = "\n--- Final Average Results ---\n"
        for k in avg_metrics.keys():
            summary_text += f"{k}: {avg_metrics[k]:.3f} ± {std_metrics[k]:.3f}\n"
            
        print(summary_text)
        
        with open("final_metrics_summary.txt", "w") as f:
            f.write(summary_text)
        print("Saved printed metrics to final_metrics_summary.txt")
        
    else:
        max_mean_dice = 0.0
        best_thr = 0.0
        best_loglist = None
        
        print("Running curve evaluation to find optimal background threshold...")
        for i in range(args.start, args.end):
            t = i / 100.0
            avg_metrics, std_metrics = do_brats_eval(args, df, threshold=t)
            print(f"Thr: {t:.2f} \t Mean Dice: {avg_metrics['Mean Dice']:.3f} \t Core: {avg_metrics['Core Dice']:.3f} \t Edema: {avg_metrics['Edema Dice']:.3f}")
            
            if avg_metrics['Mean Dice'] > max_mean_dice:
                max_mean_dice = avg_metrics['Mean Dice']
                best_thr = t
                best_loglist = (avg_metrics, std_metrics)

        print(f"\nBest background score: {best_thr:.2f} \t Best Mean Dice: {max_mean_dice:.3f}")
        do_brats_eval(args, df, threshold=best_thr)
        
        avg_metrics, std_metrics = best_loglist
        writelog(args.logfile, avg_metrics, f"{args.comment} (Best Thr: {best_thr:.2f})")
        
        summary_text = f"\n--- Final Average Results (Best Thr: {best_thr:.2f}) ---\n"
        for k in avg_metrics.keys():
            summary_text += f"{k}: {avg_metrics[k]:.3f} ± {std_metrics[k]:.3f}\n"
            
        print(summary_text)
        
        summary_file = "final_metrics_summary.txt"
        with open(summary_file, "w") as f:
            f.write(summary_text)
        print(f"Saved printed metrics to {summary_file}")