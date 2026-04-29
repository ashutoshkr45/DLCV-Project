import os
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse
from medpy.metric.binary import hd95

categories = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
              'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21, input_type='png', threshold=1.0, calc_hd95=False, printlog=False):
    TP = [multiprocessing.Value('i', 0, lock=True) for _ in range(num_cls)]
    P = [multiprocessing.Value('i', 0, lock=True) for _ in range(num_cls)]
    T = [multiprocessing.Value('i', 0, lock=True) for _ in range(num_cls)]
    
    # HD95 Accumulators (Only used in the final deep scan)
    HD95_sum = [multiprocessing.Value('f', 0.0, lock=True) for _ in range(num_cls)]
    HD95_count = [multiprocessing.Value('i', 0, lock=True) for _ in range(num_cls)]

    def compare(start, step, TP, P, T, HD95_sum, HD95_count, input_type, threshold, calc_hd95):
        for idx in range(start, len(name_list), step):
            name = name_list[idx]
            if input_type == 'png':
                predict_file = os.path.join(predict_folder, f'{name}.png')
                predict = np.array(Image.open(predict_file))
                if num_cls == 81:
                    predict = predict - 91
            elif input_type == 'npy':
                predict_file = os.path.join(predict_folder, f'{name}.npy')
                predict_dict = np.load(predict_file, allow_pickle=True).item()
                h, w = list(predict_dict.values())[0].shape
                tensor = np.zeros((num_cls, h, w), np.float32)
                for key in predict_dict.keys():
                    tensor[key+1] = predict_dict[key]
                tensor[0, :, :] = threshold 
                predict = np.argmax(tensor, axis=0).astype(np.uint8)

            gt_file = os.path.join(gt_folder, f'{name}.png')
            gt = np.array(Image.open(gt_file))
            cal = gt < 255
            mask = (predict == gt) * cal
      
            for i in range(num_cls):
                gt_i = (gt == i) * cal
                pred_i = (predict == i) * cal
                
                P_val = np.sum(pred_i)
                T_val = np.sum(gt_i)
                TP_val = np.sum((gt == i) * mask)

                with P[i].get_lock(): P[i].value += P_val
                with T[i].get_lock(): T[i].value += T_val
                with TP[i].get_lock(): TP[i].value += TP_val

                # HD95 Calculation (Strict Filter: Only if class is present in GT or Pred)
                if calc_hd95 and (T_val > 0 or P_val > 0):
                    if T_val == 0 or P_val == 0:
                        h_val = 373.12866  # Severe penalty for completely missing a present class
                    else:
                        try:
                            h_val = hd95(pred_i, gt_i, voxelspacing=(1, 1))
                        except:
                            h_val = 373.12866
                    
                    with HD95_sum[i].get_lock(): HD95_sum[i].value += h_val
                    with HD95_count[i].get_lock(): HD95_count[i].value += 1

    p_list = []
    for i in range(8):
        p = multiprocessing.Process(target=compare, args=(i, 8, TP, P, T, HD95_sum, HD95_count, input_type, threshold, calc_hd95))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()

    # Compile Final Metrics
    loglist = {}
    IoU, Dice, HD95_avg = [], [], []

    for i in range(num_cls):
        iou = TP[i].value / (T[i].value + P[i].value - TP[i].value + 1e-10)
        dice = 2 * TP[i].value / (T[i].value + P[i].value + 1e-10)
        hd = HD95_sum[i].value / (HD95_count[i].value + 1e-10) if HD95_count[i].value > 0 else 0.0

        IoU.append(iou)
        Dice.append(dice)
        HD95_avg.append(hd)
        
        loglist[categories[i]] = {'IoU': iou * 100, 'Dice': dice * 100, 'HD95': hd}

    miou = np.mean(IoU) * 100
    mdice = np.mean(Dice) * 100
    
    # Only average HD95 for classes that actually appeared in the dataset
    valid_hd95 = [hd for c, hd in zip(HD95_count, HD95_avg) if c.value > 0]
    mhd95 = np.mean(valid_hd95) if len(valid_hd95) > 0 else 0.0

    loglist['mIoU'] = miou
    loglist['mDice'] = mdice
    loglist['mHD95'] = mhd95

    if printlog:
        print(f"\n{'Class':<15} | {'IoU (%)':<10} | {'Dice (%)':<10} | {'HD95':<10}")
        print("-" * 55)
        for i in range(num_cls):
            print(f"{categories[i]:<15} | {IoU[i]*100:<10.3f} | {Dice[i]*100:<10.3f} | {HD95_avg[i]:<10.3f}")
        print("=" * 55)
        print(f"{'MEAN':<15} | {miou:<10.3f} | {mdice:<10.3f} | {mhd95:<10.3f}\n")

    return loglist

def write_final_results(filepath, loglist, comment):
    with open(filepath, 'w') as f:
        import time
        f.write(f"Evaluation Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        f.write(f"Comment: {comment}\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Class':<15} | {'IoU (%)':<10} | {'Dice (%)':<10} | {'HD95':<10}\n")
        f.write("-" * 60 + "\n")
        
        for cat in categories:
            stats = loglist[cat]
            f.write(f"{cat:<15} | {stats['IoU']:<10.3f} | {stats['Dice']:<10.3f} | {stats['HD95']:<10.3f}\n")
            
        f.write("=" * 60 + "\n")
        f.write(f"{'MEAN OVERALL':<15} | {loglist['mIoU']:<10.3f} | {loglist['mDice']:<10.3f} | {loglist['mHD95']:<10.3f}\n")
    print(f"---> Detailed results saved successfully to {filepath}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default='./VOC2012/ImageSets/Segmentation/train.txt', type=str)
    parser.add_argument("--predict_dir", default='./out_rw', type=str)
    parser.add_argument("--gt_dir", default='./VOC2012/SegmentationClass', type=str)
    parser.add_argument('--logfile', default='./evallog.txt',type=str)
    parser.add_argument('--comment', required=True, type=str)
    parser.add_argument('--type', default='png', choices=['npy', 'png'], type=str)
    parser.add_argument('--t', default=None, type=float)
    parser.add_argument('--curve', default=False, type=bool)
    parser.add_argument('--num_classes', default=21, type=int)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=60, type=int)
    args = parser.parse_args()

    if args.type == 'npy':
        assert args.t is not None or args.curve
        
    df = pd.read_csv(args.list, names=['filename'])
    name_list = df['filename'].values
    
    if not args.curve:
        # Standard Single Run
        loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, args.num_classes, args.type, args.t, calc_hd95=True, printlog=True)
        write_final_results("results_voc.txt", loglist, args.comment)
    else:
        # Threshold Search Loop (Fast, NO HD95)
        print("Starting threshold curve search (IoU only)...")
        max_mIoU = 0.0
        best_thr = 0.0
        for i in range(args.start, args.end):
            t = i / 100.0
            loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, args.num_classes, args.type, t, calc_hd95=False)
            print(f'Thr: {t:.2f} \t mIoU: {loglist["mIoU"]:.3f}%')
            
            if loglist['mIoU'] > max_mIoU:
                max_mIoU = loglist['mIoU']
                best_thr = t
            else:
                break
                
        print(f'\nBest background score: {best_thr:.3f} \t Best mIoU: {max_mIoU:.3f}%')
        print("Running final deep evaluation with Best Threshold (Calculating Dice & HD95)...")
        
        # Final Deep Run (Calculates everything and saves to file)
        final_loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, args.num_classes, args.type, best_thr, calc_hd95=True, printlog=True)
        write_final_results("results_voc.txt", final_loglist, f"{args.comment} (Best Thr: {best_thr:.2f})")
