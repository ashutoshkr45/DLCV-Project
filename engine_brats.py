import math
import sys
from typing import Iterable
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import pandas as pd
from sklearn.metrics import average_precision_score
from tqdm import tqdm  # NEW: Added tqdm import
import utils

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
           64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128,
           0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 255, 255, 255, 128, 64, 128, 0, 192, 128, 128, 192, 128,
           64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0]

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, loss_scaler, max_norm: float = 0,
                    set_training_mode=True, args=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    train_loop = tqdm(data_loader, desc=f"Train Epoch [{epoch}]", leave=False)
    for samples, targets in train_loop:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        patch_outputs = None
        c_outputs = None
        
        with torch.autocast(device_type='cuda'):
            outputs = model(samples)
            if len(outputs) == 2:
                outputs, patch_outputs = outputs
            elif len(outputs) == 3:
                outputs, c_outputs, patch_outputs = outputs

            # Base classification loss
            loss = F.multilabel_soft_margin_loss(outputs, targets)
            metric_logger.update(mct_loss=loss.item())

            # CCT Module Loss
            if c_outputs is not None:
                c_outputs = c_outputs[-args.num_cct:]
                output_cls_embeddings = F.normalize(c_outputs, dim=-1)  # 12xBxCxD
                scores = output_cls_embeddings @ output_cls_embeddings.permute(0, 1, 3, 2)  # 12xBxCxC

                ground_truth = torch.arange(targets.size(-1), dtype=torch.long, device=device)  # C
                ground_truth = ground_truth.unsqueeze(0).unsqueeze(0).expand(c_outputs.shape[0], c_outputs.shape[1],
                                                                             c_outputs.shape[2])  # 12xBxC
                regularizer_loss = torch.nn.CrossEntropyLoss(reduction='none')(scores.permute(1, 2, 3, 0),
                                                                               ground_truth.permute(1, 2, 0))  # BxCx12
                regularizer_loss = torch.mean(
                    torch.mean(torch.sum(regularizer_loss * targets.unsqueeze(-1), dim=-2), dim=-1) / (
                                torch.sum(targets, dim=-1) + 1e-8))
                metric_logger.update(attn_loss=regularizer_loss.item())
                
                loss = loss + args.loss_weight * regularizer_loss

            # Feature-Level Token Orthogonality (Anti-Collapse)
            if c_outputs is not None and c_outputs.shape[2] >= 2:
                # c_outputs shape: [Layers, B, C, D]. We take the final layer.
                final_tokens = c_outputs[-1]  # Shape: [B, C, D]
                
                # Extract the embedding vectors for Core (0) and Edema (1)
                core_token = final_tokens[:, 0, :]  # Shape: [B, D]
                edema_token = final_tokens[:, 1, :] # Shape: [B, D]
                
                # Compute Cosine Similarity between the feature vectors
                token_sim = F.cosine_similarity(core_token, edema_token, dim=-1) # Shape: [B]
                
                # We want them to be orthogonal (similarity <= 0). 
                # We only penalize positive similarities (when they share features).
                token_penalty_raw = torch.clamp(token_sim, min=0.0)
                
                # Mask: Only apply if BOTH classes are present in the image
                valid_mask = targets[:, 0] * targets[:, 1]
                
                # Compute masked mean (1e-4 prevents FP16 underflow!)
                token_penalty = (token_penalty_raw * valid_mask).sum() / (valid_mask.sum() + 1e-4)
                
                metric_logger.update(sep_loss=token_penalty.item())
                
                # Applying warmup and weight
                if epoch >= args.sep_warmup_epoch:
                    loss = loss + args.sep_loss_weight * token_penalty

            if patch_outputs is not None:
                ploss = F.multilabel_soft_margin_loss(patch_outputs, targets)
                metric_logger.update(pat_loss=ploss.item())
                loss = loss + ploss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        # NEW: Live updates on the tqdm bar
        train_loop.set_postfix(loss=f"{loss_value:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")
        
    metric_logger.synchronize_between_processes()
    print(f"--- Epoch [{epoch}] Train Summary --- | {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    mAP = []

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    val_loop = tqdm(data_loader, desc="Evaluate", leave=False)
    for images, target in val_loop:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images.shape[0]

        with torch.autocast(device_type='cuda'):
            output = model(images)

            if len(output) == 2:
                output, patch_output = output
            elif len(output) == 3:
                output, c_output, patch_output = output

            loss = criterion(output, target)
            output = torch.sigmoid(output)

            mAP_list = compute_mAP(target, output)
            mAP = mAP + mAP_list
            metric_logger.meters['mAP'].update(np.mean(mAP_list), n=batch_size)

        metric_logger.update(loss=loss.item())
        val_loop.set_postfix(loss=f"{loss.item():.4f}")

    metric_logger.synchronize_between_processes()
    print(f"--- Epoch Val Summary --- | mAP: {metric_logger.mAP.global_avg:.3f} | Loss: {metric_logger.loss.global_avg:.4f}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def compute_mAP(labels, outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    
    scores = []
    for i in range(y_true.shape[0]):
        gt = y_true[i]
        pred = y_pred[i]
        
        if np.sum(gt) == 0:
            # If the network correctly predicts no tumor (probabilities < 0.5)
            if np.sum(pred > 0.5) == 0:
                scores.append(1.0)
            else:
                scores.append(0.0)
                
        # If the slice has BOTH Core and Edema (No negative class for AP math)
        elif np.sum(gt) == len(gt):
            # Calculate simple binary accuracy for this slice
            if np.sum(pred > 0.5) == len(gt):
                scores.append(1.0)
            else:
                scores.append(0.0)
                
        else:
            ap_i = average_precision_score(gt, pred)
            scores.append(ap_i)
            
    return scores


@torch.no_grad()
def generate_attention_maps_ms(data_loader, model, device, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generating attention maps:'
    if args.attention_dir is not None:
        Path(args.attention_dir).mkdir(parents=True, exist_ok=True)
    if args.cam_npy_dir is not None:
        Path(args.cam_npy_dir).mkdir(parents=True, exist_ok=True)

    model.eval()

    # Dynamically load the correct image names directly from the CSV
    csv_path = os.path.join(args.data_path, f'{args.split}.csv')
    df = pd.read_csv(csv_path)
    img_list = df.iloc[:, 0].apply(lambda x: os.path.basename(x).replace('.png', '')).tolist()

    index = 0
    gen_loop = tqdm(data_loader, desc="Generating Maps", leave=False)
    for image_list, target in gen_loop:
        images1 = image_list[0].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images1.shape[0]
        
        img_name = img_list[index]
        index += 1

        img_temp = images1.permute(0, 2, 3, 1).detach().cpu().numpy()
        
        orig_images = (img_temp * 255.0).astype(np.float32)

        w_orig, h_orig = orig_images.shape[1], orig_images.shape[2]

        with torch.autocast(device_type='cuda'):
            cam_list = []
            vitattn_list = []
            cam_maps = None
            for s in range(len(image_list)):
                images = image_list[s].to(device, non_blocking=True)
                w, h = images.shape[2] - images.shape[2] % args.patch_size, images.shape[3] - images.shape[3] % args.patch_size
                w_featmap = w // args.patch_size
                h_featmap = h // args.patch_size

                if 'MCTformerV1' in args.model:
                    output, cls_attentions, patch_attn = model(images, return_att=True, n_layers=args.layer_index)
                    cls_attentions = cls_attentions.reshape(batch_size, args.nb_classes, w_featmap, h_featmap)
                    patch_attn = torch.sum(patch_attn, dim=0)
                else:
                    output, cls_attentions, patch_attn = model(images, return_att=True, n_layers=args.layer_index,
                                                               attention_type=args.attention_type)
                    patch_attn = torch.sum(patch_attn, dim=0)

                if args.patch_attn_refine:
                    cls_attentions = torch.matmul(patch_attn.unsqueeze(1), cls_attentions.view(cls_attentions.shape[0],cls_attentions.shape[1], -1, 1)).reshape(cls_attentions.shape[0],cls_attentions.shape[1], w_featmap, h_featmap)

                cls_attentions = F.interpolate(cls_attentions, size=(w_orig, h_orig), mode='bilinear', align_corners=False)[0]
                cls_attentions = cls_attentions.cpu().numpy() * target.clone().view(args.nb_classes, 1, 1).cpu().numpy()

                if s % 2 == 1:
                    cls_attentions = np.flip(cls_attentions, axis=-1)
                cam_list.append(cls_attentions)
                vitattn_list.append(cam_maps)

            sum_cam = np.sum(cam_list, axis=0)
            sum_cam = torch.from_numpy(sum_cam)
            sum_cam = sum_cam.unsqueeze(0).to(device)

            output = torch.sigmoid(output)

        if args.visualize_cls_attn:
            for b in range(images.shape[0]):
                cam_dict = {} # Initialize for ALL images
                norm_cam = np.zeros((args.nb_classes, w_orig, h_orig))
                
                # Only calculate CAMs if there is a tumor present
                if (target[b].sum()) > 0:
                    for cls_ind in range(args.nb_classes):
                        if target[b,cls_ind] > 0:
                            cls_score = format(output[b, cls_ind].cpu().numpy(), '.3f')
                            cls_attention = sum_cam[b,cls_ind,:]
                            cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min() + 1e-8)
                            cls_attention = cls_attention.cpu().numpy()

                            cam_dict[cls_ind] = cls_attention
                            norm_cam[cls_ind] = cls_attention

                            if args.attention_dir is not None:
                                fname = os.path.join(args.attention_dir, img_name + '_' + str(cls_ind) + '_' + str(cls_score) + '.png')
                                show_cam_on_image(orig_images[b], cls_attention, fname)

                if args.cam_npy_dir is not None:
                    np.save(os.path.join(args.cam_npy_dir, img_name + '.npy'), cam_dict)

    metric_logger.synchronize_between_processes()
    return


def show_cam_on_image(img, mask, save_path):
    img = np.float32(img) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cv2.imwrite(save_path, cam)