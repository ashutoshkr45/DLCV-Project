import argparse
import datetime
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler

from datasets_brats import build_dataset
from engine_brats import train_one_epoch, evaluate, generate_attention_maps_ms
import models
import utils

def save_training_curves(history, output_dir):
    """Generates and saves training curves inside the output directory."""
    plots_dir = os.path.join(output_dir, "training_curves")
    os.makedirs(plots_dir, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue', marker='o')
    plt.plot(epochs, history['val_loss'], label='Val Loss', color='orange', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(plots_dir, 'loss_curve.png'), bbox_inches='tight')
    plt.close()
    
    # Plot mAP Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_mAP'], label='Val mAP', color='green', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.title('Validation mAP over Epochs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(plots_dir, 'map_curve.png'), bbox_inches='tight')
    plt.close()
    
    print(f"\n---> Training curves successfully saved to {plots_dir}")

def get_args_parser():
    parser = argparse.ArgumentParser('MCTformer BraTS training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=45, type=int)

    # Model parameters
    parser.add_argument('--model', default='deit_small_MCTformerPlus', type=str, metavar='MODEL')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--weight-decay', type=float, default=0.05)
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME')
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--train-interpolation', type=str, default='bicubic')
    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT')
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--resplit', action='store_true', default=False)

    # Dataset & Run parameters
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--data-path', default='AME-CAM/src', type=str, help='base directory containing csv files')
    parser.add_argument('--data-set', default='BRATS', type=str, help='dataset mode (BRATS or BRATSMS)')
    parser.add_argument('--split', default='train', type=str, help='Target CSV split for generation (train, val, test)')
    
    parser.add_argument('--output_dir', default='saved_model', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin-mem', action='store_true')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Generating attention maps
    parser.add_argument('--gen_attention_maps', default=False, action='store_true')
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--attention-dir', type=str, default='cam-png')
    parser.add_argument('--layer-index', type=int, default=12, help='extract attention maps from the last layers')
    parser.add_argument('--patch-attn-refine', type=bool, default=True)
    parser.add_argument('--visualize-cls-attn', type=bool, default=True)
    parser.add_argument('--cam-npy-dir', type=str, default='cam-npy')
    parser.add_argument("--scales", nargs='+', type=float, default=[1.0, 0.75, 1.25])
    parser.add_argument('--attention-type', type=str, default='fused')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument("--loss-weight", default=1.0, type=float)
    parser.add_argument("--num-cct", default=12, type=int)
    # Adding argument for the Inter-Class Separability Loss weight
    parser.add_argument("--sep-loss-weight", default=0.5, type=float, help="Weight for spatial separability loss")
    parser.add_argument("--sep-warmup-epoch", default=2, type=int, help="Epoch to start applying sep loss")

    return parser


def main(args):
    print(args)
    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args, split='train')
    dataset_attn, args.nb_classes = build_dataset(is_train=False, args=args, split=args.split)
    dataset_val, _ = build_dataset(is_train=False, args=args, split='val')

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True,
    )

    data_loader_attn = torch.utils.data.DataLoader(
        dataset_attn, batch_size=1, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
    )
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val, batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False
    )

    print(f"Creating model: {args.model}")

    model = create_model(
        args.model, pretrained=False, num_classes=args.nb_classes,
        drop_rate=args.drop, drop_path_rate=args.drop_path,
        drop_block_rate=None, input_size=args.input_size
    )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        try:
            checkpoint_model = checkpoint['model']
        except:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        if args.finetune.startswith('https'):
            num_extra_tokens = 1
        else:
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches

        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)

        if args.finetune.startswith('https') and 'MCTformer' in args.model:
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens].repeat(1,args.nb_classes,1)
        else:
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]

        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        
        # Dynamically interpolate positional embeddings based on input_size
        target_size = args.input_size // args.patch_size
        if orig_size != target_size:
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(target_size, target_size), mode='bicubic', align_corners=False)
                
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)

        checkpoint_model['pos_embed_cls'] = extra_tokens
        checkpoint_model['pos_embed_pat'] = pos_tokens

        if args.finetune.startswith('https') and 'MCTformer' in args.model:
            cls_token_checkpoint = checkpoint_model['cls_token']
            new_cls_token = cls_token_checkpoint.repeat(1,args.nb_classes,1)
            checkpoint_model['cls_token'] = new_cls_token

        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    output_dir = Path(args.output_dir)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"mAP of the network on the {len(dataset_val)} test images: {test_stats['mAP']*100:.1f}%")
        return

    if args.gen_attention_maps:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        generate_attention_maps_ms(data_loader_attn, model, device, args)
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    history = {'train_loss': [], 'val_loss': [], 'val_mAP': []}

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler,
            args.clip_grad, args=args
        )

        lr_scheduler.step(epoch)
        test_stats = evaluate(data_loader_val, model, device)

        history['train_loss'].append(train_stats['loss'])
        history['val_loss'].append(test_stats['loss'])
        history['val_mAP'].append(test_stats['mAP'])

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'epoch': epoch,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    torch.save({'model': model.state_dict()}, output_dir / 'checkpoint.pth')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if args.output_dir:
        save_training_curves(history, args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('MCTformer BraTS training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)