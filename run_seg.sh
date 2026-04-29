set -e

mkdir -p saved_model/seg_weights

python seg/train_seg.py --network resnet38_seg \
                    --num_epochs 30 \
                    --seg_pgt_path MCTformer_results/MCTformer_plus/pgt-psa-rw \
                    --init_weights res38_cls.pth \
                    --save_path saved_model/seg_weights \
                    --list_path voc12/train_aug_id.txt \
                    --img_path VOCdevkit/VOC2012/JPEGImages \
                    --num_classes 21 \
                    --batch_size 4

python seg/infer_seg.py --weights saved_model/seg_weights/model_29.pth \
                      --network resnet38_seg \
                      --list_path voc12/val_id.txt \
                      --gt_path VOCdevkit/VOC2012/SegmentationClass \
                      --img_path VOCdevkit/VOC2012/JPEGImages \
                      --save_path MCTformer_results/MCTformer_plus/val_ms_crf \
                      --save_path_c MCTformer_results/MCTformer_plus/val_ms_crf_c \
                      --scales 0.5 0.75 1.0 1.25 1.5 \
                      --use_crf True