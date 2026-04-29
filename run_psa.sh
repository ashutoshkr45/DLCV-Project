######### Generating class-specific localization maps ##########
python main.py --model deit_small_MCTformerPlus \
                --data-set VOC12MS \
                --scales 1.0 \
                --img-list voc12 \
                --data-path VOCdevkit/VOC2012 \
                --resume saved_model/checkpoint.pth \
                --gen_attention_maps \
                --attention-type fused \
                --layer-index 3 \
                --visualize-cls-attn True \
                --patch-attn-refine True \
                --attention-dir MCTformer_results/MCTformer_plus/attn-patchrefine \
                --cam-npy-dir MCTformer_results/MCTformer_plus/attn-patchrefine-npy \
                # --out-crf MCTformer_results/MCTformer_plus/attn-patchrefine-npy-crf \

python psa/make_crf.py 

python psa/train_aff.py --weights res38_cls.pth \
                        --voc12_root VOCdevkit/VOC2012 \
                        --la_crf_dir MCTformer_results/MCTformer_plus/attn-patchrefine-npy-crf_1 \
                        --ha_crf_dir MCTformer_results/MCTformer_plus/attn-patchrefine-npy-crf_12 \


python psa/infer_aff.py --weights resnet38_aff.pth \
                    --infer_list voc12/train_id.txt \
                    --cam_dir MCTformer_results/MCTformer_plus/attn-patchrefine-npy \
                    --voc12_root VOCdevkit/VOC2012 \
                    --out_rw MCTformer_results/MCTformer_plus/pgt-psa-rw \

python evaluation.py --list voc12/train_id.txt \
                     --gt_dir VOCdevkit/VOC2012/SegmentationClassAug \
                     --logfile MCTformer_results/MCTformer_plus/pgt-psa-rw/_evallog.txt \
                     --type png \
                     --predict_dir MCTformer_results/MCTformer_plus/pgt-psa-rw \
                     --comment "train 1464"


python psa/infer_aff.py --weights resnet38_aff.pth \
                    --infer_list voc12/train_aug_id.txt \
                    --cam_dir MCTformer_results/MCTformer_plus/attn-patchrefine-npy \
                    --voc12_root VOCdevkit/VOC2012 \
                    --out_rw MCTformer_results/MCTformer_plus/pgt-psa-rw
