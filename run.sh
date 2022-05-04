python3 RadEfficientDet/train.py \
--snapshot "/mnt/TFM_KIKE/PRETRAINED_MODELS/efficientdet-d0.h5" \
--weighted-bifpn \
--radar-mode "concat" \
--phi 0 \
--gpu 1 \
--random-transform \
--compute-val-loss \
--freeze-backbone \
--batch-size 8 \
--epochs 103 \
--steps -1 \
--workers 4 \
--max-queue-size 64 \
--tensorboard-dir \
"/mnt/TFM_KIKE/EXPERIMENTS_KERAS/EXP_D0_CONCAT_V3_OV4/" \
csv \
"/mnt/TFM_KIKE/DATASETS/fused_imgs_v5_only_visibility_4/train.csv" \
"/mnt/TFM_KIKE/DATASETS/fused_imgs_v5_only_visibility_4/dataset_encoding.csv" \
--val-annotations-path \
"/mnt/TFM_KIKE/DATASETS/fused_imgs_v5_only_visibility_4/val.csv"