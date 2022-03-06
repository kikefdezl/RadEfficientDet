python3 RadEfficientDet/train.py \
--snapshot "/mnt/TFM_KIKE/PRETRAINED_MODELS/efficientdet-d0.h5" \
--weighted-bifpn \
--phi 0 \
--gpu 0 \
--random-transform \
--compute-val-loss \
--freeze-backbone \
--batch-size 8 \
--epochs 103 \
--steps -1 \
--workers 4 \
--max-queue-size 64 \
--tensorboard-dir \
"/mnt/TFM_KIKE/EXPERIMENTS_KERAS/EXP_D0_RADAR_MAPS/" \
csv \
"/mnt/TFM_KIKE/DATASETS/fused_imgs_v3_no_visibility_0/train.csv" \
"/mnt/TFM_KIKE/DATASETS/fused_imgs_v3_no_visibility_0/dataset_encoding.csv" \
--val-annotations-path \
"/mnt/TFM_KIKE/DATASETS/fused_imgs_v3_no_visibility_0/val.csv"