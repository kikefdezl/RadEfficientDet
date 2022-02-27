python3 EfficientRaDet/train.py \
--snapshot "/mnt2/ITP_PROJECT/PRETRAINED_MODELS/efficientdet-d0.h5" \
--weighted-bifpn \
--phi 0 \
--gpu 0 \
--random-transform \
--compute-val-loss \
--freeze-backbone \
--batch-size 8 \
--epochs 300 \
--steps -1 \
--tensorboard-dir \
"/mnt/TFM_KIKE/EXPERIMENTS_KERAS/EXP_D0_RAW_IMGS/" \
csv \
"/mnt/TFM_KIKE/DATASETS/fused_imgs_v3_no_visibility_0/train.csv" \
"/mnt/TFM_KIKE/DATASETS/fused_imgs_v3_no_visibility_0/dataset_encoding.csv" \
--val-annotations-path \
"/mnt/TFM_KIKE/DATASETS/fused_imgs_v3_no_visibility_0/val.csv"