dirname="EXP_D0_CONCAT_V3_OV4_SC"

# --radar-mode "add" \

python3 CamEfficientDet/train.py \
--snapshot "/mnt/TFM_KIKE/PRETRAINED_MODELS/efficientdet-d0.h5" \
--weighted-bifpn \
--radar-mode "concat" \
--phi 0 \
--gpu 0 \
--random-transform \
--compute-val-loss \
--freeze-backbone \
--batch-size 8 \
--epochs 100 \
--steps -1 \
--workers 4 \
--max-queue-size 64 \
--snapshot-path=checkpoints/$dirname \
--tensorboard-dir=/mnt/TFM_KIKE/EXPERIMENTS_KERAS_SC/$dirname \
csv \
"/mnt/TFM_KIKE/DATASETS/fused_imgs_v5_only_visibility_4/train_sc.csv" \
"/mnt/TFM_KIKE/DATASETS/fused_imgs_v5_only_visibility_4/dataset_encoding_sc.csv" \
--val-annotations-path \
"/mnt/TFM_KIKE/DATASETS/fused_imgs_v5_only_visibility_4/val_sc.csv"