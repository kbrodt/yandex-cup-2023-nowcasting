#!/usr/bin/env sh


set -eu  # o pipefail

GPU=${GPU:-0}

model=unet
backbone=tf_efficientnetv2_l.in21k
loss=nrmse
BS=32
incha=24

chkps_dir=./chkps
CHECKPOINTS="${chkps_dir}"/"${model}"_"${backbone}"_b"${BS}"x1_e50_"${loss}"_m_lr4_ev_ref4_4_augs2_fb_

CUDA_VISIBLE_DEVICES="${GPU}" python \
    ./src/submit.py \
        --model "${model}" \
        --backbone "${backbone}" \
        --in-channels "${incha}" \
        --batch-size "${BS}" \
        --load \
            "${CHECKPOINTS}"{3..11}_m/model_best.pth \
