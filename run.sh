#!/usr/bin/env sh


set -eu  # o pipefail

GPU=${GPU:-0,1}
PORT=${PORT:-29500}
N_GPUS=${N_GPUS:-1}

OPTIM=adamw
LR=0.0001
warmup=500
warmup=0
WD=0.01
N_EPOCHS=50
T_MAX=50

loss=nrmse
chkps_dir=./chkps

model=unet
backbone=tf_efficientnetv2_l.in21k
BS=32
incha=24

for month in {3..11}; do
    CHECKPOINT=$chkps_dir/"${model}"_"${backbone}"_b"${BS}"x"${N_GPUS}"_e"${N_EPOCHS}"_"${loss}"_m_lr4_ev_ref4_4_augs2_fb_"${month}"_m

    #MASTER_PORT="${PORT}" CUDA_VISIBLE_DEVICES="${GPU}" torchrun --nproc_per_node="${N_GPUS}" \
    CUDA_VISIBLE_DEVICES="${GPU}" python \
        ./src/train.py \
            --val-files "${month}" \
            --model "${model}" \
            --backbone "${backbone}" \
            --loss "${loss}" \
            --in-channels "${incha}" \
            --optim "${OPTIM}" \
            --learning-rate "${LR}" \
            --weight-decay "${WD}" \
            --T-max "${T_MAX}" \
            --num-epochs "${N_EPOCHS}" \
            --checkpoint-dir "${CHECKPOINT}" \
            --batch-size "${BS}" \
            --augs \
            --fp16 \
            --load $CHECKPOINT/model_last.pth \
            --resume \
            --scheduler-mode epoch \
            --warmup-t "${warmup}" \

done
