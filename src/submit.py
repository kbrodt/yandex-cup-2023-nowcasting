import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.jit
import torch.distributed
import torch.nn as nn
import torch.utils
import torch.utils.data
import tqdm

import models
import dataset


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val-files",
        type=str,
        nargs="+",
        help="path to val df",
        default=[
            "data/train/2021-02-train.hdf5",
            "data/train/2021-05-train.hdf5",
            "data/train/2021-08-train.hdf5",
            "data/train/2021-11-train.hdf5",
        ],
    )
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "unetlstm", "convlstm", "unetflow"])
    parser.add_argument("--backbone", type=str, default="tf_efficientnet_b5_ns")
    parser.add_argument("--dec-attn-type", type=str, default=None)

    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-indices", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--dec-channels", type=int, nargs="+", default=[256, 240, 224, 208, 192])
    parser.add_argument("--n-classes", type=int, default=12)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-kernels", type=int, default=32)

    parser.add_argument("--in-seq-len", type=int, default=4)
    parser.add_argument("--out-seq-len", type=int, default=12)
    parser.add_argument("--val-out-seq-len", type=int, default=12)

    parser.add_argument(
        "--num-workers", type=int, help="number of data loader workers", default=8,
    )
    parser.add_argument(
        "--num-epochs", type=int, help="number of epochs to train", default=440,
    )
    parser.add_argument("--batch-size", type=int, help="batch size", default=32)
    parser.add_argument(
        "--random-state",
        type=int,
        help="random seed",
        default=314159,
    )

    parser.add_argument(
        "--distributed", action="store_true", help="distributed training"
    )
    parser.add_argument("--syncbn", action="store_true", help="sync batchnorm")
    parser.add_argument(
        "--deterministic", action="store_true", help="deterministic training"
    )
    parser.add_argument(
        "--load",
        type=str,
        nargs="+",
        default="",
        required=True,
        help="path to pretrained model weights"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="path to pretrained model to resume training",
    )
    parser.add_argument("--fp16", action="store_true", help="fp16 training")

    args = parser.parse_args(args=args)

    return args


@torch.inference_mode()
def epoch_step_val(loader, desc, models, metric, output_file=None):
    pbar = tqdm.tqdm(total=len(loader), desc=desc, leave=False, mininterval=2)

    for (images, month, mask, _last_input_timestamp), target in loader:
        images = images.cuda(non_blocking=True)
        month = month.cuda(non_blocking=True)

        _logits = []
        for model in models:
            logits = model(images, month)
            logits += torch.flip(model(torch.flip(images, dims=[-1]), month), dims=[-1])
            logits += torch.flip(model(torch.flip(images, dims=[-2]), month), dims=[-2])
            logits += torch.flip(model(torch.flip(images, dims=[-2, -1]), month), dims=[-2, -1])
            logits /= 4.0
            _logits.append(logits)

        logits = torch.stack(_logits, dim=0).mean(0)

        if output_file is not None:
            logits = logits.squeeze(2).cpu().numpy()
            with h5py.File(output_file, mode="a") as f_out:
                assert(len(logits) == len(_last_input_timestamp))
                for output, last_input_timestamp in zip(logits, _last_input_timestamp):
                    for index, out in enumerate(output):
                        timestamp_out = str(int(last_input_timestamp) + 600 * (index + 1))
                        f_out.create_group(timestamp_out)
                        f_out[timestamp_out].create_dataset(
                            "intensity",
                            data=out,
                        )
        else:
            metric["rmse"].update(logits, target)

        torch.cuda.synchronize()
        pbar.update()

    pbar.close()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Metric:
    def __init__(self):
        self.rmses = []
        self.clean()

    def clean(self):
        self.rmses.clear()

    def update(self, preds, targets):
        # preds   [b, t, c, h, w]
        # targets [b, t, c, h, w]
        assert preds.shape[:2] == targets.shape[:2]
        assert preds.shape[1] == 12
        assert len(preds.shape) == len(targets.shape)

        target = targets.cpu().numpy()
        output = preds.cpu().numpy()

        rmse = np.sum(
            (
                np.square(target - output)
            ) * (target != -1),
            axis=(2, 3, 4)
        )  # [b, t]
        self.rmses.extend(rmse.tolist())

    def evaluate(self):
        if len(self.rmses) == 0:
            return 300.0

        rmses = np.array(self.rmses)  # [b, t]
        assert rmses.shape[1] == 12
        rmses = rmses.mean(0)
        rmse = np.sqrt(rmses).mean(0)

        return rmse


def train(args):
    torch.backends.cudnn.benchmark = True

    checkpoint = None
    models = []
    for load in args.load:
        path_to_resume = Path(load).expanduser()
        print(f"=> loading resume checkpoint '{path_to_resume}'")
        checkpoint = torch.load(
            path_to_resume,
            map_location="cpu",
        )
        _args = checkpoint["args"]

        nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint["state_dict"], "module.")
        model = build_model(_args)
        model = model.cuda().eval()
        model.load_state_dict(checkpoint["state_dict"])
        print(
            f"=> resume from checkpoint '{path_to_resume}' (epoch {checkpoint['epoch']})"
        )
        if "lstm" not in _args.backbone:
            model = model.to(memory_format=torch.channels_last)

        models.append(model)

    output_file = None
    output_file = "effv2_l_tta4_fb_r4__34567891011m.hdf5"
    if output_file is None:
        val_dataset = dataset.RadarDataset(
            list_of_files=args.val_files,
            in_seq_len=args.in_seq_len,
            out_seq_len=args.val_out_seq_len,
            mode="overlap",
            with_time=True,
        )
    else:
        val_dataset = dataset.RadarDataset(
            [
                "data/2022-test-public.hdf5",
            ],
            in_seq_len=args.in_seq_len,  # 4
            with_time=True,
            mode="overlap",
            out_seq_len=0,
        )

    val_sampler = None

    args.num_workers = min(args.batch_size, 8)
    val_batch_size = args.batch_size
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=None,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True,
        drop_last=False,
    )

    metric = {
        "rmse": Metric(),
    }

    for m in metric.values():
        m.clean()

    epoch_step_val(
        val_loader,
        "test",
        models,
        metric,
        output_file=output_file,
    )

    dev_scores = {}
    for key, m in metric.items():
        dev_scores[key] = m.evaluate()

    score = min(dev_scores.values())
    print(score)


def build_model(args):
    if args.backbone == "convlstm":
        model = models.ConvLSTMModel(
            num_channels=args.in_channels,
            num_kernels=args.n_kernels,
            num_layers=args.n_layers,
            out_seq_len=args.out_seq_len,
        )
    elif args.backbone == "unetlstm":
        model = models.UNetConvLSTMModel(
            num_channels=args.in_channels,
            num_kernels=args.n_kernels,
            out_seq_len=args.out_seq_len,
        )
    else:
        model = models.UNet(
            backbone=args.backbone,
            in_channels=args.in_channels,
            out_indices=args.out_indices,
            dec_channels=args.dec_channels,
            dec_attn_type=args.dec_attn_type,
            n_classes=args.n_classes,
            activation="relu",
        )

    return model


def main():
    args = parse_args()

    train(args)


if __name__ == "__main__":
    main()
