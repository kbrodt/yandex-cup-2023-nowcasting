import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import shlex
import textwrap
import subprocess
import random
from pathlib import Path

import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import numpy as np
import torch
import torch.jit
import torch.distributed
import torch.nn as nn
import torch.utils
import torch.utils.data
import timm
import timm.scheduler
import tqdm
from torch.utils.tensorboard.writer import SummaryWriter

import models
import dataset


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-files",
        type=str,
        nargs="+",
        help="path to train df",
        default=sorted([
            "data/train/2021-01-train.hdf5",
            "data/train/2021-02-train.hdf5",
            "data/train/2021-03-train.hdf5",
            "data/train/2021-04-train.hdf5",
            "data/train/2021-05-train.hdf5",
            "data/train/2021-06-train.hdf5",
            "data/train/2021-07-train.hdf5",
            "data/train/2021-08-train.hdf5",
            "data/train/2021-09-train.hdf5",
            "data/train/2021-10-train.hdf5",
            "data/train/2021-11-train.hdf5",
            "data/train/2021-12-train.hdf5",
        ]),
    )
    parser.add_argument(
        "--val-files",
        nargs="+",
        help="path to val df",
        type=int,
        default=[6],
    )
    parser.add_argument("--checkpoint-dir", type=str, default="logs")

    parser.add_argument("--model", type=str, default="unet", choices=["unet", "unetlstm", "convlstm", "unetflow"])
    parser.add_argument("--backbone", type=str, default="tf_efficientnetv2_s.in21k")
    parser.add_argument("--dec-attn-type", type=str, default=None)

    parser.add_argument("--loss", type=str, default="nrmse", choices=["nrmse"])
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-indices", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--dec-channels", type=int, nargs="+", default=[256, 240, 224, 208, 192])
    parser.add_argument("--n-classes", type=int, default=12)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-kernels", type=int, default=32)

    parser.add_argument("--in-seq-len", type=int, default=4)
    parser.add_argument("--out-seq-len", type=int, default=12)
    parser.add_argument("--val-out-seq-len", type=int, default=12)

    parser.add_argument("--ds-mult", type=int, default=1)

    parser.add_argument("--optim", type=str, default="adamw", help="optimizer name")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--lr-decay-scale", type=float, default=1e-2)
    parser.add_argument("--warmup-steps-ratio", type=float, default=0.2)
    parser.add_argument("--warmup-t", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=1e-2)

    parser.add_argument("--mix-beta", type=float, default=1)

    parser.add_argument("--scheduler", type=str, default="wucos", help="scheduler name")
    parser.add_argument("--scheduler-mode", type=str, default="step", choices=["step", "epoch"], help="scheduler mode")
    parser.add_argument("--T-max", type=int, default=440)
    parser.add_argument("--eta-min", type=int, default=0)

    parser.add_argument("--mixup", type=float, default=0)
    parser.add_argument("--cutmix", type=float, default=0)
    parser.add_argument("--mix-y", action="store_true", help="mix target")

    parser.add_argument("--augs", action="store_true", help="augmentations")

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
        "--load", type=str, default="", help="path to pretrained model weights"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="path to pretrained model to resume training",
    )
    parser.add_argument("--fp16", action="store_true", help="fp16 training")
    parser.add_argument("--ft", action="store_true", help="finetune")

    args = parser.parse_args(args=args)

    return args


def notify(title, summary, value=-1, server="sun"):
    print(title, summary)
    return
    cmd = textwrap.dedent(
        f"""
            ssh {server} \
                '\
                    export DISPLAY=:0 \
                    && dunstify -t 0 -h int:value:{value} "{title}" "{summary}" \
                '
        """
    )
    cmds = shlex.split(cmd)
    with subprocess.Popen(cmds, start_new_session=True):
        pass


def epoch_step_train(
    epoch,
    loader,
    desc,
    model,
    criterion,
    optimizer,
    scaler,
    scheduler=None,
    fp16=False,
    local_rank=0,
    alpha=0,
    cutmix=None,
):
    model.train()

    if local_rank == 0:
        pbar = tqdm.tqdm(total=len(loader), desc=desc, leave=False, mininterval=2)

    n_steps_per_epoch = len(loader)
    num_updates = epoch * n_steps_per_epoch
    loc_loss = n = 0
    for images, month, mask, target in loader:
        m = torch.logical_not(mask.all(1))
        if not m.any():
            continue

        images = images[m]
        target = target[m]
        month = month[m]

        images = images.cuda(local_rank, non_blocking=True)  # [b, t, c, h, w]
        month = month.cuda(local_rank, non_blocking=True)  # [b, t, c, h, w]
        target = target.cuda(local_rank, non_blocking=True)  # [b, t, c, h, w]

        with torch.cuda.amp.autocast(enabled=fp16):
            logits = model(images, month)
            loss = criterion(logits, target)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
            # error_if_nonfinite=True,
        )
        scaler.step(optimizer)
        scaler.update()
        num_updates += 1
        optimizer.zero_grad(set_to_none=True)

        bs = target.size(0)
        loc_loss += loss.item() * bs
        n += bs

        if scheduler is not None:
            scheduler.step_update(num_updates=num_updates)

        torch.cuda.synchronize()

        if local_rank == 0:
            postfix = {
                "loss": f"{loc_loss / n:.3f}",
            }
            pbar.set_postfix(**postfix)
            pbar.update()

        if np.isnan(loc_loss) or np.isinf(loc_loss):
            break

    if local_rank == 0:
        pbar.close()

    return loc_loss, n


@torch.inference_mode()
def epoch_step_val(loader, desc, model, metric, local_rank=0):
    model.eval()

    if local_rank == 0:
        pbar = tqdm.tqdm(total=len(loader), desc=desc, leave=False, mininterval=2)

    for images, month, mask, target in loader:
        images = images.cuda(local_rank, non_blocking=True)
        month = month.cuda(local_rank, non_blocking=True)

        logits = model(images, month)

        metric["rmse"].update(logits, target)

        torch.cuda.synchronize()

        if local_rank == 0:
            pbar.update()

    if local_rank == 0:
        pbar.close()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_dist(args):
    # to autotune convolutions and other algorithms
    # to pick the best for current configuration
    torch.backends.cudnn.benchmark = True

    if args.deterministic:
        set_seed(args.random_state)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_printoptions(precision=10)

    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1

    args.world_size = 1
    if args.distributed:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."


def add_weight_decay(args, model, weight_decay=1e-5, skip_list=()):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    params = [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]

    return params


def save_jit(model, args, model_path):
    return
    if hasattr(model, "module"):
        model = model.module

    model.eval()
    inp = torch.rand(1, 12, args.in_channels, args.img_size[0], args.img_size[1]).cuda(int(os.environ.get("LOCAL_RANK", 0)))

    with torch.no_grad():
        traced_model = torch.jit.trace(model, inp)

    traced_model = torch.jit.freeze(traced_model)

    traced_model.save(model_path)


def all_gather(value, n, is_dist):
    if is_dist:
        if n is not None:
            vals = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(vals, value)
            ns = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(ns, n)

            n = sum(ns)
            val = sum(vals) / n
        else:
            vals = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(vals, value)
            val = []
            for v in vals:
                val.extend(v)
    else:
        if n is not None:
            val = value / n
        else:
            val = value

    return val


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

    def all_gather(self, is_dist):
        self.rmses = all_gather(self.rmses, None, is_dist)


class NoNaNRMSE(nn.Module):
    def __init__(self, threshold=-1):
        super().__init__()

        self.threshold = threshold

    def forward(self, logits, target):#, inputs):
        # logits: [b, c, h, w]
        # target: [b, 1, c, h, w]
        # inputs: [b, t, c, h, w]
        # normalized by max

        logits = logits.squeeze(2)
        target = target.squeeze(2)
        not_nan = target > self.threshold

        diff = logits - target
        diff[~not_nan] = 0
        diff2 = torch.square(diff)
        diff2m = (diff2).sum((-1, -2)).mean(0)
        diff2msqrt = torch.sqrt(diff2m)

        rmse = diff2msqrt.mean(0)

        return rmse


def train(args):
    init_dist(args)

    torch.backends.cudnn.benchmark = True

    checkpoint_dir = Path(args.checkpoint_dir)
    summary_writer = None
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(args)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        summary_writer = SummaryWriter(checkpoint_dir / "logs")

        notify(
            checkpoint_dir.name,
            f"start training",
        )

    model = build_model(args)
    model = model.cuda(local_rank)

    checkpoint = None
    if args.load:
        path_to_resume = Path(args.load).expanduser()
        if path_to_resume.is_file():
            print(f"=> loading resume checkpoint '{path_to_resume}'")
            checkpoint = torch.load(
                path_to_resume,
                map_location="cpu",
            )

            nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint["state_dict"], "module.")
            model.load_state_dict(checkpoint["state_dict"])
            print(
                f"=> resume from checkpoint '{path_to_resume}' (epoch {checkpoint['epoch']})"
            )
        else:
            print(f"=> no checkpoint found at '{path_to_resume}'")

    if "lstm" not in args.backbone:
        model = model.to(memory_format=torch.channels_last)

    weight_decay = args.weight_decay
    if weight_decay > 0:  # and filter_bias_and_bn:
        skip = {}
        if hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()

        parameters = add_weight_decay(args, model, weight_decay, skip)
        weight_decay = 0.0
    else:
        parameters = model.parameters()

    optimizer = build_optimizer(parameters, args)

    if args.distributed:
        if args.syncbn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            # find_unused_parameters=True,
        )

    assert len(set(args.train_files)) == 12

    args.val_files = [
        args.train_files.pop(i - 1)
        for i in args.val_files
    ]

    assert len(set(args.val_files) & set(args.train_files)) == 0
    assert len(set(args.val_files) | set(args.train_files)) == 12

    train_dataset = torch.utils.data.ConcatDataset(
        [
            dataset.RadarDataset(
                list_of_files=args.train_files,
                in_seq_len=args.in_seq_len,
                out_seq_len=args.out_seq_len,
                mode="overlap",
                is_train=True,
            )
            for _ in range(args.ds_mult)
        ]
    )

    if args.ft:
        train_dataset = torch.utils.data.ConcatDataset(
            [
                train_dataset,
                dataset.RadarDataset(
                    list_of_files=args.val_files,
                    in_seq_len=args.in_seq_len,
                    out_seq_len=args.out_seq_len,
                    mode="overlap",
                    is_train=True,
                ),
            ],
        )

    val_dataset = dataset.RadarDataset(
        list_of_files=args.val_files,
        in_seq_len=args.in_seq_len,
        out_seq_len=args.val_out_seq_len,
        mode="overlap",
        is_train=False,
    )

    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    args.num_workers = min(args.batch_size, 16)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=None,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True,
        drop_last=True,
    )
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

    scheduler = build_scheduler(optimizer, args, n=len(train_loader) if args.scheduler_mode == "step" else 1)

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    criterion = build_criterion(args)
    metric = {
        "rmse": Metric(),
    }

    def saver(path, score):
        torch.save(
            {
                "epoch": epoch,
                "best_score": best_score,
                "score": score,
                "state_dict": model.state_dict(),
                "opt_state_dict": optimizer.state_dict(),
                "sched_state_dict": scheduler.state_dict()
                if scheduler is not None
                else None,
                "scaler": scaler.state_dict(),
                "args": args,
            },
            path,
        )

    res = 0
    start_epoch = 0
    best_score = float("+inf")
    score = float("+inf")
    epoch = 0
    if args.resume and checkpoint is not None:
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        if checkpoint["sched_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["sched_state_dict"])

        optimizer.load_state_dict(checkpoint["opt_state_dict"])
        scaler.load_state_dict(checkpoint["scaler"])

    for epoch in range(start_epoch, args.num_epochs):
        if epoch >= 13:
            break

        if args.distributed:
            train_sampler.set_epoch(epoch)

        desc = f"{epoch}/{args.num_epochs}"

        train_loss, n = epoch_step_train(
            epoch,
            train_loader,
            desc,
            model,
            criterion,
            optimizer,
            scaler,
            scheduler=scheduler if args.scheduler_mode == "step" else None,
            fp16=args.fp16,
            local_rank=local_rank,
        )

        train_loss = all_gather(train_loss, n, args.distributed)
        if np.isnan(train_loss) or np.isinf(train_loss):
            res = 1
            break

        for m in metric.values():
            m.clean()

        epoch_step_val(
            val_loader,
            desc,
            model,
            metric,
            local_rank=local_rank,
        )

        for m in metric.values():
            m.all_gather(args.distributed)

        dev_scores = {}
        for key, m in metric.items():
            dev_scores[key] = m.evaluate()

        if scheduler is not None:# and args.scheduler_mode == "epoch":
            scheduler.step(epoch + 1)

        if local_rank == 0:
            for idx, param_group in enumerate(optimizer.param_groups):
                lr = param_group["lr"]
                summary_writer.add_scalar(
                    "group{}/lr".format(idx), float(lr), global_step=epoch
                )

            summary_writer.add_scalar("loss/train_loss", train_loss, global_step=epoch)
            for k, score in dev_scores.items():
                summary_writer.add_scalar(f"score/dev_{k}", score, global_step=epoch)

            score = min(dev_scores.values())

            if score < best_score:
                notify(
                    checkpoint_dir.name,
                    f"epoch {epoch}: new score {score:.3f} (old {best_score:.3f}, diff {abs(score - best_score):.3f})",
                    int(100 * (epoch / args.num_epochs)),
                )
                best_score = score

                saver(checkpoint_dir / "model_best.pth", best_score)
                # save_jit(model, args, checkpoint_dir / f"model_best.pt")
                if hasattr(model, "module"):
                    torch.save(model.module, checkpoint_dir / f"modelo_best.pth")
                else:
                    torch.save(model, checkpoint_dir / f"modelo_best.pth")

            saver(checkpoint_dir / "model_last.pth", score)
            # save_jit(model, args, checkpoint_dir / "model_last.pt")

            if epoch % (2 * args.T_max) == (args.T_max - 1):
                saver(checkpoint_dir / f"model_last_{epoch + 1}.pth", score)
                # save_jit(model, args, checkpoint_dir / f"model_last_{epoch + 1}.pt")

        torch.cuda.empty_cache()

    if local_rank == 0:
        summary_writer.close()

        notify(
            checkpoint_dir.name,
            f"finished training with score {score:.3f} (best {best_score:.3f}) on epoch {epoch}",
        )

    return res


def build_model(args):
    if args.model == "convlstm":
        model = models.ConvLSTMModel(
            num_channels=args.in_channels,
            num_kernels=args.n_kernels,
            num_layers=args.n_layers,
            out_seq_len=args.out_seq_len,
        )
    elif args.model == "unetlstm":
        model = models.UNetConvLSTMModel(
            num_channels=args.in_channels,
            num_kernels=args.n_kernels,
            out_seq_len=args.out_seq_len,
            backbone=args.backbone,
            in_channels=args.in_channels,
            out_indices=args.out_indices,
            dec_channels=args.dec_channels,
            dec_attn_type=args.dec_attn_type,
            n_classes=args.n_classes,
        )
    elif args.model == "unet":
        model = models.UNet(
            backbone=args.backbone,
            in_channels=args.in_channels,
            out_indices=args.out_indices,
            dec_channels=args.dec_channels,
            dec_attn_type=args.dec_attn_type,
            n_classes=args.n_classes,
            activation="relu",
        )
    elif args.model == "unetflow":
        model = models.UNetFlow(
            backbone=args.backbone,
            in_channels=args.in_channels,
            out_indices=args.out_indices,
            dec_channels=args.dec_channels,
            dec_attn_type=args.dec_attn_type,
            n_classes=args.n_classes,
            activation="relu",
        )

    return model


def build_criterion(args):
    if args.loss == "nrmse":
        criterion = NoNaNRMSE()

    return criterion


def build_optimizer(parameters, args):
    if args.optim.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.optim.lower() == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError(f"not yet implemented {args.optim}")

    return optimizer


def build_scheduler(optimizer, args, n=1):
    scheduler = None

    if args.scheduler.lower() == "cosa":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.T_max * n,
            eta_min=args.eta_min if args.eta_min > 0 else max(args.learning_rate * 1e-1, 5e-5),
        )
    elif args.scheduler.lower() == "cosawr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.T_max,
            T_mult=1,
            eta_min=args.eta_min if args.eta_min > 0 else max(args.learning_rate * 1e-1, 5e-5),
        )
    elif args.scheduler.lower() == "wucos":
        scheduler = timm.scheduler.CosineLRScheduler(
            optimizer,
            t_initial=args.T_max * n,
            #lr_min=args.eta_min if args.eta_min > 0 else max(args.learning_rate * 1e-1, 1e-5),
            #warmup_lr_init=args.learning_rate * 1e-2,
            warmup_t=args.warmup_t,  # int(args.warmup_steps_ratio * args.num_epochs) * n,
            cycle_limit=1,  # args.T_max + 1,
            t_in_epochs=n == 1,
        )
    else:
        print("No scheduler")

    return scheduler


def main():
    args = parse_args()

    train(args)


if __name__ == "__main__":
    main()
