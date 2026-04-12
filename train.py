
import itertools
import math
import os
import os.path as osp
import random
import shutil
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import wandb
import yaml
from munch import Munch
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm

warnings.simplefilter("ignore")

from dataset import build_dataloader

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from Utils.PLBERT.util import load_plbert

from models import *
from losses import *
from utils import *

from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from optimizers import build_optimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class MyDataParallel(torch.nn.DataParallel):
    """DataParallel wrapper that proxies unknown attribute lookups to module."""

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def _device():
    return "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "cpu"


def freeze_module_parameters(model_dict, module_names):
    for name in module_names:
        if name in model_dict:
            for p in model_dict[name].parameters():
                p.requires_grad = False
            print(f"Froze: {name}")
        else:
            print(f"Module not found (skip freeze): {name}")


def print_learnable_parameters(model_dict):
    total = learnable = 0
    for key, model in model_dict.items():
        print(f"\nModule: {key}")
        for name, p in model.named_parameters():
            total += p.numel()
            if p.requires_grad:
                learnable += p.numel()
                print(f"  {name}: {p.numel()}")
    print(f"\nTotal: {total}  Learnable: {learnable} ({100*learnable/total:.2f}%)")


def _wandb_log(metrics: dict, step: int):
    try:
        wandb.log(metrics, step=step)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_dataloaders(data_params, forget_ratio, mode, batch_size, device):
    forget_df = pd.read_csv(data_params["forget_speaker_file"])
    forget_files = forget_df["speaker_files"].tolist()

    data_df = pd.read_csv(data_params["dataset_path"])
    train_df, val_df = train_test_split(
        data_df,
        test_size=data_params.get("val_size", 0.05),
        random_state=data_params.get("random_state", 42),
    )
    print(f"Train: {train_df.shape}  Val: {val_df.shape}")

    dataset_cfg = {"forget_speaker_file": forget_files, "forget_ratio": forget_ratio}
    loader_kwargs = dict(
        root_path=data_params["root_path"],
        style_root_path=data_params["style_root_path"],
        mode=mode,
        OOD_data=data_params["OOD_data"],
        min_length=data_params["min_length"],
        batch_size=batch_size,
        num_workers=data_params.get("num_workers", 2),
        device=device,
        dataset_config=dataset_cfg,
    )

    train_loader = build_dataloader(train_df, validation=False, **loader_kwargs)
    val_loader = build_dataloader(val_df, validation=True, **loader_kwargs)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main(
    config_path: str,
    mode: str = "standard",
    forget_ratio=None,
    lambda_triplet=None,
    max_iter=None,
    comment=None,
):
    assert mode in ("standard", "triplet"), (
        f"Unknown mode '{mode}'. Use 'standard' or 'triplet'."
    )

    config = yaml.safe_load(open(config_path))
    device = _device()

    # -----------------------------------------------------------------------
    # Logging (wandb)
    # -----------------------------------------------------------------------
    wandb_cfg = config.get("wandb", {})
    log_dir = config["log_dir"]

    if wandb_cfg.get("enable", True):
        run_name = wandb_cfg.get("name") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if comment:
            run_name += f"_{comment}"
        run_name += f"_{mode}"
        log_dir = osp.join(log_dir, run_name)
        wandb.login()  # relies on WANDB_API_KEY env var or prior `wandb login`
        wandb.init(
            project=wandb_cfg.get("project", "styletts2-unlearning"),
            name=run_name,
            dir=log_dir,
            config={**config, "mode": mode},
        )
    else:
        wandb.init(mode="disabled")

    os.makedirs(log_dir, exist_ok=True)
    try:
        shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    except Exception:
        pass

    # -----------------------------------------------------------------------
    # Config parsing
    # -----------------------------------------------------------------------
    epochs = config.get("epochs", 200)
    save_freq = config.get("save_freq", 2)
    batch_size = config.get("batch_size", 10)

    data_params = config["data_params"]
    loss_params = Munch(config["loss_params"])
    optimizer_params = Munch(config["optimizer_params"])
    model_params = recursive_munch(config["model_params"])

    # Allow CLI overrides for key hyperparameters
    effective_forget_ratio = (
        forget_ratio if forget_ratio is not None else data_params.get("forget_ratio", 0.8)
    )
    if mode == "triplet":
        if lambda_triplet is not None:
            loss_params.lambda_triplet = lambda_triplet
        elif not hasattr(loss_params, "lambda_triplet"):
            loss_params.lambda_triplet = 1.0

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    text_aligner = load_ASR_models(config["ASR_path"], config["ASR_config"])
    pitch_extractor = load_F0_models(config["F0_path"])
    plbert = load_plbert(config["PLBERT_dir"])

    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].to(device) for key in model]

    for key in model:
        if key not in ("mpd", "msd", "wd"):
            model[key] = MyDataParallel(model[key])

    if not (config.get("pretrained_model") and config.get("second_stage_load_pretrained")):
        raise ValueError("Unlearning requires a pretrained model. Set pretrained_model and second_stage_load_pretrained: true in config.")

    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False,
    )

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    train_loader, val_loader = load_dataloaders(
        data_params, effective_forget_ratio, mode, batch_size, device
    )

    # -----------------------------------------------------------------------
    # Optimiser
    # -----------------------------------------------------------------------
    scheduler_params = {
        "max_lr": optimizer_params.lr,
        "pct_start": 0.0,
        "epochs": epochs,
        "steps_per_epoch": len(train_loader),
    }
    sched_dict = {k: scheduler_params.copy() for k in model}
    sched_dict["bert"]["max_lr"] = optimizer_params.bert_lr * 2
    sched_dict["decoder"]["max_lr"] = optimizer_params.ft_lr * 2
    sched_dict["style_encoder"]["max_lr"] = optimizer_params.ft_lr * 2

    optimizer = build_optimizer(
        {k: model[k].parameters() for k in model},
        scheduler_params_dict=sched_dict,
        lr=optimizer_params.lr,
    )

    for g in optimizer.optimizers["bert"].param_groups:
        g.update({"betas": (0.9, 0.99), "lr": optimizer_params.bert_lr,
                   "initial_lr": optimizer_params.bert_lr, "min_lr": 0, "weight_decay": 0.01})

    for mod in ("decoder", "style_encoder"):
        for g in optimizer.optimizers[mod].param_groups:
            g.update({"betas": (0.0, 0.99), "lr": optimizer_params.ft_lr,
                       "initial_lr": optimizer_params.ft_lr, "min_lr": 0, "weight_decay": 1e-4})

    # -----------------------------------------------------------------------
    # Load pretrained checkpoint, then freeze everything except diffusion
    # -----------------------------------------------------------------------
    model, optimizer, _, _ = load_checkpoint(
        model, optimizer, config["pretrained_model"],
        load_only_params=config.get("load_only_params", True),
    )
    freeze_modules = [m for m in model.keys() if m != "diffusion"]
    freeze_module_parameters(model, freeze_modules)

    # -----------------------------------------------------------------------
    # Loss function (triplet mode only)
    # -----------------------------------------------------------------------
    triplet_loss_fn = None
    if mode == "triplet":
        margin = loss_params.get("margin", 0.3)
        triplet_loss_fn = TripletForgetLoss(margin=margin, p=2)
        print(f"Triplet loss enabled  margin={margin}  λ={loss_params.lambda_triplet}")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    running_std = []
    best_val_loss = float("inf")
    iters = 0

    torch.cuda.empty_cache()

    # When max_iter is set, cycle the dataloader indefinitely and train for
    # exactly max_iter gradient steps regardless of epoch boundaries.
    use_iter_mode = max_iter is not None
    steps_per_epoch = len(train_loader)
    if use_iter_mode:
        _train_cycle = itertools.cycle(train_loader)
        epoch_limit = math.ceil(max_iter / steps_per_epoch)
        print(f"Iteration-based training: {max_iter} steps "
              f"(~{epoch_limit} passes over the dataset).")
    else:
        epoch_limit = epochs

    for epoch in range(epoch_limit):
        _ = [model[k].eval() for k in model]
        model.diffusion.train()

        if use_iter_mode:
            steps_this_epoch = min(steps_per_epoch, max_iter - iters)
            batch_iter = tqdm(
                (next(_train_cycle) for _ in range(steps_this_epoch)),
                total=steps_this_epoch,
                desc=f"Iter {iters}/{max_iter}",
            )
        else:
            batch_iter = tqdm(train_loader, desc=f"Epoch {epoch} train")

        for batch in batch_iter:
            batch = [b.to(device) for b in batch]

            if mode == "triplet":
                texts, input_lengths, s_trg, ref_mels, ref_mels_len, neg_mels, neg_mels_len, forget_flags = batch
            else:
                texts, input_lengths, s_trg, ref_mels, ref_mels_len = batch

            with torch.no_grad():
                text_mask = length_to_mask(input_lengths).to(device)
                ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                ref = torch.cat([ref_ss, ref_sp], dim=1)

                if mode == "triplet":
                    neg_ss = model.style_encoder(neg_mels.unsqueeze(1))
                    neg_sp = model.predictor_encoder(neg_mels.unsqueeze(1))
                    neg_ref = torch.cat([neg_ss, neg_sp], dim=1)

            bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
            num_steps = np.random.randint(3, 5)

            if model_params.diffusion.dist.estimate_sigma_data:
                sigma = s_trg.std(axis=-1).mean().item()
                model.diffusion.module.diffusion.sigma_data = sigma
                running_std.append(sigma)

            s_preds = sampler(
                noise=torch.randn_like(s_trg).unsqueeze(1),
                embedding=bert_dur,
                embedding_scale=1,
                features=ref,
                embedding_mask_proba=0.1,
                num_steps=num_steps,
            ).squeeze(1)

            loss_diff = model.diffusion(s_trg.unsqueeze(1), embedding=bert_dur, features=ref).mean()
            loss_sty = F.l1_loss(s_preds, s_trg.detach())

            g_loss = loss_params.lambda_diff * loss_diff + loss_params.lambda_sty * loss_sty

            log_dict = {
                "train/loss_diff": loss_diff.item(),
                "train/loss_sty": loss_sty.item(),
            }

            if mode == "triplet":
                loss_triplet = triplet_loss_fn(
                    s_preds, s_trg.detach(), neg_ref.detach(), forget_flags
                )
                g_loss = g_loss + loss_params.lambda_triplet * loss_triplet
                log_dict["train/loss_triplet"] = loss_triplet.item()

            log_dict["train/g_loss"] = g_loss.item()

            if torch.isnan(g_loss):
                raise ValueError(f"NaN loss at epoch {epoch}, iter {iters}.")

            optimizer.zero_grad()
            g_loss.backward()
            optimizer.step("diffusion")

            iters += 1
            _wandb_log({**log_dict, "train/epoch": epoch, "train/iter": iters}, iters)

        # -------------------------------------------------------------------
        # Validation
        # -------------------------------------------------------------------
        avg_val_loss = _run_validation(
            val_loader, model, sampler, loss_params, device, mode, epoch, iters
        )
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        # -------------------------------------------------------------------
        # Checkpointing
        # -------------------------------------------------------------------
        state = {
            "net": {k: model[k].state_dict() for k in model},
            "optimizer": optimizer.state_dict(),
            "iters": iters,
            "val_loss": avg_val_loss,
            "epoch": epoch,
            "mode": mode,
        }

        if (epoch + 1) % save_freq == 0:
            ckpt_path = osp.join(log_dir, f"epoch_2nd_{epoch:05d}.pth")
            torch.save(state, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        torch.save(state, osp.join(log_dir, "last.pth"))

        if model_params.diffusion.dist.estimate_sigma_data and running_std:
            config["model_params"]["diffusion"]["dist"]["sigma_data"] = float(np.mean(running_std))
            cfg_out = osp.join(log_dir, osp.basename(config_path))
            with open(cfg_out, "w") as f:
                yaml.dump(config, f, default_flow_style=True)


def _run_validation(val_loader, model, sampler, loss_params, device, mode, epoch, iters):
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch} val"):
            batch = [b.to(device) for b in batch]

            if mode == "triplet":
                texts, input_lengths, s_trg, ref_mels, _, *_ = batch
            else:
                texts, input_lengths, s_trg, ref_mels, _ = batch

            text_mask = length_to_mask(input_lengths).to(device)
            ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
            ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
            ref = torch.cat([ref_ss, ref_sp], dim=1)

            bert_dur = model.bert(texts, attention_mask=(~text_mask).int())

            loss_diff = model.diffusion(s_trg.unsqueeze(1), embedding=bert_dur, features=ref).mean()

            try:
                s_preds = sampler(
                    noise=torch.randn_like(s_trg).unsqueeze(1),
                    embedding=bert_dur,
                    embedding_scale=1,
                    features=ref,
                    embedding_mask_proba=0.0,
                    num_steps=4,
                ).squeeze(1)
                loss_sty = F.l1_loss(s_preds, s_trg.detach())
            except Exception:
                loss_sty = torch.tensor(0.0, device=device)

            g_loss = loss_params.lambda_diff * loss_diff + loss_params.lambda_sty * loss_sty
            total_loss += g_loss.item()
            n_batches += 1

    avg = total_loss / max(1, n_batches)
    _wandb_log({"val/g_loss": avg, "val/epoch": epoch}, iters)
    return avg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="StyleTTS2 speaker-unlearning fine-tuning"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="Configs/config_unlearning.yml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "triplet"],
        default="standard",
        help="Unlearning mode: 'standard' (diffusion + style loss) or "
             "'triplet' (adds contrastive triplet loss).",
    )
    parser.add_argument(
        "--forget_ratio",
        type=float,
        default=None,
        help="Override forget_ratio from config. Probability [0,1] of replacing "
             "the reference speaker with the forget speaker during training.",
    )
    parser.add_argument(
        "--lambda_triplet",
        type=float,
        default=None,
        help="Override lambda_triplet from config (triplet mode only).",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=60000,
        help="Stop after this many gradient steps (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--comment",
        type=str,
        default=None,
        help="Optional suffix appended to the wandb run name.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        config_path=args.config,
        mode=args.mode,
        forget_ratio=args.forget_ratio,
        lambda_triplet=args.lambda_triplet,
        max_iter=args.max_iter,
        comment=args.comment,
    )
