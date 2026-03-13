from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
import yaml

from .dataset import build_dataset
from .modeling_recurrent import RecurrentPythiaConfig, RecurrentPythiaForCausalLM


CONFIGS_DIR = Path("configs")


@dataclass
class TrainConfig:
    data_path: str
    seq_len: int = 1024
    split: str = "train"
    batch_size: int = 2
    epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    max_steps: int = 0
    log_every: int = 1
    eval_every: int = 0
    eval_batches: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ffn_size: int = 2048
    max_position_embeddings: int = 4096
    window_size: int = 128
    gru_hidden: int = 256
    use_recurrence: bool = True
    truncate_bptt: bool = True
    num_workers: int = 0
    wandb_project: str = "recurrent-pythia-mvp"
    wandb_name: Optional[str] = None
    wandb_mode: str = "online"
    wandb_anonymous: str = "allow"
    wandb_tags: tuple[str, ...] = ()
    config_path: Optional[str] = None
    log_dir: str = "logs"


TRAIN_CONFIG_FIELDS = {field.name for field in fields(TrainConfig)}
TRAIN_OPTION_TO_FIELD = {
    "--data-path": "data_path",
    "--seq-len": "seq_len",
    "--split": "split",
    "--batch-size": "batch_size",
    "--epochs": "epochs",
    "--learning-rate": "learning_rate",
    "--weight-decay": "weight_decay",
    "--grad-clip": "grad_clip",
    "--max-steps": "max_steps",
    "--log-every": "log_every",
    "--eval-every": "eval_every",
    "--eval-batches": "eval_batches",
    "--device": "device",
    "--seed": "seed",
    "--hidden-size": "hidden_size",
    "--num-layers": "num_layers",
    "--num-heads": "num_heads",
    "--ffn-size": "ffn_size",
    "--max-position-embeddings": "max_position_embeddings",
    "--window-size": "window_size",
    "--gru-hidden": "gru_hidden",
    "--truncate-bptt": "truncate_bptt",
    "--num-workers": "num_workers",
    "--wandb-project": "wandb_project",
    "--wandb-name": "wandb_name",
    "--wandb-mode": "wandb_mode",
    "--wandb-anonymous": "wandb_anonymous",
    "--wandb-tags": "wandb_tags",
    "--log-dir": "log_dir",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config_file(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    elif path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a top-level mapping: {path}")
    return data


def build_train_defaults(config_data: dict[str, Any]) -> dict[str, Any]:
    defaults = {key: value for key, value in config_data.items() if key in TRAIN_CONFIG_FIELDS}
    if "wandb_tags" in defaults and isinstance(defaults["wandb_tags"], list):
        defaults["wandb_tags"] = tuple(str(tag) for tag in defaults["wandb_tags"])
    return defaults


def train_config_defaults() -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for field in fields(TrainConfig):
        if field.name == "data_path":
            continue
        defaults[field.name] = field.default
    return defaults


def cli_override_fields(argv: Optional[list[str]]) -> set[str]:
    overrides: set[str] = set()
    for token in argv or []:
        if token in TRAIN_OPTION_TO_FIELD:
            overrides.add(TRAIN_OPTION_TO_FIELD[token])
        if token == "--no-recurrence":
            overrides.add("use_recurrence")
        if token == "--config":
            overrides.add("config_path")
    return overrides


def dataset_sanity_check(data_path: str, seq_len: int, split: str, batch_size: int) -> None:
    dataset = build_dataset(data_path=data_path, seq_len=seq_len, split=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))
    print(f"split={split}")
    print(f"seq_len={seq_len}")
    print(f"dataset_sequences={len(dataset)}")
    print(f"vocab_size={dataset.vocab_size}")
    print(f"input_shape={tuple(batch['input_ids'].shape)}")
    print(f"label_shape={tuple(batch['labels'].shape)}")


def build_model(train_config: TrainConfig, vocab_size: int) -> RecurrentPythiaForCausalLM:
    model_config = RecurrentPythiaConfig(
        vocab_size=vocab_size,
        hidden_size=train_config.hidden_size,
        num_layers=train_config.num_layers,
        num_heads=train_config.num_heads,
        ffn_size=train_config.ffn_size,
        max_position_embeddings=train_config.max_position_embeddings,
        window_size=None if train_config.window_size <= 0 else train_config.window_size,
        gru_hidden=train_config.gru_hidden,
        use_recurrence=train_config.use_recurrence,
    )
    return RecurrentPythiaForCausalLM(model_config)


def detach_states(states: list[Optional[torch.Tensor]]) -> list[Optional[torch.Tensor]]:
    return [state.detach() if state is not None else None for state in states]


def run_sequence_loss(
    model: RecurrentPythiaForCausalLM,
    batch: dict[str, torch.Tensor],
    chunk_size: int,
    device: torch.device,
    train: bool,
    truncate_bptt: bool,
) -> tuple[torch.Tensor, list[Optional[torch.Tensor]]]:
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    seq_len = input_ids.size(1)
    states: list[Optional[torch.Tensor]] = [None] * len(model.layers)
    total_loss = torch.zeros((), device=device)
    total_tokens = 0

    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        chunk_input_ids = input_ids[:, chunk_start:chunk_end]
        chunk_labels = labels[:, chunk_start:chunk_end]
        logits, states = model(
            chunk_input_ids,
            states=states,
            position_offset=chunk_start,
        )
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            chunk_labels.reshape(-1),
            reduction="sum",
        )
        total_loss = total_loss + loss
        total_tokens += chunk_labels.numel()
        if train and truncate_bptt:
            states = detach_states(states)

    mean_loss = total_loss / max(total_tokens, 1)
    return mean_loss, states


def evaluate(
    model: RecurrentPythiaForCausalLM,
    loader: DataLoader,
    chunk_size: int,
    device: torch.device,
    max_batches: int,
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            loss, _ = run_sequence_loss(
                model,
                batch,
                chunk_size=chunk_size,
                device=device,
                train=False,
                truncate_bptt=False,
            )
            losses.append(loss.item())

    mean_loss = float(sum(losses) / max(len(losses), 1))
    return mean_loss, mean_loss / math.log(2)


def dry_run_train_step(
    data_path: str,
    seq_len: int,
    split: str,
    batch_size: int,
    window_size: int,
    use_recurrence: bool,
) -> None:
    train_config = TrainConfig(
        data_path=data_path,
        seq_len=seq_len,
        split=split,
        batch_size=batch_size,
        window_size=window_size,
        use_recurrence=use_recurrence,
        max_steps=1,
    )
    set_seed(train_config.seed)
    dataset = build_dataset(data_path=data_path, seq_len=seq_len, split=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = build_model(train_config, dataset.vocab_size)
    device = torch.device(train_config.device)
    model.to(device)
    batch = next(iter(loader))
    chunk_size = seq_len if window_size <= 0 else window_size
    loss, states = run_sequence_loss(
        model,
        batch,
        chunk_size=chunk_size,
        device=device,
        train=False,
        truncate_bptt=False,
    )
    logits, _ = model(batch["input_ids"][:, :chunk_size].to(device))
    print(f"logits_shape={tuple(logits.shape)}")
    print(f"state_shapes={[tuple(state.shape) for state in states if state is not None]}")
    print(f"loss={loss.item():.4f}")


def write_run_metadata(train_config: TrainConfig, model_info: dict[str, Any]) -> tuple[Path, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_slug = train_config.wandb_name or (
        Path(train_config.config_path).stem if train_config.config_path else f"train_{timestamp}"
    )
    log_dir = Path(train_config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = log_dir / f"{run_slug}.resolved.yaml"
    metadata = {
        "train_config": asdict(train_config),
        "model_info": model_info,
        "resolved_at": timestamp,
    }
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return metadata_path, run_slug


def train_model(train_config: TrainConfig) -> None:
    set_seed(train_config.seed)
    device = torch.device(train_config.device)

    train_dataset = build_dataset(
        data_path=train_config.data_path,
        seq_len=train_config.seq_len,
        split=train_config.split,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        drop_last=True,
    )

    val_loader = None
    if train_config.eval_every > 0:
        val_dataset = build_dataset(
            data_path=train_config.data_path,
            seq_len=train_config.seq_len,
            split="val",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=train_config.num_workers,
            drop_last=False,
        )

    model = build_model(train_config, train_dataset.vocab_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    chunk_size = train_config.seq_len if train_config.window_size <= 0 else train_config.window_size

    model_info = {
        "vocab_size": train_dataset.vocab_size,
        "params": sum(parameter.numel() for parameter in model.parameters()),
        "chunk_size": chunk_size,
        "device": str(device),
    }
    metadata_path, run_slug = write_run_metadata(train_config, model_info)
    print("train_config", asdict(train_config))
    print("model_config", model_info)
    print(f"resolved_config_path={metadata_path}")

    wandb_run = wandb.init(
        project=train_config.wandb_project,
        name=train_config.wandb_name or run_slug,
        mode=train_config.wandb_mode,
        anonymous=train_config.wandb_anonymous,
        tags=list(train_config.wandb_tags),
        config={**asdict(train_config), **model_info, "resolved_config_path": str(metadata_path)},
    )

    global_step = 0
    reached_step_limit = False
    try:
        model.train()
        for epoch in range(train_config.epochs):
            for batch in train_loader:
                loss, _ = run_sequence_loss(
                    model,
                    batch,
                    chunk_size=chunk_size,
                    device=device,
                    train=True,
                    truncate_bptt=train_config.truncate_bptt,
                )
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                train_loss = loss.item()
                train_bpc = train_loss / math.log(2)
                if global_step % train_config.log_every == 0:
                    print(
                        f"step={global_step} epoch={epoch + 1} "
                        f"loss={train_loss:.4f} bpc={train_bpc:.4f}"
                    )

                wandb.log(
                    {
                        "train/loss": train_loss,
                        "train/bpc": train_bpc,
                        "train/grad_norm": float(grad_norm),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch + 1,
                    },
                    step=global_step,
                )

                if val_loader is not None and train_config.eval_every > 0 and global_step % train_config.eval_every == 0:
                    val_loss, val_bpc = evaluate(
                        model,
                        val_loader,
                        chunk_size=chunk_size,
                        device=device,
                        max_batches=train_config.eval_batches,
                    )
                    print(f"eval_step={global_step} val_loss={val_loss:.4f} val_bpc={val_bpc:.4f}")
                    wandb.log(
                        {
                            "eval/loss": val_loss,
                            "eval/bpc": val_bpc,
                            "epoch": epoch + 1,
                        },
                        step=global_step,
                    )
                    model.train()

                if train_config.max_steps > 0 and global_step >= train_config.max_steps:
                    reached_step_limit = True
                    break

            if reached_step_limit:
                break
    finally:
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_path = checkpoint_dir / "last.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "train_config": asdict(train_config),
                "step": global_step,
            },
            checkpoint_path,
        )
        print(f"saved_checkpoint={checkpoint_path}")
        artifact = wandb.Artifact("last-checkpoint", type="model")
        artifact.add_file(str(checkpoint_path))
        wandb_run.log_artifact(artifact)
        wandb.finish()


def resolve_train_parser_defaults(argv: Optional[list[str]]) -> dict[str, Any]:
    argv = argv or []
    if "train" not in argv:
        return {}
    config_path = None
    for index, token in enumerate(argv):
        if token == "--config" and index + 1 < len(argv):
            config_path = argv[index + 1]
            break
    if not config_path:
        return {}
    config_data = load_config_file(config_path)
    defaults = build_train_defaults(config_data)
    defaults["config_path"] = config_path
    return defaults


def build_parser(train_defaults: Optional[dict[str, Any]] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trainer CLI for recurrent Pythia MVP")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_configs = subparsers.add_parser("list-configs", help="List available experiment configs")
    list_configs.add_argument("--config-dir", default=str(CONFIGS_DIR))

    sanity = subparsers.add_parser("dataset-sanity-check", help="Validate dataset loading")
    sanity.add_argument("--data-path", required=True)
    sanity.add_argument("--seq-len", type=int, default=1024)
    sanity.add_argument("--split", default="train")
    sanity.add_argument("--batch-size", type=int, default=2)

    dry_run = subparsers.add_parser("dry-run-train-step", help="Run one forward/loss step")
    dry_run.add_argument("--data-path", required=True)
    dry_run.add_argument("--seq-len", type=int, default=1024)
    dry_run.add_argument("--split", default="train")
    dry_run.add_argument("--batch-size", type=int, default=2)
    dry_run.add_argument("--window-size", type=int, default=128)
    dry_run.add_argument("--no-recurrence", action="store_true")

    train = subparsers.add_parser("train", help="Run a real training loop")
    if train_defaults:
        train.set_defaults(**train_defaults)
    train.add_argument("--config", help="Path to a YAML/JSON train config file")
    train.add_argument("--data-path", required=not bool(train_defaults and train_defaults.get("data_path")))
    train.add_argument("--seq-len", type=int, default=1024)
    train.add_argument("--split", default="train")
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--max-steps", type=int, default=0)
    train.add_argument("--log-every", type=int, default=1)
    train.add_argument("--eval-every", type=int, default=0)
    train.add_argument("--eval-batches", type=int, default=8)
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--hidden-size", type=int, default=512)
    train.add_argument("--num-layers", type=int, default=6)
    train.add_argument("--num-heads", type=int, default=8)
    train.add_argument("--ffn-size", type=int, default=2048)
    train.add_argument("--max-position-embeddings", type=int, default=4096)
    train.add_argument("--window-size", type=int, default=128)
    train.add_argument("--gru-hidden", type=int, default=256)
    train.add_argument("--truncate-bptt", action="store_true")
    train.add_argument("--no-truncate-bptt", action="store_true")
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--wandb-project", default="recurrent-pythia-mvp")
    train.add_argument("--wandb-name")
    train.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    train.add_argument("--wandb-anonymous", choices=["allow", "must", "never"], default="allow")
    train.add_argument("--wandb-tags", nargs="*", default=[])
    train.add_argument("--log-dir", default="logs")
    train.add_argument("--no-recurrence", action="store_true")

    return parser


def list_configs(config_dir: str) -> None:
    root = Path(config_dir)
    if not root.exists():
        raise FileNotFoundError(f"Config directory not found: {root}")

    config_paths = sorted(path for path in root.iterdir() if path.suffix in {".yaml", ".yml", ".json"})
    for path in config_paths:
        config_data = load_config_file(path)
        description = config_data.get("description", "")
        model_type = "recurrent" if config_data.get("use_recurrence", True) else "baseline"
        window = config_data.get("window_size", "n/a")
        print(f"{path.name}: type={model_type} window={window} {description}".strip())


def main(argv: Optional[list[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    train_defaults = resolve_train_parser_defaults(argv)
    parser = build_parser(train_defaults)
    args = parser.parse_args(argv)

    if args.command == "list-configs":
        list_configs(args.config_dir)
        return

    if args.command == "dataset-sanity-check":
        dataset_sanity_check(
            data_path=args.data_path,
            seq_len=args.seq_len,
            split=args.split,
            batch_size=args.batch_size,
        )
        return

    if args.command == "dry-run-train-step":
        dry_run_train_step(
            data_path=args.data_path,
            seq_len=args.seq_len,
            split=args.split,
            batch_size=args.batch_size,
            window_size=args.window_size,
            use_recurrence=not args.no_recurrence,
        )
        return

    if args.command == "train":
        config_path = args.config or train_defaults.get("config_path")
        cli_overrides = cli_override_fields(argv)
        merged = train_config_defaults()
        merged.update(train_defaults)

        if "data_path" in cli_overrides or not merged.get("data_path"):
            merged["data_path"] = args.data_path

        for field_name in TRAIN_CONFIG_FIELDS - {"data_path", "use_recurrence", "config_path"}:
            if field_name in cli_overrides:
                merged[field_name] = getattr(args, field_name)

        if "use_recurrence" in cli_overrides:
            merged["use_recurrence"] = False
        else:
            merged["use_recurrence"] = train_defaults.get("use_recurrence", True)

        if "--truncate-bptt" in argv:
            merged["truncate_bptt"] = True
        if "--no-truncate-bptt" in argv:
            merged["truncate_bptt"] = False

        merged["config_path"] = config_path
        train_model(
            TrainConfig(**merged)
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
