"""Optional Weights & Biases helpers for script entry points."""
from __future__ import annotations


def add_wandb_args(parser):
    g = parser.add_argument_group("wandb")
    g.add_argument("--wandb", action="store_true",
                   help="log metrics and checkpoints to Weights & Biases")
    g.add_argument("--wandb-project", default="deeplob-mpo")
    g.add_argument("--wandb-entity", default=None)
    g.add_argument("--wandb-run-name", default=None)
    g.add_argument("--wandb-mode", default="online",
                   choices=["online", "offline", "disabled"])
    g.add_argument("--wandb-tags", default=None,
                   help="comma-separated tags to attach to the run")


def init_wandb(args, cfg, *, job_type, extra_config=None):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed. Install it with `pip install wandb`, "
            "then run `wandb login`, or omit --wandb."
        ) from exc

    tags = ([tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
            if args.wandb_tags else None)
    config = cfg.to_dict()
    if extra_config:
        config.update(extra_config)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        tags=tags,
        job_type=job_type,
        config=config,
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    return run


def log_metrics(run, prefix, loss, metrics):
    if not run:
        return
    run.log({
        f"{prefix}/loss": loss,
        **{f"{prefix}/{name}": value for name, value in metrics.items()},
    })
