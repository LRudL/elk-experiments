from .extraction.extraction_main import run as run_extraction
from .extraction.parser import get_extraction_parser
from .training.parser import get_training_parser
from .training.train import train
from argparse import ArgumentParser
from contextlib import nullcontext, redirect_stdout
from elk.files import args_to_uuid
from transformers import AutoConfig, PretrainedConfig
import logging
import os
import torch.distributed as dist


def run():
    parser = ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "extract",
        help="Extract hidden states from a model.",
        parents=[get_extraction_parser()],
    )
    subparsers.add_parser(
        "train",
        help=(
            "Train a set of ELK probes on hidden states from `elk extract`. "
            "The first argument has to be the name you gave to the extraction."
        ),
        parents=[get_training_parser()],
    )
    subparsers.add_parser(
        "elicit",
        help=(
            "Extract and train a set of ELK probes "
            "on hidden states from `elk extract`. "
        ),
        parents=[get_extraction_parser(), get_training_parser(name=False)],
        conflict_handler="resolve",
    )

    subparsers.add_parser(
        "eval", help="Evaluate a set of ELK probes generated by `elk train`."
    )
    args = parser.parse_args()

    if model := getattr(args, "model", None):
        config = AutoConfig.from_pretrained(model)
        assert isinstance(config, PretrainedConfig)

        num_layers = getattr(config, "num_layers", config.num_hidden_layers)
        assert isinstance(num_layers, int)

        if args.layers and args.layer_stride > 1:
            raise ValueError(
                "Cannot use both --layers and --layer-stride. Please use only one."
            )
        elif args.layer_stride > 1:
            args.layers = list(range(0, num_layers, args.layer_stride))

    # Support both distributed and non-distributed training
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        dist.init_process_group("nccl")
        local_rank = int(local_rank)

    # Default to CUDA iff available
    if args.device is None:
        import torch

        if not torch.cuda.is_available():
            args.device = "cpu"
        else:
            args.device = f"cuda:{local_rank or 0}"

    # Prevent printing from processes other than the first one
    with redirect_stdout(None) if local_rank != 0 else nullcontext():
        for key in list(vars(args).keys()):
            print("{}: {}".format(key, vars(args)[key]))

        if local_rank != 0:
            logging.getLogger("transformers").setLevel(logging.ERROR)

        # TODO: Implement the rest of the CLI
        if args.command == "extract":
            run_extraction(args)
        elif args.command == "train":
            train(args)
        elif args.command == "elicit":
            args.name = args_to_uuid(args)
            try:
                train(args)
            except (EOFError, FileNotFoundError):
                run_extraction(args)

                # Ensure the extraction is finished before starting training
                if dist.is_initialized():
                    dist.barrier()

                train(args)

        elif args.command == "eval":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    run()
