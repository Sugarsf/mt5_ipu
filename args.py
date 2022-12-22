# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import argparse
import os
import sys
import yaml


config_file = os.path.join(os.path.dirname(__file__), "configs.yml")


def str_to_bool(value):
    if isinstance(value, bool) or value is None:
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise argparse.ArgumentTypeError(f'{value} is not a valid boolean value')


def parse_args(args=None):
    pparser = argparse.ArgumentParser("MT5 Configuration name", add_help=False)
    pparser.add_argument("--config",
                         type=str,
                         help="Configuration Name",
                         default='mt5_small')
    pargs, remaining_args = pparser.parse_known_args(args=args)
    config_name = pargs.config

    parser = argparse.ArgumentParser(
        "Poptorch MT5",
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    # Execution
    parser.add_argument("--batch_size", type=int, help="Set the micro batch_size",default=1)
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument("--device_iterations", type=int, help="Number of training steps to perform on the device for a single call to the model")
    parser.add_argument("--replication_factor", type=int, help="Number of replicas")
    parser.add_argument("--gradient_accumulation", type=int, help="Number of gradients to accumulate before updating the weights")
    parser.add_argument("--ipus_per_replica", type=int, help="Number of IPUs required by each replica")
    parser.add_argument("--random-seed", type=int, help="Seed for RNG")
    parser.add_argument("--matmul-proportion", type=float, nargs="+", help="Relative IPU memory proportion size allocated for matmul")
    parser.add_argument("--executable-cache-dir", type=str, default="",
                        help="Directory where Poplar executables are cached. If set, recompilation of identical graphs can be avoided. "
                        "Required for both saving and loading executables.")
    parser.add_argument("--dataloader_workers", type=int, help="The number of dataloader workers")
    parser.add_argument("--wandb", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Enabling logging to Weights and Biases")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--enable-half-partials", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Enable half partials for matmuls and convolutions globally")


    # Optimizer
    parser.add_argument("--optimizer", type=str, choices=['AdamW', 'Adam', 'SGD', 'LAMB', 'LAMBNoBias'], help="optimizer to use for the training")
    parser.add_argument("--learning_rate", type=float, help="Learning rate value for constant schedule, maximum for linear schedule.")
    parser.add_argument("--lr_schedule", type=str, choices=["constant", "linear", "cosine"],
                        help="Type of learning rate schedule. --learning-rate will be used as the max value")
    parser.add_argument("--loss_scaling", type=float, help="Loss scaling factor (recommend using powers of 2)")
    parser.add_argument("--weight_decay", type=float, help="Set the weight decay")
    parser.add_argument('--warmup_epochs', type=int, default=5, help="Number of warmup epochs")
    parser.add_argument('--state_onchip', type=str_to_bool, default=True, help="put the state of optimizer on chip")
    parser.add_argument('--eps', type=float, default=1e-6, help="set the eps for the optimizer")
    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="", help="Directory where checkpoints will be saved and restored from.\
                             This can be either an absolute or relative path. If this is not specified, only end of run checkpoint is\
                             saved in an automatically generated directory at the root of this project. Specifying directory is\
                             recommended to keep track of checkpoints.")
    parser.add_argument("--checkpoint_save_steps", type=int, default=100,
                        help="Option to checkpoint model after n steps.")
    parser.add_argument("--checkpoint_file", type=str, default="", help="Checkpoint to be retrieved for further training. This can\
                              be either an absolute or relative path to the checkpoint file.")

    parser.add_argument("--config", type=str, help="Configuration name")

    # Load the yaml
    yaml_args = dict()
    if config_name is not None:
        with open(config_file, "r") as f:
            try:
                yaml_args.update(**yaml.safe_load(f)[config_name])
            except yaml.YAMLError as exc:
                print(exc)
                sys.exit(1)

    # Check the yaml args are valid
    known_args = set(vars(parser.parse_args("")))
    unknown_args = set(yaml_args) - known_args

    if unknown_args:
        print(f" Warning: Unknown arg(s) in config file: {unknown_args}")

    parser.set_defaults(**yaml_args)
    args = parser.parse_args(remaining_args)
    

    # Expand matmul_proportion input into list representation
    if isinstance(args.matmul_proportion, float):
        args.matmul_proportion = [args.matmul_proportion] * args.ipus_per_replica

    if len(args.matmul_proportion) != args.ipus_per_replica:
        if len(args.matmul_proportion) == 1:
            args.matmul_proportion = args.matmul_proportion * args.ipus_per_replica
        else:
            raise ValueError(f"Length of matmul_proportion doesn't match ipus_per_replica: {args.matmul_proportion} vs {args.ipus_per_replica}")

    args.global_batch_size = args.replication_factor * args.gradient_accumulation * args.batch_size
    args.samples_per_step = args.global_batch_size * args.device_iterations

    return args
