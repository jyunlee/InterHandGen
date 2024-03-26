import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np

from runners.diffhand import Diffhand

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    
    parser.add_argument("--seed", type=int, default=19960905, help="Random seed")
    parser.add_argument("--config", type=str, default="default.yml", 
                        help="Path to the config file")
    parser.add_argument("--exp", type=str, default="exp", 
                        help="Path for saving running related data.")
    parser.add_argument("--doc", type=str, default='default',
                        help="A string for documentation purpose. "\
                             "Will be the name of the log folder.", )
    parser.add_argument("--verbose", type=str, default="info", 
                        help="Verbose level: info | debug | warning | critical")
    parser.add_argument("--ni", action="store_true",
                        help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')

    # Diffusion process hyperparameters
    parser.add_argument("--skip_type", type=str, default="uniform",
                        help="skip according to (uniform or quad(quadratic))")
    parser.add_argument("--eta", type=float, default=0.0, 
                        help="eta used to control the variances of sigma")
    parser.add_argument("--sequence", action="store_true")

    # load pretrained model
    parser.add_argument('--model_path', default=None, type=str,
                        help='the path of pretrain model')
    parser.add_argument('--train', action = 'store_true',
                        help='train or evluate')

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, args.doc)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    
    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    if args.train:
        if os.path.exists(args.log_path):
            overwrite = False
            if args.ni:
                overwrite = True
            else:
                response = input("Folder already exists. Overwrite? (Y/N)")
                if response.upper() == "Y":
                    overwrite = True

            if overwrite:
                shutil.rmtree(args.log_path)
                os.makedirs(args.log_path)
            else:
                print("Folder exists. Program halted.")
                sys.exit(0)
        else:
            os.makedirs(args.log_path)

        with open(os.path.join(args.log_path, "config.yml"), "w") as f:
            yaml.dump(new_config, f, default_flow_style=False)

        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    
    try:
        runner = Diffhand(args, config)
        runner.prepare_data()

        runner.create_diffusion_model(args.model_path)

        if args.train:
            runner.train()
        else:
            runner.test_hyber()
    except Exception:
        logging.error(traceback.format_exc())

    return 0

if __name__ == "__main__":
    sys.exit(main())
