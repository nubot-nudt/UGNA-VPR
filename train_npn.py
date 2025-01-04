import os
from model import get_model
from data import get_data
from training import get_trainer
from utils import parser
import click
import yaml
import warnings
from networks.mixvpr import TeacherNet
import torch
warnings.filterwarnings("ignore")


def main():
    args = parser.parse_args(training_args)
    log_path = os.path.join("logs", args.model_name)

    # start training from scratch
    if not args.resume:
        if os.path.exists(log_path):
            click.confirm(
                "experiment already exits, start training from scratch? "
                "(will overwrite log files and checkpoints); "
                "otherwise, use --resume in command line",
                abort=True,
            )

        os.makedirs(log_path, exist_ok=True)
        os.makedirs(os.path.join(log_path, "checkpoints"), exist_ok=True)

        # load model configuration
        cfg = {}
        assert os.path.exists(args.setup_cfg_path)
        with open(args.setup_cfg_path, "r") as config_file:
            setup_cfg = yaml.safe_load(config_file)
            cfg["setup"] = setup_cfg

        model_cfg_path = setup_cfg["model_cfg_path"]
        data_cfg_path = setup_cfg["data_cfg_path"]
        trainer_cfg_path = setup_cfg["trainer_cfg_path"]

        assert os.path.exists(model_cfg_path)
        with open(model_cfg_path, "r") as config_file:
            model_cfg = yaml.safe_load(config_file)
            cfg["model"] = model_cfg

        # load dataset configuration
        assert os.path.exists(data_cfg_path)
        with open(data_cfg_path, "r") as config_file:
            data_cfg = yaml.safe_load(config_file)
            cfg["data"] = data_cfg

        # load trainer configuration
        assert os.path.exists(trainer_cfg_path)
        with open(trainer_cfg_path, "r") as config_file:
            trainer_cfg = yaml.safe_load(config_file)
            cfg["trainer"] = trainer_cfg

        # record complete configuration including command line args of the experiment
        cfg.update(args.__dict__)
        with open(f"{log_path}/training_setup.yaml", "w") as config_file:
            yaml.safe_dump(cfg, config_file, default_flow_style=False)

    # resume training
    else:
        assert os.path.exists(log_path), "experiment does not exist"
        with open(f"{log_path}/training_setup.yaml", "r") as config_file:
            cfg = yaml.safe_load(config_file)

    datamodule = get_data(cfg["data"])
    network, renderer = get_model(cfg["model"])
    trainer = get_trainer(cfg["trainer"])

    encoder = TeacherNet()
    pretrained_weights_path = 'logs/MixVPR/Cambridge/ckpt_best.pth.tar'
    pretrained_state_dict = torch.load(pretrained_weights_path)
    encoder.load_state_dict(pretrained_state_dict["state_dict"])

    trainer.setup_training(datamodule, network, renderer, encoder, args.model_name, args.resume)
    trainer.start()


def training_args(parser):
    """
    Parse arguments for training setup.
    """

    # mandatory arguments
    parser.add_argument(
        "--model_name",
        "-M",
        type=str,
        required=True,
        help="unique model name",
    )
    parser.add_argument(
        "--setup_cfg_path",
        type=str,
        required=True,
        help="path to experiment setup configuration",
    )
    # arguments with default values
    parser.add_argument("--resume", action="store_true", help="continue training")
    return parser


if __name__ == "__main__":
    main()
