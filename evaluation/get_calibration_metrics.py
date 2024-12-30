import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from utils import parser, util
from evaluation.pretrained_model import PretrainedModel
import torch
import numpy as np
import tqdm
import yaml
import warnings
from data import get_data
from dotmap import DotMap
import time
import random
import matplotlib
import seaborn as sb
import pandas
import scipy.stats as stats
from datetime import datetime

matplotlib.use("TkAgg")
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


def setup_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    """
    randomly select reference images and novel view.
    record uncertainty estimation and rendering error for each test
    """
    setup_random_seed(0)

    args = parser.parse_args(calibration_args)
    log_path = os.path.join("logs", args.model_name)

    assert os.path.exists(log_path), "experiment does not exist"
    with open(f"{log_path}/training_setup.yaml", "r") as config_file:
        cfg = yaml.safe_load(config_file)

    checkpoint_path = os.path.join(log_path, "checkpoints", "best.ckpt")
    assert os.path.exists(checkpoint_path), "checkpoint does not exist"
    ckpt_file = torch.load(checkpoint_path)

    gpu_id = list(map(int, args.gpu_id.split()))
    device = util.get_cuda(gpu_id[0])

    model = PretrainedModel(cfg["model"], ckpt_file, device, gpu_id)

    datamodule = get_data(cfg["data"])
    dataset = datamodule.load_dataset("val")
    z_near = dataset.z_near
    z_far = dataset.z_far

    print("---------- start evaluation ---------- \n")

    scene_num = len(dataset)
    test_progress = tqdm.tqdm(
        total=scene_num * args.repeat,
        desc="Evaluation",
    )

    total_df = pandas.DataFrame(
        {
            "Scan": [],
            "PSNR": [],
            "SSIM": [],
            "MSE": [],
            "NLL": [],
            "AUSE": [],
            "Variance": [],
            "Log Variance": [],
            "Pearson PSNR": [],
            "Pearson MSE": [],
            "Pearson SSIM": [],
            "Time": [],
        }
    )

    # load images from each scene
    for idx in range(scene_num):
        data_instance = dataset.__getitem__(idx)
        scene = data_instance["scan_name"]
        print(f"---------- test on {scene} ---------- \n")

        images = data_instance["images"].to(device)  # (IN, 3, H, W)
        poses = data_instance["poses"].to(device)  # (IN, 4, 4)
        focal = data_instance["focal"].to(device)  # (1, 2)
        c = data_instance["c"].to(device)  # (1, 2)
        IN, _, H, W = images.shape

        cam_rays = util.gen_rays(
            poses, W, H, focal, z_near, z_far, c=c
        )  # (IN, H, W, 8)
        images_0to1 = images * 0.5 + 0.5  # (IN, 3, H, W)

        # for each scene, we run multiple inference
        psnr_record = np.zeros(args.repeat)
        ssim_record = np.zeros(args.repeat)
        mse_record = np.zeros(args.repeat)
        var_record = np.zeros(args.repeat)
        nll_record = np.zeros(args.repeat)
        ause_record = np.zeros(args.repeat)
        log_var_record = np.zeros(args.repeat)
        time_record = np.zeros(args.repeat)

        for n_run in range(args.repeat):
            view_tar = np.random.choice(IN, 1)[0]
            gt = images_0to1[view_tar].permute(1, 2, 0).cpu().numpy()

            # remove target view from reference candidates
            images_index = np.arange(IN)
            images_index = images_index[images_index != view_tar]

            views_src = np.random.choice(images_index, args.nviews, replace=False)

            with torch.no_grad():
                target_rays = cam_rays[view_tar]  # (H, W, 8)
                target_rays = target_rays.reshape(1, H * W, -1)
                model.network.encode(
                    images[views_src].unsqueeze(0),
                    poses[views_src].unsqueeze(0),
                    focal.unsqueeze(0),
                    c.unsqueeze(0),
                )

                time_start = time.time()
                predict = DotMap(model.renderer_par(target_rays))
                time_end = time.time()

                inference_time = time_end - time_start
                metrics_dict = util.calc_metrics(predict, torch.tensor(gt))

                psnr_record[n_run] = metrics_dict["psnr"]
                ssim_record[n_run] = metrics_dict["ssim"]
                mse_record[n_run] = metrics_dict["mean_mse"]
                var_record[n_run] = metrics_dict["mean_variance"]
                nll_record[n_run] = metrics_dict["nll"]
                ause_record[n_run] = metrics_dict["ause"]
                log_var_record[n_run] = metrics_dict["log_mean_variance"]
                time_record[n_run] = inference_time

                test_progress.update()

        pearson_psnr = np.corrcoef(psnr_record, log_var_record)[0, 1]
        pearson_mse = np.corrcoef(mse_record, var_record)[0, 1]
        pearson_ssim = np.corrcoef(ssim_record, var_record)[0, 1]
        spearman_psnr = stats.spearmanr(psnr_record, log_var_record)
        spearman_mse = stats.spearmanr(mse_record, var_record)
        spearman_ssim = stats.spearmanr(ssim_record, var_record)
        ause = np.mean(ause_record)
        print(
            f"{scene}: pearson_mse: {pearson_mse}, pearson_psnr: {pearson_psnr}, pearson_ssim: {pearson_ssim}, ause: {ause}"
        )
        print(
            f"{scene}: spearman_mse: {spearman_mse}, spearman_psnr: {spearman_psnr}, spearman_ssim: {spearman_ssim}, ause: {ause}"
        )

        dataframe = pandas.DataFrame(
            {
                "Scan": scene,
                "PSNR": psnr_record,
                "SSIM": ssim_record,
                "MSE": mse_record,
                "Variance": var_record,
                "NLL": nll_record,
                "AUSE": ause_record,
                "Log Variance": log_var_record,
                "Pearson PSNR": pearson_psnr,
                "Pearson MSE": pearson_mse,
                "Pearson SSIM": pearson_ssim,
                "Time": time_record,
            }
        )

        total_df = total_df.append(dataframe, ignore_index=True)

    experiment_path = os.path.join(
        "experiments",
        args.model_name,
        "calibration_experiment",
        datetime.now().strftime("%d-%m-%Y-%H-%M"),
    )
    os.makedirs(experiment_path)
    total_df.to_csv(f"{experiment_path}/dataframe.csv")

    if args.need_visual_plot:
        print("---------- process data ---------- \n")
        process_data(total_df, experiment_path)


def process_data(dataframe, experiment_path):
    mse_var_plot = sb.scatterplot(data=dataframe, x="MSE", y="Variance", hue="Scan")
    mse_var_plot.figure.savefig(
        experiment_path + "/mse_var.svg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()


def calibration_args(parser):
    """
    Parse arguments for evaluation setup.
    """

    # mandatory arguments
    parser.add_argument(
        "--model_name",
        "-M",
        type=str,
        required=True,
        help="experiments need to be evaluated",
    )

    # arguments with default values
    parser.add_argument(
        "--scene_list",
        "-sl",
        type=str,
        default="all",
        help="scene id to be evaluated on, by default is the full test list",
    )
    parser.add_argument(
        "--gpu_id", type=str, default="0", help="GPU(s) to use, space delimited"
    )
    parser.add_argument(
        "--nviews", "-nv", type=int, default=3, help="number of reference views"
    )
    parser.add_argument(
        "--repeat", "-rp", type=int, default=100, help="repeat number for each scene"
    )
    parser.add_argument(
        "--need_visual_plot", action="store_true", help="whether plot output"
    )
    return parser


if __name__ == "__main__":
    main()
