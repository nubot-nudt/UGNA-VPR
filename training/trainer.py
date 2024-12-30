import os
import numpy as np
import torch
from dotmap import DotMap
from model import loss_type
from torch.utils.tensorboard.writer import SummaryWriter
from torchinfo import summary
import tqdm
from utils import util
import matplotlib.pyplot as plt
from networks.mobilenet import TeacherNet

class Trainer:
    def __init__(
        self,
        tracking_metric,
        num_epoch_repeats,
        num_epochs,
        ray_batch_size,
        nviews,
        use_data_augmentation,
        freeze_encoder,
        print_interval,
        save_interval,
        vis_interval,
        vis_repeat,
        gpu_id,
        loss_cfg,
        optimizer_cfg,
    ):
        super().__init__()
        self.tracking_metric = tracking_metric
        self.num_epoch_repeats = num_epoch_repeats
        self.num_epochs = num_epochs
        self.ray_batch_size = ray_batch_size
        self.nviews = nviews
        self.use_data_augmentation = use_data_augmentation
        self.freeze_encoder = freeze_encoder
        self.print_interval = print_interval
        self.save_interval = save_interval
        self.vis_interval = vis_interval
        self.vis_repeat = vis_repeat
        self.gpu_id = gpu_id

        self.loss_cfg = loss_cfg
        self.optimizer_cfg = optimizer_cfg

    def setup_training(self, data_module, network, renderer, encoder, exp_name, resume):
        self.device = util.get_cuda(self.gpu_id[0])
        self.data_module = data_module
        self.network = network.to(self.device)
        self.renderer = renderer.to(self.device)
        self.encoder = encoder.to(self.device)
        self.encoder.eval()
        self.renderer_par = self.renderer.parallelize(self.network, self.gpu_id).eval()

        train_dataset = self.data_module.load_dataset(
            "train",
            use_data_augmentation=self.use_data_augmentation,
        )
        self.train_dataloader = self.data_module.get_dataloader(train_dataset)

        val_dataset = self.data_module.load_dataset("val")
        self.val_dataloader = self.data_module.get_dataloader(val_dataset)

        self.vis_data_iter = self.data_loop(self.val_dataloader)

        self.num_total_batches = len(
            self.train_dataloader.dataset
        )  # object number of training set - ON

        # get scene range of the dataset
        self.z_near = train_dataset.z_near
        self.z_far = train_dataset.z_far

        # whether train encoder
        self.network.stop_encoder_grad = self.freeze_encoder
        if self.network.stop_encoder_grad:
            self.network.encoder.eval()

        self.config_loss()
        self.config_optimizer()

        self.global_step = 0
        self.last_epoch_idx = 0

        self.log_path = os.path.join("logs", exp_name)
        self.ckpt_path = os.path.join(self.log_path, "checkpoints")
        self.last_ckpt = os.path.join(self.ckpt_path, "last.ckpt")
        self.best_ckpt = os.path.join(self.ckpt_path, "best.ckpt")
        self.writer = SummaryWriter(self.log_path)

        # set initial performance value
        if self.tracking_metric == "min":
            self.best_performance = np.inf
        elif self.tracking_metric == "max":
            self.best_performance = 0.0

        if os.path.exists(self.last_ckpt) and resume:
            self.load_state()

    def config_loss(self):
        print("configure loss function \n")
        _loss_type = self.loss_cfg["loss_type"]
        _rgb_loss_type = self.loss_cfg["rgb_loss_type"]

        if _loss_type == "logit":
            self.rgb_crit = loss_type.LogitWithUncertaintyLoss(_rgb_loss_type)
        elif _loss_type == "rgb":
            self.rgb_crit = loss_type.RGBLoss(_rgb_loss_type)
        else:
            RuntimeError("loss type not defined!")

    def config_optimizer(self):
        print("configure optimizer \n")
        lr = self.optimizer_cfg["learning_rate"]
        gamma = self.optimizer_cfg["gamma"]

        if self.renderer.is_trainable:
            params = list(self.network.parameters()) + list(self.renderer.parameters())
        else:
            params = self.network.parameters()

        self.optimizer = torch.optim.Adam(params, lr=lr)

        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=gamma
        )

    def load_state(self):
        print("loading checkpoint \n")
        checkpoint = torch.load(self.last_ckpt)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.renderer.load_state_dict(checkpoint["renderer_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.last_epoch_idx = checkpoint["epoch"]
        self.best_performance = checkpoint["best_performance"]

    def sanity_check(self):
        print("----------sanity check---------- \n")
        summary(self.network)
        summary(self.renderer)

        with torch.no_grad():
            self.network.eval()
            self.renderer.eval()
            self.val_stats = []

            test_data = next(self.vis_data_iter)
            self.validation_step(test_data)
            _ = self.validation_epoch_end()
            self.visualization_step(test_data)

    def start(self):
        self.sanity_check()

        print("----------start training---------- \n")

        for epoch in range(self.last_epoch_idx, self.num_epochs):
            print(f"\n----------epoch {epoch}---------- \n")
            train_progress = tqdm.tqdm(
                total=len(self.train_dataloader) * self.num_epoch_repeats,
                desc="Training",
            )
            val_progress = tqdm.tqdm(
                total=len(self.val_dataloader),
                desc="Validation",
            )

            self.epoch = epoch
            self.val_stats = []

            self.network.train()
            self.renderer.train()
            for _ in range(self.num_epoch_repeats):
                for data in self.train_dataloader:
                    self.training_step(data)
                    train_progress.update()

            self.network.eval()
            self.renderer.eval()
            for data in self.val_dataloader:
                self.validation_step(data)
                val_progress.update()

            val_performance = self.validation_epoch_end()

            if epoch % self.vis_interval == 0:
                self.visualization_step(next(self.vis_data_iter))

            self.lr_scheduler.step()

            # save last and best checkpoints
            if epoch % self.save_interval == 0:
                ckpt_state_dict = {
                    "epoch": epoch + 1,
                    "global_step": self.global_step + 1,
                    "best_performance": val_performance,
                    "network_state_dict": self.network.state_dict(),
                    "renderer_state_dict": self.renderer.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.lr_scheduler.state_dict(),
                }

                torch.save(ckpt_state_dict, self.last_ckpt)

                if (
                    self.tracking_metric == "min"
                    and val_performance < self.best_performance
                ):
                    self.best_performance = val_performance
                    torch.save(ckpt_state_dict, self.best_ckpt)
                elif (
                    self.tracking_metric == "max"
                    and val_performance > self.best_performance
                ):
                    self.best_performance = val_performance
                    torch.save(ckpt_state_dict, self.best_ckpt)

            train_progress.close()
            val_progress.close()

    def training_step(self, data):
        self.loss = None
        self.optimizer.zero_grad()
        self.loss = self.calc_loss(data)
        self.loss.backward()
        self.optimizer.step()

        if self.global_step % self.print_interval == 0:
            self.print_training_progress()

        self.global_step += 1

    def print_training_progress(self):
        self.writer.add_scalar("Loss/train", self.loss, self.global_step)
        self.writer.add_scalar(
            "Training_Status/learning_rate",
            self.lr_scheduler.get_last_lr()[0],
            self.global_step,
        )

        encoder_grad_norm = 0
        for params in self.network.encoder.parameters():
            if params.grad is not None:
                encoder_grad_norm += params.grad.data.norm(2).item()
        self.writer.add_scalar(
            "Training_Status/encoder_gradient_norm",
            encoder_grad_norm,
            self.global_step,
        )

        mlp_feature_grad_norm = 0
        for params in self.network.mlp_feature.parameters():
            if params.grad is not None:
                mlp_feature_grad_norm += params.grad.data.norm(2).item()
        self.writer.add_scalar(
            "Training_Status/mlp_feature_gradient_norm",
            mlp_feature_grad_norm,
            self.global_step,
        )

        mlp_out_grad_norm = 0
        for params in self.network.mlp_out.parameters():
            if params.grad is not None:
                mlp_out_grad_norm += params.grad.data.norm(2).item()
        self.writer.add_scalar(
            "Training_Status/mlp_out_gradient_norm",
            mlp_out_grad_norm,
            self.global_step,
        )

    def validation_step(self, data):
        with torch.no_grad():
            val_loss = self.calc_loss(data)
            stats = {"val_loss": val_loss.item()}
            self.val_stats.append(stats)

    def validation_epoch_end(self):
        loss = np.mean([tmp["val_loss"] for tmp in self.val_stats])
        self.writer.add_scalar("Loss/validation", loss, self.global_step)
        return loss

    def visualization_step(self, data):
        obj_id = np.random.randint(
            0, data["images"].shape[0]
        )  # pick one object to visualize
        images = data["images"][obj_id].to(self.device)  # (IN, 3, H, W)
        poses = data["poses"][obj_id].to(self.device)  # (IN, 4, 4)
        focal = data["focal"][obj_id : obj_id + 1].to(self.device)  # (1, 2)
        c = data["c"][obj_id : obj_id + 1].to(self.device)  # (1, 2)
        IN, _, H, W = images.shape

        images_0to1 = images * 0.5 + 0.5  # (IN, 3, H, W)

        # randomly select source views and target view
        view_tar = np.random.choice(IN, 1)[0]
        gt = images_0to1[view_tar].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)

        # remove target view from reference candidates
        images_index = np.arange(IN)
        images_index = images_index[images_index != view_tar]

        self.writer.add_image(
            "ground truth", (gt * 255).astype(np.uint8), 0, dataformats="HWC"
        )
        psnr_list = []
        ssim_list = []
        mse_list = []
        var_list = []
        log_var_list = []

        for i in range(self.vis_repeat):
            ref_nviews = self.nviews[torch.randint(0, len(self.nviews), (1,)).item()]
            views_src = np.random.choice(images_index, ref_nviews, replace=False)
            source_views = (
                images_0to1[views_src]
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
                .reshape(-1, H, W, 3)
            )
            self.writer.add_images(
                "source views",
                (source_views * 255).astype(np.uint8),
                global_step=i,
                dataformats="NHWC",
            )

            with torch.no_grad():
                #target_rays = cam_rays[view_tar]  # (H, W, 8)
                self.network.encode(
                    images[views_src].unsqueeze(0),
                    poses[views_src].unsqueeze(0),
                    focal,
                    c=c if c is not None else None,
                )

                images_0to1_view_tar = images_0to1[view_tar].unsqueeze(0)
                poses_view_tar=poses[view_tar].unsqueeze(0).unsqueeze(0)
                gt_des,_=self.encoder(images_0to1_view_tar)    # (1,3,244,244)->(1,d_feature)
                predict = DotMap(self.network(poses_view_tar))  # (1,d_feature)
                #print(predict.des.shape)
                #print("uncertainty")
                #print(predict.uncertainty)
                util.tb_visualizer(
                    predict, gt_des, self.writer, H, W, i
                )
                metrics_dict = util.calc_metrics(predict, torch.tensor(gt_des))

                mse_list.append(metrics_dict["mean_mse"])
                var_list.append(metrics_dict["mean_variance"])
                psnr_list.append(metrics_dict["psnr"])
                ssim_list.append(metrics_dict["ssim"])
                log_var_list.append(metrics_dict["log_mean_variance"])

        fig, axs = plt.subplots(3, 1)
        axs[0].scatter(np.array(mse_list), np.array(var_list))
        axs[1].scatter(np.array(psnr_list), np.array(log_var_list))
        axs[2].scatter(np.array(ssim_list), np.array(var_list))
        self.writer.add_figure("calibration", fig)
        plt.close(fig)
        """
        self.writer.add_scalar(
            "Performance/avg_psnr", np.average(psnr_list), global_step=self.global_step
        )

        self.writer.add_scalar(
            "Performance/best_psnr", np.max(psnr_list), global_step=self.global_step
        )
        self.writer.add_scalar(
            "Performance/avg_ssim", np.average(ssim_list), global_step=self.global_step
        )

        self.writer.add_scalar(
            "Performance/best_ssim", np.max(ssim_list), global_step=self.global_step
        )
        """
    def calc_loss(self, data):
        # ON-object number, IN-image number, RN-reference number, H-height, W-width
        all_images = data["images"]  # (ON, IN, 3, H, W)
        all_poses = data["poses"]  # (ON, IN, 4, 4)
        all_focals = data["focal"].to(self.device)  # (ON, 2)
        all_c = data["c"].to(self.device)  # (ON, 2)
        ON, IN, _, _, _ = all_images.shape
        #print(all_focals.shape)
        #print(all_c.shape)
        all_rgb_gt = []
        all_poses_gt = []
        print("ON")
        print(ON)
        # randomly pick number of reference views
        ref_nviews = self.nviews[torch.randint(0, len(self.nviews), ()).item()]
        image_ord = torch.empty((ON, ref_nviews), dtype=torch.long)

        for obj_idx in range(ON):
            images = all_images[obj_idx]  # (IN, 3, H, W)
            poses = all_poses[obj_idx]  # (IN, 4, 4)
            focal = all_focals[obj_idx]
            c = all_c[obj_idx]

            ref_index = torch.from_numpy(
                np.random.choice(IN, ref_nviews, replace=False)
            )
            image_ord[obj_idx] = ref_index

            # generate boolean mask for filtering out reference images
            boolean_mask = np.ma.make_mask(np.ones(IN))
            boolean_mask[ref_index] = False

            H, W = images.shape[-2:]
            images_0to1 = images * 0.5 + 0.5  # as final prediction ranges from 0 to 1
            rgb_gt_all = images_0to1

            pose_gt_all = (poses[boolean_mask])  # (IN-RN, 3, H, W)
            rgb_gt_all = (rgb_gt_all[boolean_mask])  # (IN-RN, 4, 4)


            rgb_gt = rgb_gt_all.to(self.device)

            all_rgb_gt.append(rgb_gt)
            all_poses_gt.append(pose_gt_all)


        all_rgb_gt = torch.stack(all_rgb_gt)  # (ON, IN-RN, 3, H, W)
        all_poses_gt = torch.stack(all_poses_gt)  # (ON, IN-RN,  4, 4)
        #print(all_rgb_gt.shape)
        #print(all_poses_gt.shape)
        # randomly select reference views and poses from object images
        ref_images = util.batched_index_select_nd(all_images, image_ord).to(
            self.device
        )  # (ON, RN, 3, H, W)
        ref_poses = util.batched_index_select_nd(all_poses, image_ord).to(
            self.device
        )  # (ON, RN, 4, 4)

        all_poses = all_images = None  # release memory
        #print("ref")
        #print(ref_images.shape)
        #print(ref_poses.shape)
        #print(all_focals.shape)
        #print(all_c.shape)
        self.network.encode(
            ref_images,
            ref_poses,
            all_focals,
            all_c,
        )
        all_poses_gt=all_poses_gt.to(self.device)
        #print(all_poses_gt.shape)
        ON,N_RN,_,_,_=all_rgb_gt.shape
        all_rgb_gt = all_rgb_gt.reshape(-1, *images.shape[1:])  # (ON*(N-RN)), 3, H, W)
        all_des_gt, _ = self.encoder(all_rgb_gt)  # (ON*(N-RN)),512)
        all_des_gt=all_des_gt.reshape(ON, N_RN, -1) # (ON, N-RN, feature=512)
        predict=DotMap(self.network(all_poses_gt))  # (ON, N-RN, feature=512)
        #print(predict.all_uncertainty.shape)
        loss = self.rgb_crit(predict, all_des_gt)
        #print("loss")
        #print(loss)
        return loss

    @staticmethod
    def data_loop(dl):
        """
        Loop an iterable infinitely
        """

        while True:
            for x in iter(dl):
                yield x

    @classmethod
    def init_from_cfg(cls, cfg):
        return cls(
            tracking_metric=cfg["tracking_metric"],
            num_epoch_repeats=cfg["num_epoch_repeats"],
            num_epochs=cfg["num_epochs"],
            ray_batch_size=cfg["ray_batch_size"],
            nviews=cfg["nviews"],
            use_data_augmentation=cfg["use_data_augmentation"],
            freeze_encoder=cfg["freeze_encoder"],
            print_interval=cfg["print_interval"],
            save_interval=cfg["save_interval"],
            vis_interval=cfg["vis_interval"],
            vis_repeat=cfg["vis_repeat"],
            gpu_id=cfg["gpu_id"],
            loss_cfg=cfg["loss"],
            optimizer_cfg=cfg["optimizer"],
        )
