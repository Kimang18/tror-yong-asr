import logging
import math

import hydra
import lightning as L
import torch
import torchmetrics as tm
import whisper
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    TQDMProgressBar,
)
from omegaconf import DictConfig
from safetensors.torch import save_model
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer
from tror_yong_asr import TrorYongASRConfig

from .src.data import ASRDataModule
from .src.muon_adam import MuonWithAuxAdam
from .src.net import TrorYongASRDev

# Set the logging level for httpx to WARNING to hide INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)


class TrorYongModule(L.LightningModule):
    def __init__(self, net, train_encoder_layer=-1, compile=False):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net

        self.val_acc = tm.Accuracy(
            task="multiclass",
            num_classes=self.net.vocab_size,
            ignore_index=-100,  # in labels, the padding token is -100
            # average="micro",
        )

    def training_step(self, batch: dict, batch_idx: int):
        bs = batch["mels"].shape[0]
        loss, len_loss = self.net.decoder_forward(
            *self.net.encoder(batch["mels"], batch["mel_masks"]),
            batch["input_ids"],
            batch["labels"],
        )
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=bs,
        )
        for i in range(len(self.trainer.optimizers[0].param_groups)):
            cur_lr = self.trainer.optimizers[0].param_groups[i]["lr"]
            self.log(f"lr/pg_{i + 1}", cur_lr, on_step=True)
        # if self.trainer.fit_loop.epoch_loop.total_batch_idx + 1 > self.trainer.max_steps:
        #     self.trainer.should_stop = True
        # track attention logits, for debugging only
        # with torch.no_grad():
        #     max_logit = torch.stack([self.net.decoder.self_attn.max_logits, self.net.decoder.cross_attn.max_logits]).max()
        # self.log('train_attn_logit', max_logit, on_step=True, prog_bar=False)
        return loss + len_loss

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_acc.reset()

    def validation_step(self, batch, batch_idx):
        bs = batch["mels"].shape[0]
        outputs = self.net(**batch)
        self.val_acc.update(
            outputs.logits.reshape(-1, self.net.vocab_size),
            batch["labels"].reshape(-1),
        )
        self.log("val/loss", outputs.loss, prog_bar=True, batch_size=bs)
        self.log("val/acc", self.val_acc, prog_bar=True)

    # def on_validation_epoch_end(self):
    #     loss = self.val_loss.compute()
    #     self.val_loss_best(loss)
    #     self.log('val_loss_best', self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net, dynamic=True)

    def on_save_checkpoint(self, checkpoint):
        # facilitate data streaming over checkpoints
        train_dataloader = self.trainer.train_dataloader
        if isinstance(train_dataloader, StatefulDataLoader):
            checkpoint["dataset_state_dict"] = train_dataloader.state_dict()

    def on_load_checkpoint(self, checkpoint):
        # facilitate data streaming over checkpoints
        self._restore_dataset = checkpoint.get("dataset_state_dict", None)

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val, gradient_clip_algorithm
    ):
        # only clip the gradient of parameters optimized by Adam
        torch.nn.utils.clip_grad_norm_(self.clip_params, max_norm=gradient_clip_val)

    def configure_optimizers(self):
        """Use hybrid optimizer: AdamW + Muon"""
        # freeze encoder
        assert (
            0 <= self.hparams.train_encoder_layer <= len(self.net.encoder.blocks)
            or self.hparams.train_encoder_layer == -1
        ), (
            "train_encoder_layer must be between 0 and len(self.net.encoder.blocks) or -1 for full training"
        )
        if self.hparams.train_encoder_layer >= 0:
            for block in self.net.encoder.blocks[
                : len(self.net.encoder.blocks) - self.hparams.train_encoder_layer
            ]:
                for param in block.parameters():
                    param.requires_grad = False

        param_groups, clip_params = self.net.configure_optimizer()
        optimizer = MuonWithAuxAdam(param_groups)
        self.clip_params = clip_params

        optimizer_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = int(0.015 * optimizer_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, optimizer_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # scheduler = get_cosine_schedule_with_warmup(
        #     optimizer, num_warmup_steps=warmup_steps, num_training_steps=optimizer_steps
        # )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(seed=cfg.seed, workers=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "KrorngAI/TrorYongASR-small",
        language="km",
        task="transcribe",
        trust_remote_code=True,
    )
    # NOTE: language and task tokens are not used. We just want the number of sot_sequence, not the tokens themselves.

    dm = ASRDataModule(
        n_mels=cfg.net.size.n_mels,
        pad_id=tokenizer.eot,
        transcribe_id=tokenizer.transcribe,
        translate_id=tokenizer.translate,
        seed=cfg.seed,
        **cfg.data,
    )

    # lm_head outputs logit over 5 special tokens too (km, en, transcribe, translate, nospeech) this is because embedding layer must contain these tokens and lm_head shares weight with embedding layer
    config = TrorYongASRConfig(
        vocab_size=len(tokenizer),
        n_audio_ctx=cfg.net.n_audio_ctx,
        n_text_ctx=cfg.net.n_text_ctx,
        dropout=cfg.net.dropout,
        bias=cfg.net.bias,
        pad_token_id=tokenizer.eot,
        eot_token_id=tokenizer.eot,
        transcribe_id=tokenizer.transcribe,
        **cfg.net.size,
    )
    net = TrorYongASRDev(config, len(tokenizer.sot_sequence))
    net.encoder.load_weights_from_whisper(
        whisper.load_model(cfg.net_size).encoder.state_dict()
    )

    model = TrorYongModule(net, **cfg.model)

    # save_dir = "/content/drive/MyDrive/Checkpoint_TrorYongASR"
    # os.makedirs(save_dir, exist_ok=True)
    weights_callback = ModelCheckpoint(
        # dirpath=save_dir,
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        # Adding {step} to the filename is best practice for step-based training
        filename="model-{step:06d}-{val_acc:.3f}",
        # every_n_train_steps=500, # Ensures the callback checks the monitor every 500 steps
    )
    # logger = WandbLogger(project=f"TrorYongASR-{cfg.net_size}", log_model=True)
    # logger.watch(net)

    class JupyterNoProgressBar(TQDMProgressBar):
        def init_validation_tqdm(self):
            bar = super().init_validation_tqdm()
            bar.disable = True  # Turns off the validation progress bar completely
            return bar

    torch.set_float32_matmul_precision("high")
    trainer = L.Trainer(
        num_sanity_val_steps=0,  # skip sanity check
        enable_progress_bar=True,
        check_val_every_n_epoch=None,  # Important for streaming!
        callbacks=[
            JupyterNoProgressBar(refresh_rate=200),
            # LearningRateMonitor(logging_interval='step'),
            # ckpt_callback,
            weights_callback,
        ],
        # logger=logger,
        **cfg.trainer,
    )

    trainer.fit(model, dm)
    # trainer.fit(model, dm, ckpt_path="/content/drive/MyDrive/Checkpoint_TrorYongASR/model-step=007611-val_acc=0.863.ckpt")
    # torch.save(model.net.state_dict(), "best_model_params1.pt")
    save_model(net, "model.safetensors")
    print(weights_callback.best_model_path)


if __name__ == "__main__":
    main()
