from dataclasses import dataclass

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from datasets import Audio, interleave_datasets, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader


@dataclass
class WhisperDataCollatorWithPadding:
    """
    batching waveforms and sequences of tokens
    lightning module will convert waveforms into mel spectrogram
    """

    def __call__(self, features: list[dict]) -> dict:
        waveforms, wave_lengths = [], []
        token_ids, num_tokens = [], []
        for f in features:
            waveforms.append(f["audio"]["array"])
            wave_lengths.append(len(waveforms[-1]))

            token_ids.append(f["token_ids"])
            num_tokens.append(len(token_ids[-1]) - 1)

        max_n_samples = max(wave_lengths)
        waveforms = torch.as_tensor(
            np.array(
                [
                    np.pad(
                        arr,
                        (0, max_n_samples - n_samples),
                        "constant",
                        constant_values=0.0,
                    )
                    for arr, n_samples in zip(waveforms, wave_lengths)
                ],
                dtype=np.float32,
            )
        )

        max_num_tok = max(num_tokens)
        max_num_tok = (max_num_tok + 7) // 8 * 8  # optimal for T4
        # max_num_tok = (max_num_tok + 31) // 32 * 32 # optimal for A100
        token_ids = torch.as_tensor(
            np.array(
                [
                    np.pad(
                        tok, (0, max_num_tok - n_tok), "constant", constant_values=-100
                    )
                    for tok, n_tok in zip(token_ids, num_tokens)
                ]
            )
        )

        return {
            "waveforms": waveforms,
            "wave_lengths": torch.tensor(wave_lengths, dtype=torch.long),
            "token_ids": token_ids.long(),
        }


class ASRDataModule(L.LightningDataModule):
    def __init__(
        self,
        n_mels,
        pad_id,
        transcribe_id,
        translate_id,
        bs_per_device=4,
        n_worker_per_device=0,
        seed=3407,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.n_fft = 400
        self.hop_length = 160

        self.mel_transform = T.MelSpectrogram(
            n_mels=self.hparams.n_mels,
            sample_rate=16_000,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            power=2.0,
            center=True,
            pad_mode="reflect",
            f_min=0.0,
            f_max=8000.0,
            norm="slaney",
            mel_scale="slaney",
            window_fn=torch.hann_window,
        )

        self.time_mask = T.TimeMasking(time_mask_param=100)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=27)

    def on_after_batch_transfer(self, batch: dict, dataloader_idx: int) -> dict:
        """
        Prepare (input_ids, labels) pair and convert waveforms to mel spectrogram
        """
        token_ids = batch.pop("token_ids")

        batch["input_ids"] = token_ids[:, :-1].detach().clone()
        batch["input_ids"].masked_fill_(batch["input_ids"] == -100, self.hparams.pad_id)

        token_ids[:, 2].masked_fill(
            (token_ids[:, 2] == self.hparams.transcribe_id)
            | (token_ids[:, 2] == self.hparams.translate_id),
            -100,
        )
        batch["labels"] = token_ids[:, 1:]

        mel_specs = self.mel_transform(batch.pop("waveforms"))[..., :-1]
        # NOTE: to get log_mel, we need to process one waveform at a time because AmplitudeToDB use torch.max() that
        # computes max over all dimensions (including batch dimension), yet it should compute max for each sample in the batch
        # log_mels = torch.stack([self.db_transform(mel) for mel in mel_specs])
        # or manually without using T.AmplitudeToDB
        mel_specs = torch.clamp(mel_specs, min=1e-10).log10()
        mel_specs = torch.maximum(
            mel_specs, mel_specs.amax(dim=(-2, -1), keepdim=True) - 8.0
        )

        # normalization
        # mel_specs = (mel_specs + 40.0) / 40.0  # 40.0 and not 4.0 when using AmplitudeToDB because Decibel is 10xbel
        # or normalize by mean and std
        # mel_specs = (mel_specs - mel_specs.mean(dim=-1, keepdim=True)) / mel_specs.std(dim=-1, keepdim=True).clamp(min=1e-5)
        # or normalize with sliding window
        # mel_specs = self._slide_window_cmn(mel_specs)

        mel_specs = (mel_specs + 4.0) / 4.0

        if self.trainer.training:
            # data augmentation
            mel_specs = self.freq_mask(mel_specs)
            mel_specs = self.freq_mask(mel_specs)

            mel_specs = self.time_mask(mel_specs)
            mel_specs = self.time_mask(mel_specs)

        # spec_lengths = ((batch.pop("wave_lengths") - self.n_fft) // self.hop_length) + 1 # for center=False in MelSpectrogram
        spec_lengths = batch.pop("wave_lengths") // self.hop_length
        max_frames = spec_lengths.max()
        max_frames = (max_frames + 7) // 8 * 8
        pad = max_frames - mel_specs.shape[-1]
        mel_specs = F.pad(mel_specs, (0, pad, 0, 0, 0, 0), "constant", 0.0)
        batch["mels"] = mel_specs
        batch["mel_masks"] = (
            torch.arange(max_frames, device=mel_specs.device)[None, :]
            < spec_lengths[:, None]
        )
        return batch

    def setup(self, stage=None):
        """
        Manual spltiting training set:
        Split across rank -> split across workers
        """
        ds1 = load_dataset(
            "Kimang18/compact-DDD-Cambodia-khmer-speech-dataset-v0-2-0", streaming=True
        )
        ds2 = load_dataset(
            "Kimang18/compact-DDD-Cambodia-khmer-speech-dataset-en-translate-v0-2-0",
            streaming=True,
        )
        ds3 = load_dataset(
            "Kimang18/compact-librispeech-clean-100-v0-2-0", streaming=True
        )
        ds4 = load_dataset(
            "Kimang18/compact-librispeech-clean-100-km-translate-v0-2-0", streaming=True
        )
        ds5 = load_dataset("Kimang18/compact-nonspeech7k-v0-2-0", streaming=True)

        ds1 = ds1.cast_column("audio", Audio(sampling_rate=16_000))
        ds2 = ds2.cast_column("audio", Audio(sampling_rate=16_000))
        ds3 = ds3.cast_column("audio", Audio(sampling_rate=16_000))
        ds4 = ds4.cast_column("audio", Audio(sampling_rate=16_000))
        ds5 = ds5.cast_column("audio", Audio(sampling_rate=16_000))

        self.ds_train = [
            split_dataset_by_node(
                ds1["train"],
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size,
            ),
            split_dataset_by_node(
                ds2["train"],
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size,
            ),
            split_dataset_by_node(
                ds3["train"],
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size,
            ),
            split_dataset_by_node(
                ds4["train"],
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size,
            ),
            split_dataset_by_node(
                ds5["train"],
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size,
            ),
        ]

        self.ds_validation = interleave_datasets(
            [ds1["test"], ds2["test"], ds3["validation"], ds4["test"]],
            probabilities=[0.22, 0.21, 0.36, 0.21],
            stopping_strategy="all_exhausted_without_replacement",
            seed=self.hparams.seed,
        )

        self.mel_transform.to("cuda")
        self.time_mask.to("cuda")
        self.freq_mask.to("cuda")

    def get_dataloader(self, dataset, batch_size, num_workers, train=True):
        def worker_init_fn(worker_id: int):
            worker_info = get_worker_info()
            if worker_info is not None:
                worker_ds = worker_info.dataset
                new_raw_ds = [
                    source.shard(
                        num_shards=worker_info.num_workers, index=worker_info.id
                    )
                    for source in worker_ds.raw_ds
                ]
                worker_ds.raw_ds = new_raw_ds

        class PyTorchHFIterableDataset(torch.utils.data.IterableDataset):
            def __init__(self, raw_ds, buffer_size=1_000, seed=0):
                self.raw_ds, self.buffer_size, self.seed = (
                    raw_ds,
                    buffer_size,
                    seed,
                )
                self.epoch = 0

            def _build_pass(self, epoch):
                """shuffle within sources then interleave"""
                seed = self.seed + epoch
                sources = []
                for i in range(len(self.raw_ds)):
                    self.raw_ds[i].set_epoch(epoch)
                    sources.append(self.raw_ds[i])

                return interleave_datasets(
                    sources,
                    probabilities=[0.30, 0.23, 0.32, 0.1, 0.05],
                    stopping_strategy="all_exhausted",
                    seed=seed,
                )

            def __iter__(self):
                epoch = self.epoch
                while True:
                    yield from self._build_pass(epoch)
                    epoch += 1

            def state_dict(self):
                state_dict = self.hf_ds.state_dict()
                state_dict["dataset_epoch_count"] = self.epoch
                return state_dict

            def load_state_dict(self, state_dict):
                self.hf_ds.load_state_dict(state_dict)
                self.epoch = state_dict.get("dataset_epoch_count", 0)

        if train:
            dataset = PyTorchHFIterableDataset(
                dataset,
                seed=self.hparams.seed,
            )

        return StatefulDataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=WhisperDataCollatorWithPadding(),
            num_workers=num_workers,
            pin_memory=(
                num_workers
                > 0,  # pin_memory will lock memory while in this training, memory is quite scarce.
            ),
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True,
            worker_init_fn=worker_init_fn,
        )

    def train_dataloader(self):
        loader = self.get_dataloader(
            self.ds_train,
            self.hparams.bs_per_device,
            self.hparams.n_worker_per_device,
            train=True,
        )

        # restore loader if needed
        if (
            self.trainer is not None
            and hasattr(self.trainer.lightning_module, "_restore_dataset")
            and self.trainer.lightning_module._restore_dataset
        ):
            loader.load_state_dict(self.trainer.lightning_module._restore_dataset)
            self.trainer.lightning_module._restore_dataset = None
        return loader

    def val_dataloader(self):
        return self.get_dataloader(
            self.ds_validation, self.hparams.bs_per_device, 0, train=False
        )

    # def test_dataloader(self):
    #     return DataLoader(
    #         self.ds_test,
    #         batch_size=self.hparams.batch_size,
    #         collate_fn=WhisperDataCollatorWithPadding(pad_id=self.hparams.pad_id),
    #     )


if __name__ == "__main__":
    dm = ASRDataModule()
