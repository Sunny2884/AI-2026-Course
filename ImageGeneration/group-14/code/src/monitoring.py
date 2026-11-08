"""Sample-weighted distributed diffusion diagnostics and TensorBoard events."""

import json
import time

import torch


class TrainingMonitor:
    def __init__(self, accelerator, output, config, log_every=10, histogram_every=100,
                 image_every=100):
        self.accelerator = accelerator
        self.output = output
        self.log_every = log_every
        self.histogram_every = histogram_every
        self.image_every = image_every
        self.writer = None
        if accelerator.is_main_process:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(output / "tensorboard"), flush_secs=20,
                                        purge_step=config.get("resume_step"))
            self.writer.add_text("configuration", json.dumps(config, indent=2))
        self.pending = torch.zeros(14, device=accelerator.device, dtype=torch.float64)
        self.interval_sum = self.interval_count = 0
        self.started = time.perf_counter()

    def start_epoch(self):
        self.epoch_sum = self.epoch_count = 0
        self.started = time.perf_counter()

    def observe(self, losses, timesteps, num_timesteps, prediction, target):
        values = self.pending
        values[0] += losses.detach().double().sum()
        values[1] += losses.numel()
        bins = (timesteps * 5 // num_timesteps).clamp(max=4)
        for index in range(5):
            selected = losses.detach()[bins == index]
            values[2+2*index] += selected.double().sum()
            values[3+2*index] += selected.numel()
        values[12] += prediction.detach().float().square().flatten(1).mean(1).double().sum()
        values[13] += target.detach().float().square().flatten(1).mean(1).double().sum()

    def update(self, step, epoch, learning_rates, grad_norm, model, condition, timesteps):
        reduced = self.accelerator.reduce(self.pending, reduction="sum").cpu().tolist()
        self.pending.zero_()
        elapsed = time.perf_counter() - self.started
        self.started = time.perf_counter()
        total, count = reduced[:2]
        self.epoch_sum += total
        self.epoch_count += count
        self.interval_sum += total
        self.interval_count += count
        record = dict(step=step, epoch=epoch, loss=total/count, samples=count,
                      learning_rate=learning_rates, grad_norm=float(grad_norm),
                      seconds_per_update=elapsed, images_per_second=count/max(elapsed, 1e-9),
                      prediction_rms=(reduced[12]/count)**0.5,
                      target_rms=(reduced[13]/count)**0.5)
        if self.writer:
            self.writer.add_scalar("train/epoch", epoch, step)
            self.writer.add_scalar("train/processed_samples_this_update", count, step)
            for key in ("loss", "grad_norm", "prediction_rms", "target_rms"):
                self.writer.add_scalar("train/"+key, record[key], step)
            for index, rate in enumerate(learning_rates):
                self.writer.add_scalar("learning_rate/"+("lora" if index == 0 else "adapter"), rate, step)
            for index in range(5):
                if reduced[3+2*index]:
                    self.writer.add_scalar("loss_by_timestep/bin_{}".format(index),
                                           reduced[2+2*index]/reduced[3+2*index], step)
                    self.writer.add_scalar("samples_by_timestep/bin_{}".format(index), reduced[3+2*index], step)
            for key in ("seconds_per_update", "images_per_second"):
                self.writer.add_scalar("performance/"+key, record[key], step)
        memory_values = ([torch.cuda.memory_allocated()/2**30, torch.cuda.memory_reserved()/2**30,
                          torch.cuda.max_memory_allocated()/2**30]
                         if self.accelerator.device.type == "cuda" else [0., 0., 0.])
        memory = torch.tensor(memory_values, device=self.accelerator.device)
        memory = self.accelerator.gather(memory).reshape(-1, 3).cpu().tolist()
        record["gpu_memory_gib"] = memory
        if self.writer:
            for rank, values in enumerate(memory):
                for key, value in zip(("allocated", "reserved", "peak_allocated"), values):
                    self.writer.add_scalar("gpu_{}/{}_gib".format(rank, key), value, step)
            if self.histogram_every and step % self.histogram_every == 0:
                self.writer.add_histogram("rank0/timesteps", timesteps.detach().cpu(), step)
                for name, value in condition.items():
                    self.writer.add_histogram("rank0/condition/"+name, value.detach().float().cpu(), step)
                for name, parameter in self.accelerator.unwrap_model(model).named_parameters():
                    if parameter.requires_grad:
                        self.writer.add_histogram("parameters/"+name, parameter.detach().float().cpu(), step)
            with (self.output/"training_log.jsonl").open("a") as handle:
                handle.write(json.dumps(record)+"\n")
        if step % self.log_every == 0 or step == 1:
            mean = self.interval_sum/self.interval_count
            self.accelerator.print("epoch {} step {} loss {:.6f}; interval mean {:.6f}".format(
                epoch, step, record["loss"], mean), flush=True)
            if self.writer:
                self.writer.add_scalar("train/interval_loss", mean, step)
            self.interval_sum = self.interval_count = 0

    def epoch_end(self, epoch, step):
        mean = self.epoch_sum/self.epoch_count
        self.accelerator.print("epoch {} mean loss {:.6f} ({} processed samples)".format(
            epoch, mean, int(self.epoch_count)), flush=True)
        if self.writer:
            self.writer.add_scalar("train/epoch_loss", mean, step)
            with (self.output/"epoch_log.jsonl").open("a") as handle:
                handle.write(json.dumps(dict(epoch=epoch, step=step, loss=mean,
                                             processed_samples=self.epoch_count))+"\n")
            self.writer.flush()

    def images(self, step, batch, noisy, prediction, timesteps, vae, scheduler):
        if not self.writer or not self.image_every or step % self.image_every:
            return
        with torch.no_grad():
            count = min(4, len(noisy))
            alpha = scheduler.alphas_cumprod.to(noisy.device)[timesteps[:count]].reshape(-1, 1, 1, 1)
            if scheduler.config.prediction_type == "epsilon":
                clean = (noisy[:count].float() - (1-alpha).sqrt()*prediction[:count].float())/alpha.sqrt()
            else:
                clean = alpha.sqrt()*noisy[:count].float() - (1-alpha).sqrt()*prediction[:count].float()
            decoded = vae.decode(clean/vae.config.scaling_factor).sample.float().clamp(-1, 1)
            self.writer.add_images("train/teacher_target", (batch["target"][:count]+1)/2, step)
            self.writer.add_images("train/predicted_x0_not_free_generation", (decoded+1)/2, step)

    def close(self):
        if self.writer:
            self.writer.close()
