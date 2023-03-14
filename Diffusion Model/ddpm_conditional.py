"""
This code train a conditional diffusion model on CIFAR.
It is based on @dome272.

@wandbcode{condition_diffusion}
"""

import argparse
from contextlib import nullcontext
import os
import copy
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from types import SimpleNamespace
from fastprogress import progress_bar, master_bar
from utils import *
from modules import UNet_conditional, EMA
import logging
import wandb

config = SimpleNamespace(    
    run_name = "DDPM_conditional",
    epochs = 100,
    noise_steps=1000,
    seed = 42,
    batch_size = 10,
    img_size = 64,
    num_classes = 13,
    dataset_path = "../data",                    #TODO Change directories to accord with working environment              
    embedding_path = "../all_embeddings_pca.pt", #TODO Change directories to accord with working environment  
    name_path = "../all_file_names.csv",         #TODO Change directories to accord with working environment 
    train_folder = "train",
    val_folder = "validation",
    device = "cuda",
    slice_size = 1,
    use_wandb = True,
    do_validation = True,
    fp16 = True,
    log_every_epoch = 10,
    num_workers=10,
    lr = 3e-4)


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, num_classes=13, c_in=3, c_out=3, device="cuda"):  # TODO classes
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.model = UNet_conditional(c_in, c_out, num_classes=num_classes).to(device)   # TODO classes
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)  #TODO Check ema_model code
        self.device = device
        self.c_in = c_in
        self.num_classes = num_classes  #TODO classes

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_images(self, x, t):
        "Add noise to images at instant t"
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    @torch.inference_mode()
    def sample(self, use_ema, embeddings, cfg_scale=3):  # TODO labels == classes
        n = embeddings.shape[0]  # TODO labels == classes
        logging.info(f"Sampling {n} new images....")
        model = self.ema_model if use_ema else self.model
        model.eval()
        with torch.inference_mode():
            x = torch.randn((n, self.c_in, self.img_size, self.img_size)).to(self.device)  # TODO creates n samples of random noise where n == number of classes
            for i in progress_bar(reversed(range(1, self.noise_steps)), total=self.noise_steps-1, leave=False):
                t = (torch.ones(n) * i).long().to(self.device) #TODO creates 1D tensor of containing timestep value n times 
                predicted_noise = model(x, t, embeddings) # TODO Labels in this case is a tensor of values 0 --> num_classes. 
                                                      # TODO This means that an embedding will come from each class during sampling. 
                                                      # TODO I will need to change this to extract a random audio embedding from each class.
                                                      # TODO The label object is defined in the first line of code of the log_images method


                if cfg_scale > 0:                     
                    uncond_predicted_noise = model(x, t, None)  # TODO Make sure that when None unconditional estimation occurs 
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def train_step(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.step_ema(self.ema_model, self.model)
        self.scheduler.step()

    def one_epoch(self, train=True, use_wandb=False):
        avg_loss = 0.
        if train: self.model.train()
        else: self.model.eval()
        pbar = progress_bar(self.train_dataloader, leave=False)
        for i, (images, embeddings) in enumerate(pbar):  #TODO extract embeddings rather than labels from dataloader
            with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()):
                images = images.to(self.device)
                embeddings = embeddings.to(self.device)  #TODO Send embeddings to device
                t = self.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.noise_images(images, t)
                if np.random.random() < 0.1:
                    embeddings = None
                predicted_noise = self.model(x_t, t, embeddings)  # TODO adjust model to take embeddings rather than labels
                loss = self.mse(noise, predicted_noise)
                avg_loss += loss
            if train:
                self.train_step(loss)
                if use_wandb: 
                    wandb.log({"train_mse": loss.item(),
                                "learning_rate": self.scheduler.get_last_lr()[0]})
            pbar.comment = f"MSE={loss.item():2.3f}"        
        return avg_loss.mean().item()

    def log_images(self, args, use_wandb=False): # I added args
        # MY CODE 
        ## Reads embedding and embedding name files 
        embedding_file = torch.load(args.embedding_path)
        names_file = np.loadtxt(args.name_path, delimiter=',', dtype=str)

        ## Gets all instrument names in order 
        class_names = glob.glob(os.path.join(args.dataset_path, args.train_folder) + "/*")
        class_names = [name.split("/")[-1] for name in class_names]
        class_names.sort()

        ## Extracts a random embedding for each instrument
        embedding_list = []
        for cls in class_names:
            files = [file for file in os.listdir(os.path.join(args.dataset_path, args.train_folder, cls)) if file[-3:] == "jpg"]
            idx = np.random.randint(0, len(files))
            sample_file = files[idx]
            emb_idx = np.where(names_file == sample_file)[0][0]
            embedding_list.append(embedding_file[emb_idx])

        embeddings = torch.stack(embedding_list).to(self.device)  # Formats embeddings to (num_classes, embedding_dim)



        "Log images to wandb and save them to disk"
        #labels = torch.arange(self.num_classes).long().to(self.device)  #TODO This creates tensor of numbers 0 --> num_classes such that an instance from each class is sampled
        sampled_images = self.sample(use_ema=False, embeddings=embeddings)  #TODO I will need to adjust labels to be random audio embeddings from each class 
        ema_sampled_images = self.sample(use_ema=True, embeddings=embeddings) #TODO I will need to adjust labels to be random audio emneddings from each class. 
                                                                      #TODO These embeddings are used in the sample() method to condition the sampling process
        # plot_images(sampled_images)  #to display on jupyter if available
        if use_wandb:
            wandb.log({"sampled_images":     [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in sampled_images]})
            wandb.log({"ema_sampled_images": [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in ema_sampled_images]})

    def load(self, model_cpkt_path, model_ckpt="ckpt.pt", ema_model_ckpt="ema_ckpt.pt"):  #TODO IF I WANT TO LOAD CHECKPOINTS I WILL NEED TO ADJUST FILENAMES TO INCLUDE THE DESIRED EPOCH NUMBER
        self.model.load_state_dict(torch.load(os.path.join(model_cpkt_path, model_ckpt)))
        self.ema_model.load_state_dict(torch.load(os.path.join(model_cpkt_path, ema_model_ckpt)))

    def save_model(self, run_name, use_wandb=False, epoch=-1):
        "Save model locally and on wandb"
        torch.save(self.model.state_dict(), os.path.join("models", run_name, f"ckpt.pt"))          #TODO ADD EPOCH TO SAVE IF CHECKPOINTING os.path.join("models", run_name, f"ckpt_{epoch}.pt"))      
        torch.save(self.ema_model.state_dict(), os.path.join("models", run_name, f"ema_ckpt.pt"))  #TODO ADD EPOCH TO SAVE IF CHECKPOINTING os.path.join("models", run_name, f"ema_ckpt_{epoch}.pt"))   
        torch.save(self.optimizer.state_dict(), os.path.join("models", run_name, f"optim.pt"))     #TODO ADD EPOCH TO SAVE IF CHECKPOINTING os.path.join("models", run_name, f"optim_{epoch}.pt"))
        if use_wandb:
            at = wandb.Artifact("model", type="model", description="Model weights for DDPM conditional", metadata={"epoch": epoch})
            at.add_dir(os.path.join("models", run_name))
            wandb.log_artifact(at)

    def prepare(self, args):
        mk_folders(args.run_name)
        device = args.device
        self.train_dataloader, self.val_dataloader = get_data(args)  #TODO get_data() is a big component, make sure this code works with the re-written version
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=0.001)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr, 
                                                 steps_per_epoch=len(self.train_dataloader), epochs=args.epochs)
        self.mse = nn.MSELoss()
        self.ema = EMA(0.995)
        self.scaler = torch.cuda.amp.GradScaler()

    def fit(self, args):
        for epoch in progress_bar(range(args.epochs), total=args.epochs, leave=True):
            logging.info(f"Starting epoch {epoch}:")
            _  = self.one_epoch(train=True, use_wandb=args.use_wandb)
            
            ## validation
            if args.do_validation:
                avg_loss = self.one_epoch(train=False, use_wandb=args.use_wandb)
                if args.use_wandb:
                    wandb.log({"val_mse": avg_loss})
            
            # log predicitons
            if epoch % args.log_every_epoch == 0:
                self.log_images(args, use_wandb=args.use_wandb)
                # self.save_model(run_name=args.run_name, use_wandb=args.use_wandb, epoch=epoch)  #TODO Uncomment this to save model checkpoints every time images are saved

        # save model
        self.save_model(run_name=args.run_name, use_wandb=args.use_wandb, epoch=epoch)  




def parse_args(config):
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--run_name', type=str, default=config.run_name, help='name of the run')
    parser.add_argument('--epochs', type=int, default=config.epochs, help='number of epochs')
    parser.add_argument('--seed', type=int, default=config.seed, help='random seed')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parser.add_argument('--img_size', type=int, default=config.img_size, help='image size')
    parser.add_argument('--num_classes', type=int, default=config.num_classes, help='number of classes')
    parser.add_argument('--dataset_path', type=str, default=config.dataset_path, help='path to dataset')
    parser.add_argument('--embedding_path', type=str, default=config.embedding_path, help='path to all_embeddings.pt')
    parser.add_argument('--name_path', type=str, default=config.name_path, help='path to all_names.csv')
    parser.add_argument('--device', type=str, default=config.device, help='device')
    parser.add_argument('--use_wandb', type=bool, default=config.use_wandb, help='use wandb')
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
    parser.add_argument('--slice_size', type=int, default=config.slice_size, help='slice size')
    parser.add_argument('--noise_steps', type=int, default=config.noise_steps, help='noise steps')
    args = vars(parser.parse_args())
    
    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)


if __name__ == '__main__':
    parse_args(config)

    ## seed everything
    set_seed(config.seed)

    diffuser = Diffusion(config.noise_steps, img_size=config.img_size, num_classes=config.num_classes)
    with wandb.init(project="train_sd", group="train", config=config) if config.use_wandb else nullcontext():
        diffuser.prepare(config)
        diffuser.fit(config)
