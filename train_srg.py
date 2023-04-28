import os
import torch
import wandb
import argparse
from tqdm import tqdm

# My file
from loss import Loss
from utils import show_tensor_images, save_images, get_data, show_tensor_images_wandb
from modules import Generator, Discriminator

from GPUtil import showUtilization as gpu_usage

from torch._inductor import config
config.compile_threads = 1

# Parse torch version for autocast
# ######################################################
version = torch.__version__
version = tuple(int(n) for n in version.split('.')[:-1])
has_autocast = version >= (1, 6)
# ######################################################




def train_srgan(args):
    device = args.device
    total_steps = args.epochs
    display_step = args.display_step

    
    generator = torch.load(args.gen_path)
    generator = generator.to(device).train()

    discriminator = Discriminator(n_blocks=1, base_channels=8)
    discriminator = discriminator.to(device).train()
    
    loss_fn = Loss(device=device)

    dataloader = get_data(args)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
    g_scheduler = torch.optim.lr_scheduler.LambdaLR(g_optimizer, lambda _: 0.1)
    d_scheduler = torch.optim.lr_scheduler.LambdaLR(d_optimizer, lambda _: 0.1)

    lr_step = total_steps // 2
    cur_step = 0

    mean_g_loss = 0.0
    mean_d_loss = 0.0
    
    
    # show image in wandb
    columns=["id", "lr_real", "hr_fake", "hr_real"]

    while cur_step < total_steps:
        for hr_real, lr_real in tqdm(dataloader, position=0):
            hr_real = hr_real.to(device)
            lr_real = lr_real.to(device)

            # Enable autocast to FP16 tensors (new feature since torch==1.6.0)
            # If you're running older versions of torch, comment this out
            # and use NVIDIA apex for mixed/half precision training
            if has_autocast:
                with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                    g_loss, d_loss, hr_fake = loss_fn(
                        generator, discriminator, hr_real, lr_real,
                    )
            else:
                g_loss, d_loss, hr_fake = loss_fn(
                    generator, discriminator, hr_real, lr_real,
                )

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            mean_g_loss += g_loss.item() / display_step
            mean_d_loss += d_loss.item() / display_step
            
            wandb.log({"G_Loss": mean_g_loss}, {"D_Loss": mean_d_loss})

            if cur_step == lr_step:
                g_scheduler.step()
                d_scheduler.step()
                print('Decayed learning rate by 10x.')

            if cur_step % display_step == 0 and cur_step > 0:
                # 완드비 로그
                
                
                
                print('Step {}: Generator loss: {:.5f}, Discriminator loss: {:.5f}'.format(cur_step, mean_g_loss, mean_d_loss))
                show_tensor_images(lr_real * 2 - 1)
                show_tensor_images(hr_fake.to(hr_real.dtype))
                show_tensor_images(hr_real)
                
                show_lr_real = show_tensor_images_wandb(lr_real * 2 - 1)
                show_hr_fake = show_tensor_images_wandb(hr_fake.to(hr_real.dtype))
                show_hr_real = show_tensor_images_wandb(hr_real)
                
                
                test_table = wandb.Table(columns=columns)
                test_table.add_data(cur_step, wandb.Image(show_lr_real), wandb.Image(show_hr_fake), wandb.Image(show_hr_real))
                wandb.log({"srgan_predictions" : test_table})
                
                
                

                save_images(lr_real, os.path.join("./img_srg/lr_real", f"lr_real_{cur_step}.jpg"))
                save_images(hr_fake, os.path.join("./img_srg/hr_fake", f"hr_fake_{cur_step}.jpg"))
                save_images(hr_real, os.path.join("./img_srg/hr_real", f"hr_real_{cur_step}.jpg"))

                mean_g_loss = 0.0
                mean_d_loss = 0.0
                torch.save(generator, './model/generator.pt')
                torch.save(discriminator, './model/discriminator.pt')
                gpu_usage()

            cur_step += 1
            if cur_step == total_steps:
                break


if __name__ == "__main__":

  
 

 # ARGUMENTS PARSER
  p = argparse.ArgumentParser()
  
  p.add_argument("--project_name", type=str, default="srgan_project", help='(글자) 프로젝트 이름을 정해주세요!')
  
  p.add_argument("--epochs", type=float, default=2e5, help='(부동소수점) 훈련 횟수를 정해주세요')
  p.add_argument("--batch_size", type=int, default=16, help='(정수)배치 사이즈를 정해주세요')
  p.add_argument("--display_step", type=int, default=500, help='(정수) 디스플레이 스탭을 정해주세요')
  p.add_argument("--device", type=str, default="cuda", help='(글자)GPU 로 돌릴지 CPU로 돌릴지 정해주세요!')
  p.add_argument("--lr", type=float, default=1e-4, help='(부동소수점)러닝레이트를 넣어주세요')
  p.add_argument("--gen_path", type=str, default="./model/srresnet.pt", help='(글자)srresnet model 경로를 설정해주세요!')
                  
  args = p.parse_args()
  # wandb  login
  wandb.login(key='')
  run = wandb.init(
  # Set the project where this run will be logged
    project=args.project_name,
  # Track hyperparameters and run metadata
    config=args)
  

  train_srgan(args)
  
