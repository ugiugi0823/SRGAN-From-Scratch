import os
import torch
import wandb
import argparse

from tqdm import tqdm


# My file
from loss import Loss
from utils import show_tensor_images, save_images, get_data, show_tensor_images_wandb
from modules import Generator

from GPUtil import showUtilization as gpu_usage

from torch._inductor import config
config.compile_threads = 1

# Parse torch version for autocast
# ######################################################
version = torch.__version__
version = tuple(int(n) for n in version.split('.')[:-1])
has_autocast = version >= (1, 6)
# ######################################################

def train_srresnet(args):
    device = args.device
    total_steps = args.epochs
    display_step = args.display_step
    wandb_step = args.wandb_step

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    srresnet = Generator(n_res_blocks=16, n_ps_blocks=2)
    dataloader = get_data(args)

    srresnet = srresnet.to(device).train()
    optimizer = torch.optim.Adam(srresnet.parameters(), lr=args.lr)

    cur_step = 0
    mean_loss = 0.0
    
    # show image in wandb
    columns=["id", "lr_real", "hr_fake", "hr_real"]
    test_table = wandb.Table(columns=columns)
    test_table2 = wandb.Table(columns=columns)
    
    while cur_step < total_steps:
        for hr_real, lr_real in tqdm(dataloader, position=0):
            hr_real = hr_real.to(device)
            lr_real = lr_real.to(device)

            # Enable autocast to FP16 tensors (new feature since torch==1.6.0)
            # If you're running older versions of torch, comment this out
            # and use NVIDIA apex for mixed/half precision training
            if has_autocast:
                with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                    hr_fake = srresnet(lr_real)
                    loss = Loss.img_loss(hr_real, hr_fake)
            else:
                hr_fake = srresnet(lr_real)
                loss = Loss.img_loss(hr_real, hr_fake)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mean_loss += loss.item() / display_step
            
            wandb.log({"SRResnet_Loss": mean_loss})

            if cur_step % display_step == 0 and cur_step > 0:
                print('Step {}: SRResNet loss: {:.5f}'.format(cur_step, mean_loss))
                # print('lr_real', lr_real.detach().cpu().numpy().shape)
                # print('hr_fakr', hr_fake.detach().cpu().numpy().shape)
                # print('hr_real', hr_real.detach().cpu().numpy().shape)
                show_tensor_images(lr_real * 2 - 1)
                show_tensor_images(hr_fake.to(hr_real.dtype))
                show_tensor_images(hr_real)  


                show_lr_real = show_tensor_images_wandb(lr_real * 2 - 1)
                show_hr_fake = show_tensor_images_wandb(hr_fake.to(hr_real.dtype))
                show_hr_real = show_tensor_images_wandb(hr_real)
                test_table.add_data(cur_step, wandb.Image(show_lr_real), wandb.Image(show_hr_fake), wandb.Image(show_hr_real))              




                test_table2.add_data(cur_step, wandb.Image(show_lr_real), wandb.Image(show_hr_fake), wandb.Image(show_hr_real))              
                wandb.log({"SRResnet_predictions2" : test_table2})



                mean_loss = 0.0
                save_images(lr_real, os.path.join("./img_srr/lr_real", f"lr_real_{cur_step}.jpg"))
                save_images(hr_fake, os.path.join("./img_srr/hr_fake", f"hr_fake_{cur_step}.jpg"))
                save_images(hr_real, os.path.join("./img_srr/hr_real", f"hr_real_{cur_step}.jpg"))
                torch.save(srresnet, './model/srresnet.pt')
                gpu_usage()

            elif cur_step % wandb_step == 0 and cur_step > 0:
                print('wandb 5000번째 로그')
              
                wandb.log({"SRResnet_predictions" : test_table})
              

            cur_step += 1
            if cur_step == total_steps:
                break


if __name__ == "__main__":


 # ARGUMENTS PARSER
  p = argparse.ArgumentParser()
  
  p.add_argument("--project_name", type=str, default="srresnet_project", help='(글자) 프로젝트 이름을 정해주세요!')
  p.add_argument("--run_name", type=str, default="Mickey_srresnet_bs-16_lr-1e-4", help='(글자) 프로젝트 이름을 정해주세요!')
  
  p.add_argument("--epochs", type=float, default=1e5, help='(부동소수점) 훈련 횟수를 정해주세요')
  p.add_argument("--batch_size", type=int, default=16, help='(정수)배치 사이즈를 정해주세요')
  p.add_argument("--display_step", type=int, default=500, help='(정수) 디스플레이 스탭을 정해주세요')
  p.add_argument("--wandb_step", type=int, default=5000, help='(정수) 완드비 스탭을 정해주세요')
  p.add_argument("--device", type=str, default="cuda", help='(글자)GPU 로 돌릴지 CPU로 돌릴지 정해주세요!')
  p.add_argument("--lr", type=float, default=1e-4, help='(부동소수점)러닝레이트를 넣어주세요')
                  
  args = p.parse_args()
  
  # wandb  login
  wandb.login(key='')
  run = wandb.init(
  # Set the project where this run will be logged
    project=args.project_name,
    name = args.run_name,
  # Track hyperparameters and run metadata
    config=args)
  

  train_srresnet(args)
  
      
