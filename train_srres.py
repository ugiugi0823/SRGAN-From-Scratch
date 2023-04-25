#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def train_srresnet(srrestnet, dataloader, device, lr=1e-4, total_steps=1e6, display_step=500):
    srrestnet = srrestnet.to(device).train()
    optimizer = th.optim.Adam(srrestnet.parameters(), lr=lr)
    
    cur_step = 0
    mean_loss = 0.0
    while cur_step < total_steps:
        for hr_real, lr_real in tqdm(dataloader, position=0):
            hr_real = hr_real.to(device)
            lr_real = lr_real.to(device)
            
            # Enable autocast to EP16 tensors (new feature since torch==1.6.0)
            # If you're running older versions of torch, comment this out
            # and use NVIDIA apex for mixed/half precision training
            if has_autocast:
                with th.cuda.amp.autocast(enabled=(device=='cuda')):
                    hr_fake = srrestnet(lr_real)
                    loss = Loss.img_loss(hr_real, hr_fake)
                    
            else:
                hr_fake = srrestnet(lr_real)
                loss = Loss.img_loss(hr_real, hr_fake)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            mean_loss += loss.item() / display_step
            
            
            if cur_step % display_step == 0 and cur_step > 0:
                print('Step {}: SRResNet loss: {:.5f}'.format(cur_step, mean_loss))
                show_tensor_images(lr_real * 2 - 1)
                show_tensor_images(hr_fake.to(hr_real.dtype))
                show_tensor_images(hr_real)
                mean_loss = 0.0
                
            cur_step += 1
            if cur_step == total_steps:
                break
                
                

                
                
device = 'cuda' if th.cuda.is_available() else 'cpu'
generator = Generator(n_res_blocks=16, n_ps_blocks=2)

# Uncomment the following lines if you're using ImageNet
# dataloader = torch.utils.data.DataLoader(
#     Dataset('data', 'train', download=True, hr_size=[384, 384], lr_size=[96, 96]),
#     batch_size=16, pin_memory=True, shuffle=True,
# )
# train_srresnet(generator, dataloader, device, lr=1e-4, total_steps=1e6, display_step=500)
# torch.save(generator, 'srresnet.pt')

# Uncomment the following lines if you're using STL


dataloader = th.utils.data.DataLoader(
    Dataset('data', 'train', download=True, hr_size=[96, 96], lr_size=[24, 24]),
    batch_size=16, pin_memory=True, shuffle=True,
)
train_srresnet(generator, dataloader, device, lr=1e-4, total_steps=1e5, display_step=1000)
th.save(generator, 'srresnet.pt')

                

