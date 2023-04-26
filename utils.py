import torch
from tqdm import tqdm
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import numpy as np
from torchvision.utils import save_image
from PIL import Image
from torch.utils.data import DataLoader
from datasets import Dataset

# Parse torch version for autocast
# ######################################################
version = torch.__version__
version = tuple(int(n) for n in version.split('.')[:-1])
has_autocast = version >= (1, 6)
# ######################################################

def get_data(args):
  dataloader = torch.utils.data.DataLoader(
    Dataset('data', 'train', download=True, hr_size=[96, 96], lr_size=[24, 24]),
    batch_size=args.batch_size, pin_memory=True, shuffle=True,
  )  
  return dataloader



def show_tensor_images(image_tensor):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:4], nrow=4)
    plt.axis('off')
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()





def save_images(images, path, **kwargs):
    save_image(images[:1], path)





def show_images(path, **kwargs):
  image = Image.open(path)
  np_array = np.array(image)

  np_array.shape
  pil_image=Image.fromarray(np_array)
  pil_image.show()
  
    
