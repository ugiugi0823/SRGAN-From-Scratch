import torch
from tqdm import tqdm
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import numpy as np

# Parse torch version for autocast
# ######################################################
version = torch.__version__
version = tuple(int(n) for n in version.split('.')[:-1])
has_autocast = version >= (1, 6)
# ######################################################

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
    print('이제 사진을 저장합니다~~~~!!ㅏㅇ너리ㅏㄴ어링너ㅏㅣㄹㅇㄴ')
    # (0,1,2,3)
    images = (images + 1) / 2
    images = images[:1]
    ndarr = images.squeeze()
    ndarr = images.to('cpu').numpy()
    
    # im = Image.fromarray((ndarr * 255).astype(np.uint8))
    im = Image.fromarray(ndarr)
    im.save(path)
