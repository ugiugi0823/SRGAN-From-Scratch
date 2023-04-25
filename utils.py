#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tqdm import tqdm
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


# Parse torch verison for autocast
# #######################################################
version = th.__version__
version = tuple(int(n)) for n in version.split('.')[:-1]
has_autocast = version >= (1, 6)
# #######################################################

def show_tensor_images(image_tensor):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    """
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:4], nrow=4)
    plt.axis('off')
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

