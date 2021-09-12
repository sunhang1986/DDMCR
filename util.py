import numpy as np
import torch
import random
import math
import torch.autograd as autograd

def padding_image(image, h, w):
    assert h >= image.size(2)
    assert w >= image.size(3)
    padding_top = (h - image.size(2)) // 2
    padding_down = h - image.size(2) - padding_top
    padding_left = (w - image.size(3)) // 2
    padding_right = w - image.size(3) - padding_left
    out = torch.nn.functional.pad(image, (padding_left, padding_right, padding_top,padding_down), mode='reflect')
    return out, padding_left, padding_left + image.size(3), padding_top,padding_top + image.size(2)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            m.bias.data.zero_()
    if isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def compute_gradient_penalty(D, real_samples, fake_samples):
    """
    Calculates the gradient penalty loss for WGAN GP
    :param D:
    :param real_sanmples:
    :param fake_samples:
    :return:
    """
    # Random weight term for interpolation between real fake samples
    alpha = torch.cuda.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).reauires_grad_(True)
    validity = D(interpolates)
    fake = autograd.Variable(torch.cuda.FloatTensor(np.ones(validity.shape)), requires_grad=False)
    # Get gradient w.r.t interpolates
    gradients = autograd.grad(outputs=validity, inputs=interpolates,
                              grad_outputs=fake, create_graph=True, retain_graph=True,
                              only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    # gradients_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    gradients_penalty = (torch.clamp(gradients.norm(2, dim=1) - 1., min=0.)).mean()
    return gradients_penalty

class ImagePool():
    """
    This class implements an image buffer that stores previously generated images.
    This buffer enable us to updatae discriminateors using a history of generated images
    rather than the ones produced by the latest generators.
    """
    def __init__(self, pool_size=50):
        """
        Initialize the ImagePool class
        Parameters:
            pool_size(int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0: # create an empty pool
            self.num_imgs = 0
            self.images =[]
    def query(self, images):
        """
         Return an image from the pool
         Parameters:
             images: the latest generated images from the generator

         Returns images from the buffer
             By 50/100, the buffer will return input images.
             By 50/100, the buffer will return images previously started in the buffer and insert the current images to the buffer.
        """
        if self.pool_size == 0: #if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size: # if the buffer is not full, keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5: # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1) #randint is inclusive
                    tmp =self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp) ###
                else: #by anthor 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0) # collect all the images and return
        return return_images

def spatial_pyramid_pool(self, previous_conv, num_sample, previous_conv_size, out_pool_size):
    """
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer

    returns: a tensor vector with shape [1*n] is the concentration of multi-level pooling
    """
    for i in range(len(out_pool_size)):
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = (h_wid*out_pool_size[i] - previous_conv_size[0] +1)/2
        w_pad = (w_wid * out_pool_size[i] - previous_conv_size[1] + 1) / 2
        maxpool = torch.nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if i == 0:
            spp = x.view(num_sample, -1)
        else:
            spp = torch.cat((spp, x.view(num_sample, -1)), 1)
    return spp















