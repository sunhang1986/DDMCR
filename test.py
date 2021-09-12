from model import Generator_base, Generator, Discriminator
from VGG16_FE import VGG16_25,LossNetwork
from dataloader import TrainDataset, TestDataset
import torch
import adabound
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as tt
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm
import itertools
from PIL import Image
import cv2
import argparse, os
import shutil
import random
import time
import numpy as np
import math
from util import padding_image, weights_init_normal, weights_init, ImagePool
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts

parser = argparse.ArgumentParser(description = "VHDR")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--gpus", default="0,1,2,3", type=str, help="gpu ids (default: 0)")
parser.add_argument("--test", action="store_true", help="for test?")
parser.add_argument("--test_ori", action="store_true", help="just output one image?")
parser.add_argument("--batch_size", default=16, type=int, help="train batch size")
parser.add_argument("--optim", default="adam", type=str, help="optim")
parser.add_argument("--epoch", default=0, type=int, help="start train epochs")
parser.add_argument("--n_epochs", default=400, type=int, help="total training epochs nums")
parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
#parser.add_argument("--idt_weight", default=1, type=float, help="the identity mapping loss weight Default=0.001")
parser.add_argument("--idt_weight", default=1, type=float, help="the identity mapping loss weight Default=0.001")
#parser.add_argument("--source_weight", default=1, type=float, help="the identity mapping loss weight Default=0.001")
parser.add_argument("--source_weight", default=1, type=float, help="the identity mapping loss weight Default=0.001")
#parser.add_argument("--cycle_weight", default=20, type=float, help="the cycle consistency  loss weight Default=0.0001")
parser.add_argument("--cycle_weight", default=10, type=float, help="the cycle consistency  loss weight Default=0.0001")
parser.add_argument("--inputset", default="RESIDE_ITS_haze", type=str, help="Training input dataset")
parser.add_argument("--labelset", default="RESIDE_ITS_clear", type=str, help="Training label dataset")
#parser.add_argument("--g_lr", default=0.0001, type=float, help="Generator Learning Rate. Default=0.0001")
parser.add_argument("--g_lr", default=0.001, type=float, help="Generator Learning Rate. Default=0.0001")
parser.add_argument("--g_final_lr", default=0.1, type=float, help="Generator Final Learning Rate for Adabound. Default=0.1")
#parser.add_argument("--d_lr", default=0.0004, type=float, help="Discriminator Learning Rate. Default=0.0004")
parser.add_argument("--d_lr", default=0.004, type=float, help="Discriminator Learning Rate. Default=0.0004")
parser.add_argument("--d_final_lr", default=0.1, type=float, help="Discriminator Final Learning Rate for Adabound. Default=0.1")
parser.add_argument("--gamma", default=1e-3, type=float, help="Adam b1 Default=0.1")
parser.add_argument("--wd", default=5e-4, type=float, help="weight_decay Default=5e-4")
parser.add_argument("--b1", default=0.9, type=float, help="Adam b1 Default=0.1")
parser.add_argument("--b2", default=0.999, type=float, help="Adam b2 Default=0.1")
parser.add_argument("--test_path", default="test", type=str, help="Test path")
parser.add_argument("--Gx_model_path", default="checkpoint_new_pcycle_pvgg_27.17/OTS_haze_256_OTS_clear_256/epoch_252_Gx.pth", type=str, help="Gx_model path")
parser.add_argument("--Gx1_model_path", default="checkpoint/five_dataset_512_HDR-512/epoch_20_Gx1.path", type=str, help="Gx1_model path")
parser.add_argument("--Gy_model_path", default="checkpoint/five_dataset_512_HDR-512/epoch_20_Gy.path", type=str, help="Gy_model path")
parser.add_argument("--Gy1_model_path", default="checkpoint/five_dataset_512_HDR-512/epoch_20_Gy1.path", type=str, help="Gy1_model path")

opt = parser.parse_args()
print(opt)

if opt.cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
        raise Exception("No GPU found or wrong gpu id, please run without --cuda")

opt.seed = random.randint(1, 10000)
torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

cudnn.benchmark = True

if not opt.test:
    train_data = TrainDataset(opt.inputset, opt.labelset)
    # test_data = TestDataset(opt.testset)
    train_loader = DataLoader(dataset=train_data, num_workers=2, pin_memory=True, batch_size=opt.batch_size, shuffle=True)
    #test_loader = DataLoader(dataset=test_data, num_workers=1, batch_size=1, shuffle=False)

l1_loss = torch.nn.L1Loss()
mse_loss = torch.nn.MSELoss()

Gx_base = Generator_base()
# print(Gx_base.conv_down1.weight_data)
Gx = Generator(Gx_base)
# print(Gx_base.conv_down1.weight_data)
Gx1 = Generator(Gx_base)
Gy_base = Generator_base()
Gy = Generator(Gy_base)
Gy1 = Generator(Gy_base)

Dx = Discriminator()
Dy = Discriminator()
Dx_P = Discriminator(True,32)
Dy_P = Discriminator(True,32)

if opt.cuda:
    Gx_base.cuda()
    Gy_base.cuda()
    Gx.cuda()
    Gx1.cuda()
    Gy.cuda()
    Gy1.cuda()
    Dx.cuda()
    Dy.cuda()
    Dx_P.cuda()
    Dy_P.cuda()
    l1_loss.cuda()
    mse_loss.cuda()

def test(opt, epoch=0, load_model=True):
    #print(opt.Gx_model_path)
    #exit()
    if load_model:
        if not opt.test_ori:
            Gx.load_state_dict(torch.load(opt.Gx_model_path))
            Gy.load_state_dict(torch.load(opt.Gy_model_path))
            Gy1.load_state_dict(torch.load(opt.Gy1_model_path))
        #Gy1.load_state_dict(torch.load(opt.Gy1_model_path))
        Gx1.load_state_dict(torch.load(opt.Gx1_model_path))
        #Gx.load_state_dict(torch.load(opt.Gx_model_path))

    if opt.test:
        result_path = "test_out"
    else:
        result_path = "result_img/%s_%s" % (opt.inputset, opt.labelset)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    for img_name in tqdm(os.listdir(opt.test_path)):
        name, _ =os.path.splitext(img_name)
        img = Image.open(os.path.join(opt.test_path, img_name)).convert("RGB")
        w, h = img.size
        with torch.no_grad():
            Gx.eval()
            Gx1.eval()
            Gy.eval()
            Gy1.eval()

            max_h = int(math.ceil(h/16))*16
            max_w = int(math.ceil(w/16))*16
            input_tensor = torch.unsqueeze(tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(tt.ToTensor()(img)), 0)
            input_tensor_pad, ori_left, ori_right, ori_top, ori_down = padding_image(input_tensor, max_h, max_w)
            input = Variable(input_tensor_pad, requires_grad=False)

            if opt.cuda:
                input = input.cuda()

            if not opt.test_ori:
                out = Gx(input)
                out_inv = Gy1(out)
                out1 = Gy(input)
                out1_inv = Gx1(out1)
                out1_out = Gx1(input)

                out = out.data[:, :, ori_top:ori_down, ori_left:ori_right]
                out_inv = out_inv.data[:, :, ori_top:ori_down, ori_left:ori_right]
                out1 = out1.data[:, :, ori_top:ori_down, ori_left:ori_right]
                out1_inv = out1_inv.data[:, :, ori_top:ori_down, ori_left:ori_right]
                out1_out = out1_out.data[:, :, ori_top:ori_down, ori_left:ori_right]
                img_sample = torch.cat((out.cpu(), out_inv.cpu(), input_tensor, out1.cpu(), out1_inv.cpu(),out1_out.cpu()), 0)
                save_image(img_sample, "%s/%s_out_%d.png" % (result_path, name, epoch), nrow=3, normalize=True)
            else:
                start = time.time()
                out = Gx1(input)
                #out = Gx(input)
                out = out.data[:, :, ori_top:ori_down, ori_left:ori_right]
                print("time: ", time.time() - start)
                save_image(out, "%s/%s_out_%d.png" % (result_path, name, epoch), nrow=1, normalize=True)

if opt.test:
    test(opt, epoch=300)
    exit()
else:
    train(opt)
    exit()





