import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
class VGG16_25(nn.Module):#该类并未使用
    def __init__(self):
        super(VGG16_25,self).__init__()
        vgg_model=models.vgg16(pretrained=True)
        self.vgg_model2=torch.nn.Sequential(*list(vgg_model.children())[0][:9])
        self.vgg_model5=torch.nn.Sequential(*list(vgg_model.children())[0][:30])
        for param in self.vgg_model2.parameters():
            param.requires_grad=False
        for param in self.vgg_model5.parameters():
            param.requires_grad=False
    def forward(self,x):
        x_p2=self.vgg_model2(x)
        x_p5=self.vgg_model5(x)
        return x_p2,x_p5
class LossNetwork(torch.nn.Module):
    def __init__(self):
        super(LossNetwork, self).__init__()
        vgg_model =models.vgg16(pretrained=True).features[:14]
        self.l1loss=nn.L1Loss()
        #vgg_model = vgg_model.to(opt.device)
        for param in vgg_model.parameters():
            param.requires_grad = False
        
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            #'3': "relu1_2",
            #'8': "relu2_2",
            #'15': "relu3_3"
           # '1': "relu1_2",
            #'3': "relu2_2",
            '5': "relu3_3",
            '9': "relu4_2",
            '13': "relu5_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            #print(name)
            x = module(x)
            #print(x.size())
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            #loss.append(F.mse_loss(dehaze_feature, gt_feature))
            loss.append(self.l1loss(dehaze_feature, gt_feature))

        #loss[0:5]=[1/32*loss[0],1/16*loss[1],1/8*loss[2],1/4*loss[3],loss[4]]
        #loss[0:4]=[1/16*loss[0],1/8*loss[1],1/4*loss[2],loss[3]]
        loss[0:3]=[1/8*loss[0],1/4*loss[1],loss[2]]
        #return sum(loss)/len(loss)
        return sum(loss)


