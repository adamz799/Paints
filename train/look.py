from config import *
import torchvision
from utils10 import get_tensors, get_image
from tensorboardX import SummaryWriter

writer = SummaryWriter()
net = torch.load('v10_3_vgg.net').to(device).eval()

for f in file_list:
    i=0
    inputs_batch, hint_batch, label_batch = get_tensors([f])
    outputs,L = net(inputs_batch, hint_batch, ITERATION)

    for layer in L:         
        layer = layer.transpose(0, 1)# C，B, H, W  ---> B，C, H, W
        img_grid = torchvision.utils.make_grid(layer, normalize=True, scale_each=True, nrow=16)  
        writer.add_image(str(i), img_grid, global_step=0)
        print(layer.shape)
        i+=1
    writer.close()
    a=1