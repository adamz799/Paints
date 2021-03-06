from config10 import *
from utils_app import  get_image, get_tensors, device, delta_E1994,get_paint

BATCH_SIZE = 1
epoch = 120000

torch.backends.cudnn.benchmark=True

net = torch.load('v10_3_vgg_b.net').to(device).eval()

for i in range(1, epoch+1):
    filenames = random.sample(file_list, BATCH_SIZE)
    inputs_batch, hint_batch = get_paint(filenames)
    outputs = net(inputs_batch, hint_batch, ITERATION)
    
    hint_pos = hint_batch[0,0:1]
    hint_pos = torch.cat((hint_pos, hint_pos, hint_pos),0).cpu().detach().numpy()
    hint_pos = np.transpose(hint_pos,(1,2,0))
    hint_pos = np.where(hint_pos==-256, False, True)

    sketch = get_image('sketch',inputs_batch)      
    hint = get_image('hint', hint_batch)
       
    inputs = np.where(hint_pos, hint, sketch)
    r = get_image('result', outputs)
    #get_image('result1', outputs1)
    result = np.concatenate( [inputs,r],1)
    cv2.imshow('Result', result)
    cv2.imshow('hint',hint)
   