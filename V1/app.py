from config10 import *
import argparse
from utils_app import  get_image, device, get_paint

BATCH_SIZE = 1
epoch = 120000

torch.backends.cudnn.benchmark=True

parser = argparse.ArgumentParser()
parser.add_argument(
    '--lineart',
    type=str,
    required=True,
    help='lineart_dir'
)

parser.add_argument(
    '--ref_img',
    type=str,
    required=True,
    help='ref_img_dir'
)

FLAGS = parser.parse_args()

lineart_dir = FLAGS.lineart
ref_dir = FLAGS.ref_img

net = torch.load('p.net').to(device).eval()

for i in range(1, epoch+1):
    filenames = random.sample(file_list, BATCH_SIZE)
    inputs_batch, hint_batch = get_paint(lineart_dir, ref_dir)
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
   