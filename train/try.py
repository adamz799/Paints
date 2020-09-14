from config10 import *
from utils10 import  get_image, get_tensors, device, color_distance, delta_E1994,gaussian_kernel
import matplotlib.pyplot as plt
from Unet10 import GaussianBlur
from Unet12 import InputNet
BATCH_SIZE = 1
epoch = 120000

torch.backends.cudnn.benchmark=True

blurer = GaussianBlur(gaussian_kernel()).to(device)

def get_dis(tensor1, tensor2):
    t = tensor1.cpu().detach().numpy()
    t = np.rint(t*2.55).astype(np.int32)
    i, m = np.unique(t, return_counts=True)
    t = tensor2.cpu().detach().numpy()
    t = np.clip(t,0,100)
    t = np.rint(t*2.55).astype(np.int32)
    i1, m1 = np.unique(t, return_counts=True)
    plt.plot(i,m, 'b-', i1, m1, 'g-')
    plt.xlabel('Light')
    plt.ylabel('Amount')
    #plt.title('Histogram of IQ')
    #plt.text(60,.025, r'$\mu=100,\ \sigma=15$')
    # plt.axis([-10,300,0,10000])
    # plt.grid(True)
    plt.show()
    #print(i,m)

def get_dis1(ratio, hint_point, tensor1, tensor2):#, tensor3):
    t = tensor1.cpu().detach().numpy()
    t = (t*ratio).astype(np.int32)
    i, m = np.unique(t, return_counts=True)
    t = tensor2.cpu().detach().numpy()
    #t = np.clip(t,0,100)
    t = (t*ratio).astype(np.int32)
    i1, m1 = np.unique(t, return_counts=True)
    # t = tensor3.cpu().detach().numpy()
    # #t = np.clip(t,0,100)
    # t = (t*ratio).astype(np.int32)
    # i2, m2 = np.unique(t, return_counts=True)
    plt.plot(i,m, 'r^', i1, m1, 'b^')#, i2, m2, 'b^')
    plt.xlabel('a')
    plt.ylabel('Amount')
    #plt.title('Histogram of IQ')
    #plt.text(60,.025, r'$\mu=100,\ \sigma=15$')
    # plt.axis([-10,300,0,10000])
    # plt.grid(True)
    
    t = hint_point.cpu().detach().numpy()
    t = (t*ratio).astype(np.int32)
    i, m = np.unique(t, return_counts=True)
    #plt.vlines(i, 0, 50000)
    plt.plot(i,m, 'y+')
    plt.show()
    #print(i,m)

net1 = torch.load('v10_3_vgg.net').to(device).eval()
net = torch.load('v10_3_vgg_b.net').to(device).eval()

# base_net_dict = net.state_dict()
# self_dict = net1.state_dict()
# base_net_dict = {name:weight for name,weight in base_net_dict.items() if name in self_dict}
# self_dict.update(base_net_dict)
# net1.load_state_dict(self_dict)
# torch.save(net1, 'net_test.net')


#net_t = torch.load('v10_3_2_t.net').to(device).eval()
#net1 = torch.load('../v9/inputnet_v9.pkl').to(device).eval()
#net1 = torch.load('v10_3_3.net').to(device).eval()
def copy_parameter(IN, BN):
    IN.bias = BN.bias
    IN.weight = BN.weight
    # IN.running_mean = BN.running_mean
    # IN.running_var = BN.running_var

def copy_parameter_d(IN, BN):
    copy_parameter(IN.fuse[2],BN.fuse[2])
    copy_parameter(IN.unsqueeze1[2],BN.unsqueeze1[2])

def copy_parameter_g(IN, BN):
    copy_parameter(IN.net[2],BN.net[2])
    copy_parameter(IN.net[5],BN.net[5])
    copy_parameter(IN.net[8],BN.net[8])

def copy_parameter_p(IN, BN):
    copy_parameter(IN.f[0],BN.f[0])
    copy_parameter(IN.f[3],BN.f[3])
    
for i in range(1, epoch+1):
    filenames = random.sample(file_list, BATCH_SIZE)
    inputs_batch, hint_batch, label_batch = get_tensors(filenames, data_augm=False)
    outputs = net(inputs_batch, hint_batch, ITERATION)
    outputs1 = net1(inputs_batch, hint_batch, ITERATION)
    # outputs = blurer(outputs)
    get_image('result111', outputs1)
    # outputs = net(inputs_batch, label_batch, ITERATION)

    hint_pos = hint_batch[0,0:1]
    hint_pos = torch.cat((hint_pos, hint_pos, hint_pos),0).cpu().detach().numpy()
    hint_pos = np.transpose(hint_pos,(1,2,0))
    hint_pos = np.where(hint_pos==-256, False, True)

    sketch = get_image('sketch',inputs_batch)      
    hint = get_image('hint', hint_batch)
    
    
    inputs = np.where(hint_pos, hint, sketch)
    r = get_image('result', outputs)
    #get_image('result1', outputs1)
    cv2.imshow('r', np.concatenate( [inputs,r],0))
    cv2.imshow('hint',hint)
    cv2.imshow('sketch',sketch) 
    get_image('label', label_batch)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('l', img)
    # for i in range(4):
    #     get_image(filenames[0]+' {}'.format(i), outputs1[i])
    #get_image('result', outputs)
    #get_image('result1', outputs1)

    #label_batch = tensor_YUV2RGB(label_batch)
    # o = tensor_YUV2RGB(outputs)
    # o1 = tensor_YUV2RGB(outputs1[0])

    # loss = color_distance(outputs, label_batch)
    # loss1 = color_distance(o1, label_batch)

    #loss = loss_func(outputs, label_batch)
    #loss1 = loss_func(outputs1, label_batch)

    #print('epoch {0}, loss {1:.6f}, loss1 {2:.6f}'.format(i,loss.item(),loss1.item()))
    #print(loss.item())
    #cv2.waitKey(0)
    channel =1
    get_dis1(8, hint_batch[0,channel], label_batch[0,channel],outputs[0,channel])#, outputs1[0,channel])
    #for i in range(4):
    #cv2.destroyAllWindows()
    #get_image(filenames[0]+' {}'.format(i), outputs1[i])