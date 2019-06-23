from config10 import *
import torch.distributions.normal as N


def tensor_YUV2RGB(YUV_tensor):
    #[BATCH_SIZE, CHANNEL, H, W]
    #To compatible with OpenCV's convertion
    Y = YUV_tensor[:,0,None]
    U = YUV_tensor[:,1,None]-128
    V = YUV_tensor[:,2,None]-128
    R = Y+1.14*V
    G = Y-0.395*U-0.581*V
    B = Y + 2.032*U+0.001*V
    rgb_img = torch.cat([R,G,B], 1)#.clamp(0, 255)#pytorch issue 7002
    # zeros = torch.zeros(rgb_img.shape).to(device)
    # rgb_img = torch.where(rgb_img<0, zeros, rgb_img)
    #rgb_img_c = torch
    
    return rgb_img

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225] 

M=[[0.4124,0.3576,0.1805],
[0.2126,0.7152,0.0722],
[0.0193,0.1192,0.9505]]

Xn, Yn, Zn = 0.950456, 1.000, 1.088754

delta = 6/29
tt = delta**3
k = 1/(3*delta**2)
b = 4/29

_tt = delta
_k = 1/k

def f(t):   
    if isinstance(t,torch.Tensor):
        return torch.where(t>tt,t**(1/3),k*t+b)
    elif isinstance(t,np.ndarray):
        return np.where(t>tt,t**(1/3),k*t+b)
    else:
        print('Unknown tensor type')
    return 
    #return t**(1/3) if t>0.00885645 else 7.78703703*t+4/29

def f_reverse(t):
    if isinstance(t,torch.Tensor):
        return torch.where(t>_tt,t**3,_k*(t-b))
    elif isinstance(t,np.ndarray):
        return np.where(t>_tt,t**3,_k*(t-b))
    else:
        print('Unknown tensor type')
    return 
    #return t**3 if t>0.20689655 else 0.02656935503*(t-4/29)

def BGR2Lab(BGR_tensor):
    BGR_tensor=BGR_tensor/255.0
    B = BGR_tensor[:,:,0:1]
    G = BGR_tensor[:,:,1:2]
    R = BGR_tensor[:,:,2:3]
    X = 0.412453*R + 0.357580*G + 0.180423*B
    Y = 0.212671*R + 0.715160*G + 0.072169*B
    Z = 0.019334*R + 0.119193*G + 0.950227*B
    fy = f(Y/Yn)
    #0≤L≤100  , −127≤a≤127 , −127≤b≤127
    L = 116*fy-16 
    a = 500*(f(X/Xn)-fy) 
    b = 200*(fy-f(Z/Zn))
    return np.concatenate([L,a,b],axis= 2)

def tensor_BGR2Lab(BGR_tensor):
    BGR_tensor=BGR_tensor/255.0
    B = BGR_tensor[:,0:1]
    G = BGR_tensor[:,1:2]
    R = BGR_tensor[:,2:3]
    #sRGB D65
    X = 0.412453*R + 0.357580*G + 0.180423*B
    Y = 0.212671*R + 0.715160*G + 0.072169*B
    Z = 0.019334*R + 0.119193*G + 0.950227*B
    fy = f(Y/Yn)
    #0≤L≤100  , −127≤a≤127 , −127≤b≤127
    #from RGB color space to CIE Lab with D65 
    # >>> lab_max
    # array([[[100.     ,  98.23516,  94.47579]]], dtype=float32)
    # >>> lab_min
    # array([[[   0.      ,  -86.18125 , -107.861755]]], dtype=float32)
    L = 116*fy-16 
    a = 500*(f(X/Xn)-fy) 
    b = 200*(fy-f(Z/Zn))
    if isinstance(BGR_tensor,torch.Tensor):
        return torch.cat([L,a,b],1)
    elif isinstance(BGR_tensor,np.ndarray):
        return np.concatenate([L,a,b],axis= 1)
    else:
        print('Unknown tensor type')
    return 

def tensor_Lab2BGR(Lab_tensor):
    L = Lab_tensor[:,0:1]
    a = Lab_tensor[:,1:2]
    b = Lab_tensor[:,2:3]
    L_ = (L+16)/116
    X = Xn*f_reverse(L_+a/500)
    Y = Yn*f_reverse(L_)
    Z = Zn*f_reverse(L_-b/200)
    R =  3.240479*X - 1.537150*Y - 0.498535*Z
    G = -0.969256*X + 1.875992*Y + 0.041556*Z
    B =  0.055648*X - 0.204043*Y + 1.057311*Z
    if isinstance(Lab_tensor,torch.Tensor):
        return torch.cat([B,G,R],1)*255.0
    elif isinstance(Lab_tensor,np.ndarray):
        return np.concatenate([B,G,R],axis= 1)*255.0
    else:
        print('Unknown tensor type')
    return 

def tensor_Lab2RGB(Lab_tensor):
    L = Lab_tensor[:,0:1]
    a = Lab_tensor[:,1:2]
    b = Lab_tensor[:,2:3]
    L_ = (L+16)/116
    X = Xn*f_reverse(L_+a/500)
    Y = Yn*f_reverse(L_)
    Z = Zn*f_reverse(L_-b/200)
    R =  3.240479*X - 1.537150*Y - 0.498535*Z
    G = -0.969256*X + 1.875992*Y + 0.041556*Z
    B =  0.055648*X - 0.204043*Y + 1.057311*Z
    if isinstance(Lab_tensor,torch.Tensor):
        return torch.cat([R,G,B],1)*255.0
    elif isinstance(Lab_tensor,np.ndarray):
        return np.concatenate([R,G,B],axis= 1)*255.0
    else:
        print('Unknown tensor type')
    return 

def tensor_Lab2torch_rgb(Lab_tensor):
    L = Lab_tensor[:,0:1]
    a = Lab_tensor[:,1:2]
    b = Lab_tensor[:,2:3]
    L_ = (L+16)/116
    X = Xn*f_reverse(L_+a/500)
    Y = Yn*f_reverse(L_)
    Z = Zn*f_reverse(L_-b/200)
    R = (( 3.240479*X - 1.537150*Y - 0.498535*Z)-mean[0])/std[0]
    G = ((-0.969256*X + 1.875992*Y + 0.041556*Z)-mean[1])/std[1]
    B = (( 0.055648*X - 0.204043*Y + 1.057311*Z)-mean[2])/std[2]

    return torch.cat([R,G,B],1)
    
    return 

def delta_E1994(Lab1, Lab2):
    C1 = torch.sqrt(Lab1[:,1]**2+Lab1[:,2]**2)
    C2 = torch.sqrt(Lab2[:,1]**2+Lab2[:,2]**2)
    delta_C = C1-C2
    delta_2 = (Lab1-Lab2)**2
    delta_H2 = delta_2[:,1]+delta_2[:,2]-delta_C**2
    zeros = torch.zeros(delta_H2.shape).to(device)
    delta_H2 = torch.where(delta_H2<0, zeros, delta_H2)

    K_L = 1
    K_C = 1
    K_H = 1
    K1 = 0.045
    K2 = 0.015
    S_L = 1
    S_C = 1 + K1*C1
    S_H = 1 + K2*C1
    delta_E = torch.sqrt(delta_2[:,0]+(delta_C/S_C)**2+delta_H2/(S_H**2))
    return delta_E.mean()


def gaussian_kernel(kernel_size = 11,sigma = 4):
    
    center = kernel_size//2
    
    if sigma == 0:
        sigma = ((kernel_size-1)*0.5 - 1)*0.3 + 0.8
    
    s = 2*(sigma**2)
    kernel = np.fromfunction(lambda i,j: (i-center)**2+(j-center)**2, (kernel_size,kernel_size))
    kernel = np.exp(-kernel / s)
    kernel /= kernel.sum()  
    return kernel.astype(np.float32)

def color_distance(rgb1, rgb2):#RGB
    zeros = torch.zeros(rgb1.shape).to(device)
    rgb1_t = torch.where(rgb1<0, zeros, rgb1)

    up_bound = torch.zeros(rgb1.shape).to(device)+255
    rgb1_t = torch.where(rgb1>255, up_bound, rgb1_t)
    
    r_mean = (rgb1_t[:,0]+rgb2[:,0])/2.0
    r_square = (rgb1-rgb2)**2
    c = (2+r_mean/256)*r_square[:,0]+4*r_square[:,1]+(2+(255-r_mean)/256)*r_square[:,2]
    c = torch.sqrt(c)
    return c.mean()

def np_color_distance(rgb1,rgb2):#numpy
    r_mean = (rgb1[:,:,0]+rgb2[:,:,0])/2.0
    r_square = (rgb1-rgb2)**2
    c = (2+r_mean/256)*r_square[:,:,0]+4*r_square[:,:,1]+(2+(255-r_mean)/256)*r_square[:,:,2]
    c = np.sqrt(c)
    return c

grand_label = np.zeros([256,256,3]).astype(np.float32)+127

def gen_seg(block_img):   
    dis = np.rint(np_color_distance(grand_label, block_img)/8).astype(np.int32)
    uniq, counts = np.unique(dis, return_counts=True)
    for i in range(uniq.shape[0]):
        if counts[i]<10:
            k = i+1
            if k==uniq.shape[0]:
                dis = np.where(dis==uniq[i], uniq[i-1], dis)
            else:
                dis = np.where(dis==uniq[i], uniq[k], dis)
    uniq = np.unique(dis)
    return uniq, dis


# def tensor_color_distance(rgb1,rgb2):#tensor
#     r_mean = (rgb1[0]+rgb2[0])/2.0
#     r_square = (rgb1-rgb2)**2
#     c = (2+r_mean/256)*r_square[0]+4*r_square[1]+(2+(255-r_mean)/256)*r_square[2]
#     c = np.sqrt(c)
#     return c

# grand_label_tensor = torch.zeros([3,256,256])+127 

# def tensor_gen_seg(block_img):   
#     dis = torch.round(tensor_color_distance(grand_label_tensor, block_img)/8).type(torch.int32)
#     uniq = torch.unique(dis, sorted=True)
#     for i in range(uniq.shape[0]):
#         e = torch.where(dis==uniq[i],1,0)
#         if e.sum()<10:
#             k = i+1
#             if k==uniq.shape[0]:
#                 dis = np.where(e, uniq[i-1], dis)
#             else:
#                 dis = np.where(e, uniq[k], dis)
#     uniq = np.unique(dis)
#     return uniq, dis


def get_image(name, tensor):
    if tensor.shape[1]==1:
        img = tensor[0,0].cpu().detach().numpy()[:,:,None].astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #cv2.imshow(name, img)
        return img
        

                #img = np.rint(img)
                # tensor = torch.clamp(tensor, -127,127)
                # tensor[:,0] = torch.clamp(tensor[:,0], 0, 100)

                #     >>> lab_max
                # array([[[100.     ,  98.23516,  94.47579]]], dtype=float32)
                # >>> lab_min
                # array([[[   0.      ,  -86.18125 , -107.861755]]], dtype=float32)
    tensor[:,0] = torch.clamp(tensor[:,0],0,100)
    tensor[:,1] = torch.clamp(tensor[:,1],-86.18125,98.23516)
    tensor[:,2] = torch.clamp(tensor[:,2],-107.861755,94.47579)
    tensor = tensor_Lab2BGR(tensor)[0]
    img = tensor.cpu().detach().numpy()
    img = np.clip(img,0,255).astype(np.uint8)

    # max_v, min_v = img.max(),img.min()
    # img = (img -min_v)/(max_v-min_v)*255
    img = np.transpose(img,(1,2,0))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    cv2.imshow(name,img)
    #cv2.waitKey(0)
    return img






#print('sample_amount: ',sample_amount)


def get_tensors(names, data_augm = True):

    inputs_batch = [None]*len(names)
    # seg_batch = [None]*len(names)
    # idx_batch = [None]*len(names)
    label_batch = [None]*len(names)
    hint_batch = [None]*len(names)
    

    for idx in range(0,len(names)):
        f = names[idx]
        
        # Read file
        lineart = cv2.imread(data_dir+f, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        color = cv2.imread(label_dir+f).astype(np.float32)
        color = BGR2Lab(color)

        max_width = 5
        patch_width = 42
        patch_num = int(256/patch_width)
        sample_amount = patch_num*patch_num
    
        if data_augm:
            t = int(random.random()*4)
            p_noise = random.random()

            # Randimly flip the image
            if t==0:
                lineart = lineart[::-1]
                #block = block[::-1]
                color = color[::-1]
            elif t == 1:
                lineart = lineart[:,::-1]
                #block = block[:,::-1]
                color = color[:,::-1]
            elif t == 2:
                lineart = lineart[::-1,::-1]
                #block = block[::-1,::-1]
                color = color[::-1,::-1]
            elif t == 3:
                lineart = np.rot90(lineart, 1)
                #block = np.rot90(block, 1)
                color = np.rot90(color, 1)
            elif t == 4:
                lineart = np.rot90(lineart, -1)
                #block = np.rot90(block, -1)
                color = np.rot90(color, -1)
            
            # Add noise
            if p_noise<0.15:             
                noise = np.random.normal(0, 3, lineart.shape).astype(lineart.dtype)
                lineart += noise
                # noise = np.random.normal(0, 3, color.shape).astype(color.dtype)
                # color += noise

        # Make hint
        availible_region = np.where(lineart >254, True, False)

        hint = np.zeros(color.shape).astype(np.float32)
        hint[:,:,0]-=256

        availible_region = np.array(np.where(availible_region))
        offset = int(availible_region.shape[1]/sample_amount)
        random_pos = np.random.randint(0,offset, size = sample_amount*2)
        random_width = np.random.randint(1,max_width, size = sample_amount*2)
        random_prob = np.random.random_sample(size = sample_amount)

        # amount = random.randint(20, 25)
        # points = np.random.randint(0,len(availible_region[0]),size=amount)
        
        # for i in points:
        #     x,y = availible_region[0][i]+max_width, availible_region[1][i]+max_width
        #     #color_value = color[x,y]
        #     hint[x-width:x+width,y-width:y+width] = color[x,y]#color_value
        j=0
        offset_t=-offset

        def draw_a_point(pos):
            p = availible_region[:,offset_t+pos]
            width = random_width[j]
            x,y = np.clip(p,width, 256-width)
            hint[x-width:x+width,y-width:y+width] = color[x,y]


        for i in range(sample_amount):
            offset_t+=offset
            if random_prob[i]<0.2:
                continue            
            draw_a_point(random_pos[j])
            j+=1
            if random_prob[i]>0.9:
                draw_a_point(random_pos[j])
                j+=1
                
        #print(j)

        # hint1 = np.zeros(color.shape).astype(np.float32)-256
        # for i in points:
        #     x,y = availible_region[0][i]+max_width, availible_region[1][i]+max_width
        #     color_value = color[x-width:x+width,y-width:y+width]
        #     hint[x-width:x+width,y-width:y+width] = color_value
        #     gray_label = gray_labels[x,y]
        #     gray_pos = np.where(gray_labels==gray_label,True,False)
        #     for j in range(3):           
        #         hint1[:,:,j] = np.where(gray_pos, np.median(color_value[:,:,j]), hint1[:,:,j])

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))  
        # hint1 = cv2.dilate(hint1, kernel)  
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
        # hint = cv2.dilate(hint, kernel)
        # hint_pos = np.where(hint[:,:,0]==-256.,True,False)
        # for j in range(3):        
        #     hint[:,:,j] = np.where(hint_pos, hint1[:,:,j], hint[:,:,j])

        # Make seg img
        #block = torch.from_numpy(np.transpose(block), (2,0,1)).to(device)
        #feature_idx, feature_map = gen_seg(block)
            
        # Send to pytorch
        lineart = lineart[None]
        #feature_map = feature_map[None]
        #lineart = np.concatenate([lineart]*2, axis =0)
        color = np.transpose(color, (2,0,1))
        hint =  np.transpose(hint, (2,0,1))

        inputs_batch[idx] = torch.from_numpy(lineart.copy())
        # seg_batch[idx] = torch.from_numpy(feature_map)
        # idx_batch[idx] = torch.from_numpy(feature_idx).to(device)
        label_batch[idx] = torch.from_numpy(color.copy())
        hint_batch[idx] = torch.from_numpy(hint.copy())


    # Send to GPU
    inputs_batch = torch.stack(inputs_batch).to(device)
    #seg_batch = torch.stack(seg_batch).to(device)
    #idx_batch = torch.cat(idx_batch,0).to(device)
    label_batch = torch.stack(label_batch).to(device)
    hint_batch = torch.stack(hint_batch).to(device)
   

    return inputs_batch, hint_batch, label_batch#, seg_batch, idx_batch



drawing = False # 鼠标左键按下时，该值为True，标记正在绘画
mode = True # True 画矩形，False 画圆
ix, iy = -1, -1 # 鼠标左键按下时的坐标
refer_img = np.zeros([300,300],dtype = np.uint8)
refer_lab = np.zeros([300,300],dtype = np.float32)
color = (0,0,0)
color_lab = (0,0,0)
img1 = np.zeros((128,128,3), np.uint8)
line = np.zeros([300,300],dtype = np.uint8)
hint = np.zeros([300,300],dtype = np.float32)
result = np.zeros([300,300],dtype = np.uint8)


def pick_color(event, x, y, flags, param):
    global color,color_lab, refer_img, img1,refer_lab

    if event == cv2.EVENT_LBUTTONDOWN:
        # 鼠标左键按下事件
        color = refer_img[y,x]
        color_lab = refer_lab[y,x]
        img1[:,:]=color
        cv2.imshow('Color',img1)

def pick_color_e(event, x, y, flags, param):
    global color,color_lab, refer_img, img1,refer_lab,result

    if event == cv2.EVENT_LBUTTONDOWN:
        # 鼠标左键按下事件
        color = result[y,x]
        color_lab = hint[y,x]
        img1[:,:]=color
        cv2.imshow('Color',img1)

def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode,color, color_lab, hint
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # 鼠标左键按下事件
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        # 鼠标移动事件
        if drawing == True:
            if mode == True:
                line[iy:y,ix:x]=color
                hint[iy:y,ix:x]=color_lab
                #cv2.rectangle(img, (ix, iy), (x, y), (0,240,0), -1)
            

    elif event == cv2.EVENT_LBUTTONUP:
        # 鼠标左键松开事件
        drawing = False
        # if mode == True:
        #     cv2.rectangle(img, (ix, iy), (x, y), (c[0],c[1],c[2]), -1)
        # else:
        #     cv2.circle(img, (x, y), 5, (c[0],c[1],c[2]), -1)


cv2.namedWindow('Lineart')
cv2.setMouseCallback('Lineart', draw_circle) # 设置鼠标事件的回调函数

cv2.namedWindow('Color_ref')
cv2.setMouseCallback('Color_ref', pick_color) # 设置鼠标事件的回调函数
cv2.namedWindow('Color')
cv2.namedWindow('Result')
cv2.setMouseCallback('Result', pick_color_e) # 设置鼠标事件的回调函数
i=0

def get_paint(names):

    inputs_batch = [None]*len(names)
    #label_batch = [None]*len(names)
    hint_batch = [None]*len(names)
    
    for idx in range(0,len(names)):
        f = names[idx]
        global refer_img, line, refer_lab, hint
        # Read file
        line = cv2.imread('D:/Paints/pics/256_data/test/289002.jpg')
        m = max(line.shape[:2])
        size = 400
        ratio = (size+0.0)/m
        line = cv2.resize(line,(0,0),fx=ratio,fy = ratio, interpolation= cv2.INTER_AREA)

        if line.shape[1]<size:
            d = int((size - line.shape[1])/2)+1
            line = cv2.copyMakeBorder(line,0,0,d,d,cv2.BORDER_CONSTANT,value=[255,255,255])

        if line.shape[0]<size:
            d = int((size - line.shape[0])/2)+1
            line = cv2.copyMakeBorder(line,d,d,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
        

        line = line[:size,:size]
        #line = cv2.resize(line,dsize = (0,0), fx=0.8,fy=0.8)
        line = np.where(line>245,255,line-50)
        line = np.clip(line,0,255)
        lineart = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        refer_img = cv2.imread(label_dir+f)
        cv2.imshow('Color_ref', refer_img)
        refer_lab = BGR2Lab(refer_img.astype(np.float32))
        global i
        if i==0:
            hint = np.zeros(line.shape).astype(np.float32)
            hint[:,:,0]-=256
            i+=1

        global mode
        while(1):
            cv2.imshow('Lineart', line)
            
            k = cv2.waitKey(1) & 0xFF
            if k == ord('m'):
                mode = not mode
            elif k == 27:
                break
            elif k == ord('r'):
                break
       
        # Send to pytorch
        lineart = lineart[None]
        #feature_map = feature_map[None]
        #lineart = np.concatenate([lineart]*2, axis =0)
        #color = np.transpose(color, (2,0,1))
        hint_t =  np.transpose(hint, (2,0,1))

        inputs_batch[idx] = torch.from_numpy(lineart.copy())
        # seg_batch[idx] = torch.from_numpy(feature_map)
        # idx_batch[idx] = torch.from_numpy(feature_idx).to(device)
        #label_batch[idx] = torch.from_numpy(color.copy())
        hint_batch[idx] = torch.from_numpy(hint_t.copy())


    # Send to GPU
    inputs_batch = torch.stack(inputs_batch).to(device)
    #label_batch = torch.stack(label_batch).to(device)
    hint_batch = torch.stack(hint_batch).to(device)
   

    return inputs_batch, hint_batch#, label_batch#, seg_batch, idx_batch