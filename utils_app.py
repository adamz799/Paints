from config10 import *

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
    

def get_image(name, tensor):
    if tensor.shape[1]==1:
        img = tensor[0,0].cpu().detach().numpy()[:,:,None].astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img
        
    # lab_max
    # array([[[100.     ,  98.23516,  94.47579]]], dtype=float32)
    #  lab_min
    # array([[[   0.      ,  -86.18125 , -107.861755]]], dtype=float32)
    tensor[:,0] = torch.clamp(tensor[:,0],0,100)
    tensor[:,1] = torch.clamp(tensor[:,1],-86.18125,98.23516)
    tensor[:,2] = torch.clamp(tensor[:,2],-107.861755,94.47579)
    tensor = tensor_Lab2BGR(tensor)[0]
    img = tensor.cpu().detach().numpy()
    img = np.clip(img,0,255).astype(np.uint8)
    img = np.transpose(img,(1,2,0))
    cv2.imshow(name,img)
    return img


drawing = False 
mode = True 
ix, iy = -1, -1 
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
        color = refer_img[y,x]
        color_lab = refer_lab[y,x]
        img1[:,:]=color
        cv2.imshow('Color',img1)

def pick_color_e(event, x, y, flags, param):
    global color,color_lab, refer_img, img1,refer_lab,result

    if event == cv2.EVENT_LBUTTONDOWN:
        color = result[y,x]
        color_lab = hint[y,x]
        img1[:,:]=color
        cv2.imshow('Color',img1)

def draw(event, x, y, flags, param):
    global ix, iy, drawing, mode,color, color_lab, hint
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:      
        if drawing == True:
            if mode == True:
                line[iy:y,ix:x]=color
                hint[iy:y,ix:x]=color_lab
               

    elif event == cv2.EVENT_LBUTTONUP:     
        drawing = False
       


cv2.namedWindow('Lineart')
cv2.setMouseCallback('Lineart', draw) 
cv2.namedWindow('Color_ref')
cv2.setMouseCallback('Color_ref', pick_color) 
cv2.namedWindow('Color')
cv2.namedWindow('Result')
cv2.setMouseCallback('Result', pick_color_e)
i=0

def get_paint(names):
    inputs_batch = [None]*len(names)
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
        hint_t =  np.transpose(hint, (2,0,1))

        inputs_batch[idx] = torch.from_numpy(lineart.copy())
        hint_batch[idx] = torch.from_numpy(hint_t.copy())


    # Send to GPU
    inputs_batch = torch.stack(inputs_batch).to(device)
    hint_batch = torch.stack(hint_batch).to(device)
   
    return inputs_batch, hint_batch