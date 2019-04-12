import torch , cv2, random, os
import numpy as np

class PaintDataset(torch.utils.data.Dataset):
    def __init__(self,root,train=True, transform = None, target_transform=None, with_clue = False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.with_clue = with_clue

        if self.train:
            self.train_data = os.listdir(root+'/bw/')
            self.train_labels = os.listdir(root+'/color/')
        else:
            self.test_data = os.listdir(root+'/bw/')
            self.test_labels = os.listdir(root+'/color/')


    def __getitem__(self,index):
        if self.train:
            target = self.train_labels[index]
            img =  target.split('.')[0]+'.png'
        else:
            pass
            #img, target = self.test_data[index], self.test_labels[index]
        

        #for idx in range(0,len(names)):
        #f = names[idx]
        t = int(random.random()*4)
        p_noise = random.random()
        
        lineart = cv2.imread(self.root+'bw/'+img ,cv2.IMREAD_GRAYSCALE)
        color = cv2.imread(self.root+'color/'+target)

        # Randimly flip the image
        if t==0:
            lineart = lineart[::-1]
            color = color[::-1]
        elif t == 1:
            lineart = lineart[:,::-1]
            color = color[:,::-1]
        elif t == 2:
            lineart = lineart[::-1,::-1]
            color = color[::-1,::-1]

        lineart = lineart.astype(np.float32)
        lineart /= 255.0
        lineart = np.expand_dims(lineart, 0)
        color = np.transpose(color, (2,0,1)).astype(np.float32)
        color /= 255.0
        
        if self.with_clue:
            points = np.random.randint(5,len(lineart[0])-5,size=[72*5,2])#(256*256/9)*0.01 = 72.8
            mask = np.zeros((3,len(lineart[0]), len(lineart[0][0])), np.float32)
            # Set each channel to 0.0, 0.5 and 1.0 respectively
            mask[1] += 0.5
            mask[2] += 1.0
            #mask = np.insert(mask, [1,1], [0.5,1.0], axis=0) 
            for p in points:
                mask[:,p[0]-2:p[0]+1,p[1]-2:p[1]+1] = color[:,p[0]-2:p[0]+1,p[1]-2:p[1]+1]
                #print(mask[:,p[0]-2:p[0]+1,p[1]-2:p[1]+1])
            lineart = np.concatenate([lineart, mask], axis =0)

        if p_noise<0.15: # Add noise            
            noise = np.random.normal(0, 5, lineart.shape).astype(np.float32)/255.0
            lineart += noise
            noise = np.random.normal(0, 5, color.shape).astype(np.float32)/255.0
            color += noise

        return torch.from_numpy(lineart), torch.from_numpy(color)


    def __len__(self):
        return len(self.train_labels)