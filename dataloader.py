import os
import sys
import xmltodict
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, phase, image_dir, classes_dir, segvecs_dir):
        self.image_dir = image_dir
        self.classes_dir = classes_dir
        self.segvecs_dir = segvecs_dir
        classes_candidates = [name.replace('.xml', '') for name in os.listdir(classes_dir)]
        segvecs_candidates = set([name.replace('.png', '') for name in os.listdir(segvecs_dir)])
        idx_list = [idx for idx in classes_candidates if idx in segvecs_candidates]

        if phase=='train':
            self.idx_list = idx_list[:int(len(idx_list) * 0.8)]
        else:
            self.idx_list = idx_list[int(len(idx_list) * 0.8):]

    def __len__(self):
        return len(self.idx_list)

    def get_image(self,item):
        idx = self.idx_list[item]
        x = cv2.imread(os.path.join(self.image_dir, idx + '.png'))
        x = x[:1184, :, :]
        x = x.transpose((2, 1, 0))
        return x

    def get_label(self,item):
        try:
            classmap = np.zeros((1208, 1920, 2), dtype=np.float32)
            vectormap = np.zeros((1208, 1920, 2), dtype=np.float32)
            idx = self.idx_list[item]

            img = cv2.imread(os.path.join(self.segvecs_dir, idx + '.png'))
            img = img.sum(axis=-1)
            ###print("max:", img.max(), ' min:', img.min())
            img= img!=0.0
            img = img.astype(np.float32)
            ###print("Image maxuimum:", img.max())
            #img = img / img.max()
            classmap = img

            vectormap[:, :, 0] = cv2.imread(os.path.join(self.segvecs_dir, idx + '_w.png')).sum(axis=-1)/255
            vectormap[:, :, 1] = cv2.imread(os.path.join(self.segvecs_dir, idx + '_h.png')).sum(axis=-1) / 255

            classmap = classmap[:1184, :]
            classmap = classmap.transpose((1, 0))
            vectormap = vectormap[:1184, :,:]
            vectormap = vectormap.transpose((2, 1, 0))
            return (classmap, vectormap)
        except:
            print("Encountered a problem with item:",item," idx", self.idx_list[item])
            return self.get_label((item+1)%self.__len__())


    def __getitem__(self, item):
        #print("Item:", item)
        x = self.get_image(item)
        y = self.get_label(item)
        return x, y



if __name__ == '__main__':
    image_dir = '/raid/group-data/uc150429/AID_DATA_201905/batch-123/original_image/'
    segvecs_dir = '/raid/group-data/uc150429/AID_DATA_201905/batch-123/center_vectors/'
    classes_dir = '/raid/group-data/uc150429/AID_DATA_201905/batch-123/pixelwise_annotation_xml'

    dataloader = MyDataset(image_dir = image_dir, classes_dir = classes_dir, segvecs_dir = segvecs_dir)
    x, (classmap, vectormap) = dataloader[0]
    import matplotlib.pyplot as plt
    plt.figure(); plt.imshow(classmap[:,:,0])
    plt.figure(); plt.imshow(classmap[:,:,1])
    plt.figure(); plt.imshow(vectormap[:,:,0])
    plt.figure(); plt.imshow(vectormap[:,:,1])
    plt.show()

    print("classmap.shape", classmap.shape)
    print("vectormap.shape", vectormap.shape)