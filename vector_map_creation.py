"""Import statements"""
import os
import numpy as np

import sys
import cv2
import matplotlib.pyplot as plt

import xmltodict
from tqdm import tqdm as tqdm

"""Custom classes"""
class DataGetter():
    def __init__(self, png_dir, annotation_dir):
        self.png_dir = png_dir
        self.annotation_dir = annotation_dir
    def get_idx_list(self):
        annotations = set([label.replace('.xml','') for label in os.listdir(self.annotation_dir)])
        pngs = set([png.replace('.png','') for png in os.listdir(self.png_dir)])
        self.idxs = annotations.intersection(pngs)
    def get_img(self, idx):
        path = os.path.join(self.png_dir,idx+'.png')
        img = cv2.imread(path)
        return img
    def get_annotation(self, idx):
        path = os.path.join(self.annotation_dir,idx+'.xml')
        with open(path, 'r') as f:
            xmlo = xmltodict.parse(f.read())
        return xmlo


def create_vectorfield(xmlo):
    vectorfield = np.zeros((1208, 1920, 2), dtype=np.float32)
    for polygon in xmlo['Document']['Polygons']['Polygon']:
        vectorfield_mask = np.zeros((1208, 1920, 3), dtype=np.uint8)
        LayerID = polygon['LayerID']
        if LayerID not in ['100', '101', '102', '103', '104']:
            continue
        points = polygon['Points']['Point']

        points = [[int(float(s)) for s in line.split(',')] for line in points]
        points = np.array(points, np.int32)
        center_w, center_h = np.mean(points, axis=0).astype(np.float32)

        M = cv2.moments(points)
        center_w = int(M["m10"] / M["m00"])
        center_h = int(M["m01"] / M["m00"])

        #print("Center", center_w, center_h)
        # points = points.reshape((-1,1,2))

        # img = cv2.fillPoly(img,[points],(0,255,255))
        vectorfield_mask = cv2.fillPoly(vectorfield_mask, [points], (0, 0, 1))

        # This was just to verify that the center is correct
        # cv2.circle(vectorfield_mask, (int(center_w), int(center_h)), 50, (1,1,0), 5)
        vectorfield_mask = np.sum(vectorfield_mask, axis=-1)  # at this point a hav a binary amsk for the current image
        W, H = np.meshgrid(np.linspace(0, 1919, 1920), np.linspace(0, 1207, 1208))

        # limit delta to the current object
        Dw = (center_w - W) * vectorfield_mask.astype(np.float32)
        Dh = (center_h - H) * vectorfield_mask.astype(np.float32)

        maxw = Dw.max()
        minw = Dw.min()
        maxh = Dh.max()
        minh = Dh.min()

        Dw = 1 - 1 / maxw * Dw * (Dw >= 0) - 1 / minw * Dw * (Dw < 0)
        Dw *= vectorfield_mask.astype(np.float32)

        Dh = 1 - 1 / maxh * Dh * (Dh >= 0) - 1 / minh * Dh * (Dh < 0)
        Dh *= vectorfield_mask.astype(np.float32)

        vectorfield[:, :, 0] *= (Dw == 0)
        vectorfield[:, :, 1] *= (Dh == 0)

        vectorfield[:, :, 0] += Dw
        vectorfield[:, :, 1] += Dh

        #plt.figure();
        #plt.imshow(vectorfield_mask)
    return vectorfield


"""paths"""
png_dir='/raid/group-data/uc150429/AID_DATA_201905/batch-123/original_image/'
annotation_dir='/raid/group-data/uc150429/AID_DATA_201905/batch-123/pixelwise_annotation_xml'
center_vec_dir='/raid/group-data/uc150429/AID_DATA_201905/batch-123/center_vec' \
               'tors'

"""Create DataGetter"""
datagetter = DataGetter(annotation_dir=annotation_dir,
                        png_dir=png_dir)

datagetter.get_idx_list()






print("Splitting import lsit")

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
fulllist = list(datagetter.idxs)
arg_instances = list(chunks(fulllist, int(len(fulllist)/60)))
#print(len(arg_instances))
#assert(1==2)


def process_list(idx_list):
    for idx in idx_list:

        xmlo = datagetter.get_annotation(idx)
        vectorfield = create_vectorfield(xmlo)
        np.nan_to_num(vectorfield)

        plt.imsave(arr=vectorfield.sum(axis=-1), fname='/raid/group-data/uc150429/AID_DATA_201905/batch-123/center_vectors/'+idx+'.png', cmap = plt.cm.gray)
        plt.imsave(arr=vectorfield[:,:,0],
                   fname='/raid/group-data/uc150429/AID_DATA_201905/batch-123/center_vectors/' + idx + '_w.png', cmap = plt.cm.gray)
        plt.imsave(arr=vectorfield[:,:,1],
                   fname='/raid/group-data/uc150429/AID_DATA_201905/batch-123/center_vectors/' + idx + '_h.png', cmap = plt.cm.gray)
        plt.imshow(vectorfield[:,:,0])
        plt.show()
        print("aaa")

        #np.save(file='/raid/group-data/uc150429/AID_DATA_201905/batch-123/center_vectors/'+idx+'.npy',
        #        arr=vectorfield)


# """Parallel execution"""
# from joblib import Parallel, delayed
# results = Parallel(n_jobs=60, verbose=1, backend="multiprocessing")\
#     (map(delayed(process_list), arg_instances))


"""Single processor execution"""
process_list(fulllist)
