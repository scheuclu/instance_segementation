"""Imports"""

import torch
import torchvision.models as models
import hiddenlayer as hl
import segmentation_models_pytorch as smp
from easydict import EasyDict as edict

from dataloader import DataLoader

"""Parameters"""
params = edict({'batch_size':1,
                'width': 1920,
                'height': 1184,
                'numclasses': 2})




""" Load the model and make sure it works """
inputs = torch.zeros([2, 3, 1920, 1184])
model = smp.Unet('resnet34', classes=params.numclasses+2, encoder_weights='imagenet')
output = model.forward(inputs)
print("output-shape:", output.shape)

preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name = 'resnet34', pretrained='imagenet')



"""dataloader"""
image_dir = '/raid/group-data/uc150429/AID_DATA_201905/batch-123/original_image/'
segvecs_dir = '/raid/group-data/uc150429/AID_DATA_201905/batch-123/center_vectors/'
classes_dir = '/raid/group-data/uc150429/AID_DATA_201905/batch-123/pixelwise_annotation_xml'
dataloader = DataLoader(image_dir = image_dir, classes_dir = classes_dir, segvecs_dir = segvecs_dir)
x, y = dataloader[0]
y_classmap, y_vectormap = y

print(x.shape)




X = torch.Tensor(x[None,:,:,:])
Y_class = torch.Tensor(y_classmap[None,:,:]).type(torch.long)
Y_vector = torch.Tensor(y_vectormap[None,:,:,:])
print("X", X.shape, X.dtype)
print("Y_class", Y_class.shape, Y_class.dtype)
print("Y_vector", Y_vector.shape, Y_vector.dtype)

result = model.forward(X)
# torch.Size([1, 4, 1920, 1184])

pred_class_logits = result[:, :2, :, :] #torch.Size([1, 2, 1920, 1184])
pred_class_vectors = result[:, 2:, :, :] #torch.Size([1, 2, 1920, 1184])

#class_preds = torch.nn.Softmax(dim=-1)(class_logits)



""" Cross entropy loss """
cross_entropy_loss_fn = loss = torch.nn.CrossEntropyLoss()
pred_class_logits = torch.reshape(pred_class_logits, (params.batch_size,params.numclasses,params.width*params.height))
Y_class = torch.reshape(Y_class, (params.batch_size,params.width*params.height))
cross_entropy_loss= cross_entropy_loss_fn(pred_class_logits, Y_class)

""" L2 vector loss """
# pred_class_vectors:  torch.Size([1, 2, 1920, 1184])
# Y_vector: torch.Size([1, 2, 1920, 1184])
vector_loss_fn = torch.nn.MSELoss()
vector_loss = vector_loss_fn(pred_class_vectors, Y_vector)

total_loss = cross_entropy_loss + vector_loss

print("Calcualted loss for current batch as:", total_loss)