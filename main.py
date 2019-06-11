"""Imports"""

import torch
import torchvision.models as models
import hiddenlayer as hl
import segmentation_models_pytorch as smp
from easydict import EasyDict as edict

from dataloader import MyDataset
from plots import VisdomLinePlotter
from train import run_epoch
import torch.optim as optim
from torch.optim import lr_scheduler

import os

"""Parameters"""
params = edict({'batch_size':3,
                'width': 1920,
                'height': 1184,
                'numclasses': 2,
                'train_identifier': 'my_instance_segmentation'})

device = 'cuda:0'


"""Visdom stuff"""
from visdom import Visdom
visdom_log_path = os.path.join("/tmp")
#visdom_log_path = outdir
print("Saving visdom logs to", visdom_log_path)
viz = Visdom(port=6065, log_to_filename=visdom_log_path)
# for env in viz.get_env_list():
viz.delete_env(params.train_identifier)
viz.log_to_filename = os.path.join(visdom_log_path,params.train_identifier+".visdom")
plotter  = VisdomLinePlotter(env_name=params.train_identifier, plot_path=visdom_log_path)




""" Load the model and make sure it works """
#inputs = torch.zeros([2, 3, 1920, 1184])
model = smp.Unet('resnet34', classes=params.numclasses+2, encoder_weights='imagenet')
model = model.to(device)
#output = model.forward(inputs)
#print("output-shape:", output.shape)

preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name = 'resnet34', pretrained='imagenet')



"""dataloader"""
image_dir = '/raid/group-data/uc150429/AID_DATA_201905/batch-123/original_image/'
segvecs_dir = '/raid/group-data/uc150429/AID_DATA_201905/batch-123/center_vectors/'
classes_dir = '/raid/group-data/uc150429/AID_DATA_201905/batch-123/pixelwise_annotation_xml'
dataset = MyDataset(image_dir = image_dir, classes_dir = classes_dir, segvecs_dir = segvecs_dir)
x, y = dataset[0]
y_classmap, y_vectormap = y

print(x.shape)


train_sampler = None
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=params.batch_size, shuffle=(train_sampler is None),
    num_workers=20, pin_memory=True, sampler=train_sampler)

dataloader = {'train':train_loader, 'val': train_loader}

print("Created train loader")


""" Run a full epoch"""
optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9)

scheduler = lr_scheduler.StepLR(
    optimizer,
    step_size=20,
    gamma=0.5)

for epoch in range(10):
    epoch_loss = run_epoch(
        params=params,
        phase='train',
        num_batches=10,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=dataloader['train'],
        device=device)

    plotter.plot(var_name='train_loss',
                 split_name='split_name',
                 title_name='title_name',
                 x=epoch,
                 y=epoch_loss)

assert(1==2)


# val_loader = torch.utils.data.DataLoader(
#     datasets.ImageFolder(valdir, transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize,
#     ])),
#     batch_size=opts.batchsize, shuffle=False,
#     num_workers=20, pin_memory=True)






X = torch.Tensor(x[None,:,:,:])
Y_class = torch.Tensor(y_classmap[None,:,:]).type(torch.long)
Y_vector = torch.Tensor(y_vectormap[None,:,:,:])
print("X", X.shape, X.dtype)
print("Y_class", Y_class.shape, Y_class.dtype)
print("Y_vector", Y_vector.shape, Y_vector.dtype)

""" put input and target to device"""
Y_class = Y_class.to(device)
Y_vector = Y_vector.to(device)
X = X.to(device)

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