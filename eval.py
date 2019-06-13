""" Evaluation module.

Checkpoint evaluation module.

Currently, no real metrics are calculated yet.
Instead, for every eval image, the following comparisons are created:
 - ground-truth class vs predicted class
 - ground-truth x-distance vs predicted x-distance
 - ground-truth y-distance vs predicted y-distance

TODO:
    * Find a way to identify object instances from the predictions

"""

import os
import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

from dataloader import MyDataset


checkpoint = '/raid/user-data/lscheucher/tmp/my_instance_segmentation/checkpoints/350my_instance_segmentation_SGD.pth'


parent_dir = checkpoint.rsplit('/', maxsplit=1)[0]
eval_folder = 'eval_'+checkpoint.rsplit('/', maxsplit=1)[1].split('.')[0]
eval_dir = os.path.join(parent_dir, eval_folder)
if not os.path.isdir(eval_dir):
    os.mkdir(eval_dir)


image_dir = '/raid/group-data/uc150429/AID_DATA_201905/batch-123/original_image/'
segvecs_dir = '/raid/group-data/uc150429/AID_DATA_201905/batch-123/center_vectors/'
classes_dir = '/raid/group-data/uc150429/AID_DATA_201905/batch-123/pixelwise_annotation_xml'

dataset_eval = MyDataset(phase='val', image_dir = image_dir, classes_dir = classes_dir, segvecs_dir = segvecs_dir)

sampler = None
eval_loader = torch.utils.data.DataLoader(
    dataset_eval,
    batch_size=3, shuffle=(sampler is None),
    num_workers=2, pin_memory=True, sampler=sampler)

model = smp.Unet('resnet34', classes=4)
model.load_state_dict(torch.load(checkpoint))
model.eval()

# Loop over evaluation images
for i, (x,y) in zip(range(10), eval_loader):
    y_classmap, y_vectormap = y
    X = torch.Tensor(x.type(torch.float))
    Y_class = torch.Tensor(y_classmap).type(torch.long)
    Y_vector = torch.Tensor(y_vectormap[:, :, :, :])

    result = model.forward(X)

    # Loop over images in batch
    for batch_index in range(result.shape[0]):
        fig = plt.figure()
        fig.set_size_inches(20, 20)
        fig.add_subplot(4, 2, 2)
        class_pred = result[batch_index, :2, :, :]
        class_pred_argmax = class_pred.argmax(dim=0)
        plt.imshow(class_pred_argmax.transpose(1, 0).numpy())
        plt.title("class - Prediction")
        fig.add_subplot(4, 2, 1)
        plt.imshow(Y_class[batch_index, :, :].numpy().transpose((1, 0)))
        plt.title("class - Ground truth")

        gt_w = Y_vector[batch_index, 0, :, :].numpy().transpose((1, 0))
        gt_h = Y_vector[batch_index, 1, :, :].numpy().transpose((1, 0))
        pred_w = result[batch_index, 2, :, :].detach().numpy().transpose((1, 0))
        pred_h = result[batch_index, 3, :, :].detach().numpy().transpose((1, 0))

        fig.add_subplot(4, 2, 3)
        plt.imshow(gt_w)
        plt.title("w - Ground truth")

        fig.add_subplot(4, 2, 4)
        plt.imshow(pred_w)
        plt.title("w - Prediction")

        fig.add_subplot(4, 2, 5)
        plt.imshow(gt_h)
        plt.title("h - Ground truth")

        fig.add_subplot(4, 2, 6)
        plt.imshow(pred_h)
        plt.title("h - Prediction")

        fig.add_subplot(4, 2, 7)
        plt.imshow(X[batch_index].numpy().transpose((2, 1, 0))[:, :, ::-1] / 255)
        plt.title("Image")

        outfile = os.path.join(eval_dir, str(i)+'_'+str(batch_index)+'.png')
        print("Saving results to:", outfile)
        plt.savefig(outfile)
        plt.close(fig)