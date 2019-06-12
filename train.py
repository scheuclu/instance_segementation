import torch
import itertools as it

def run_epoch(params, phase, num_batches, model, optimizer, scheduler, dataloader, device):

        print(phase.upper())
        if phase == 'train':
            scheduler.step()
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0

        # Iterate over data once.
        batchidx = 0
        for i, (x, y) in zip(it.count(num_batches),dataloader):
            print(i,end=' ')
            batchidx += 1
            if batchidx > num_batches:
                print(" ")
                break

            y_classmap, y_vectormap = y
            X = torch.Tensor(x.type(torch.float))
            Y_class = torch.Tensor(y_classmap).type(torch.long)
            Y_vector = torch.Tensor(y_vectormap[:, :, :, :])

            """ put input and target to device"""
            Y_class = Y_class.to(device)
            Y_vector = Y_vector.to(device)
            X = X.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(X)

                pred_class_logits = outputs[:, :2, :, :]  # torch.Size([1, 2, 1920, 1184])
                pred_class_vectors = outputs[:, 2:, :, :]  # torch.Size([1, 2, 1920, 1184])

                """ Cross entropy loss """
                cross_entropy_loss_fn = loss = torch.nn.CrossEntropyLoss()
                pred_class_logits = torch.reshape(pred_class_logits,
                                                  (params.batch_size, params.numclasses, params.width * params.height))
                Y_class = torch.reshape(Y_class, (params.batch_size, params.width * params.height))
                cross_entropy_loss = cross_entropy_loss_fn(pred_class_logits, Y_class)

                """ L2 vector loss """
                # pred_class_vectors:  torch.Size([1, 2, 1920, 1184])
                # Y_vector: torch.Size([1, 2, 1920, 1184])
                vector_loss_fn = torch.nn.MSELoss()
                vector_loss = vector_loss_fn(pred_class_vectors, Y_vector)

                total_loss = cross_entropy_loss + vector_loss

                # backward + optimize only if in training phase
                if phase == 'train':
                    total_loss.backward()
                    optimizer.step()

            # statistics
            running_loss += total_loss.item() * X.size(0)

        epoch_loss = running_loss / num_batches

        return epoch_loss