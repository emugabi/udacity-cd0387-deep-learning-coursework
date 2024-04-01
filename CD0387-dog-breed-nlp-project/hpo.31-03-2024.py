#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader

import argparse
import json
import logging
import os
import boto3
import io
import sys

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )
    average_accuracy = correct/len(test_loader.dataset)
    average_loss = test_loss/len(test_loader.dataset)
    logger.info(f'Test set: Average loss: {average_loss}, Accuracy: {100*average_accuracy}%')
    
    # Logger info
    if average_accuracy > 0.9:
        logger.info('Great job!')
    elif average_accuracy > 0.8:
        logger.info('Good job! Improve the accuracy.')
    else:
        logger.warning('Keep working.')
    
    # Size of the dataset
    num_images = len(test_loader.dataset)
    batch_size = test_loader.batch_size
    num_batches = len(test_loader)
    logger.info(f'Tested {num_images} images in {num_batches} batches of size {batch_size}.')
        
        

def train2(model, train_loader, valid_loader, criterion, optimizer, device, epochs):
    loss_counter = 0
    best_valid_loss = float('inf')
    samples = 0
    data_loaders = {'train': train_loader, 'valid': valid_loader}

    for epoch in range(epochs):
        print(f"Epoch:{epoch}")
        for phase in ['train', 'valid']:
            print(f"Phase:{phase}")
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, predictions = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == labels.data)
                samples += len(inputs)

                # NOTE: Comment lines below to train and test on whole dataset
                if samples > (0.2 * len(data_loaders[phase].dataset)):
                    break

            epoch_loss = running_loss // len(data_loaders[phase])
            epoch_acc = running_corrects // len(data_loaders[phase])

            if phase == 'valid':
                if epoch_loss < best_valid_loss:
                    best_valid_loss = epoch_loss
                else:
                    loss_counter += 1

            print('{} loss: {:.3f}, accuracy: {:.2f}, best valid loss: {:.3f}'
                  .format(phase, epoch_loss, epoch_acc, best_valid_loss))

        if loss_counter == 1:
            break
        if epoch == 0:
            break

    return model

def train(model, train_loader, cost, optimizer, epochs, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    for epoch in range(1, epochs + 1):
        model.train()
        for e in range(epoch):
            running_loss = 0
            correct = 0
            for data, target in train_loader:
                optimizer.zero_grad()

                pred = model(data)
                loss = cost(pred, target)
                running_loss += loss
                loss.backward()
                optimizer.step()
                pred = pred.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            print(f"Epoch {e}: Loss {running_loss/len(train_loader.dataset)}, Accuracy {100*(correct/len(train_loader.dataset))}%")
    return model
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained = True)

    for param in model.parameters():
        param.requires_grad = False
    # Adding new layers for fine-tuning
    num_features = model.fc.in_features
    num_classes = 133
    model.fc = nn.Sequential(nn.Linear(num_features, 256), 
                              nn.ReLU(),
                              nn.Linear(256, 128),
                              nn.ReLU(),
                              nn.Linear(128,  num_classes),
                              nn.LogSoftmax(dim=1)
                            )
    # Counting number of layers and trainable parameters
    num_layers = len(list(model.parameters()))
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f'Model created: {num_layers} layers, {num_trainable_params} trainable parameters.')
    
    return model

def create_data_loaders2(data, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.ImageFolder(data, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader

def create_data_loaders(loader_type, batch_size, transformer):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    
    bucket_name = 'sagemaker-us-east-1-433073421675'
    
    if(loader_type == 'train'):
        dataset = S3ImageDataset(bucket_name, "sagemaker/cd0387-project-resnet-50/train", transform=transformer)
        return DataLoader(dataset, batch_size = batch_size, shuffle=False)
    else:
        dataset = S3ImageDataset(bucket_name, "sagemaker/cd0387-project-resnet-50/test", transform=transformer)
        return DataLoader(dataset, batch_size = batch_size, shuffle=True)

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = net()
    model = model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), args.lr)

    '''
    TODO: Create dataloaders
    '''
    logger.info("Creating dataloaders.")

    #train_loader = create_data_loaders('train', args.batch_size, data_transforms['train'])
    train_loader = create_data_loaders2(os.environ['SM_CHANNEL_TRAINING'], args.batch_size)
    
    valid_loader = create_data_loaders2(os.environ['SM_CHANNEL_VALIDATION'], args.batch_size)

    test_loader = create_data_loaders2(os.environ['SM_CHANNEL_TEST'], args.batch_size)
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    logger.info("Training the model.")
    
    #model = train(model, train_loader, loss_criterion, optimizer, args.epochs, device)
    model = train2(model, train_loader, valid_loader, loss_criterion, optimizer, device, args.epochs)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info("Testing the model.")
    test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    logger.info("Saving the model.")
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.state_dict(), path)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    # Container environment
    parser.add_argument(
        "--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"]
    )
    parser.add_argument(
        "--model-dir", type=str, default=os.environ["SM_MODEL_DIR"]
    )
    
    args = parser.parse_args()
    print(args)
    
    main(args)