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
from torch.utils.data import DataLoader

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
    num_images = len(test_loader.dataset)
    logger.info(f'Test image size: {num_images}, Test average loss: {average_loss}, Test accuracy: {100*average_accuracy}%')
        
        

def train(model, train_loader, valid_loader, criterion, optimizer, device, epochs):
    for epoch in range(epochs):

        for phase in ['train', 'valid']:
            running_loss = 0
            running_correct = 0
        
            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = valid_loader
                
            for data, target in data_loader:
                data = data.to(device)
                target = target.to(device)
            
                outputs = model(data)
                loss = criterion(outputs, target)
            
                _, preds = torch.max(outputs, 1)
            
                running_loss += loss.item() * data.size(0)

                with torch.no_grad():
                    running_correct += torch.sum(preds == target).item()
            
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_correct / len(data_loader.dataset)
        
            print(f'Epoch : {epoch}-{phase}, epoch loss = {epoch_loss}, epoch_acc = {epoch_acc}')

            
    return model 
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained = True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    num_classes = 133
    model.fc = nn.Sequential(nn.Linear(num_features, 256), 
                              nn.ReLU(),
                              nn.Linear(256,  num_classes),
                              nn.LogSoftmax(dim=1)
                            )
    
    return model

def create_data_loaders(data, batch_size):
    train_path = os.path.join(data, 'train')
    test_path = os.path.join(data, 'test')
    validation_path = os.path.join(data, 'valid')
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    
    train_transform = data_transforms["train"]

    test_transform = data_transforms["test"]
    
    valid_transform = data_transforms["valid"]
    
    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=train_transform)    
    test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=test_transform)
    validation_dataset = torchvision.datasets.ImageFolder(root=validation_path, transform=valid_transform)
    
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    validation_data_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    
    return train_data_loader, test_data_loader, validation_data_loader

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
    train_loader, test_loader, valid_loader = create_data_loaders(args.data_dir, args.batch_size)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    logger.info("Training the model.")
    
    #model = train(model, train_loader, loss_criterion, optimizer, args.epochs, device)
    model = train(model, train_loader, valid_loader, loss_criterion, optimizer, device, args.epochs)
    
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