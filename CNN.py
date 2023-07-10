import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import time
import torch.multiprocessing
#import multiprocessing
from torch.multiprocessing import Pool, Process, set_start_method, Queue
from functools import partial
import threading
from scipy.special import softmax

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define a Loss function and optimizer (Let’s use a Classification Cross-Entropy loss and SGD with momentum.)
import torch.optim as optim


criterion = nn.CrossEntropyLoss()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, include):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        temp = torch.zeros(x.shape)
        if len(include) > 0:
          for i in range(x.size(0)):
            for j in include:
              temp[i][j] = x[i][j]
          x = temp
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def trainModel(include, train_loader, test_loader, batch_size, classes, y):
  included = include.copy()
  included.append(y)
  if y < 0:
    included = []
  if y != -1 and y in include:
    keyValue = (y, 0.0)
    #queueOutput.put(keyValue)
    return keyValue
    #featureAccuracies[y] = 0.0
  else:
    #print("hello1" , flush = True)
    #List to store loss to visualize
    train_losslist = []
    valid_loss_min = np.Inf # track change in validation loss
    print("Feature Thread Started " + str(y), flush = True)
    model = Net()
    """
    if train_on_gpu:
        #model.cuda()
        model = model.to(device)
    """
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #print("hello2" , flush = True)
    for epoch in range(1):
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        running_loss = 0.0
        ###################
        # train the model #
        ###################
        #print("hello3" , flush = True)
        model.train()
        #print("hello4" , flush = True)
        #print(train_loader)
        for data, target in train_loader:
            #print("hello5" , flush = True)
            """
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                included = torch.tensor(included)
                #data, target = data.cuda(), target.cuda()
                data, target, included = data.to(device), target.to(device), included.to(device)
            """
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            #print("hello6" , flush = True)
            output = model(data, included)
            # calculate the batch loss
            #print("hello7" , flush = True)
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)
            # print statistics
            running_loss += loss.item()
        ######################
        # validate the model #
        ######################
        """
        model.eval()
        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                #data, target = data.cuda(), target.cuda()
                data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data, [])
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss
            valid_loss += loss.item()*data.size(0)
        """
        # calculate average losses
        train_loss = train_loss/len(train_loader.dataset)
        #valid_loss = valid_loss/len(valid_loader.dataset)
        train_losslist.append(train_loss)

        # print training/validation statistics
        #print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss), flush = True)

    # track test loss
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    #model.to(device)
    model.eval()
    # iterate over test data
    for data, target in test_loader:
        """
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
          temp = []
          temp = torch.tensor(temp)
          #data, target = data.cuda(), target.cuda()
          data, target, temp = data.to(device), target.to(device), temp.to(device)
        """
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data, [])
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    print("Feature Number included: " + str(y), flush = True)
    # average test loss
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss), flush = True)

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])), flush = True)
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]), flush = True)

    print('Test Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)), flush = True)
    #featureAccuracies[y] = np.sum(class_correct) / np.sum(class_total)
    print(flush = True)
    if y < 0:
      return (y, np.sum(class_correct) / np.sum(class_total))
    keyValue = (y, np.sum(class_correct) / np.sum(class_total))
    #queueOutput.put(keyValue)
    return keyValue
