import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import CNN
import time
import torch.multiprocessing
#import multiprocessing
from torch.multiprocessing import Pool, Process, set_start_method, Queue, cpu_count
from functools import partial
import threading
from scipy.special import softmax

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define a Loss function and optimizer (Let’s use a Classification Cross-Entropy loss and SGD with momentum.)
import torch.optim as optim

print(cpu_count())
#GLOBAL VARIABLES
# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20
# percentage of training set to use as validation
valid_size = 0.0

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# choose the training and test datasets
train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
    num_workers=num_workers)

# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

criterion = nn.CrossEntropyLoss()
queueOutput = Queue()
featureAccuracies = {}

def featureThreading(include):
  include = include
  start = time.time()
  if len(include) > 0 and include[0] == -1:
    baseTest = CNN.trainModel(include, -1, queueOutput, train_loader, test_loader, batch_size, classes)
    featureAccuracies[baseTest[0]] = baseTest[1]
  else:
    #featureAccuracies = {}
    print(featureAccuracies)
    processes = [Process(target = CNN.trainModel, args = (include, i, queueOutput, train_loader, test_loader, batch_size, classes)) for i in range(0, 2)]
    for p in processes:
      p.start()
    for p in processes:
      p.join()

    #p = Pool(8)
    #results = p.map(trainModel, (include, range(0,2)))
    #for i in range(0, 2):
    #  print(queueOutput.get())
    #  print(results.next())
    results = [queueOutput.get() for p in processes]
    print(results)
    print(featureAccuracies)
    """
    threads = []
    for i in range(0, 2):
        threads.append(threading.Thread(target = trainModel, args = (include, i,)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    """
  end = time.time()

  sortedfeatureAccuracies = {k: featureAccuracies[k] for k in sorted(featureAccuracies)}
  sortedfeatureAccuracies = list(sortedfeatureAccuracies.values())
  #softMaxFeatures = list(softmax(sortedfeatureAccuracies))
  max_val = 0.0
  baseTestAccuracy = 0.0
  if len(sortedfeatureAccuracies) > 1:
    max_val = max(sortedfeatureAccuracies)
    if max_val == 0.0:
      pass
    idx_max = sortedfeatureAccuracies.index(max_val)
    print("Maximum Test Accuracy among Features: " + str(max_val))
    #include.append(idx_max)
  else:
    baseTestAccuracy = sortedfeatureAccuracies[0]
    sortedfeatureAccuracies = sortedfeatureAccuracies[1:]
    print("Base Test Accuracy: " + str(baseTestAccuracy))
  print("Time taken: " + str(end - start))
  print("Feature Accuracies Unsorted: " + str(featureAccuracies))

  print("Sorted Feature Accuracies: " + str(sortedfeatureAccuracies))
  print("Included Indices: " + str(include))
  print()
  return sortedfeatureAccuracies, include, max_val, baseTestAccuracy

import functools
if __name__ == '__main__':
    """
    try:
      set_start_method('spawn', force=True)
      print("spawned")
    except RuntimeError:
      pass
    """
    """
    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()
    print(train_on_gpu)
    device_name = torch.cuda.get_device_name()
    n_gpu = torch.cuda.device_count()
    print(f"Found device: {device_name}, n_gpu: {n_gpu}")
    device = torch.device("cuda")
    """
    beamSearch = 2
    #Calculate base test accuracy
    include = [-1]
    baseTest = CNN.trainModel(include, train_loader, test_loader, batch_size, classes, -1)
    featureAccuracies[baseTest[0]] = baseTest[1]
    featureAccuracies.pop(-1)
    #First iteration outside of while loop to initialize variables
    include = []


    #params = functools.partial(CNN.trainModel, include, train_loader, test_loader, batch_size, classes)
    #indicies = [i for i in range(0,2)]
    #with Pool(processes = 8) as pool:
    #  results = pool.map(params, indicies)
    #  pool.close()
    #  pool.join()
    #for i in range(0, 2):
    #  print(queueOutput.get())
    #  print(results.next())
    max_val = 0.0
    baseTestAccuracy = 0.0
    sortedfeatureAccuracies = [99]
    sortedfeatureAccuracies, include, max_val, baseTestAccuracy = featureThreading(include)
    sortedfeatureAccuracies = np.array(sortedfeatureAccuracies)

    negbeamSearch = beamSearch * -1
    idx = np.argpartition(sortedfeatureAccuracies, negbeamSearch)[negbeamSearch:]
    topValues = sortedfeatureAccuracies[idx]
    print("Top " + str(beamSearch) + " values: " + str(topValues))
    topIdx = list(idx)
    includedResults = [[]]
    while(max(sortedfeatureAccuracies) >= max_val):
      includedInput = []
      #threads = []
      #results = []

      """
      for i in range(0, beamSearch):
        threads.append(threading.Thread(target = featureThreading, args = (include, results,)))
        for thread in threads:
          thread.start()
        for thread in threads:
          thread.join()
      """
      for x in topIdx:
        #print(topIdx)
        temp = includedResults[x // 2].copy()
        temp.append(x)
        includedInput.append(temp)

      """
      pool = ThreadPool(processes=beamSearch)

      for x in included:
        print(x)
        async_result = pool.apply_async(featureThreading, (x,)) # tuple of args for foo

      # do some other stuff in the main process

      return_val = async_result.get()  # get the return value from your function.
      for x in return_val:
        print(x)

      pool.close()
      pool.join()
      """
      maxAccuracies = []
      includedResults = []
      max_valResult = []
      for x in range(0, len(includedInput)):
        print("BEAMSEARCH: " + str(x))
        sortedfeatureAccuracies, include, max_val, _ = featureThreading(includedInput[x])
        if max_val == 0.0:
          break
        maxAccuracies += sortedfeatureAccuracies
        includedResults.append(include)
        max_valResult.append(max_val)
      if max_val == 0.0:
        break
      print(includedResults)
      print(maxAccuracies)
      max_val = max(max_valResult)
      topIdx = list(np.argpartition(maxAccuracies, negbeamSearch)[negbeamSearch:])
