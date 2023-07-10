import csv
import numpy as np
import matplotlib.pyplot as plt
import os

beamWidths = [1,2,3,4,5,6,7]
numfeaturesSelected = [5, 10,15,20,25,30,35,40,45,50]
overleaf = []
for numfeature in numfeaturesSelected:
    baseAccuracy = 0.0
    total = 0
    numfeatureMax = []
    for beamWidth in beamWidths:
        beamWidthData = []
        arr = os.listdir()
        for file in arr:
            data = []
            if "beamSearch" + str(beamWidth) in file:
                if ".csv" in file:
                    total += 1
                    for x in range(0, beamWidth):
                        data.append([])
                    with open(file, "r") as f:
                        reader = csv.reader(f, delimiter="\t")
                        count = 0
                        for line in reader:
                            if len(data[0]) > numfeature:
                                continue
                            if len(line) > 0:
                                if 'Maximum' in line[0]:
                                    data[count%beamWidth].append(float((line[0].split(":",1)[1]).strip()))
                                    count += 1
                                if 'Base Test Accuracy' in line[0]:
                                    if float((line[0].split(":",1)[1]).strip()) > 0:
                                        baseAccuracy += float((line[0].split(":",1)[1]).strip())
                        beamWidthData.append(data)
        branchMax = []
        for y in range(0,len(beamWidthData)):
            data = beamWidthData[y]
            firstIdx = 0
            maxs = []
            for x in range(0, len(data)):
                data[x] = data[x][:-1]
                if x == 0:
                    firstIdx = data[x][0]
                else:
                    data[x].insert(0, firstIdx)
                plt.plot(data[x], label = "Beam Branch " + str(x + 1))
                #plt.plot(data[x][data[x].index(max(data[x]))], max(data[x]), label = "Max Value for Beam Branch " + str(x + 1))
                maxs.append(max(data[x]))
                #annot_max(data[x], data[x])
            print("Beamwidth " + str(beamWidth) + " iteration " + str(y + 1))
            print("Beamwidth " + str(beamWidth) + " maximum branch accuracies with " + str(numfeature) + " features selected: " + str(maxs))
            print("Beamwidth " + str(beamWidth) + " maximum accuracy of branches with " + str(numfeature) + " features selected: " + str(max(maxs)))
            plt.ylabel('Test Set Accuracy')
            plt.xlabel('Number of Features Included')
            plt.title('Test Accuracies of Beam Width = ' + str(beamWidth) + ' vs Number of Features Included (d=' + str(numfeature) + ')')
            plt.legend()
            plt.show()
            branchMax.append(max(maxs))
        print("Beamwidth " + str(beamWidth) + " average maximum accuracy among iterations with " + str(numfeature) + " features selected: " + str(sum(branchMax) / len(branchMax)))
        numfeatureMax.append(sum(branchMax) / len(branchMax))
        print()
    overleaf.append(numfeatureMax)
    print("Average Base Case Accuracy: " + str(baseAccuracy / total))
writeLines = []
for x in range(0,len(overleaf)):
    temp = ''
    print(str((x+1)*5), end =" ")
    temp = temp + str((x+1)*5) + ' '
    print('& ', end =" ")
    temp = temp + '& '
    for y in overleaf[x]:
        print(str(y) + ' & ', end =" ")
        temp = temp + str(y) + ' & '
    temp = temp + '\\'
    print('\\\\')
    writeLines.append(temp)
