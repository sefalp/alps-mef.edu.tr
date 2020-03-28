
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import math
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn.tree import export_graphviz

def plotterScatter(Dataset):
    df = pd.DataFrame(Dataset)
    colors = ['blue','orange', 'purple', 'pink', 'brown']
    df[2] = df[2].astype(int)
    classlbl = df[2]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)

    colors = np.asarray(colors)
    colorslist = colors[classlbl]
    
    for x, y, c in zip(df[0], df[1], colorslist):
        ax.scatter(x, y, color=c, cmap='viridis', alpha=1)
    
    #plt.plot([2.99,2.99], [0,8],'y',label="first X point(level two)")
    #plt.plot([7.04,7.04], [0,8],'r',label="second X point(level three")
    #plt.plot([0,8], [3.99,3.99],'b',label="only Y point(level one)")
    #plt.legend()
    return plt.show()

def plot_bar_x(elements,elements2):
    # this is for plotting purpose
    index1 = np.arange(len(elements))/1000
    index2 = np.arange(len(elements2))/1000
    plt.plot(index1, elements,'b')
    plt.plot(index2, elements2,'r')
    plt.xlabel('Xcoord', fontsize=10)
    plt.ylabel('Ycoord', fontsize=10)
    plt.title("İnformation Gain for each coordinate (X(blue) and Y(red))")
    plt.legend()

def entropy(firstProb,secondProb):
    entro = -(firstProb*math.log2(firstProb))-(secondProb*math.log2(secondProb)) 
    return entro

def findEntropy(Dataset):
    blue = 0.00001
    red  = 0.00001
    for data in Dataset:
        if(data[2] == 1):
            blue = blue + 1
        if(data[2] == 2):
            red = red + 1
    total = red + blue
    probB = blue / total
    probR = red / total
    res = entropy(probR,probB)
    #print("red value : {} , blue value : {} , total value : {} ".format(red,blue,total))
    #print("prob of red value : {} , prob of blue value : {} ".format(probR,probB))
    return res

def entropyArray(Dataset,coordVal,xORy,dataLen):
    leftList = []
    rightList = []
    if(xORy == 1):
        for data in Dataset:
            if(data[0] <= coordVal):
                leftList.append(data)
            else:
                rightList.append(data)
        leftEntropy = findEntropy(leftList)
        rightEntropy = findEntropy(rightList)
        totalEntropy = (len(leftList)/dataLen)*leftEntropy + (len(rightList)/dataLen)*rightEntropy
    
    else: 
        for data in Dataset:
            if(data[1] <= coordVal):
                leftList.append(data)
            else:
                rightList.append(data)
        leftEntropy = findEntropy(leftList)
        rightEntropy = findEntropy(rightList)
        totalEntropy = (len(leftList)/dataLen)*leftEntropy + (len(rightList)/dataLen)*rightEntropy
        
    return totalEntropy

def findBestCoord(Dataset):
    df = pd.DataFrame(Dataset)
    dataLen = df.shape[0]
    allDataEntropy = findEntropy(Dataset)
    bestXcoord = [0,0,1] # first one x coordinate value and second element is information gain value 
    bestYcoord = [0,0,2] # first one y coordinate value and second element is information gain value 
    igXarray = []
    igYarray = []
    xMaxVal = df.loc[df[0].idxmax()][0]
    xMinVal = df.loc[df[0].idxmin()][0]
    yMaxVal = df.loc[df[1].idxmax()][1]
    yMinVal = df.loc[df[1].idxmin()][1]
    for i in np.arange(xMinVal,xMaxVal,0.001):
        infoGainX = allDataEntropy - entropyArray(Dataset,i,1,dataLen)
        igXarray.append(infoGainX)
        if(infoGainX > bestXcoord[1]):
            bestXcoord[0] = i
            bestXcoord[1] = infoGainX
    for i in np.arange(yMinVal,yMaxVal,0.001):
        infoGainY = allDataEntropy - entropyArray(Dataset,i,2,dataLen)
        igYarray.append(infoGainY)
        if(infoGainY > bestYcoord[1]):
            bestYcoord[0] = i
            bestYcoord[1] = infoGainY
            
    return bestXcoord,bestYcoord,igXarray,igYarray


def removeUnnecessaryData(Dataset,xBest,yBest):
    if(xBest[1] > yBest[1]):
        leftList = []
        rightList = []
        for data in Dataset:
            if(data[1] <= yBest[0]):
                leftList.append(data)
            else:
                rightList.append(data)
        leftEntropy = findEntropy(leftList)
        rightEntropy = findEntropy(rightList)
        if(leftEntropy < rightEntropy):
            Dataset = rightList
        else:
            Dataset = leftList
        
    else:
        leftList = []
        rightList = []
        for data in Dataset:
            if(data[0] <= xBest[0]):
                leftList.append(data)
            else:
                rightList.append(data)
        leftEntropy = findEntropy(leftList)
        rightEntropy = findEntropy(rightList)
        if(leftEntropy < rightEntropy):
            Dataset = rightList
        else:
            Dataset = leftList
    return Dataset


f = open('data.txt', "r")
Dataset = []
num_lines = sum(1 for line in open('data.txt'))
for i in range(num_lines):
    x = f.readline().split(",")
    x = np.array(x)
    y = x.astype(np.float)
    Dataset.append(y)
allDataEntropy = findEntropy(Dataset)
bestX,bestY,igXarr,igYarr = findBestCoord(Dataset) # best information gain coodinates for x and y axis
plot_bar_x(igXarr,igYarr)
datasetNew = removeUnnecessaryData(Dataset,bestX,bestY) # For better ig determination removes unnecessary data from dataset
newBestX,newBestY,igXarr,igYarr = findBestCoord(datasetNew)
plotterScatter(Dataset)


#****************************    BONUS    ***********************************************


from sklearn.tree import DecisionTreeClassifier

df = pd.DataFrame(Dataset)

X = df[[0, 1]]
y = df[2]

tree = DecisionTreeClassifier(max_depth = 3,criterion = "entropy")
tree.fit(X,y)

from sklearn.tree import export_graphviz
export_graphviz(tree,out_file =  "sectree.dot",filled = True)


















