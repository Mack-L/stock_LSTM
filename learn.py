import numpy as n
import random as r

data = n.load(r"MLdata\data.npy")
weights = n.load(r"MLdata\weights.npy")
forgets = weights[0]
inputs = weights[1]
candidates = weights[2]
outputs = weights[3]
biases = n.load(r"MLdata\biases.npy") # f,i,c,o
conv1 = n.load(r"MLdata\conv1.npy")
conv2 = n.load(r"MLdata\conv2.npy")
conv3 = n.load(r"MLdata\conv3.npy")
finalmat = n.load(r"MLdata\final.npy")
finalbias = n.load(r"MLdata\fbias.npy")

length = len(data)
n.seterr(over='ignore')

def sigmoid(x):
    return n.where(x >= 15, 1.0, n.where(x <= 15, 0.0, 1 / (1 + n.exp(-x))))

def tanh(x):
    return n.tanh(x)

def convlayer(inputd, weights):
    channels = len(inputd[0])
    length = len(inputd)
    k = len(weights[0])
    outchannels = len(weights)
    pad = n.zeros((k//2, channels), dtype=n.float64)
    padded = n.concatenate((pad, inputd, pad))
    output = n.empty((length, outchannels), dtype=n.float64)
    for i in range(0,length):
        for j in range(0,outchannels):
            filterw = weights[j]
            section = padded[i:i+k]
            output[i][j] = n.sum(filterw * section)  # FIP
    return output
    #


def LSTM(ct, ht, xt):
    hxt = n.concatenate((ht,xt))
    ft = sigmoid(forgets @ hxt + biases[0])
    it = sigmoid(inputs @ hxt + biases[1])
    ot = sigmoid(outputs @ hxt + biases[2])
    cat = tanh(candidates @ hxt + biases[3])
    ct1 = ct*ft + it*cat
    ht1 = tanh(ct1)*ot
    return ct1, ht1, ft, it, cat, ot
    


stockcompleted = []
#while len(stockcompleted) != length:
stock = r.randint(0,length)
while stock in stockcompleted:
    stock = r.randint(0,length)
initial = data[stock]
completed = []

#while len(completed) != 110:
#
#
portion = 10*r.randint(10,120)
while portion in completed:
    portion = 10*r.randint(10,120)

subdata = initial[portion-100:portion]

# forward pass
layer1 = convlayer(subdata, conv1)
layer2 = convlayer(layer1, conv2)
layer3 = convlayer(layer2, conv3)
layerfull = n.concatenate((layer3, subdata), axis = 1)# 100rows of 74

cells = n.zeros((101, 32), dtype=n.float64)
hiddens = n.zeros((101, 32), dtype=n.float64)
forgots = n.zeros((100, 32), dtype=n.float64)
inps = n.zeros((100, 32), dtype=n.float64)
cands = n.zeros((100, 32), dtype=n.float64)
outs = n.zeros((100, 32), dtype=n.float64)


for i in range(0,100):
    cells[i+1], hiddens[i+1], forgots[i], inps[i], cands[i], outs[i] = LSTM(cells[i], hiddens[i], layerfull[i])

answer = finalmat @ hiddens[100] #+ finalbias
print(answer)
## end of forward pass

