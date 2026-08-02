import numpy as n
import random as r
eta = 0.01

data = n.load(r"MLdata\data.npy")
weights = n.load(r"NN1\weights.npy")
forgets = weights[0]
inputs = weights[1]
candidates = weights[2]
outputs = weights[3]
biases = n.load(r"NN1\biases.npy") # f,i,c,o
conv1 = n.load(r"NN1\conv1.npy")
conv2 = n.load(r"NN1\conv2.npy")
conv3 = n.load(r"NN1\conv3.npy")
finalmat = n.load(r"NN1\final.npy")
finalbias = n.load(r"NN1\fbias.npy")

length = len(data)
n.seterr(over='ignore')

def sigmoid(x):
    return n.where(x >= 15, 1.0, n.where(x <= -15, 0.0, 1 / (1 + n.exp(-x))))

def tanh(x):
    return n.tanh(x)

def leakyrelu(x):
    return n.where(x < 0, 0.01*x, x)

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
    return leakyrelu(output)
    #


def LSTM(ct, ht, xt):
    hxt = n.concatenate((ht,xt))
    ft = sigmoid(forgets @ hxt + biases[0])
    it = sigmoid(inputs @ hxt + biases[1])
    ot = sigmoid(outputs @ hxt + biases[2])
    cat = tanh(candidates @ hxt + biases[3])
    ct1 = ct*ft + it*cat
    ht1 = tanh(ct1)*ot
    return ct1, ht1, ft, it, cat, ot, hxt
    

def getrues(future):
    true = []
    current = future[0][3]
    short = future[1][3]
    long = future[-1][3]
    if current < short:
        true.append(1.0)
    elif short < current:
        true.append(-1.0)
    else:
        true.append(0.0)
    true.append(abs(10*((short-current)/current)))
    if current < long:
        true.append(1.0)
    elif long < current:
        true.append(-1.0)
    else:
        true.append(0.0)
    true.append(abs(10*((long-current)/current)))
    return n.array(true)

runs = 0
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
    portion = 10*r.randint(10,118)

subdata = initial[portion-100:portion] #100 rows of 10

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
zeds = n.zeros((100, 106), dtype=n.float64)


for i in range(0,100):
    cells[i+1], hiddens[i+1], forgots[i], inps[i], cands[i], outs[i], zeds[i] = LSTM(cells[i], hiddens[i], layerfull[i])

final32 = hiddens[100]
answer = finalmat @ final32 + finalbias # short(prob, mag), long(prob, mag)
print(answer)
## end of forward pass

trues = getrues(initial[portion-1:portion+24]) # 25 rows of 10
print(trues)
cost = n.sum((answer-trues)**2)
print(cost)

# backprop
if runs == 0:
    conv1_g = n.zeros((32, 7 ,10), dtype=n.float64)
    conv2_g = n.zeros((64, 5, 32), dtype=n.float64)
    conv3_g = n.zeros((64, 3, 64), dtype=n.float64)
    biases_g = n.zeros((4, 32), dtype=n.float64)
    weights_g = n.zeros((4, 32, 106), dtype=n.float64) #f,i,c,o
    finalmat_g = n.zeros((4, 32), dtype=n.float64)
    finalbias_g = n.zeros((4), dtype=n.float64)


deltaf = answer - trues #dC/d answer
for i in range(0,4):
    for j in range(0,32):
        finalmat_g[i][j] -= eta* deltaf[i] * final32[j]
    finalbias_g[i] -= eta* deltaf[i]

finalmatT = finalmat.transpose()
deltaf32 = finalmatT @ deltaf #dc/d final32    length = 32

dht1 = deltaf32
dct1 = n.zeros((32), dtype=n.float64)

#LSTM backprop
for t in range(99,-1,-1): # cells, hiddens, forgots, inps, cands, outs
    dot = dht1 * tanh(cells[t+1])
    dct1 += dht1 * outs[t] * (1 - (tanh(cells[t+1]))**2)
    dft = cells[t] * dct1
    dit = cands[t] * dct1
    dcat = inps[t] * dct1
    dct1 = forgots[t] * dct1
    dpot = dot * outs[t] * (1-outs[t])
    dpft = dft * forgots[t] * (1-forgots[t])
    dpit = dit * inps[t] * (1-inps[t])
    dpcat = dcat * cands[t] * (1-cands[t])
    print(t)
    
    for i in range(0,32):
        for j in range(0,106):
            weights_g[0][i][j] -= eta* dpft[i] * zeds[t][j]
            weights_g[1][i][j] -= eta* dpit[i] * zeds[t][j]
            weights_g[2][i][j] -= eta* dpcat[i] * zeds[t][j]
            weights_g[3][i][j] -= eta* dpot[i] * zeds[t][j]
            
    forgetsT = forgets.transpose()
    inputsT = inputs.transpose()
    candidatesT = candidates.transpose()
    outputsT = outputs.transpose()
    
    dzt = forgetsT @ dpft + inputsT @ dpit + candidatesT @ dpcat + outputsT @ dpot
    dht1 = dzt[:32]
#

##conv backprop

#
print("done")

