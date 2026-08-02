import numpy as n

conv1 = n.random.uniform(-1, 1, size=(32, 7 ,10))
conv2 = n.random.uniform(-1, 1, size=(64, 5, 32))
conv3 = n.random.uniform(-1, 1, size=(64, 3, 64))

n.save(r"NN1\conv1.npy", conv1)
n.save(r"NN1\conv2.npy", conv2)
n.save(r"NN1\conv3.npy", conv3)

biases = n.random.uniform(-1, 1, size=(4, 32))
weights = n.random.uniform(-1, 1, size=(4, 32, 106))
final = n.random.uniform(-1, 1, size=(4, 32))
fbias = n.random.uniform(-1, 1, size=(4))

n.save(r"NN1\weights.npy", weights)
n.save(r"NN1\biases.npy", biases)
n.save(r"NN1\final.npy", final)
n.save(r"NN1\fbias.npy", fbias)