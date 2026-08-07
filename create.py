import numpy as n

conv1 = n.random.uniform(-1, 1, size=(32, 7 ,10))
conv2 = n.random.uniform(-1, 1, size=(64, 5, 32))
conv3 = n.random.uniform(-1, 1, size=(64, 3, 64))
convbias = n.random.uniform(-1, 1, size=(5, 32))

biases = n.random.uniform(-1, 1, size=(4, 32))
weights = n.random.uniform(-1, 1, size=(4, 32, 106))
final = n.random.uniform(-1, 1, size=(6, 32))
fbias = n.random.uniform(-1, 1, size=(6))


n.save(r"NN1\conv1.npy", conv1)
n.save(r"NN1\conv2.npy", conv2)
n.save(r"NN1\conv3.npy", conv3)
n.save(r"NN1\convbias.npy", convbias)
n.save(r"NN1\weights.npy", weights)
n.save(r"NN1\biases.npy", biases)
n.save(r"NN1\final.npy", final)
n.save(r"NN1\fbias.npy", fbias)

n.save(r"NN2\conv1.npy", conv1.copy())
n.save(r"NN2\conv2.npy", conv2.copy())
n.save(r"NN2\conv3.npy", conv3.copy())
n.save(r"NN2\convbias.npy", convbias.copy())
n.save(r"NN2\weights.npy", weights.copy())
n.save(r"NN2\biases.npy", biases.copy())
n.save(r"NN2\final.npy", final.copy())
n.save(r"NN2\fbias.npy", fbias.copy())