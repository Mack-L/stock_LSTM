import numpy as n

conv1 = n.random.uniform(-1, 1, size=(32, 7 ,10))
conv2 = n.random.uniform(-1, 1, size=(64, 5, 32))
conv3 = n.random.uniform(-1, 1, size=(64, 3, 64))

n.save(r"MLdata\conv1.npy", conv1)
n.save(r"MLdata\conv2.npy", conv2)
n.save(r"MLdata\conv3.npy", conv3)

biases = n.random.uniform(-1, 1, size=(4, 32))
weights = n.random.uniform(-1, 1, size=(4, 32, 106))
final = n.random.uniform(-1, 1, size=(4, 32))
fbias = n.random.uniform(-1, 1, size=(4))

n.save(r"MLdata\weights.npy", weights)
n.save(r"MLdata\biases.npy", biases)
n.save(r"MLdata\final.npy", final)
n.save(r"MLdata\fbias.npy", final)