from sys import argv
from matplotlib import pyplot as plt
import os

res_path = argv[1]
acc1 = []
acc5 = []
with open(res_path) as f:
    lines = f.readlines()
    for line in lines:
        l = line.split(" ")
        acc1.append(float(l[6][:-1]))
        acc5.append(float(l[9]))

x = []
for i, a in enumerate(acc1):
    x.append(i+1)
plt.plot(x, acc1, color='g', label='acc1')
plt.plot(x, acc5, color='r', label='acc5')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()

plt.title('local epoch : 3, client : 10, fraction : 0.5')
plt.savefig('result.png')
