import matplotlib.pyplot as plt
import csv
import numpy as np

def plot_accuracy(save_path):
    accuracies = []
    i = 0
    with open(save_path + '/accuracy.csv') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            accuracies.append([i, float(row[0])])
            i += 1

    plt.figure()
    line1 = plt.plot([a[1] for a in accuracies])
    plt.xticks([a[0] for a in accuracies])
    plt.xlabel("Training steps")
    plt.ylabel("Loss/Accuracy")
    plt.legend((line1),("accuracy"))
    plt.show()
