import numpy as np
import matplotlib.pyplot as plt


def generate_data(k=2, d=2, file_name='data.csv'):
    means = (np.random.rand(k, d) - 0.5) * 20
    mixing = np.random.rand(k).reshape(k, -1)
    mixing /= mixing.sum()

    element_count = 1000
    element_size = (element_count, d)

    values_mean_0 = np.random.exponential(1, element_size) * (np.random.randint(0, 2, element_size) - 0.5) * 2

    count = np.round((mixing * element_size[0]))

    samples = np.zeros(element_size)

    index = 0
    for classIndex, [amount] in enumerate(count):
        start = index
        end = int(start + amount)
        index = end

        data = values_mean_0[start:end, :] + means[classIndex, :]

        samples[start:end, :] = data

    np.savetxt(file_name, samples)

    plt.figure()
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.scatter(means[:, 0], means[:, 1])
    plt.show(block=False)
