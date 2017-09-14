import numpy as np
import matplotlib.pyplot as plt


def calculate_probability(instances, means):
    """
    Given a matrix consisting of rows of instances and columns of dimensions calculate the probability according
    to a single exponential distribution.
    """
    shifted_instances = instances - means
    distances = np.sqrt(np.sum(shifted_instances ** 2, axis=1).reshape((-1, 1)))
    return np.exp(-distances)


def calculate_responsibilities(instances, means, mixtures):
    """
    Calculates the responsibilities for each instance based on the mean and mixture coefficients
    """
    probabilities = np.zeros((instances.shape[0], means.shape[0]))

    for class_index in range(means.shape[0]):
        probabilities[:, class_index] = (calculate_probability(instances, means[class_index, :]) * mixtures[class_index]).reshape((1, -1))

    probabilities /= probabilities.sum(axis=1).reshape((-1, 1)) * mixtures[class_index]

    return probabilities


def update_parameters(responsibilities, instances, means, mixtures):
    """
    returns the new means and mixture co-efficients based on math
    """
    new_mixes = np.zeros(mixtures.shape)
    new_means = np.zeros(means.shape)

    for class_index in range(means.shape[0]):
        class_responsibility = responsibilities[:, class_index].reshape((-1, 1))
        instance_distance_from_mean = np.abs(instances - means[class_index, :])
        instance_distance_from_mean += (instance_distance_from_mean == 0) * 0.000001
        numerator = class_responsibility * instances / instance_distance_from_mean
        denominator = class_responsibility / instance_distance_from_mean

        new_mean = numerator.sum(axis=0) / denominator.sum(axis=0)
        mix = class_responsibility.sum() / responsibilities.sum()

        new_mixes[class_index, :] = mix.reshape((1, -1))
        new_means[class_index, :] = new_mean.reshape((1, -1))

    return new_means, new_mixes


def run_em(k=2, instance_file_name='data.csv'):
    instances = np.loadtxt(instance_file_name)
    d = instances.shape[1]

    means = (np.random.rand(k, d) - 0.5) * 20
    mixes = np.random.rand(k).reshape(k, -1)
    mixes /= mixes.sum()

    x_coords = [[] for _ in range(k)]
    y_coords = [[] for _ in range(k)]

    for iteration in range(100):
        for i in range(means.shape[0]):
            x_coords[i].append(means[i, 0])
            y_coords[i].append(means[i, 1])

        if iteration % 10 == 0:
            print("Iteration %d" % iteration)
        means, mixes = update_parameters(calculate_responsibilities(instances, means, mixes), instances, means, mixes)

    plt.figure()
    plt.scatter(instances[:, 0], instances[:, 1])
    for i in range(k):
        plt.plot(x_coords[i], y_coords[i], color="C%d" % (i + 1))

    plt.show(block=False)
