
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class Linspace(object):
    def __init__(self, min, max, resolution):
        self.min = min
        self.max = max
        self.resolution = resolution

    def normalize(self, x):
        return (x - self.min)/(self.max-self.min)

    def denormalize(self, x):
        return x*(self.max-self.min)+self.min

    @property
    def extent(self):
        return (self.min, self.max)

    @property
    def edges(self):
        return np.linspace(self.min, self.max, self.resolution + 1)

    @property
    def centers(self):
        edges = self.edges
        return (edges[:-1] + edges[1:])/2


class Grid():
    def __init__(self, latitude, longitude):
        # Define the grid
        self.latitude = latitude
        self.longitude = longitude

    def center_mesh(self):
        return np.meshgrid(self.latitude.centers, self.longitude.centers)

    def normalize(self, latitude, longitude):
        return (self.latitude.normalize(latitude),
                self.longitude.normalize(longitude))

    def denormalize(self, latitude, longitude):
        return (self.latitude.denormalize(latitude),
                self.longitude.denormalize(longitude))


class Blobs(object):
    def __init__(self, n_centers, seed=1):
        self.n_centers = n_centers

        # Use RandomState so the behavior is repeatable.
        self.rng = np.random.RandomState(seed)

        # The price data will be a sum of Gaussians, at random locations.
        self.centers = self.rng.rand(self.n_centers, 2)  # shape: (centers, dimensions)

        # Each Gaussian has a maximum price contribution, at the center.
        self.price_delta = 0.5 + 2 * self.rng.rand(self.n_centers)

        # Each Gaussian also has a standard-deviation and variance.
        self.std = 0.2 * self.rng.rand(self.n_centers)  # shape: (centers)

    @property
    def var(self):
        return self.std ** 2

    def __call__(self, x, y):
        # Cache the shape, and flatten the inputs.
        shape = x.shape
        assert y.shape == x.shape
        x = x.flatten()
        y = y.flatten()

        # Convert x, y examples into an array with shape (examples, dimensions)
        xy = np.array([x, y]).T

        # Calculate the square distance from each example to each center.
        # shape: (examples, centers, dimensions)
        components2 = (xy[:, None, :] - self.centers[None, :, :]) ** 2
        r2 = components2.sum(axis=2)  # shape: (examples, centers)

        # Calculate the z**2 for each example from each center.
        z2 = r2 / self.var[None, :]
        result = (np.exp(-z2) * self.price_delta).sum(1)  # shape: (examples,)

        # Restore the original shape.
        return result.reshape(shape)

class GridPlotter(object):
    def __init__(self, grid, vmin, vmax):
        self.grid = grid
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, values):
        plt.imshow(
            values,
            # The color axis goes from `price_min` to `price_max`.
            vmin=self.vmin, vmax=self.vmax,
            # Put the image at the correct latitude and longitude.
            extent=self.grid.longitude.extent+self.grid.longitude.extent,
        )
