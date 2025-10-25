import torch
import numpy as np
from numpy import arange
import pandas as pd


class SpikingNN(torch.nn.Module):
    def __init__(self):
        self.T = 50
        self.dt = 0.125
        self.time = arange(0, self.T + self.dt, self.dt)
        self.Pref = 0  # resting potential
        self.Pmin = -1  # minimum potential
        self.Pth = 25  # threshold
        self.D = 0.25  # leakage factor
        self.Pspike = 4  # spike potential

        self.count = 0  # refractory counter
        self.t_ref = 5  # refractory period
        self.t_rest = 0

    def loss(self):
        return None

    def update(self, Pn, S):
        for i, t in enumerate(self.time):
            if i == 0:
                Pn[i] = S[i] - self.D
            else:
                if t <= self.t_rest:
                    Pn[i] = self.Pref
                elif t > self.t_rest:
                    if Pn[i - 1] > self.Pmin:
                        Pn[i] = Pn[i - 1] + S[i] - self.D
                    else:
                        Pn[i] = 0
                if Pn[i] >= self.Pth:
                    Pn[i] += self.Pspike
                    self.t_rest = t + self.t_ref
        return Pn


if __name__ == "__main__":
    network = SpikingNN()

    N = 500

    input_vector = np.random.normal(0, 5, N)
    other_input = np.random.normal(0, 4, N)

    for time in range(100):
        P = network.update(input_vector, other_input)
        print("trace", P)
