import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset

import common
import coords
import fastmerl

torch.set_default_dtype(torch.float32)

Xvars = ['hx', 'hy', 'hz', 'dx', 'dy', 'dz']
Yvars = ['brdf_r', 'brdf_g', 'brdf_b']
loss_name = 'mean_absolute_logarithmic_error'
batch_size = 512
epochs = 100
verbose = 2
learning_rate = 5e-4
np.random.seed(0)
torch.manual_seed(0)

device = 'cpu'  # 'cuda' or 'cpu'
losses = []


def mean_absolute_logarithmic_error(y_true, y_pred):
    return torch.mean(torch.abs(torch.log(1 + y_true) - torch.log(1 + y_pred)))


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = torch.nn.Linear(in_features=6, out_features=21, bias=True)
        self.fc2 = torch.nn.Linear(in_features=21, out_features=21, bias=True)
        self.fc3 = torch.nn.Linear(in_features=21, out_features=3, bias=True)

        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.zeros_(self.fc3.bias)

        self.fc1.weight = torch.nn.Parameter(torch.zeros((6, 21), dtype=torch.float32).uniform_(-0.05, 0.05).T,
                                             requires_grad=True)
        self.fc2.weight = torch.nn.Parameter(torch.zeros((21, 21), dtype=torch.float32).uniform_(-0.05, 0.05).T,
                                             requires_grad=True)
        self.fc3.weight = torch.nn.Parameter(torch.zeros((21, 3), dtype=torch.float32).uniform_(-0.05, 0.05).T,
                                             requires_grad=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(torch.exp(self.fc3(x)) - 1.0)  # additional relu is max() op as in code in nn.h
        return x


class MerlDataset(Dataset):
    def __init__(self, merlPath, batchsize):
        super(MerlDataset, self).__init__()

        self.bs = batchsize
        self.BRDF = fastmerl.Merl(merlPath)

        self.reflectance_train = generate_nn_datasets(self.BRDF, nsamples=800000, pct=0.8)
        self.reflectance_test = generate_nn_datasets(self.BRDF, nsamples=800000, pct=0.2)

        self.train_samples = torch.tensor(self.reflectance_train[Xvars].values, dtype=torch.float32, device=device)
        self.train_gt = torch.tensor(self.reflectance_train[Yvars].values, dtype=torch.float32, device=device)

        self.test_samples = torch.tensor(self.reflectance_test[Xvars].values, dtype=torch.float32, device=device)
        self.test_gt = torch.tensor(self.reflectance_test[Yvars].values, dtype=torch.float32, device=device)

    def __len__(self):
        return self.train_samples.shape[0]

    def get_trainbatch(self, idx):
        return self.train_samples[idx:idx + self.bs, :], self.train_gt[idx:idx + self.bs, :]

    def get_testbatch(self, idx):
        return self.test_samples[idx:idx + self.bs, :], self.test_gt[idx:idx + self.bs, :]

    def shuffle(self):
        r = torch.randperm(self.train_samples.shape[0])
        self.train_samples = self.train_samples[r, :]
        self.train_gt = self.train_gt[r, :]

    def __getitem__(self, idx):
        pass


def brdf_to_rgb(rvectors, brdf):
    hx = torch.reshape(rvectors[:, 0], (-1, 1))
    hy = torch.reshape(rvectors[:, 1], (-1, 1))
    hz = torch.reshape(rvectors[:, 2], (-1, 1))
    dx = torch.reshape(rvectors[:, 3], (-1, 1))
    dy = torch.reshape(rvectors[:, 4], (-1, 1))
    dz = torch.reshape(rvectors[:, 5], (-1, 1))

    theta_h = torch.atan2(torch.sqrt(hx ** 2 + hy ** 2), hz)
    theta_d = torch.atan2(torch.sqrt(dx ** 2 + dy ** 2), dz)
    phi_d = torch.atan2(dy, dx)
    wiz = torch.cos(theta_d) * torch.cos(theta_h) - \
          torch.sin(theta_d) * torch.cos(phi_d) * torch.sin(theta_h)
    rgb = brdf * torch.clamp(wiz, 0, 1)
    return rgb


def brdf_values(rvectors, brdf=None, model=None):
    if brdf is not None:
        rangles = coords.rvectors_to_rangles(*rvectors)
        brdf_arr = brdf.eval_interp(*rangles).T
    elif model is not None:
        # brdf_arr = model.predict(rvectors.T)        # nnModule has no .predict
        raise RuntimeError("Should not have entered that branch at all from the original code")
    else:
        raise NotImplementedError("Something went really wrong.")
    brdf_arr *= common.mask_from_array(rvectors.T).reshape(-1, 1)
    return brdf_arr


def generate_nn_datasets(brdf, nsamples=800000, pct=0.8):
    rangles = np.random.uniform([0, 0, 0], [np.pi / 2., np.pi / 2., 2 * np.pi], [int(nsamples * pct), 3]).T
    rangles[2] = common.normalize_phid(rangles[2])

    rvectors = coords.rangles_to_rvectors(*rangles)
    brdf_vals = brdf_values(rvectors, brdf=brdf)

    df = pd.DataFrame(np.concatenate([rvectors.T, brdf_vals], axis=1), columns=[*Xvars, *Yvars])
    df = df[(df.T != 0).any()]
    df = df.drop(df[df['brdf_r'] < 0].index)
    return df


model = MLP().to(device)

optim = torch.optim.Adam(model.parameters(),
                         lr=learning_rate,
                         betas=(0.9, 0.999),
                         eps=1e-15,  # eps=None raises error
                         weight_decay=0.0,
                         amsgrad=False)

# read merl brdf:
merlpath = 'alum-bronze.binary'
merl = MerlDataset(merlpath, batchsize=batch_size)

# for saving npy weights
merlname = 'alum-bronze'
outpath = './'

start = time.time()
train_losses = []  # helper, for plotting

for epoch in range(epochs):

    losses = []  # batch-losses per epoch
    merl.shuffle()
    epochStart = time.time()
    num_batches = int(merl.train_samples.shape[0] / batch_size)

    # iterate over batches
    for k in range(num_batches):
        optim.zero_grad()

        # get batch from MERL data, feed into model to get prediction
        mlp_input, groundTruth = merl.get_trainbatch(k * merl.bs)
        output = model(mlp_input).to(device)

        # convert to RGB data
        rgb_pred = brdf_to_rgb(mlp_input, output)
        rgb_true = brdf_to_rgb(mlp_input, groundTruth)

        loss = mean_absolute_logarithmic_error(y_true=rgb_true, y_pred=rgb_pred)
        loss.backward()
        optim.step()
        losses.append(loss.item())

    epoch_loss = sum(losses) / len(losses)
    train_losses.append(epoch_loss)
    print("Epoch {}/{} - Loss {:.7f} - Time: {:.4f}s".format(epoch + 1, epochs,
                                                             epoch_loss,
                                                             time.time() - epochStart))

print("Trained for {} epochs in {} seconds".format(epochs, time.time() - start))

plt.plot(train_losses)
plt.savefig("loss.png")

for el in model.named_parameters():
    param_name = el[0]   # either fc1.bias or fc1.weight
    weights = el[1]
    segs = param_name.split('.')
    if segs[-1] == 'weight':
        param_name = segs[0]
    else:
        param_name = segs[0].replace('fc', 'b')

    filename = '_{}.npy'.format(param_name)
    filepath = os.path.join(outpath, filename)
    curr_weight = weights.detach().cpu().numpy().T  # transpose bc mitsuba code was developed for TF convention
    np.save(filepath, curr_weight)
