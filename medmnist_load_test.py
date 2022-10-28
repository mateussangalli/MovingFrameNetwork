import numpy as np
import plotly.graph_objects as go

dataset = np.load('medmnist/adrenalmnist3d.npz')

print(dataset)
for k in dataset.keys():
    print(k)

train_images = dataset['train_images']
train_labels = dataset['train_labels']

print(np.unique(train_labels))

print(train_images.shape)
for i in np.unique(train_labels):
    volume = train_images[train_labels[:, 0] == i, ...]
    volume = train_images[np.random.randint(0, volume.shape[0]), ...]

    X, Y, Z = np.meshgrid(np.linspace(-1., 1., volume.shape[0]),
                          np.linspace(-1., 1., volume.shape[1]),
                          np.linspace(-1., 1., volume.shape[2]))
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=volume.flatten(),
        isomin=volume.min(),
        isomax=volume.max(),
        opacity=0.1,  # needs to be small to see through all surfaces
        surface_count=20,  # needs to be a large number for good volume rendering
    ))
    fig.show()
