import matplotlib.pyplot as plt

# functional placeholder - will be fixed in later versions

def plot_tensor(tensor_float):
    plt.imshow(tensor_float.numpy().reshape(28, 28), cmap='grey')
    plt.show()