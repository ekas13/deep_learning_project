import matplotlib.pyplot as plt
import numpy as np

def plot_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss', color='blue', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_samples(samples, path):
    idx, dim, classes = 0, 28, np.sqrt(len(samples)).astype(int)

    # create empty canvas
    canvas = np.zeros((dim * classes, classes * dim))

    # fill with tensors
    for i in range(classes):
        for j in range(classes):
            # Detach the tensor and convert it to a NumPy array
            canvas[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim] = samples[idx].cpu().detach().reshape((dim, dim)).numpy()
            idx += 1

        print(str(i) + ' sample')

    # visualize matrix of tensors as gray scale image
    plt.figure(figsize=(4, 4))
    plt.axis('off')
    plt.imshow(canvas, cmap='gray')
    plt.title('MNIST handwritten digits')
    plt.savefig(path)
