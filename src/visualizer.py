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
            canvas[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim] = samples[idx].reshape((dim, dim))
            idx += 1

        print(str(i) + ' sample')

    # visualize matrix of tensors as gray scale image
    plt.figure(figsize=(4, 4))
    plt.axis('off')
    plt.imshow(canvas, cmap='gray')
    plt.title('MNIST handwritten digits')
    if path:
        plt.savefig(path)
    else:
        plt.show()

def plot_samples_cifar(samples, path):
    idx, dim, classes = 0, 32, np.sqrt(len(samples)).astype(int)

    # create empty canvas
    canvas = np.zeros((dim * classes, classes * dim))

    # fill with tensors
    for i in range(classes):
        for j in range(classes):
            # Detach the tensor and convert it to a NumPy array
            canvas[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim] = samples[idx].reshape((dim, dim))
            idx += 1

        print(str(i) + ' sample')

    # visualize matrix of tensors as gray scale image
    plt.figure(figsize=(4, 4))
    plt.axis('off')
    plt.imshow(canvas, cmap='gray')
    plt.title('MNIST handwritten digits')
    if path:
        plt.savefig(path)
    else:
        plt.show()

if __name__ == "__main__":

  EPOCH = "500"
  samples = np.load(f"experiments/jan_experiments/9_samples_epoch_{EPOCH}.npy")
  # Plot the 25 RGB samples
  fig, axes = plt.subplots(3, 3, figsize=(10, 10))
  # # Iterate through each sample and plot
  
  for i, ax in enumerate(axes.flat):
    sample = samples[i].transpose(1, 2, 0)  # Transpose to (32, 32, 3) for RGB format
    ax.imshow(sample)

    # ax.imshow(samples[i], cmap='gray')
    
    ax.axis('off')  # Turn off axis for cleaner visualization

  plt.savefig(f"{EPOCH}.png")
  plt.tight_layout()
  plt.show()
