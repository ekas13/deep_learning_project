import matplotlib.pyplot as plt

def plot_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss', color='blue', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Time')
    plt.legend()
    plt.grid(True)
    plt.show()