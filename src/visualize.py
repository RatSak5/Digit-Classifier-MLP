import matplotlib.pyplot as plt


def see(pixels):
    pixels = pixels.view(28, 28)
    plt.matshow(pixels, cmap='gray')
    plt.show()


def plot_confusion_matrices(matrices, titles):
    # Matrices shown in the form of grey-scale images
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    axes = axes.ravel()

    showmat = []
    for mat in matrices:
        mat = mat.clone()
        mat.fill_diagonal_(0)
        mat = mat / mat.sum(dim=1, keepdim=True)
        showmat.append(mat)

    for ax, mat, title in zip(axes, showmat, titles):
        im = ax.matshow(mat, cmap=plt.cm.gray)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()
