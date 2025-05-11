from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # Your implementation goes here!
    loaded = np.load(filename)
    meanLoad = np.mean(loaded, axis=0)
    center = loaded - meanLoad
    return center
    #raise NotImplementedError

def get_covariance(dataset):
    # Your implementation goes here!
    transpose = np.transpose(dataset)
    point = (np.dot(transpose, dataset) / (dataset.size - 1))
    print(point.shape)
    return point
    #raise NotImplementedError

def get_eig(S, k):
    # Your implementation goes here!
    val, vect = eigh(S, subset_by_index = [len(S) - k, len(S) - 1])
    print(len(S), S.shape)
    val = np.flip(val)
    val = np.diag(val)
    vect = np.fliplr(vect)
    return val, vect
    #raise NotImplementedError

def get_eig_prop(S, prop):
    # Your implementation goes here!
    summation = np.trace(S)
    val, vect = eigh(S, subset_by_value= [summation * prop, np.inf])
    val = np.flip(val)
    val = np.diag(val)
    vect = np.fliplr(vect)
    return val, vect
    #raise NotImplementedError

def project_and_reconstruct_image(image, U):
    # Your implementation goes here!
    project = np.zeros(len(image))
    for i in range(len(U[0])):
        project += np.dot(U[:, i], image) * (U[:, i])
    return project
    #raise NotImplementedError

def display_image(im_orig_fullres, im_orig, im_reconstructed):
    # Please use the format below to ensure grading consistency
    high_orig = im_orig_fullres.reshape(218, 178, 3)
    orig = im_orig.reshape(60, 50)
    recon = im_reconstructed.reshape(60, 50)
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(9,3), ncols=3)
    image1 = ax1.imshow(high_orig, cmap='Greens_r', aspect='equal')
    image2 = ax2.imshow(orig, cmap='Greens_r', aspect='equal')
    image3 = ax3.imshow(recon, cmap='Greens_r', aspect='equal')
    bar1 = fig.colorbar(image1, ax=ax1)
    bar2 = fig.colorbar(image2, ax=ax2)
    bar3 = fig.colorbar(image2, ax=ax3)
    fig.tight_layout()
    ax1.set_title("Original High Res")
    ax2.set_title("Original")
    ax3.set_title("Reconstructed")
    # Your implementation goes here!
    return fig, ax1, ax2, ax3

def perturb_image(image, U, sigma):
    # Your implementation goes here!
    weights = np.dot(U.T, image)
    pert = np.random.normal(0, sigma, weights.shape)
    pert_weight = weights + pert
    pert_image = np.dot(U, pert_weight)


    return pert_image
    #raise NotImplementedError
