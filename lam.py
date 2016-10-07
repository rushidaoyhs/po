import numpy as np
import numpy.linalg as lin
import pickle as pickle
import matplotlib.pyplot as plt

file = './safe_samples/safe_samples.pickle'
with open(file, 'rb') as handler:
    safe_samples = pickle.load(handler)
image_width, image_height = safe_samples['state'][0].shape

num_features = image_width + image_height
num_actions = 2

file = 'objects'
with open(file, 'rb') as handler:
    objects = pickle.load(handler)
print(objects)

bk_color = 144
bk_image = np.ones((image_width, image_height), dtype = int) * 144
def phi_sum_row(img):
    ps = np.zeros((image_height))
    for r in range(image_height):
        if np.max(img[r, :]) == 0:
            feat = np.min(img[r, :])
        else:
            feat = np.max(img[r, :])
        ps[r] = feat
    return ps


def phi_sum_col(img):
    ps = np.zeros((image_width))
    for c in range(image_width):
        if np.max(img[:, c]) == 0:
            feat = np.min(img[:, c])
        else:
            feat = np.max(img[:, c])
        ps[c] = feat
    return ps

def phi_sum(img):
    img -= bk_image
    return np.hstack([phi_sum_row(img), phi_sum_col(img)])


states = safe_samples['state']
num_samples = states.shape[0]
# Phi = np.zeros((num_samples, num_features))
# for i in range(num_samples):
#     Phi[i] = phi_sum(states[i])
# print(Phi.shape)
image = safe_samples['state'][0]
# ps_r = phi_sum_row(image)
# print(np.nonzero(ps_r))
# ps_c = phi_sum_col(image)
# print(np.nonzero(ps_c))
ps = phi_sum(image)
# print(np.nonzero())
# print(phi_sum(image).shape)
print(np.count_nonzero(ps))

# import  sample_analysis
# sample_analysis.display(image)
# plt.show()

