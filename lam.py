import numpy as np
import numpy.linalg as lin
import pickle as pickle
import matplotlib.pyplot as plt

num_features = 10
num_actions = 2

file = '../pong/safe_samples/safe_samples.pickle'
with open(file, 'rb') as handler:
    safe_samples = pickle.load(handler)
image_width, image_height = safe_samples['state'][0].shape


file = '../pong/objects'
with open(file, 'rb') as handler:
    objects = pickle.load(handler)
feature_images = np.zeros((num_features, image_width, image_height))
feature_images[0] = np.ones((image_width, image_height)) * objects['ball']['color']
feature_images[1] = np.ones((image_width, image_height)) * 144 #bk color
feature_images[2] = np.ones((image_width, image_height)) * objects['me']['color']
feature_images[3] = np.ones((image_width, image_height)) * objects['opponent']['color']
plt.figure(1)
plt.imshow(safe_samples['state'][0] - feature_images[0])
plt.figure(2)
plt.imshow(safe_samples['state'][0] - feature_images[1])
plt.figure(3)
plt.imshow(safe_samples['state'][0] - feature_images[2])
plt.show()

def phi(img):
    assert(feature_images.shape[0] == num_features)
    ps = np.zeros((num_features, 1))
    for i in range(num_features):
        ps[i] = np.exp(lin.norm(img - feature_images[i]))
    return ps

ps1 = phi(np.zeros((image_width, image_height)))
print(ps1)

# https://drive.google.com/file/d/0B4yKRDF2fvC8S0hBeUlrX1lDa2c/view
def ls_model(S, A, Snext, R):
    assert(S.shape[0] ==  len(A))
    assert(len(A) == len(R))
    assert(S.shape == Snext.shape)
    N = S.shape[0]

    H = np.zeros((num_actions, num_features, num_features))
    E = np.zeros((num_actions, num_features, num_features))
    e = np.zeros((num_actions, num_features))
    for i in range(N):
        ps = phi(S[i])
        psnext = phi(Snext[i])
        a = A[i]
        H[a] += ps * ps.T
        E[a] += psnext * ps.T
        e[a] += ps * R[i]
    F = np.zeros((num_actions, num_features, num_features))
    f = np.zeros((num_actions, num_features))
    for a in range(num_actions):
        F[a] = E[a] * np.linalg.pinv(H[a])
        f[a] = np.linalg.solve(E[a], e[a])
    return F, f


gamma = 0
def lam_lstd(F, f, Phi):
    w = np.random.randn(f.shape[0])
    A = np.zeros(num_features, num_features)
    b = np.zeros(num_features)
    for ps in Phi:
        Qmax = 0.0
        abest = -1
        for a in range(F.shape[0]):
            Q = f[a].dot(ps) + gamma * w.dot(F[a] * ps)
            if Q > Qmax:
                Qmax = Q
                abest = a
        phi_next = F[abest] * ps
        r = f[a].dot(ps)
        A += ps * (gamma*phi_next - ps)
        b += ps * r
    w = - np.linalg.solve(A, b)
    return w