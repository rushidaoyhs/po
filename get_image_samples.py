import matplotlib.pyplot as plt
import matplotlib.image as matimage
import pickle as pickle
import numpy as np

with open('background.png', 'rb') as handler:
    background = pickle.load(handler)
print(background.shape)
plt.imshow(background)

with open('objects', 'rb') as handler:
    objects = pickle.load(handler)
print(objects)


def has_a_collision(locations, objects):
    for o1 in objects.keys():
        for o2 in objects.keys():
            if o2 is o1: continue
            if two_collide(objects[o1], locations[o1], objects[o2], locations[o2]):
                return True
    return False


def two_collide(obj1, loc1, obj2, loc2):
    return loc1['start_row'] <= loc2['start_row'] <= loc1['start_row'] + obj1['rows'] and \
           loc1['start_col'] <= loc2['start_col'] <= loc1['start_col'] + obj1['cols']


def exchange_me_opponent_if_needed(locs):
    if locs['me']['start_col'] < locs['opponent']['start_col']:
        tmp = locs['opponent']['start_col']
        locs['opponent']['start_col'] = locs['me']['start_col']
        locs['me']['start_col'] = tmp
    return locs

def non_colliding_locations(background_img):
    collided = True
    locations = {}
    while collided:
        for obj in objects.keys():
            start_r = np.random.random_integers(0, background_img.shape[0] - objects[obj]['rows'])
            start_c = np.random.random_integers(0, background_img.shape[1] - objects[obj]['cols'])
            locations[obj] = {'start_row': start_r, 'start_col': start_c}
        collided = has_a_collision(locations, objects)
        exchange_me_opponent_if_needed(locations)
    return locations


def rand_sample(background_img):
    img = background_img.copy()
    locations = non_colliding_locations(background_img)
    for obj in objects.keys():
        for r in range(locations[obj]['start_row'], locations[obj]['start_row'] + objects[obj]['rows']):
            for c in range(locations[obj]['start_col'], locations[obj]['start_col'] + objects[obj]['cols']):
                img[r, c] = objects[obj]['color']
    if locations['ball']['start_col'] >= locations['me']['start_col'] + objects['me']['cols']:
        reward = 0 #-1.0
    elif locations['ball']['start_col'] + objects['ball']['cols'] <= locations['opponent']['start_col']:
        reward = 2 #1.0
    else:
        reward = 1 #0.0
    return img, reward


# plt.figure(2)
# sample = draw(ball_color, ball_rows, ball_cols, start_r, start_c)
# sample = draw(my_color, my_rows, my_cols, start_r, start_c)
# background_test = np.ones((40, 30, 3), dtype=np.float32) * 255
# plt.imshow(background_test)
# plt.show()
# sample, reward = rand_sample(background)
# plt.imshow(sample)
# plt.show()

num_samples = 10000
width, height = background.shape[0], background.shape[1]
print(background.shape)
background_1 = background[:,:]
print(background_1.shape)
safe_samples_x = np.zeros((num_samples, width, height), dtype=np.int)
safe_samples_y = np.zeros(num_samples, dtype=np.int)

import time
t0=time.clock()
for i in range(num_samples):
    print('i=%d' % i)
    safe_samples_x[i, :, :], safe_samples_y[i] = rand_sample(background_1)
safe_samples = {
    'state': safe_samples_x,
    'reward': safe_samples_y
    }

file = './safe_samples/safe_samples.pickle'
with open(file, 'wb') as handler:
    pickle.dump(safe_samples, handler)
print(time.clock() - t0)

