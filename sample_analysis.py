import pickle as pickle
import numpy as np
import sys
import matplotlib.pyplot as plt

file_header = 'samples_episodes/samples_random_policy_small.pickle'

IMAGE_TO_DISPLAY = 26

with open(file_header+'1', 'rb') as handle:
    samples = pickle.load(handle)

image_width, image_height = samples['states'][0][0].shape
print('image_width, image_height=%d, %d' %(image_width, image_height))
rewards = samples['rewards']
unique_rewards, rewards_count = np.unique(np.hstack(rewards), return_counts=True)
print(rewards_count.shape)
# num_negatives, num_zeros, num_positives = rewards_count[0], rewards_count[1], rewards_count[2]

states = samples['states']

assert(len(states) == len(rewards))
image_size = (image_width, image_height)

def display(img):
    one_image = img.reshape(image_width, image_height)
    plt.axis('off')
    plt.imshow(one_image)


img_has_ball = states[0][IMAGE_TO_DISPLAY]
img_unchanged = np.copy(img_has_ball)
print(img_has_ball.shape)
display(img_unchanged)

img_column = np.reshape(img_has_ball,  image_width * image_height)
uniq_colors, color_count = np.unique(img_has_ball, return_counts=True)
print(uniq_colors, color_count)

my_color = 92 #[92, 186, 92]
background_color = 144#[144, 72, 17]
ball_color = 236#[236, 236, 236]
opponent_color = 213#[213, 130, 74]

display(img_has_ball)

# measure the size of each objects
def find_start_ro_column(object_color):
    for row in range(img_has_ball.shape[0]):
        for col in range(img_has_ball.shape[1]):
            if np.abs(img_has_ball[row, col] - object_color) < 1e-8:
                return row, col

def find_row_size_of_object(object_color):
    r_start, c_start = find_start_ro_column(object_color)
    row_size = 0
    for x in range(r_start, img_has_ball.shape[0]):
        if np.abs(img_has_ball[x, c_start] - object_color) < 1e-8:
            row_size += 1
        else:
            break
    return row_size

def find_col_size_of_object(object_color):
    r_start, c_start = find_start_ro_column(object_color)
    col_size = 0
    for y in range(c_start, img_has_ball.shape[1]):
        if np.abs(img_has_ball[r_start, y]  - object_color) < 1e-8:
            col_size += 1
        else:
            break
    return col_size

def find_object_size(object_color):
    return find_row_size_of_object(object_color), find_col_size_of_object(object_color)

ball_rows, ball_cols = find_object_size(ball_color)
print('ball_rows = %d, ball_columns=%d ' %(ball_rows, ball_cols))

my_rows, my_cols = find_object_size(my_color)
print('my_rows = %d, my_columns=%d ' %(my_rows, my_cols))

opponent_rows, opponent_cols = find_object_size(opponent_color)
print('opponent rows = %d, opponent columns=%d ' %(opponent_rows, opponent_cols))


# erase object from image
def erase_object(color, row_start, num_rows, col_start, num_cols):
    for r in range(row_start, row_start + num_rows):
        for c in range(col_start, col_start + num_cols):
            img_has_ball[r, c] = background_color

ball_row_start, ball_col_start = find_start_ro_column(ball_color)
erase_object(ball_color, ball_row_start, ball_rows, ball_col_start, ball_cols)

my_row_start, my_col_start = find_start_ro_column(my_color)
erase_object(my_color, my_row_start, my_rows, my_col_start, my_cols)

opponent_row_start, opponent_col_start = find_start_ro_column(opponent_color)
erase_object(opponent_color, opponent_row_start, opponent_rows, opponent_col_start, opponent_cols)

with open('background.png', 'wb') as handler:
    pickle.dump(img_has_ball, handler)

objects = {
    'ball': {'color': ball_color, 'rows': ball_rows, 'cols': ball_cols},
    'me': {'color': my_color, 'rows': my_rows, 'cols': my_cols},
    'opponent': {'color': opponent_color, 'rows': opponent_rows, 'cols': opponent_cols}
}
with open('objects', 'wb') as handler:
    pickle.dump(objects, handler)


