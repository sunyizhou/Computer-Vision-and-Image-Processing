import numpy as np
from PIL import Image
from multiprocessing import Pool
from functools import partial
from utils import load_images
import time

######################
### integral_img

def convert_to_integral_img(img):
    col_sum = img
    integral_img = np.zeros((img.shape[0] + 1, img.shape[1] + 1))

    #print(img)
    for x in range(1, img.shape[0]):
        for y in range(0, img.shape[1]):
            col_sum[x, y] = col_sum[x-1, y] + img[x, y]

    #print(col_sum)
    for x in range(1, img.shape[0]+1):
        for y in range(1, img.shape[1]+1):
            integral_img[x, y] = integral_img[x, y-1] + col_sum[x-1, y-1]
    
    return integral_img


def get_region_sum(integral_img, top_left, bottom_right):
    if top_left == bottom_right:
        return 0
    
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])
    return integral_img[bottom_right[0], bottom_right[1]] - integral_img[top_right[0], top_right[1]] \
                - integral_img[bottom_left[0], bottom_left[1]] + integral_img[top_left[0] ,top_left[1]]


#########################
### haar feature
    
def get_score(img, feature_type, top_left, width, height):
    score = 0
    bottom_right = (top_left[0] + height, top_left[1] + width)

    if feature_type == "TWO_VERTICAL":
        white = get_region_sum(img, top_left, (int(top_left[0] + height/2), top_left[1] + width))
        black = get_region_sum(img, (int(top_left[0] + height/2), top_left[1]), bottom_right)
        score = white - black
    elif feature_type == "TWO_HORIZONTAL":
        white = get_region_sum(img, top_left, (top_left[0] + height, int(top_left[1] +  width/2)))
        black = get_region_sum(img, (top_left[0], int(top_left[1] + width/2)), bottom_right)
        score = white - black
    elif feature_type == "THREE_VERTICAL":
        white1 = get_region_sum(img, top_left, (int(top_left[0] + height/3), top_left[1] + width))
        black = get_region_sum(img, (int(top_left[0] + height/3), top_left[1]), (int(top_left[0] + 2*height/3), top_left[1] + width))
        white2 = get_region_sum(img, (int(top_left[0] + 2*height/3), top_left[1]), bottom_right)
        score = white1 - black + white2
    elif feature_type == "THREE_HORIZONTAL":
        white1 = get_region_sum(img, top_left, (top_left[0] + height, int(top_left[1] + width/3)))
        black = get_region_sum(img, (top_left[0], int(top_left[1] + width/3)), (top_left[0] + height, int(top_left[1] + 2*width/3)))
        white2 = get_region_sum(img, (top_left[0], int(top_left[1] + 2*width/3)),  bottom_right)
        score = white1 - black + white2
    elif feature_type == "FOUR_DIAGONAL":
        # top left area
        white1 = get_region_sum(img, top_left, (int(top_left[0] + height/2), int(top_left[1] + width/2)))
        # top right area
        black1 = get_region_sum(img, (top_left[0], int(top_left[1] + width/2)), (int(top_left[0] + height/2), top_left[1] + width))
        # bottom left area
        black2 = get_region_sum(img, (int(top_left[0] + height/2), top_left[1]), (top_left[0] + height, int(top_left[1] + width/2)))
        # bottom right area
        white2 = get_region_sum(img, (int(top_left[0] + height/2), int(top_left[1] + width/2)),  bottom_right)
        score = white1 - black1 - black2 + white2

    #print(feature_type, top_left, width, height, score)
    #print()
    return score


def get_vote(score, polarity, threshold):
    #score = get_score(img, feature_type, top_left, width, height)
    tmp = 0
    if  polarity*score <  polarity*threshold:
        tmp = 1
    else:
        tmp = 0
    return tmp


def create_features(img_list, img_height, img_width):
    print('Extracting haar features...')
    feature_type = ['TWO_VERTICAL', 'TWO_HORIZONTAL', 'THREE_VERTICAL', 'THREE_HORIZONTAL', 'FOUR_DIAGONAL']
    feature_stride = [[1, 2], [2, 1], [1, 3], [3, 1], [2, 2]]
    max_width = 12
    max_height = 12
    haar_list = []
    print('img total num: ', len(img_list))
    
    for i in range(5):
        start_width = max(feature_stride[i][0], 6)

        for width in range(start_width, max_width+1, feature_stride[i][0]*3):
            start_height = max(feature_stride[i][1], 6)

            for height in range(start_height, max_height+1, feature_stride[i][1]*3):
                for x in range(img_width - width+1):
                    for y in range(img_height - height+1):
                            haar_list.append([feature_type[i], width, height, x, y])

    #pool = Pool(processes=4)
    
    img_feature_list = []
    for i in range(len(img_list)):
        if i%500 == 0:
            print('finished feature extraction: ', i)

        img = img_list[i]
        img_feature_list.append(parrallel_work(img, haar_list))

    print('...done. ' + str(len(img_feature_list)) + ' features created.\n')
    print('feature length: ', str(len(img_feature_list[0])))
    return img_feature_list, haar_list


def parrallel_work(img, haar_list):
    # print(x,y,width,height,i)
    feature_list = []
    integral_img = convert_to_integral_img(img)
    for key in haar_list:    
        score = get_score(integral_img, key[0], [key[4], key[3]], key[1], key[2])
        feature_list.append(score)

    #print('feature vector len: ', len(feature_list))
    del integral_img
    return feature_list


if __name__ == '__main__':
    #test_matrix = np.array([[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15], [16,17,18,19,20]])
    # ret = convert_to_integral_img(test_matrix)
    # print(ret)

    #create_features([test_matrix], 4, 5)
    
    
    start = time.clock()
    tmp = load_images('../train/negative')
    print('num of readed pic: ', len(tmp))

    create_features(tmp, 24, 24)

    elapsed = (time.clock() - start)
    print("Time used:",elapsed)
    