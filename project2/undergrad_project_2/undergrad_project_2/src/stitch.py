import numpy as np
import cv2
import sys
import os
import math
import random


def random_proj(match_list, kps1, kps2, mode):
    r1 = (int)(random.random() * len(match_list))

    r2 = (int)(random.random() * len(match_list))
    while r2 == r1:
        r2 = (int)(random.random() * len(match_list))

    r3 = (int)(random.random() * len(match_list))
    while r3 == r2 or r3 == r1:
        r3 = (int)(random.random() * len(match_list))

    r4 = (int)(random.random() * len(match_list))
    while r4 == r3 or r4 == r2 or r4 == r1:
        r4 = (int)(random.random() * len(match_list))

    proj_list = (r1, r2, r3, r4)

    if mode == 'left':
        points1_list = [match_list[r1]['c1'], match_list[r2]['c1'],
                        match_list[r3]['c1'], match_list[r4]['c1']]
        points2_list = [match_list[r1]['c2'], match_list[r2]['c2'],
                        match_list[r3]['c2'], match_list[r4]['c2']]

        transform = cv2.getPerspectiveTransform( np.array(points1_list, dtype="float32"),  np.array(points2_list, dtype="float32"))
        return transform, proj_list
    else:
        points1_list = [match_list[r1]['c3'], match_list[r2]['c3'],
                        match_list[r3]['c3'], match_list[r4]['c3']]
        points2_list = [match_list[r1]['c2'], match_list[r2]['c2'],
                        match_list[r3]['c2'], match_list[r4]['c2']]

        transform = cv2.getPerspectiveTransform( np.array(points1_list, dtype="float32"),  np.array(points2_list,dtype="float32"))
        return transform, proj_list


def main():
    data_dir = sys.argv[1]
    file_name_list = []
    print('start')
    for i in os.listdir(data_dir):
        temp = os.path.join(data_dir, i)
        if os.path.isdir(temp):
            pass
        else:
            file_name_list.append(temp)

    if(len(file_name_list) == 2):
        img_left = cv2.imread(str(file_name_list[0]))          
        img_middle = cv2.imread(str(file_name_list[1])) 
        img_right = cv2.imread(str(file_name_list[1])) 
    else:
        img_left = cv2.imread(str(file_name_list[0]))          
        img_middle = cv2.imread(str(file_name_list[1])) 
        img_right = cv2.imread(str(file_name_list[2])) 

    width = img_middle.shape[1]
    height = img_middle.shape[0]

    bg_img_middle = np.zeros((height*2, width*3, 3), dtype=np.uint8)
    for i in range(width, 2*width):
        for j in range(height//2, height//2 + height):
            bg_img_middle[j][i] = img_middle[j - height//2][i - width]

    width = bg_img_middle.shape[1]
    height = bg_img_middle.shape[0]
    
    # cv2.imwrite("bg_img_middle.jpg", bg_img_middle)

    detector = cv2.AKAZE_create()

    # find the keypoints and descriptors with SIFT
    kps1, feat1 = detector.detectAndCompute(img_left, None)
    kps2, feat2 = detector.detectAndCompute(bg_img_middle, None)
    kps3, feat3 = detector.detectAndCompute(img_right, None)

    match_left_list = []
    match_right_list = []

    for i in range(len(feat1)):
        for j in range(len(feat2)):
            dist = np.sqrt(np.sum((feat1[i] - feat2[j])**2))
            if dist < 10:
                match_left_list.append({'c1':kps1[i].pt, 'c2':kps2[j].pt, 'dist':dist})
    
    for i in range(len(feat2)):
        for j in range(len(feat3)):
            dist = np.sqrt(np.sum((feat2[i] - feat3[j])**2))
            if dist < 10:
                match_right_list.append({'c3':kps3[j].pt, 'c2':kps2[i].pt, 'dist':dist})

    print('match left num', len(match_left_list))
    print('match right num', len(match_right_list))
    print('\n\n')

    #best_left_proj = []
    best_left_transform = []
    max_left_count = 0

    #best_right_proj = []
    best_right_transform = []
    max_right_count = 0

    for i in range(200):
        s_left = set([])
        s_right = set([])
        candidate_left_transform, candidate_left_proj = random_proj(match_left_list, kps1, kps2, 'left')
        candidate_right_transform, candidate_right_proj = random_proj(match_right_list, kps3, kps2, 'right')
        s_left.add(candidate_left_proj)
        s_right.add(candidate_right_proj)
        
        left_count = 0
        right_count = 0
        for j in range(2000):
            l1 = len(s_left)
            l2 = len(s_right)

            _, proj1 = random_proj(match_left_list, kps1, kps2, 'left')
            _, proj2 = random_proj(match_right_list, kps3, kps2, 'right')
            s_left.add(proj1)
            s_right.add(proj2)

            while len(s_left) == l1:
                _, proj1 = random_proj(match_left_list, kps1, kps2, 'left')
                s_left.add(proj1)
            while len(s_right) == l2:
                _, proj2 = random_proj(match_right_list, kps3, kps2, 'right')
                s_right.add(proj2)
            
            error_left = 0
            error_right = 0

            for k in range(4):
                A = np.array([[match_left_list[proj1[k]]['c1']]], dtype=np.float32)
                res1 = cv2.perspectiveTransform(A, candidate_left_transform)

                B = np.array([[match_right_list[proj2[k]]['c3']]], dtype=np.float32)
                res2 = cv2.perspectiveTransform(B, candidate_right_transform)

                err1 = 0
                err2 = 0

                err1 +=  (match_left_list[proj1[k]]['c2'][0] - res1[0][0][0]) ** 2
                err1 +=  (match_left_list[proj1[k]]['c2'][1] - res1[0][0][1]) ** 2
                error_left = math.sqrt(err1)

                err2 +=  (match_right_list[proj2[k]]['c2'][0] - res2[0][0][0]) ** 2
                err2 +=  (match_right_list[proj2[k]]['c2'][1] - res2[0][0][1]) ** 2
                error_right = math.sqrt(err2)

            if error_left < 3:
                left_count += 1
            if error_right < 3:
                right_count += 1
        # print('---------', i)
        if left_count > max_left_count:
            print('left: ', left_count)
            max_left_count = left_count
           #best_left_proj = candidate_left_proj
            best_left_transform = candidate_left_transform
        if right_count > max_right_count:
            print('right: ', right_count)
            max_right_count = right_count
            #best_right_proj = candidate_right_proj
            best_right_transform = candidate_right_transform

    warp_img_left = cv2.warpPerspective(img_left, best_left_transform, (width, height))
    warp_img_right = cv2.warpPerspective(img_right, best_right_transform, (width, height))


    left, right = 0, 10000
    for col in range(0, width):
        if bg_img_middle[:, col].any() and warp_img_left[:, col].any():
            left = col
            break
    for col in range(width-1, 0, -1):
        if bg_img_middle[:, col].any() and warp_img_left[:, col].any():
            right = col
            break

    print(left, right)

    for row in range(0, height):
        if row%100 == 0:
            print(row, height)
        for col in range(0, width):
            if not bg_img_middle[row, col].any():
                bg_img_middle[row, col] = warp_img_left[row, col]
            elif not warp_img_left[row, col].any():
                pass
            else:
                left_len = float(abs(col - left))
                right_len = float(abs(col - right))
                alpha = left_len / (left_len + right_len)
                bg_img_middle[row, col] = np.clip(bg_img_middle[row, col] * (1-alpha) + warp_img_left[row, col] * alpha, 0, 255)

    print('left done')

    left, right = 0, 10000
    # right
    for col in range(0, width):
        if bg_img_middle[:, col].any() and warp_img_right[:, col].any():
            left = col
            break
    for col in range(width-1, 0, -1):
        if bg_img_middle[:, col].any() and warp_img_right[:, col].any():
            right = col
            break

    print(left, right)

    for row in range(0, height):
        if row%100 == 0:
            print(row, height)
        for col in range(0, width):
            if not bg_img_middle[row, col].any():
                bg_img_middle[row, col] = warp_img_right[row, col]
            elif not warp_img_right[row, col].any():
                pass
            else:
                left_len = float(abs(col - left))
                right_len = float(abs(col - right))
                alpha = left_len / (left_len + right_len)
                bg_img_middle[row, col] = np.clip(bg_img_middle[row, col] * (1-alpha) + warp_img_right[row, col] * alpha, 0, 255)

    print('righ done')
    print('complete')
    save_path = os.path.join(data_dir, "panorama.jpg")
    cv2.imwrite(save_path, bg_img_middle) 


if __name__ == "__main__":
    main()
