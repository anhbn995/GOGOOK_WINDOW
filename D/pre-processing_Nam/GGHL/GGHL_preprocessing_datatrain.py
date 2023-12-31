import os
import numpy as np
import cv2
from tqdm import tqdm


label_path = '/mnt/Nam/Shipdetection/data/labelTxt/'
store_path = '/mnt/Nam/Shipdetection/data/Annotations/'
txt_path = "/mnt/Nam/Shipdetection/train.txt"

classes = ['car']

label_list = os.listdir(store_path)

def minAreaRect2longSideFormat(rectangle_inf):
    width = rectangle_inf[1][0]
    height = rectangle_inf[1][1]
    theta = rectangle_inf[-1]
    longSide = max(width, height)
    shortSide = min(width, height)
    if theta == 90:
        if longSide == width:
            pass
        else:
            theta = 0
        if np.around(longSide, 2) == np.around(shortSide, 2):
            theta = 0
    else:
        if np.around(longSide, 2) == np.around(shortSide, 2):
            pass
        else:
            if longSide == width:
                pass
            else:
                theta += 90

    if 179 < theta <= 180:
        theta = 179

    return (rectangle_inf[0], (longSide, shortSide), theta)


for label in tqdm(label_list):
    boxes = []
    label_info_path = label_path + label.replace('.tif', '.txt')
    with open(label_info_path) as label_info:
        new_boxes = []
        new_boxes1 = []
        for box in label_info.readlines():
            box = list(map(str, box.split()))
            corners = np.array([[int(float(box[0])), int(float(box[1]))], [int(float(box[2])), int(float(box[3]))],
                                [int(float(box[4])), int(float(box[5]))], [int(float(box[6])), int(float(box[7]))]])

            corners_sort = corners[np.argsort(corners[:, 1])]
            corners_sorted = corners_sort[np.append(np.argsort(
                corners_sort[0:2, 0]), np.argsort(corners_sort[2:4, 0])[::-1] + 2)]

            if corners_sorted[0, 0] == corners_sorted[1, 0]:
                temp_max = corners_sorted[0, 1]
                if corners_sorted[0, 0] < min(corners_sorted[2, 0], corners_sorted[3, 0]):
                    if temp_max > corners_sorted[1, 1]:
                        pass
                    else:
                        corners_sorted = corners_sorted[[1, 0, 2, 3]]
                else:
                    if temp_max > corners_sorted[1, 1]:
                        corners_sorted = corners_sorted[[1, 0, 2, 3]]
                    else:
                        pass

            if corners_sorted[2, 0] == corners_sorted[3, 0]:
                temp_max = corners_sorted[2, 1]
                if corners_sorted[2, 0] < min(corners_sorted[0, 0], corners_sorted[1, 0]):
                    if temp_max > corners_sorted[3, 1]:
                        pass
                    else:
                        corners_sorted = corners_sorted[[0, 1, 3, 2]]
                else:
                    if temp_max > corners_sorted[3, 1]:
                        corners_sorted = corners_sorted[[0, 1, 3, 2]]
                    else:
                        pass

            corners_x_max = corners[np.argmax(corners[:, 0])][0]
            corners_y_max = corners[np.argmax(corners[:, 1])][1]
            corners_x_min = corners[np.argmin(corners[:, 0])][0]
            corners_y_min = corners[np.argmin(corners[:, 1])][1]
            corners_fixed = corners_sorted

            if corners_x_max == corners_sorted[2, 0] and corners_x_min == corners_sorted[
                0, 0] and corners_y_max == corners_sorted[3, 1] and corners_y_min == corners_sorted[1, 1]:

                corners_fixed = corners_fixed[[1, 2, 3, 0]]


            elif corners_x_max == corners_sorted[1, 0] and corners_x_min == corners_sorted[
                3, 0] and corners_y_max == corners_sorted[3, 1] and corners_y_min == corners_sorted[1, 1]:

                corners_fixed[0, 1] = corners_y_min
                corners_fixed[2, 1] = corners_y_max

            elif corners_x_max == corners_sorted[2, 0] and corners_x_min == corners_sorted[
                0, 0] and corners_y_max == corners_sorted[2, 1] and corners_y_min == corners_sorted[0, 1]:

                corners_fixed[1, 0] = corners_x_max
                corners_fixed[3, 0] = corners_x_min

            elif corners_x_max == corners_sorted[1, 0] and corners_x_min == corners_sorted[
                3, 0] and corners_y_max == corners_sorted[3, 1] and corners_y_min == corners_sorted[0, 1]:

                corners_fixed[2, 1] = corners_y_max

            elif corners_x_max == corners_sorted[1, 0] and corners_x_min == corners_sorted[
                3, 0] and corners_y_max == corners_sorted[2, 1] and corners_y_min == corners_sorted[1, 1]:

                corners_fixed[0, 1] = corners_y_min

            elif corners_x_max == corners_sorted[2, 0] and corners_x_min == corners_sorted[
                0, 0] and corners_y_max == corners_sorted[3, 1] and corners_y_min == corners_sorted[0, 1]:

                corners_fixed[1, 1] = corners_y_min

                corners_fixed = corners_fixed[[1, 2, 3, 0]]


            elif corners_x_max == corners_sorted[2, 0] and corners_x_min == corners_sorted[
                0, 0] and corners_y_max == corners_sorted[2, 1] and corners_y_min == corners_sorted[1, 1]:

                corners_fixed[3, 1] = corners_y_max

                corners_fixed = corners_fixed[[1, 2, 3, 0]]

            elif corners_x_max == corners_sorted[1, 0] and corners_x_min == corners_sorted[
                0, 0] and corners_y_max == corners_sorted[2, 1] and corners_y_min == corners_sorted[0, 1]:

                corners_fixed[3, 0] = corners_x_min

            elif corners_x_max == corners_sorted[1, 0] and corners_x_min == corners_sorted[
                0, 0] and corners_y_max == corners_sorted[2, 1] and corners_y_min == corners_sorted[1, 1]:

                corners_fixed[0, 1] = corners_y_min
                corners_fixed[3, 0] = corners_x_min

            elif corners_x_max == corners_sorted[1, 0] and corners_x_min == corners_sorted[
                0, 0] and corners_y_max == corners_sorted[3, 1] and corners_y_min == corners_sorted[0, 1]:

                corners_fixed[2, 1] = corners_y_max
                corners_fixed[3, 0] = corners_x_min

            elif corners_x_max == corners_sorted[1, 0] and corners_x_min == corners_sorted[
                0, 0] and corners_y_max == corners_sorted[3, 1] and corners_y_min == corners_sorted[1, 1]:

                corners_fixed[2, 0] = corners_x_max

                corners_fixed = corners_fixed[[1, 2, 3, 0]]

            # X_max Xmin Y_max Y_min => X3 X4 X3 X1 @ 13
            elif corners_x_max == corners_sorted[2, 0] and corners_x_min == corners_sorted[
                3, 0] and corners_y_max == corners_sorted[2, 1] and corners_y_min == corners_sorted[0, 1]:

                corners_fixed[1, 0] = corners_x_max

            elif corners_x_max == corners_sorted[2, 0] and corners_x_min == corners_sorted[
                3, 0] and corners_y_max == corners_sorted[2, 1] and corners_y_min == corners_sorted[1, 1]:

                corners_fixed[0, 1] = corners_y_min
                corners_fixed[1, 0] = corners_x_max

            elif corners_x_max == corners_sorted[2, 0] and corners_x_min == corners_sorted[
                3, 0] and corners_y_max == corners_sorted[3, 1] and corners_y_min == corners_sorted[0, 1]:

                corners_fixed[1, 0] = corners_x_max
                corners_fixed[2, 1] = corners_y_max

            elif corners_x_max == corners_sorted[2, 0] and corners_x_min == corners_sorted[
                3, 0] and corners_y_max == corners_sorted[3, 1] and corners_y_min == corners_sorted[1, 1]:

                corners_fixed[0, 0] = corners_x_min

                corners_fixed = corners_fixed[[1, 2, 3, 0]]
                
            basic_info = [corners_x_min, corners_y_min,
                          corners_x_max, corners_y_max, classes.index(box[8])]
            corners_info = [x for corner in corners_fixed.tolist()
                            for x in corner]
            all_info = basic_info + corners_info

            if box[9] != '2':
                r = cv2.contourArea(corners_fixed) / (float(corners_x_max - corners_x_min) * float(corners_y_max - corners_y_min))
                all_info.append(r)

                cnt = np.array([[int(corners_fixed[0][0]), int(str(corners_fixed[0][1]))],
                                [int(corners_fixed[1][0]), int(corners_fixed[1][1])],
                                [int(corners_fixed[2][0]), int(corners_fixed[2][1])],
                                [int(corners_fixed[3][0]), int(corners_fixed[3][1])]])
                rect = cv2.minAreaRect(cnt)
                longSide_inf = minAreaRect2longSideFormat(rect)
                angle = longSide_inf[-1]

                all_info.append(angle)

                new_boxes.append(",".join([str(x) for x in all_info]))

                labelout = str(classes.index(box[8])) + ' ' + str(corners_fixed[0][0]) + ' ' + \
                           str(corners_fixed[0][1]) + ' ' + str(corners_fixed[1][0]) + ' ' + \
                           str(corners_fixed[1][1]) + ' ' + str(corners_fixed[2][0]) + ' ' + \
                           str(corners_fixed[2][1]) + ' ' + str(corners_fixed[3][0]) + ' ' + \
                           str(corners_fixed[3][1]) + '\n'

        each_boxes = (store_path + label.split('.')[0] + '.tif' + " " + " ".join(new_boxes) + "\n")
        temp = store_path + label.split('.')[0] + '.tif' + " " + "\n"
        if each_boxes != temp:
            boxes.append(each_boxes)

    with open(txt_path, 'a') as file:
        file.writelines(boxes)