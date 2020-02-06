from functools import partial
import numpy as np
from feature import create_features,get_vote,convert_to_integral_img,get_score,parrallel_work
from utils import load_images
from PIL import Image
import cv2
from sklearn.svm import SVC
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import Imputer
import json
import os
import time
import sys



def train(num_classifiers=80):

    pos_samples = load_images('../train/positive')
    neg_samples = load_images('../train/negative')

    pos_num = len(pos_samples)
    neg_num = len(neg_samples)

    total_num = pos_num + neg_num
    img_height = 24
    img_width = 24

    # Create initial weights and labels
    pos_weights = np.ones(pos_num) * 1. / (2 * pos_num)
    neg_weights = np.ones(neg_num) * 1. / (2 * neg_num)

    weights = np.hstack((pos_weights, neg_weights))
    labels = np.hstack((np.ones(pos_num), np.zeros(neg_num)))

    images = pos_samples + neg_samples

    # Create features for all sizes and locations
    feature_value_list, feature_list = create_features(images, 24, 24)

    
    num_features = len(feature_list)
    feature_indexes = list(range(num_features))
    
    # select classifiers

    classifiers = []

    print('Selecting classifiers..')
    for _ in range(num_classifiers):

        # normalize weights
        weights *= 1. / np.sum(weights)

        # select best classifier based on the weighted error
        total_err_dict = {}
        for f in range(len(feature_indexes)):
            if feature_indexes[f] < 0:
                print('!!!!!!! ', f)
                continue
            if f%1000 == 0:
                print('num', _, 'classifier  round:', f)
            for polarity in [-1, 1]:
                for threshold in range(-30, 40, 10):
                    threshold /= 10
                    error = 0
                    for i in range(total_num):
                        #print(feature_value_list[i][f],  polarity, threshold)
                        if get_vote(feature_value_list[i][f],  polarity, threshold) != labels[i]:
                            error += weights[i]
                    # print('--------', f, polarity, error)

                    total_err_dict[error] = [f, polarity, threshold]

        sort_keys = sorted(total_err_dict)

        best_error = sort_keys[0]
        best_feature_idx = total_err_dict[best_error][0]
        best_polarity = total_err_dict[best_error][1]
        best_threshold = total_err_dict[best_error][2]


        best_feature = feature_list[best_feature_idx]
        classifier_weight = np.log((1 - best_error) / best_error)

        print()
        print('============')
        print(best_error, best_feature_idx, best_polarity, best_threshold, classifier_weight)
        

        ### [weight, feature[len:5], polarity, threshold]
        classifiers.append([classifier_weight, best_feature, best_polarity, best_threshold])

        # update image weights
        

        for i in range(total_num):
            if get_vote(feature_value_list[i][best_feature_idx], best_polarity, best_threshold) != labels[i]:
                weights[i] *= 1
            else:
                weights[i] *= best_error/(1-best_error)

        # remove feature (a feature can't be selected twice)
        feature_indexes[best_feature_idx] = -1

    save_model(classifiers, 'classifier.pkl')
    return classifiers


def train_backup():

    pos_samples = load_images('../train/positive')
    neg_samples = load_images('../train/negative')

    pos_num = len(pos_samples)
    neg_num = len(neg_samples)

    total_num = pos_num + neg_num
    img_height = 24
    img_width = 24


    # Create initial weights and labels
    pos_weights = np.ones(pos_num) * 1. / (2 * pos_num)
    neg_weights = np.ones(neg_num) * 1. / (2 * neg_num)

    weights = np.hstack((pos_weights, neg_weights))
    labels = np.hstack((np.ones(pos_num), np.zeros(neg_num)))

    images = pos_samples + neg_samples

    # Create features for all sizes and locations
    feature_value_list, feature_list = create_features(images, 24, 24)

    train_x = np.array(feature_value_list, dtype=np.float32)
    train_x = Imputer().fit_transform(train_x)
    train_y = np.array(labels, dtype=np.float32)

    print(train_x.shape, train_y.shape)
    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=30, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=100, learning_rate=0.8)
    model.fit(train_x, train_y)
    print('train over')
    save_model(model)
    return model, feature_list


def save_model(model, model_name):
    '''save a model in .pkl file'''

    fileObject = open(model_name, 'wb')
    pickle.dump(model, fileObject)                                 
    fileObject.close()


def read_model(model_name):
    '''load a model from a .pkl file'''

    fileName = model_name
    fileObject = open(fileName, 'rb')
    model = pickle.load(fileObject) 
    return model


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)



def test( classifiers, img_dir = '../test'):

    data = []
    for _file in os.listdir(img_dir):
        if _file.endswith('.jpg'):
            #print(os.path.join(path, _file))
            #show_img = cv2.imread(os.path.join(img_dir, _file))

            img = Image.open((os.path.join(img_dir, _file))).convert("L")
           

            img_width = img.size[0]
            img_height = img.size[1]

            scale_factor = 1

            feature_weight_sum =0
            for c in classifiers:
                feature_weight_sum += c[0]
            
            while(img_width>400 or img_height>400):
                img_width = (int)(img_width/2)
                img_height = (int)(img_height/2)
                img = img.resize((img_width, img_height))
                scale_factor *= 2
                #show_img = cv2.resize(show_img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            for kk in range (10, 40, 10):
                k = (float)(kk / 10)
                tmp_scale_factor = scale_factor * k
                t_count = 0
                for i in range(0, img_width-(int)(24*k), 2):
                    #if i% 40 == 0:
                        #print(i, img_width-24*k, k, feature_weight_sum/2)
                    for j in range(0, img_height-(int)(24*k), 2):
                        total_score = 0
                        #tmp1 = list(range(i, i+24*k))
                        #tmp2 = list(range(j, j+24*k))

                        #region = img[tmp2]
                        #region = region[:, tmp1]
                        region = img.crop((i, j, i+(int)(24*k), (int)(j+24*k)))
                        region = region.resize((24, 24))

                        region = np.array(region, dtype = np.float32) 
                        region /= region.max()
                        integral_region = convert_to_integral_img(region)

                        for c in classifiers:
                            key = c[1]
                            # key: [feature_type, width, height, x, y]

                            # c: [classifier_weight, best_feature, best_polarity, best_threshold]

                            #print(i, j, key[3], key[4], img_width-24, img_height-24)
                            score = get_score(integral_region, key[0], [key[4], key[3]], key[1], key[2])
                            total_score += c[0] * (get_vote(score, c[2], c[3]))
                            
                    
                            if total_score > feature_weight_sum/30*19:
                                # print('------------   ', i, j, total_score, feature_weight_sum/2)
                                t_count += 1
                                #cv2.rectangle(show_img, (int(i), int(j)), (int(i+(int)(24*k)-1), int(j+(int)(23*k)-1)), (0,0,255), 1)
                                data.append( {"iname":_file.split('/')[-1], "bbox": [(int)(i*tmp_scale_factor), (int)(j*tmp_scale_factor), (int)(24*k), (int)(24*k)]}) 

                print(str(_file), k, t_count)
    with open("result.json","w") as f:
        json.dump(data, f, cls=MyEncoder)
    
    #cv2.imshow("img", show_img)
    #cv2.waitKey(0)

                
def test_backup(feature_list, model):
    test_samples = load_images('../test')

    for img in test_samples:
        #img = img.resize((400, 300))
        img_width = img.shape[1]
        img_height = img.shape[0]
        show_img = cv2.imread('../test/test.jpg')

        count=0
        for i in range(0, img_width-23, 2):
            for j in range(0, img_height-23, 2):
                
                tmp1 = list(range(i, i+24))
                tmp2 = list(range(j, j+24))

                region = img[tmp2]
                region = region[:, tmp1]
    
                img_feature_list = []
                img_feature_list.append(parrallel_work(region, feature_list))

                y_pred = model.predict(img_feature_list)

                count += 1
                if count %100 == 0:
                    print(count, y_pred)

                if y_pred == 1:
                    cv2.rectangle(show_img, (int(i), int(j)), (int(i+23), int(j+23)), (0,0,255), 1)
        
        cv2.imshow("img", show_img)
        cv2.waitKey(0)
                

if __name__ == '__main__':
    #classifiers = train()
    test_dir = sys.argv[1]
    #print(test_dir)
    classifiers = read_model('classifier.pkl')
    test(classifiers = classifiers, img_dir = test_dir)

    #model, feature_list = train_backup()
    #test_backup(feature_list, model)

