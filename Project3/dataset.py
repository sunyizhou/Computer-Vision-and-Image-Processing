import cv2
import os
import numpy as np
import random

def convert_to_bbox():      
    annotation_dir = '../FDDB-folds'   
    pos_sample_dir = '../train/positive/'  
    neg_sample_dir = '../train/negative/'  
    test_num = 0 
    count = 0
    annotation_dict = {}   
    for i in range(2):        
        annotation_path = annotation_dir + "/FDDB-fold-%0*d-ellipseList.txt"%(2,i+1)        
        annotation_file=open(annotation_path)   
        while(True):            
            f_name = annotation_file.readline()[:-1]
            file_name = f_name+".jpg"            
            if not file_name:                
                break            
            face_num = annotation_file.readline()            
            if not face_num:                
                break            
            face_num=(int)(face_num)
            face_img = cv2.imread("../"+file_name)

            tmp_img = cv2.imread("../"+file_name)

            print(file_name)
            bounding_box = []
            for j in range(face_num):  
                count += 1
                print(count)              
                line = annotation_file.readline().strip().split() 
                major_axis_radius=(float)(line[0])                
                minor_axis_radius=(float)(line[1])                
                angle=(float)(line[2])                
                center_x=(float)(line[3])                
                center_y=(float)(line[4])                
                score=(float)(line[5])                
                angle = angle / 3.1415926*180                
                cv2.ellipse(face_img, ((int)(center_x), (int)(center_y)), ((int)(major_axis_radius), (int)(minor_axis_radius)), angle, 0., 360.,(255, 0, 0))                 
                                  
                mask=np.zeros((face_img.shape[0], face_img.shape[1]), dtype=np.uint8)                   
                cv2.ellipse(mask, ((int)(center_x), (int)(center_y)), ((int)(major_axis_radius), (int)(minor_axis_radius)), angle, 0., 360.,(255, 255, 255))                                        
                image, contours, hierarchy=cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)                    
                         
                r=cv2.boundingRect(contours[0])                        
                x_min=r[0]                        
                y_min=r[1]                        
                x_max=r[0]+r[2]                        
                y_max=r[1]+r[3]     
                bounding_box.append([x_min, y_min, x_max, y_max])  
                
                # save positive samples
                cropped_face = tmp_img[y_min:y_max, x_min:x_max]  
                cropped_face = cv2.resize(cropped_face, (24, 24), interpolation = cv2.INTER_AREA)
                # 裁剪坐标为[y0:y1, x0:x1]
                # print(pos_sample_dir + str(count) + '_' + str(j)  + '.jpg')
                cv2.imwrite(pos_sample_dir + str(count) + '_' + str(j)  + '.jpg', cropped_face)

                # save negative samples
                for i in range(1,5):
                    len = 24*i
                    c = 0
                    flag = True
                    tmp_width = random.randint(0, face_img.shape[1]-len)
                    tmp_height = random.randint(0, face_img.shape[0]-len)
                    while (tmp_width>x_min and tmp_width<x_max) or (tmp_width+len>x_min and tmp_width+len<x_max) or \
                            (tmp_height>y_min and tmp_height<y_max) or (tmp_height+len>y_min and tmp_height+len<y_max):
                        c += 1
                        tmp_width = random.randint(0, face_img.shape[1]-len)
                        tmp_height = random.randint(0, face_img.shape[0]-len)
                        if c > 100:
                            flag = False
                            break
                        #print()
                        #print(tmp_height, tmp_width)
                    if flag:
                        bg = tmp_img[tmp_height:tmp_height+len, tmp_width:tmp_width+len]
                        # print(neg_sample_dir + str(count) + '_' + str(j) + '_'+ str(i) + '.jpg','     ', tmp_height,tmp_height+24, tmp_width,tmp_width+24)
                        bg = cv2.resize(bg, (24, 24), interpolation = cv2.INTER_AREA)
                        cv2.imwrite(neg_sample_dir + str(count) + '_' + str(j) + '_'+ str(i) + '.jpg', bg)
        
                #cv2.rectangle(face_img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,0,255))
                #print(x_max-x_min, y_max-y_min)
            #cv2.imshow("img", face_img)
            #cv2.waitKey(0)
            test_num += face_num
            #print(file_name, face_num, line)            
        #print('=======================')  
        #print('picture num: ', len(annotation_dict))
    print('=======================')
    print('bbox transform done') 
    print('total face: ', test_num)
    return

if __name__ == '__main__':
    convert_to_bbox()