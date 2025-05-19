import numpy as np
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2
import math
import matplotlib.pyplot as plt
import sys
from sklearn.linear_model import LinearRegression

cryst_area = list()
user_choice = input("Please choose one of the options below \n"
                    "ia- for image or,cl- for 'contour length' or ca- for 'contour area': ")

class distance:
    # used to calculate distance between two black pixels to select black objects one at a time
    # and reject which is not associated wwith 
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def cal_dist(self):
        dist = math.sqrt((self.x1 - self.x2) ** 2 + (self.y1 - self.y2) ** 2)
        return(dist)

class in_process:
    def __init__(self, nlist):
        self.nlist = nlist

    def cal_contour_points(self):
        ylist = list()
        contour_points = list()
        nlist = sorted(self.nlist, key=lambda coord: coord[0])
        l = len(nlist)
        i = 1
        j = 0
        while j < l:
            x, y = nlist[j]
            ylist.append(y)
#            j += 1
            while i <= l-1:
                x1, y1 = nlist[i]
                if x1 == x:
                    ylist.append(y1)
                else:
                    sorted_ylist = sorted(ylist)
                    tylist = len(sorted_ylist)
                    if tylist > 1:
                        x2 = x
                        y2 = sorted_ylist[0]
                        count = 0
                        for k in range(tylist):
                            if y2 == sorted_ylist[k]:
                                count += 1
                                contour_points.append((x2, y2))
                        x3 = x
                        y3 = sorted_ylist[tylist-1]
                        count = 0
                        for k in range(tylist):
                            if y3 == sorted_ylist[k]:
                                count += 1
                                contour_points.append((x3, y3))
                    else:
                        x2 = x
                        y2 = sorted_ylist[0]
                        contour_points.append((x2, y2))
                    ylist = list()
                    break
                i=i+1
            j=i+1
        return(contour_points)

    def cal_contour_area(self):
        nnlist = sorted(self.nlist, key=lambda coord: coord[0])
        x1 = nnlist[0]
        xn = nnlist[len(nnlist)-1]
        contour_area = 0
        contour_area = list()

        for i in range(x1, xn):
            ylist = list()
            for x, y in nnlist:
                if i == x:
                    ylist.append(y)
            sorted_ylist = sorted(ylist)
            miny = sorted_ylist[0]
            maxy = sorted_ylist[len(sorted_ylist)-1]
            val = (maxy - miny) + 1
            contour_area = contour_area + val
        return(contour_area)


class x_data:
    def __init__(self,cryst_area):
        self.y=cryst_area
        self.x=self.xgen()
    def xgen(self):
        x1=list(range(len(self.y)))
        return np.array(x1).reshape(-1,1)

# Fit linear regression model
class plot_data(x_data):
    def regression(self,save_path=None):
        model=LinearRegression()
        model.fit(self.x, self.y)
        self.y_pred = model.predict(self.x)
        plt.scatter(self.x, self.y, color='blue', label='Coordinates')
        plt.plot(self.x, self.y_pred, color='red', label='Linear Regression')
        plt.xlabel('Time')
        if user_choice=='ia':
            plt.ylabel('Area of crystals')
            plt.title('Growth of the area of crystals with time ')
        if user_choice=='cl':
            plt.ylabel('contour length of crystals')
            plt.title('Growth of the contour length of crystals with time ')
        if user_choice=='ca':
            plt.ylabel('contour area of crystals')
            plt.title('Growth of the contour area of crystals with time ')
        if save_path:
            plt.savefig(save_path)

        plt.show()

directory_path='/home/sourav/Desktop/crystal_growth_image_analysis/jpg_data'
if user_choice=='ia':
    save_path='/home/sourav/Desktop/crystal_growth_image_analysis/figs/area.png'
if user_choice=='cl':
    save_path='/home/sourav/Desktop/crystal_growth_image_analysis/figs/cont_length.png'
if user_choice=='ca':
    save_path='/home/sourav/Desktop/crystal_growth_image_analysis/figs/cont_area.png'

file_names = [f for f in os.listdir(directory_path) if\
        os.path.isfile(os.path.join(directory_path, f))] # get a list of file names

def extract_serial_number(file_name):
    return int(''.join(filter(str.isdigit, file_name)))

file_names.sort(key=extract_serial_number)

print(file_names)

#sys.exit()

key_pressed = None  # Global variable to track key

def on_key(event):
    global key_pressed
    key_pressed = event.key
    print(f"Key pressed: {key_pressed}")
    plt.close()

# main loop starts below

kk = 0
nn = 10
n = 1
for file_name in file_names:
    kk = kk+1  
    #print(file_name, kk)
    if kk > nn:
        break
    image_path = os.path.join(directory_path, file_name)
    image = cv2.imread(image_path)
   # print(f"file name:{file_name},kk")

    # Check if the image was loaded successfully
    if image is None:
        print("Image not found.")
        sys.exit()
    else:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
        _, mask = cv2.threshold(grayscale_image, 50, 255, cv2.THRESH_BINARY) # Threshold to create a mask for dark areas

        black_pixels = np.where(mask == 0)
        black_coordinates = list(zip(black_pixels[0], black_pixels[1]))
        l = len(black_coordinates)
        i = 0
        part = 0
        while i < l:
            #print('iiii',i)
            y, x = black_coordinates[i]

            part = part +1 # running number of cropped image of each picture

            cropped_image = mask[y - 100: y + 100, x - 100: x + 100]

            # Display the image using matplotlib
            plt.imshow(cropped_image, cmap='gray')
            plt.title(f"{part}- Cropped Dark Region of picture {kk} \n"
                      "Please check & close the image to continue")
            plt.axis('off')  # Hide axes
            plt.show()
            
            # Simulate key press using input
            key = input("Press 'y' to accept this region: ").strip().lower()
            
            if key == 'y':
                black_pixels1 = np.where(cropped_image == 0)
                print(black_pixels1)
                black_coordinates_cropped = list(zip(black_pixels1[1], black_pixels1[0]))
                print('you said yes')
                length1=len(black_coordinates_cropped)
                print('length1',length1)
                if user_choice == 'ia':
                    cryst_area.append(len(black_coordinates_cropped))
                if user_choice == 'cl':
                    cplx=in_process(len(black_coordinates_cropped))
                    contour_pointsx_list = cplx.cal_contour_points()
                    swapped_black_coordinates_cropped = [(y, x) for x, y in black_coordinates_cropped]
                    cply=in_process(swapped_black_coordinates_cropped)
                    contour_pointsy_list=cply.cal_contour_points()
                    swapped_contour_pointsy_list = [(y, x) for x, y in counter_pointsy_list]
                    contour_points_list=list(set(contour_pointsx_list+swapped_contour_pointsy_list))
                    cryst_area.append(len(contour_points_list))
                    image = np.zeros((400, 400, 1), dtype=np.units)

                    for m, n in coutour_points_list: #draw circlrs at each pixel coordinate
                        cv2_circle(image, (n, m), 2, (255, 255, 255), -1) # -1 fills the circles

                    cv2.inshow('Image with Points', image)
                    cv2.waitKey(100)
                    cv2.imwrite('contour image_1.tif',image)

                if user_choice == 'ca':
                    cplx=in_process(black_coordinates_cropped)
                    contour_pointsx_list = cplx.cal_contour_points()
                    swapped_black_coordinates_cropped = [(y, x) for x, y in black_coordinates_cropped]
                    cply=in_process(swapped_black_coordinates_cropped)
                    contour_pointsy_list = cply.cal_contour_points()
                    swapped_contour_pointsy_list = [(y, x) for x, y in contour_pointsy_list]
                    contour_pointsy_list=list(set(contour_pointsx_list+swapped_contour_pointsy_list))
                    ca = in_process(contour_points_list)
                    contour_area = ca.cal_contour_area()
                    cryst_area.append(contour_area)
            ##    break
            ##else:
            ##    break
            #else:
            #    black_pixels = np.where(cropped_image == 0)
            #    black_coordinates_cropped =list(zip(black_pixels[0], black_pixels[1]))
            #    print('black_coordinates_cropped',black_coordinates_cropped)
            #    length=len(black_coordinates_cropped)
            #    print('length',length)
            #    sys.exit()
            #    if length<=length1+500 and length>=length1-500:
            #        black_coordinates_cropped = list(zip(black_pixels[0], black_pixels[1]))
            #        length=len(black_coordinates_cropped)
            #        if user_choice == 'ia':
            #            cryst_area.append(length)
            #        if user_choice == 'ci':
            #            cplx=in_process(black_coordinates_cropped)
            #            contour_pointsx_list=cplx.cal_contour_points()
            #            swapped_black_coordinates_cropped = [(y, x) for x, y in black_coordinates_cropped]
            #            cply=in_process(swapped_black_coordinates_cropped)
            #            contour_pointsy_list=cply.cal_contour_points()
            #            swapped_contour_pointsy_list = [(y, x) for x, y in contour_pointsy_list]
            #            contour_points_list = list(set(contour_points_list+swapped_contour_pointsy_list))
            #            cryst_area.append(len(contour_points_list))
            #        if user_choice == 'ca':
            #            cplx=in_process(black_coordinates_cropped)
            #            contour_pointsx_list=cplx.cal_contour_points()
            #            swapped_black_coordinates_cropped = [(y, x) for x, y in black_coordinates_cropped]
            #            cply=in_process(swapped_black_coordinates_cropped)
            #            contour_pointsy_list=cply.cal_contour_points()
            #            swapped_contour_pointsy_list = [(y, x) for x, y in contour_pointsy_list]
            #            contour_points_list = list(set(contour_points_list+swapped_contour_pointsy_list))
            #            ca=in_process(contour_points_list)
            #            contour_area=ca.cal_contour_area()
            #            cryst_area.append(contour_area)
            #        cv2.imshow('Cropped image', cropped_image)
            #        cv2.waitKey(100)
            #        break

            j=1
            while j < l-101:
                y1_check, x1_check = black_coordinates[j + 100]
                Dist = distance(x, y, x1_check, y1_check)
                D = Dist.cal_dist()
                if D < 5:
                    j=j+100
                    break
                j=j+100
            i=i+j

print('crystal_area', cryst_area)
plot = plot_data(cryst_area)
plot.regression(save_path)

