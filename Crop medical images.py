# import packages
import cv2
import os
from tkinter import *
import tkinter.messagebox
import xlwt
import xlrd
from xlutils.copy import copy
import numpy as np

# open the image
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype = np.uint8), -1)
    return cv_img

# locate the mouse over the image
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)
        print(x, y)


# extract the location numbers of the mouse from the window
def getInput(title, message):
    def return_callback(event):
        print('quit...')
        root.quit()

    def close_callback():
        #message.showinfo('message', 'no click...')
        print('quit...')
        root.quit()

    root = Tk(className=title)
    root.wm_attributes('-topmost', 1)
    screenwidth, screenheight = root.maxsize()
    width = 300
    height = 100
    size = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
    root.geometry(size)
    root.resizable(0, 0)
    lable = Label(root, height=2)
    lable['text'] = message
    lable.pack()
    entry = Entry(root)
    entry.bind('<Return>', return_callback)
    entry.pack()
    entry.focus_set()
    root.protocol("WM_DELETE_WINDOW", close_callback)
    root.mainloop()
    str = entry.get()
    root.destroy()
    return str

# define the style for the excel sheet for saving the labels
def set_style(name, height, bold=False):
    style = xlwt.XFStyle()  

    font = xlwt.Font()  
    font.name = name  # 'Times New Roman'
    font.bold = bold
    font.color_index = 4
    font.height = height

    style.font = font

    return style

# create an Excel file
def create_excel_xls(path, sheet_name, attributes):
    workbook = xlwt.Workbook(encoding='utf-8')  # create a new workbook
    sheet = workbook.add_sheet(sheet_name)  # create a new sheet
    for i in range(0, len(attributes)):
        sheet.write(0, i, attributes[i], set_style('Times New Roman', 220, True))
    workbook.save(path)  # save the workbook

# add in the image information in the excel file
def write_excel_xls_append(path, value):
    index = len(value)  
    workbook = xlrd.open_workbook(path)  
    sheets = workbook.sheet_names()  # extract all the sheets from the workbook
    worksheet = workbook.sheet_by_name(sheets[0])  
    rows_old = worksheet.nrows  
    new_workbook = copy(workbook)  # make xlrd copy to xlwt
    new_worksheet = new_workbook.get_sheet(0)  
    for i in range(0, index):
        new_worksheet.write(rows_old, i, value[i],
                            set_style('Times New Roman', 220, True))  
    new_workbook.save(path)  


if __name__ == '__main__':
    width = 1000
    height = 600

    # save the image path
    input_dir = r'C:\Users\...\CT\...' # location of the images
    
    output_dir_LR = r'C:\...\Cropped CT\Left and Right' 
   
    output_dir_W = r'C:\Users\...\Cropped CT\Whole'


    if not os.path.exists(output_dir_LR):  
        os.makedirs(output_dir_LR)

    if not os.path.exists(output_dir_W):  
        os.makedirs(output_dir_W)

    # the file path for the labels
    Labels = r'C:\...\CT Label'
    
    Leftdata_labels = Labels + r'/Left_labels.xls'
    Rightdata_labels = Labels + r'/Right_labels.xls'
    Fulldata_labels = Labels + r'/Whole_labels.xls'
    
    attributes = ['img_path', 'labels']

    
    if not os.path.exists(Labels):
        os.makedirs(Labels)

    if not os.path.exists(Leftdata_labels):
        create_excel_xls(Leftdata_labels, 'Left_labels', attributes)

    if not os.path.exists(Rightdata_labels):
        create_excel_xls(Rightdata_labels, 'Right_labels', attributes)

    if not os.path.exists(Fulldata_labels):
        create_excel_xls(Fulldata_labels, 'Full_labels', attributes)

    # read all the file names
    all_files = os.listdir(input_dir)
    
    for file_idx in range(len(all_files)):
        # create new full_path
        # print(type(file))
        print(file_idx)
        file = all_files[file_idx]
        case_path = input_dir + r'/' + file
        
        if os.path.isdir(case_path):
            
            all_imgs = os.listdir(case_path)
            idx = 1
            for img_name in all_imgs:
                # print(type((img)))
                if os.path.splitext(img_name)[-1] == '.png':
                    img_path = case_path + r'/' + img_name

                    #################################whether a screenshot is applicable#####################################


                    img = cv_imread(str(img_path))
                    cv2.namedWindow("image", 0)
                    cv2.resizeWindow("image", (width, height))
                    cv2.imshow("image", img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # cv2.waitKey(1)


                    if not tkinter.messagebox.askyesno('Hint', 'Whether to screenshot'):
                        pass
                    else:  # if yes
                        ####################################screenshot the left-side##########################################
                        print(str(file_idx) + '_' + file + '_left')
                        L = [] 

                        img = cv_imread(str(img_path))
                        a = []
                        b = []
                        cv2.namedWindow("image", 0)
                        cv2.resizeWindow("image", (width, height))
                        cv2.imshow("image", img)

                        
                        cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
                        cv2.imshow("image", img)
                        cv2.waitKey(0)

                        if len(b) == 0:
                            pass
                        else:
                            x_start = b[-2]
                            x_end = b[-1]
                            y_start = a[-2]
                            y_end = a[-1]
                            # screenshot from (x_start，y_start) to (x_end,y_end)
                            cropImg = img[x_start:x_end, y_start:y_end]
                            
                            file_name = r'CT' + file + r'_L' + str(idx) + '.png'
                            image_path = output_dir_LR + r'\\' + file_name
                            cv2.imwrite(image_path, cropImg)

                            Left_name = r'CT' + file + r'_L' + str(idx)

                            L.append(Left_name)
                            # extract the label
                            text = getInput('Left label', 'Which class?')
                            L.append(text)
                            print(L)
                            
                            write_excel_xls_append(Leftdata_labels, L)

                        ####################################Screenshot the right-side##########################################
                        print(str(file_idx) + '_' + file + '_right')
                        L = []

                        img = cv_imread(str(img_path))
                        a = []
                        b = []

                        cv2.namedWindow("image", 0)
                        cv2.resizeWindow("image", (width, height))
                        cv2.imshow("image", img)

                        
                        cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
                        cv2.imshow("image", img)
                        cv2.waitKey(0)

                        if len(b) == 0:
                            pass
                        else:
                            x_start = b[-2]
                            x_end = b[-1]
                            y_start = a[-2]
                            y_end = a[-1]
                            # Screenshot from (x_start，y_start) to (x_end,y_end)
                            cropImg = img[x_start:x_end, y_start:y_end]
                            
                            file_name = r'CT' + file + r'_R' + str(idx) + '.png'
                            image_path = output_dir_LR + r'\\' + file_name
                            cv2.imwrite(image_path, cropImg)
                            Right_name = r'CT' + file + r'_R' + str(idx)
                            L.append(Right_name)
                            text = getInput('Right label', 'Which class?')
                            L.append(text)
                            write_excel_xls_append(Rightdata_labels, L)

                        ##################################Screenshot the whole##########################################
                        print(str(file_idx) + '_' + file + '_full')
                        L = []

                        img = cv_imread(str(img_path))
                        a = []
                        b = []

                        cv2.namedWindow("image", 0)
                        cv2.resizeWindow("image", (width, height))
                        cv2.imshow("image", img)

                        
                        cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
                        cv2.imshow("image", img)
                        cv2.waitKey(0)

                        if len(b) == 0:
                            pass
                        else:
                            x_start = b[-2]
                            x_end = b[-1]
                            y_start = a[-2]
                            y_end = a[-1]
                            # Screenshot from (x_start，y_start) to (x_end,y_end)
                            cropImg = img[x_start:x_end, y_start:y_end]
                            
                            file_name = r'CT' + file + '_' + str(idx) + '.png'

                            image_path = output_dir_W + r'\\' + file_name
                            cv2.imwrite(image_path, cropImg)
                            Full_name = r'CT' + file + r'_' + str(idx)
                            L.append(Full_name)
                            text = getInput('Whole label', 'Which class?')
                            L.append(text)
                            write_excel_xls_append(Fulldata_labels, L)

                            idx += 1

