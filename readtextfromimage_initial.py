# USAGE
# python readtextfromimage.py --image Input-2.jpg

# import the necessary packages
import cv2
import pytesseract, re
import numpy as np
import argparse

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

import os, sys
from PIL import Image
import shutil

from datetime import datetime
from pymongo import MongoClient
client = MongoClient()
db = client.ic4pro

# from mongoengine import *
# connect('ic4pro', host='localhost', port=27017)

# class ic4_ocr(Document):
#     _id = StringField(required=True, max_length=10)
#     ticket_image = StringField(required=True, max_length=50)
#     ticket_name = StringField(required=True, max_length=50)
#     ticket_date = DateTimeField(default=datetime.datetime.now)
#     account_no = StringField(default='null', max_length=10)
#     account_name = StringField(default='null', max_length=50)
#     amount = DecimalField(default='null')
#     amount_word = StringField(default='null', max_length=200)
#     cheque_no = StringField(default='null', max_length=15)
#     bank_name = StringField(default='null', max_length=50)
#     signature = StringField(default='null', max_length=20)
#     stamp = StringField(default='null', max_length=20)
#     extractn_date = DateTimeField(default=datetime.datetime.now().date().strftime("%Y%d%m"))
#     extractn_time = DateTimeField(default=datetime.datetime.now().time().strftime("%H:%M:%S"))
#     remark = StringField(default='null', max_length=100)
#     comment = StringField(default='null', max_length=100)
#     rejectn_reason = StringField(default='null', max_length=100)


# class ic4_ocr_ex(Document):        
#     ticket_image = StringField(required=True, max_length=50)        
#     extractn_date = DateTimeField(default=datetime.datetime.now().date().strftime("%Y%d%m"))
#     extractn_time = DateTimeField(default=datetime.datetime.now().time().strftime("%H:%M:%S"))        
#     rejectn_reason = StringField(default=null, max_length=100)

# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image to be OCR'd")
# # ap.add_argument("-c", "--min-conf", type=int, default=0,
# # 	help="mininum confidence value to filter weak text detection")
# args = vars(ap.parse_args())

os.makedirs('withImage', exist_ok=True)     # create withImage folder intended for storing successfully extracted images, but not yet used for the purpose
# source = './inimg/'  # Source path 
source = "D:\\iC4_Pro_Project\\ic4_pro_ocr\\input\\"  # Source path  
# source = './'  # Source path  
# destinatn = './outimg/'  # Destination path  
destinatn = "D:\\iC4_Pro_Project\\ic4_pro_ocr\\processed\\"  # Destination path  
# exceptn = './eximg/'  # Exception path
exceptn = "D:\\iC4_Pro_Project\\ic4_pro_ocr\\exceptn\\"  # Exception path
# filepath = "./inimg/"
# filepath = "."



# change the current working directory to a newly created one before doing any operations in it
os.chdir("D:\\iC4_Pro_Project\\ic4_pro_ocr\\input")
# Get the path of current working directory 
filepath = os.getcwd() 

# for filename in os.listdir('.'):      # Loop over all files in the working directory.
for filename in os.listdir(filepath):   # Loop over all files in the working directory.
    # if not (filename.endswith('.png') or filename.endswith('.jpg')) \     # initial code with break line format
    # or filename == LOGO_FILENAME:                                         # initial code with break line format
    if not (filename.endswith('.png') or filename.endswith('.jpg')):         
        shutil.move(source + filename, exceptn + filename)  # Move the content of source to destination 
        
        posts_ex = db.ic4_ocr_ex
        post_ex_data = {            
            'ticket_image': filename,            
            'extractn_date': datetime.now().date().strftime("%Y%d%m"),
            'extractn_time': datetime.now().time().strftime("%H:%M:%S"),            
            'rejectn_reason': 'incompatible image format'
        }
        result = posts_ex.insert_one(post_ex_data)
        print('One rejected: {0}, {1}'.format(result.inserted_id, 'incompatible image format'))

        # post_ic4_ocr_ex = ic4_ocr_ex(         
        #     ticket_image = filename        
        #     rejectn_reason = 'incompatible image format'
        # )
        # post_ic4_ocr_ex.save()       # This will perform an insert
        # print('One rejected: {0}, {1}'.format(post_ic4_ocr_ex._id, 'incompatible image format'))

        continue    # skip non-image files and the logo file itself

    # im = Image.open(filename)     # using "from PIL import Image" module
    # width, height = im.size       # using "from PIL import Image" module
    im = filename
  
    try:
        ########## Use numpy slicing to extract Voucher Name ##########
        # img1 = cv2.imread(args["image"])
        # img1 = cv2.imread(args["image"], cv2.IMREAD_GRAYSCALE)  # read image in grayscale
        img1 = cv2.imread(im, cv2.IMREAD_GRAYSCALE)  # read image in grayscale
        x = 10
        y = 50
        h = 105
        w = 380

        # x = 12
        # y = 46
        # h = 45
        # w = 320

        # x = 5
        # y = 40
        # h = 120
        # w = 410
        global crop_img1
        crop_img1 = img1[y:y+h, x:x+w]
        # cv2.imshow("Voucher Name Before Morph", crop_img1)
        # crop_img1 = cv2.threshold(crop_img1, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]    # convert image to binary format

        # morphological operations to remove noise around the characters using erosion and dilation
        kernel = np.ones((2, 1), np.uint8)
        crop_img1 = cv2.erode(crop_img1, kernel, iterations=1)
        crop_img1 = cv2.dilate(crop_img1, kernel, iterations=1)
        cv2.imshow("Voucher Name", crop_img1)
    
    except FileNotFoundError as fnf_error:
        print(fnf_error)
        shutil.move(source + filename, exceptn + filename)  # Move unsuccessful image from source to exception
        
        posts_ex = db.ic4_ocr_ex
        post_ex_data = {            
            'ticket_image': filename,            
            'extractn_date': datetime.now().date().strftime("%Y%d%m"),
            'extractn_time': datetime.now().time().strftime("%H:%M:%S"),            
            'rejectn_reason': 'FileNotFoundError'
        }
        result = posts_ex.insert_one(post_ex_data)
        print('One rejected: {0}, {1}'.format(result.inserted_id, 'FileNotFoundError'))

        # post_ic4_ocr_ex = ic4_ocr_ex(         
        #     ticket_image = filename        
        #     rejectn_reason = 'FileNotFoundError'
        # )
        # post_ic4_ocr_ex.save()       # This will perform an insert
        # print('One rejected: {0}, {1}'.format(post_ic4_ocr_ex._id, 'FileNotFoundError'))

        continue
    except Exception as e:
        print(e)
        shutil.move(source + filename, exceptn + filename)  # Move unsuccessful image from source to exception
        
        posts_ex = db.ic4_ocr_ex
        post_ex_data = {            
            'ticket_image': filename,            
            'extractn_date': datetime.now().date().strftime("%Y%d%m"),
            'extractn_time': datetime.now().time().strftime("%H:%M:%S"),            
            'rejectn_reason': e
        }
        result = posts_ex.insert_one(post_ex_data)
        print('One rejected: {0}, {1}'.format(result.inserted_id, e))

        # post_ic4_ocr_ex = ic4_ocr_ex(         
        #     ticket_image = filename        
        #     rejectn_reason = e
        # )
        # post_ic4_ocr_ex.save()       # This will perform an insert
        # print('One rejected: {0}, {1}'.format(post_ic4_ocr_ex._id, e))

        continue
    except:
        print("Exception3 raised while extracting")
        shutil.move(source + filename, exceptn + filename)  # Move unsuccessful image from source to exception
        
        posts_ex = db.ic4_ocr_ex
        post_ex_data = {            
            'ticket_image': filename,            
            'extractn_date': datetime.now().date().strftime("%Y%d%m"),
            'extractn_time': datetime.now().time().strftime("%H:%M:%S"),            
            'rejectn_reason': "Exception3 raised while extracting"
        }
        result = posts_ex.insert_one(post_ex_data)
        print('One rejected: {0}, {1}'.format(result.inserted_id, "Exception3 raised while extracting"))

        # post_ic4_ocr_ex = ic4_ocr_ex(         
        #     ticket_image = filename        
        #     rejectn_reason = "Exception3 raised while extracting"
        # )
        # post_ic4_ocr_ex.save()       # This will perform an insert
        # print('One rejected: {0}, {1}'.format(post_ic4_ocr_ex._id, "Exception3 raised while extracting"))

        continue
    finally:
        cv2.waitKey(0)

    # text1 = pytesseract.image_to_string(crop_img1)
    text1 = pytesseract.image_to_string(crop_img1, lang='eng', config='--oem 3 --psm 6')
    searchlist1 = re.findall(r"[A-Z]{3,20}[ ]?", text1)      # first rank
    # searchlist1 = re.findall(r"([A-Z]{3,20}\s?)", text1)      # first rank
    # print(text1)    
    if searchlist1:
        # print("ACCOUNT NAME:", "".join(str(item) for item in searchlist))
        # print("ACCOUNT NAME:", "".join(searchtext))
        
        searchstring1 = "".join(str(item) for item in searchlist1)
        
        # creating substring from start of string 
        # define length upto which substring required 
        substring1 = searchstring1[:20] 
        # substring1 = substring1[:-1]    # remove linebreak which is the last character of the string
    else:substring1 = ""

    print(substring1)

    try:
        ########## Use numpy slicing to extract Voucher Number ##########
        # img2 = cv2.imread(args["image"])
        # img2 = cv2.imread(args["image"], cv2.IMREAD_GRAYSCALE)  # read image in grayscale
        img2 = cv2.imread(im, cv2.IMREAD_GRAYSCALE)  # read image in grayscale
        
        if substring1 == 'CASH DEPOSIT SLIP':      
            x2 = 420      # Cash Deposit Slip
            y2 = 155      # Cash Deposit Slip
            h2 = 130      # Cash Deposit Slip
            w2 = 470      # Cash Deposit Slip
        elif substring1 == 'CASH DEPOSIT SLIP ':      
            x2 = 410      # Cash Deposit Slip
            y2 = 140      # Cash Deposit Slip
            h2 = 150      # Cash Deposit Slip
            w2 = 500      # Cash Deposit Slip
            substring1 = 'CASH DEPOSIT SLIP'
        elif substring1 == 'CHEQUE DEPOSIT SLIP':
            x2 = 460        # Cheque Deposit Slip
            y2 = 110        # Cheque Deposit Slip
            h2 = 140        # Cheque Deposit Slip
            w2 = 520        # Cheque Deposit Slip
        elif substring1 == 'CHEQUE DEPOSIT SLIP ':
            x2 = 460        # Cheque Deposit Slip
            y2 = 110        # Cheque Deposit Slip
            h2 = 140        # Cheque Deposit Slip
            w2 = 520        # Cheque Deposit Slip
            substring1 = 'CHEQUE DEPOSIT SLIP'
        elif substring1 == 'CASH WITHDRAWAL SLIP':      
            x2 = 230     # Cash Deposit Slip
            y2 = 60      # Cash Deposit Slip
            h2 = 120      # Cash Deposit Slip
            w2 = 290      # Cash Deposit Slip
        elif substring1 in 'CASH WITHDRAWAL SLIP':      
            x2 = 230      # Cash Deposit Slip
            y2 = 60      # Cash Deposit Slip
            h2 = 130      # Cash Deposit Slip
            w2 = 290      # Cash Deposit Slip
            substring1 = 'CASH WITHDRAWAL SLIP'
        else:
            print("Exception1 raised while extracting ticket_name")
            shutil.move(source + filename, exceptn + filename)  # Move unsuccessful image from source to exception
            
            posts_ex = db.ic4_ocr_ex
            post_ex_data = {            
                'ticket_image': filename,            
                'extractn_date': datetime.now().date().strftime("%Y%d%m"),
                'extractn_time': datetime.now().time().strftime("%H:%M:%S"),            
                'rejectn_reason': "Exception1 raised while extracting ticket_name"
            }
            result = posts_ex.insert_one(post_ex_data)
            print('One rejected: {0}, {1}'.format(result.inserted_id, "Exception1 raised while extracting ticket_name"))

            # post_ic4_ocr_ex = ic4_ocr_ex(         
            #     ticket_image = filename        
            #     rejectn_reason = "Exception1 raised while extracting ticket_name"
            # )
            # post_ic4_ocr_ex.save()       # This will perform an insert
            # print('One rejected: {0}, {1}'.format(post_ic4_ocr_ex._id, "Exception1 raised while extracting ticket_name"))

            continue

        global crop_img2
        crop_img2 = img2[y2:y2+h2, x2:x2+w2]
        # cv2.imshow("Voucher Number Before Morph", crop_img2)
        # crop_img2 = cv2.threshold(crop_img2, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]    # convert image to binary format

        # morphological operations to remove noise around the characters using erosion and dilation
        kernel = np.ones((2, 1), np.uint8)
        crop_img2 = cv2.erode(crop_img2, kernel, iterations=1)
        crop_img2 = cv2.dilate(crop_img2, kernel, iterations=1)
        cv2.imshow("Voucher Number", crop_img2)
    except:
        print("Exception raised while identifying ticket")
        shutil.move(source + filename, exceptn + filename)  # Move unsuccessful image from source to exception
        
        posts_ex = db.ic4_ocr_ex
        post_ex_data = {            
            'ticket_image': filename,            
            'extractn_date': datetime.now().date().strftime("%Y%d%m"),
            'extractn_time': datetime.now().time().strftime("%H:%M:%S"),            
            'rejectn_reason': "Exception2 raised while extracting ticket_id"
        }
        result = posts_ex.insert_one(post_ex_data)
        print('One rejected: {0}, {1}'.format(result.inserted_id, "Exception raised while identifying ticket"))

        # post_ic4_ocr_ex = ic4_ocr_ex(         
        #     ticket_image = filename        
        #     rejectn_reason = "Exception2 raised while extracting ticket_id"
        # )
        # post_ic4_ocr_ex.save()       # This will perform an insert
        # print('One rejected: {0}, {1}'.format(post_ic4_ocr_ex._id, "Exception2 raised while extracting ticket_id"))

        continue
    finally:
        cv2.waitKey(0)

    # # img = cv2.imread('bitcoin.jpg')             # initial code line 
    # img = cv2.imread('cash-deposit-blank.png')
    # text = pytesseract.image_to_string(img)     # initial code line
    # print(text)                                # initial code line
    # out_below = pytesseract.image_to_string(img)        # initial line of code

    ############################# Moved up ###############################################
    # # text1 = pytesseract.image_to_string(crop_img1)
    # text1 = pytesseract.image_to_string(crop_img1, lang='eng', config='--oem 3 --psm 6')
    # searchlist1 = re.findall(r"[A-Z]{3,20}[ ]?", text1)      # first rank
    # # searchlist1 = re.findall(r"([A-Z]{3,20}\s?)", text1)      # first rank
    # # print(text1)
    # # print(searchtext1)
    # if searchlist1:
    #     # print("ACCOUNT NAME:", "".join(str(item) for item in searchlist))
    #     # print("ACCOUNT NAME:", "".join(searchtext))

    #     # searchstring1 = "".join(str(item) for item in searchlist1)
    #     searchstring1 = "".join(str(item) for item in searchlist1)
        
    #     # creating substring from start of string 
    #     # define length upto which substring required 
    #     substring1 = searchstring1[:20] 
    #     # substring1 = substring1[:-1]    # remove linebreak which is the last character of the string
    # else:substring1 = ""

    # print(substring1)
    ######################################################################################

    # text2 = pytesseract.image_to_string(crop_img2)    # use default settings
    # text2 = pytesseract.image_to_string(crop_img2, config='digits')    # use this if 'C:\Program Files (x86)\Tesseract-OCR\tessdata\configs\digits' is modified to suit requirement
    # text2 = pytesseract.image_to_string(crop_img2, lang='eng',config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789')
    text2 = pytesseract.image_to_string(crop_img2, lang='eng', config='--oem 3 --psm 6')
    searchlist2 = re.findall(r"[0-9]{7,9}", text2)      # first rank
    # searchlist2 = re.findall(r'[0-9]{8}', text2)   
    # searchlist2 = re.findall(r'[0-9]', text2)
    # searchlist2 = re.findall(r'\d{7,9}', text2)   
    # searchlist2 = re.findall(r'\d', text2)    

    # print(text2)
    # print(searchtext2)
    if searchlist2:
        # print("ACCOUNT NAME:", "".join(str(item) for item in searchlist))
        # print("ACCOUNT NAME:", "".join(searchtext))

        # searchstring2 = "".join(str(item) for item in searchlist2)
        # # creating substring from start of string 
        # # define length upto which substring required 
        # substring2 = searchstring2[:20] 
        # OR
        substring2 = ''.join(searchlist2)
    else:
        substring2 = ""
        print("Exception4 raised while extracting ticket_id")
        shutil.move(source + filename, exceptn + filename)  # Move unsuccessful image from source to exception
        
        posts_ex = db.ic4_ocr_ex
        post_ex_data = {            
            'ticket_image': filename,            
            'extractn_date': datetime.now().date().strftime("%Y%d%m"),
            'extractn_time': datetime.now().time().strftime("%H:%M:%S"),            
            'rejectn_reason': "Exception4 raised while extracting ticket_id"
        }
        result = posts_ex.insert_one(post_ex_data)
        print('One rejected: {0}, {1}'.format(result.inserted_id, "Exception4 raised while extracting ticket_id"))

        # post_ic4_ocr_ex = ic4_ocr_ex(         
        #     ticket_image = filename        
        #     rejectn_reason = "Exception4 raised while extracting ticket_id"
        # )
        # post_ic4_ocr_ex.save()       # This will perform an insert
        # print('One rejected: {0}, {1}'.format(post_ic4_ocr_ex._id, "Exception4 raised while extracting ticket_id"))

        continue

    print(substring2)

    shutil.move(source + filename, destinatn + filename)  # Move successful image from source to destination 

    # itemlist = [text1,text2]
    # itemlist = [substring1,substring2]
    itemline = [substring1,",",substring2]

    with open("D:\\iC4_Pro_Project\\ic4_pro_ocr\\outfile.txt", "a") as outfile:
        # outfile.write(",".join(itemlist))    
        outfile.write("" .join(itemline))
        outfile.write("\n")

    
    posts = db.ic4_ocr
    post_data = {
        '_id': substring2,
        'ticket_image': filename,
        'ticket_name': substring1,
        'ticket_date': 'null',
        'account_no': 'null',
        'account_name': 'null',
        'amount': 'null',
        'amount_word': 'null',
        'cheque_no': 'null',
        'micr_digit4': 'null',
	    'micr_digit5': 'null',
	    'micr_digit6': 'null',
        'bank_name': 'null',
        'signature': 'null',
        'stamp': 'null',
        'extractn_date': datetime.now().date().strftime("%Y%d%m"),
        'extractn_time': datetime.now().time().strftime("%H:%M:%S"),
        'remark': 'record extracted',
        'comment': 'required fields extracted',
        'rejectn_reason': 'null',
        'callover_agent': 'null',
        'callover_date': 'null',
        'callover_time': 'null'
    }
    result = posts.insert_one(post_data)
    print('One post: {0}, {1}'.format(result.inserted_id, substring1))


    # with open('./outfile2.txt', 'w') as filehandle:
    #     for listitem in itemlist:
    #         filehandle.write('%s,' % listitem)


    # with open('./outfile2.txt', 'w') as filehandle:
    #     filehandle.writelines("%s" % listitem for listitem in itemlist)

    
    # post_ic4_ocr = ic4_ocr(    
    #     _id = substring2
    #     ticket_image = filename
    #     ticket_name = substring1
    #     ticket_date = 'null'
    #     account_no = 'null'
    #     account_name = 'null'
    #     amount = 'null'
    #     amount_word = 'null'
    #     cheque_no = 'null'
    #     bank_name = 'null'
    #     signature = 'null'
    #     stamp = 'null'
    #     # extractn_date = DateTimeField(default=datetime.datetime.now().date().strftime("%Y%d%m"))
    #     # extractn_time = DateTimeField(default=datetime.datetime.now().time().strftime("%H:%M:%S"))
    #     remark = 'record extracted'
    #     comment = 'required fields extracted'
    #     rejectn_reason = 'null'
    # )
    # post_ic4_ocr.save()       # This will perform an insert
    # print(post_ic4_ocr._id + ', ' + substring1)
    # # post_ic4_ocr.title = 'A Better Post Title'
    # # post_ic4_ocr.save()       # This will perform an atomic edit on "title"
    # # print(post_ic4_ocr.title)