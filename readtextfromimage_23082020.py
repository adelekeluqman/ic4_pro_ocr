# USAGE
# python readtextfromimage.py --image Input-2.jpg

# import the necessary packages
import cv2
import pytesseract, re
import numpy as np
import argparse
from skimage.segmentation import clear_border
from imutils import contours
import imutils
# from bank_check_ocr import extract_digits_and_symbols

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

# # def extract_digits_and_symbols(image, charCnts, minW=5, minH=15):		# initial line of code
# def extract_digits_and_symbols(image, charCnts, minW=5, minH=10):		# modified line that enhance detection of micr xter
#     # grab the internal Python iterator for the list of character
#     # contours, then  initialize the character ROI and location
#     # lists, respectively
#     charIter = charCnts.__iter__()
#     rois = []
#     locs = []

#     # keep looping over the character contours until we reach the end
#     # of the list
#     while True:
#         try:
#             # grab the next character contour from the list, compute
#             # its bounding box, and initialize the ROI
#             c = next(charIter)
#             (cX, cY, cW, cH) = cv2.boundingRect(c)
#             roi = None

#             # check to see if the width and height are sufficiently
#             # large, indicating that we have found a digit
#             if cW >= minW and cH >= minH:
#                 # extract the ROI
#                 roi = image[cY:cY + cH, cX:cX + cW]
#                 rois.append(roi)
#                 locs.append((cX, cY, cX + cW, cY + cH))

#             # otherwise, we are examining one of the special symbols
#             else:
#                 # MICR symbols include three separate parts, so we
#                 # need to grab the next two parts from our iterator,
#                 # followed by initializing the bounding box
#                 # coordinates for the symbol
#                 parts = [c, next(charIter), next(charIter)]
#                 (sXA, sYA, sXB, sYB) = (np.inf, np.inf, -np.inf,
#                     -np.inf)

#                 # loop over the parts
#                 for p in parts:
#                     # compute the bounding box for the part, then
#                     # update our bookkeeping variables
#                     (pX, pY, pW, pH) = cv2.boundingRect(p)
#                     sXA = min(sXA, pX)
#                     sYA = min(sYA, pY)
#                     sXB = max(sXB, pX + pW)
#                     sYB = max(sYB, pY + pH)

#                 # extract the ROI
#                 roi = image[sYA:sYB, sXA:sXB]
#                 rois.append(roi)
#                 locs.append((sXA, sYA, sXB, sYB))

#         # we have reached the end of the iterator; gracefully break
#         # from the loop
#         except StopIteration:
#             break

#     # return a tuple of the ROIs and locations
#     return (rois, locs)

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

micrfont = "D:/iC4_Pro_Project/ic4_pro_ocr/micrfont/templateMicr.png"  # MICR font image file path
micrpath = "D:\\iC4_Pro_Project\\ic4_pro_ocr\\micrfolder\\"  # MICR digit reference file path
# micrfile = "D:\\iC4_Pro_Project\\ic4_pro_ocr\\micrfolder\\micr_e13b_reference.png"
micrfile = " "
for filename2 in os.listdir(micrpath):
    if (filename2.endswith('.png') or filename2.endswith('.jpg')): 
        micrfile = filename2
        # print('micrfile string: ' + micrfile)


# change the current working directory to a newly created one before doing any operations in it
os.chdir("D:\\iC4_Pro_Project\\ic4_pro_ocr\\input")
# Get the path of current working directory 
filepath = os.getcwd() 

# for filename in os.listdir('.'):      # Loop over all files in the working directory.
for filename in os.listdir(filepath):   # Loop over all files in the working directory.
    # if not (filename.endswith('.png') or filename.endswith('.jpg')) \     # initial code with break line format
    # or filename == LOGO_FILENAME:                                         # initial code with break line format
    if not (filename.endswith('.png') or filename.endswith('.jpg')):    
        print("----------- file moved to exceptn folder 1 -----------")     
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



    # ---------------------- check if im is cheque -------------------------------    
    # # initialize the list of reference character names, in the same
    # # order as they appear in the reference image where the digits
    # # their names and:
    # # T = Transit (delimit bank branch routing transit #)
    # # U = On-us (delimit customer account number)
    # # A = Amount (delimit transaction amount)
    # # D = Dash (delimit parts of numbers, such as routing or account)
    # charNames = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    #     "T", "U", "A", "D"]

    # # load the reference MICR image from disk, convert it to grayscale,
    # # and threshold it, such that the digits appear as *white* on a
    # # *black* background  # "./out/output-image.png"
    # ref = cv2.imread(micrpath + micrfile, 0)
    # # ref = cv2.imread("D:\\iC4_Pro_Project\\ic4_pro_ocr\\micrfolder\\micr_e13b_reference.png")
    # # ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    # ref = imutils.resize(ref, width=400)
    # ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY_INV |
    #     cv2.THRESH_OTSU)[1]

    # # find contours in the MICR image (i.e,. the outlines of the
    # # characters) and sort them from left to right
    # refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
    #     cv2.CHAIN_APPROX_SIMPLE)
    # refCnts = imutils.grab_contours(refCnts)
    # refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]

    # # extract the digits and symbols from the list of contours, then
    # # initialize a dictionary to map the character name to the ROI
    # # refROIs = extract_digits_and_symbols(ref, refCnts,		# initial line of code
    # # 	minW=10, minH=20)[0]									# initial line of code
    # refROIs = extract_digits_and_symbols(ref, refCnts, minW=10, minH=20)[0]
    # chars = {}

    # # loop over the reference ROIs
    # for (name, roi) in zip(charNames, refROIs):
    #     # resize the ROI to a fixed size, then update the characters
    #     # dictionary, mapping the character name to the ROI
    #     # roi = cv2.resize(roi, (36, 36)) 		# initial line of code
    #     roi = cv2.resize(roi, (36, 36)) 
    #     chars[name] = roi

    # # initialize a rectangular kernel (wider than it is tall) along with
    # # an empty list to store the output of the check OCR
    # # rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))	# initial line of code
    # rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))		# better segmented rectangular kernel
    # output = []

    # load the input image, grab its dimensions, and apply array slicing
    # to keep only the bottom 20% of the image (that's where the account
    # information is)
    # # image = cv2.imread(im, 0)
    # image = cv2.imread(im)
    # # image = cv2.imread('D:/iC4_Pro_Project/ic4_pro_ocr/input/sample1c.png', 0)
    # (h, w,) = image.shape[:2]
    # delta = int(h - (h * 0.2))		# initial line of code
    # # delta = int(h - (h * 0.3))	# adjust the bottom % of the cheque image captured for scanning
    # bottom = image[delta:h, 0:w]

    # # convert the bottom image to grayscale, then apply a blackhat
    # # morphological operator to find dark regions against a light
    # # background (i.e., the routing and account numbers)
    # # gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)     # initial code line, throws error: (Invalid number of channels in input image:'VScn::contains(scn)' where 'scn' is 1)
    # gray = bottom
    # # gray = cv2.cvtColor(bottom, cv2.CV_8UC1)
    # # gray = cv2.cvtColor(bottom, cv2.CV_32SC1)
    # blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

    # # compute the Scharr gradient of the blackhat image, then scale
    # # the rest back into the range [0, 255]
    # gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0,
    #     ksize=-1)
    # gradX = np.absolute(gradX)
    # (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    # gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    # gradX = gradX.astype("uint8")

    # # apply a closing operation using the rectangular kernel to help
    # # close gaps in between rounting and account digits, then apply
    # # Otsu's thresholding method to binarize the image
    # gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    # thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]     # initial code line
    # # thresh = cv2.threshold(gradX, 0, 255, cv2.CV_8UC1)[1]
    # # thresh = cv2.threshold(gradX, 0, 255, cv2.CV_32SC1)[1]

    # # remove any pixels that are touching the borders of the image (this
    # # simply helps us in the next step when we prune contours)
    # thresh = clear_border(thresh)

    # # find contours in the thresholded image, then initialize the
    # # list of group locations
    # # groupCnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # groupCnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # ------ code modification on line 164-107, initial line is 104 ------ # 18-05-2020
    # # groupCnts = groupCnts[0] if imutils.is_cv2() else groupCnts[1]
    # groupCnts = groupCnts[0] #if imutils.is_cv2() else groupCnts[1]

    # if groupCnts :
    #     # os.system('python bank_check_ocr.py --image D:/iC4_Pro_Project/ic4_pro_ocr/input/sample1c.png  --reference D:/iC4_Pro_Project/towardsdatascience/micrfolder/micr_e13b_reference.png')
    #     # os.system('python bank_check_ocr.py --image D:/iC4_Pro_Project/ic4_pro_ocr/input/sample1c.png  --reference D:/iC4_Pro_Project/towardsdatascience/micrfolder/micr_e13b_reference.png')
    #     import bank_check_ocr
    #     shutil.move(source + filename, destinatn + filename)  # Move successful image from source to destination 
    #     print("----------- enter this line1 -----------")
    #     continue
    

    import numpy as np_
    from PIL import Image as Image_

    # im_haystack = Image_.open(r"lenna.png")
    # im_needle   = Image_.open(r"eye.png")
    im_haystack = Image_.open(im)
    im_needle   = Image_.open(micrfont)
    # found = False

    def find_matches(haystack, needle):
        arr_h = np_.asarray(haystack)
        arr_n = np_.asarray(needle)
        # arr_h = np_.array(haystack)
        # arr_n = np_.array(needle)

        y_h, x_h = arr_h.shape[:2]
        y_n, x_n = arr_n.shape[:2]

        xstop = x_h - x_n + 1
        ystop = y_h - y_n + 1

        matches = []
        for xmin in range(0, xstop):
            for ymin in range(0, ystop):
                xmax = xmin + x_n
                ymax = ymin + y_n

                arr_s = arr_h[ymin:ymax, xmin:xmax]     # Extract subimage
                arr_t = (arr_s == arr_n)                # Create test matrix
                # if arr_t.all():                       # Only consider exact matches
                if arr_t:                         # Only consider exact matches
                    matches.append((xmin,ymin))                    
                    # return matches
        
        return matches
        # return False

    # print(find_matches(im_haystack, im_needle))

    if find_matches(im_haystack, im_needle) :
    # if found :    
        # print(find_matches(im_haystack, im_needle))
        print("----------- enter this line3 -----------")
        import bank_check_ocr
        print("----------- enter this line4 -----------")
        shutil.move(source + im, destinatn + im)  # Move successful image from source to destination 
        print("----------- enter this line1 -----------")
        continue
    

    # display information to the screen
    print("----------- enter this line2 -----------")
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
        print("----------- file moved to exceptn folder 2 -----------") 
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
        print("----------- file moved to exceptn folder 2 -----------") 
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
        print("----------- file moved to exceptn folder 3 -----------") 
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
    else: 
        substring1 = ""
        print('Voucher Name: ' + substring1)
        print("----------- file moved to exceptn folder 3 -----------") 
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

    print('Voucher Name: ' + substring1)

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
            print("----------- file moved to exceptn folder 4 -----------") 
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
        print("----------- file moved to exceptn folder 5 -----------") 
        shutil.move(source + filename, exceptn + filename)  # Move unsuccessful image from source to exception
        
        posts_ex = db.ic4_ocr_ex
        post_ex_data = {            
            'ticket_image': filename,            
            'extractn_date': datetime.now().date().strftime("%Y%d%m"),
            'extractn_time': datetime.now().time().strftime("%H:%M:%S"),            
            'rejectn_reason': "Exception raised while identifying ticket"
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
        print("----------- file moved to exceptn folder 6 -----------") 
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
    print("----------- file moved to destinatn folder 1 -----------") 
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
    
    # # def extract_digits_and_symbols(image, charCnts, minW=5, minH=15):		# initial line of code
    # def extract_digits_and_symbols(image, charCnts, minW=5, minH=10):		# modified line that enhance detection of micr xter
    #     # grab the internal Python iterator for the list of character
    #     # contours, then  initialize the character ROI and location
    #     # lists, respectively
    #     charIter = charCnts.__iter__()
    #     rois = []
    #     locs = []

    #     # keep looping over the character contours until we reach the end
    #     # of the list
    #     while True:
    #         try:
    #             # grab the next character contour from the list, compute
    #             # its bounding box, and initialize the ROI
    #             c = next(charIter)
    #             (cX, cY, cW, cH) = cv2.boundingRect(c)
    #             roi = None

    #             # check to see if the width and height are sufficiently
    #             # large, indicating that we have found a digit
    #             if cW >= minW and cH >= minH:
    #                 # extract the ROI
    #                 roi = image[cY:cY + cH, cX:cX + cW]
    #                 rois.append(roi)
    #                 locs.append((cX, cY, cX + cW, cY + cH))

    #             # otherwise, we are examining one of the special symbols
    #             else:
    #                 # MICR symbols include three separate parts, so we
    #                 # need to grab the next two parts from our iterator,
    #                 # followed by initializing the bounding box
    #                 # coordinates for the symbol
    #                 parts = [c, next(charIter), next(charIter)]
    #                 (sXA, sYA, sXB, sYB) = (np.inf, np.inf, -np.inf,
    #                     -np.inf)

    #                 # loop over the parts
    #                 for p in parts:
    #                     # compute the bounding box for the part, then
    #                     # update our bookkeeping variables
    #                     (pX, pY, pW, pH) = cv2.boundingRect(p)
    #                     sXA = min(sXA, pX)
    #                     sYA = min(sYA, pY)
    #                     sXB = max(sXB, pX + pW)
    #                     sYB = max(sYB, pY + pH)

    #                 # extract the ROI
    #                 roi = image[sYA:sYB, sXA:sXB]
    #                 rois.append(roi)
    #                 locs.append((sXA, sYA, sXB, sYB))

    #         # we have reached the end of the iterator; gracefully break
    #         # from the loop
    #         except StopIteration:
    #             break

    #     # return a tuple of the ROIs and locations
    #     return (rois, locs)