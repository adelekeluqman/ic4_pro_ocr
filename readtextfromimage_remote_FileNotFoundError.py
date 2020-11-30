# USAGE
# python readtextfromimage.py --image Input-2.jpg

# import the necessary packages

import cv2
import pytesseract, re
import numpy as np
import argparse
from skimage.segmentation import clear_border
import imutils
from imutils import contours
import os, sys
import shutil
from PIL import Image

from datetime import datetime
from pymongo import MongoClient
client = MongoClient()
db = client.ic4pro

# from PIL import Image
# import PIL.Image
# from pytesseract import image_to_string
# import pytesseract, re

cwd = os.getcwd() 
# pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
## OR
pytesseract.pytesseract.tesseract_cmd = os.path.join(cwd, "Tesseract-OCR\\tesseract.exe")
# pytesseract.pytesseract.tesseract_cmd = 'D:/iC4_Pro_Project/ic4_pro_ocr/Tesseract-OCR/tesseract.exe'
# pytesseract.pytesseract.tesseract_cmd = './Tesseract-OCR/tesseract'
# TESSDATA_PREFIX = './Tesseract-OCR'


# from mongoengine import *
# connect('ic4pro', host='localhost', port=27017)

# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image to be OCR'd")
# # ap.add_argument("-c", "--min-conf", type=int, default=0,
# # 	help="mininum confidence value to filter weak text detection")
# args = vars(ap.parse_args())

# def extract_digits_and_symbols(image, charCnts, minW=5, minH=15):		# initial line of code
def extract_digits_and_symbols(image, charCnts, minW=5, minH=10):		# modified line that enhance detection of micr xter
    # grab the internal Python iterator for the list of character
    # contours, then  initialize the character ROI and location
    # lists, respectively
    charIter = charCnts.__iter__()
    rois = []
    locs = []

    # keep looping over the character contours until we reach the end
    # of the list
    while True:
        try:
            # grab the next character contour from the list, compute
            # its bounding box, and initialize the ROI
            c = next(charIter)
            (cX, cY, cW, cH) = cv2.boundingRect(c)
            roi = None

            # check to see if the width and height are sufficiently
            # large, indicating that we have found a digit
            if cW >= minW and cH >= minH:
                # extract the ROI
                roi = image[cY:cY + cH, cX:cX + cW]
                rois.append(roi)
                locs.append((cX, cY, cX + cW, cY + cH))

            # otherwise, we are examining one of the special symbols
            else:
                # MICR symbols include three separate parts, so we
                # need to grab the next two parts from our iterator,
                # followed by initializing the bounding box
                # coordinates for the symbol
                parts = [c, next(charIter), next(charIter)]
                (sXA, sYA, sXB, sYB) = (np.inf, np.inf, -np.inf,
                    -np.inf)

                # loop over the parts
                for p in parts:
                    # compute the bounding box for the part, then
                    # update our bookkeeping variables
                    (pX, pY, pW, pH) = cv2.boundingRect(p)
                    sXA = min(sXA, pX)
                    sYA = min(sYA, pY)
                    sXB = max(sXB, pX + pW)
                    sYB = max(sYB, pY + pH)

                # extract the ROI
                roi = image[sYA:sYB, sXA:sXB]
                rois.append(roi)
                locs.append((sXA, sYA, sXB, sYB))

        # we have reached the end of the iterator; gracefully break
        # from the loop
        except StopIteration:
            break

    # return a tuple of the ROIs and locations
    return (rois, locs)


def get_accname(image):

    # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required = True, help="path to input image")	    	    
    # ap.add_argument("-r", "--reference", required=True, help="path to reference MICR E-13B font")	    	    
    # args = vars(ap.parse_args())

    # convert the image to an inverted binary image
    # img = cv2.imread("D:/iC4_Pro_Project/ic4_pro_ocr/input/sample1c.png")
    img = cv2.imread(image)
    
    # load the input image, grab its dimensions, and apply array slicing
    # to keep only the bottom 20% of the image (that's where the account
    # information is)
    ### image = cv2.imread(args["image"])
    (h, w,) = img.shape[:2]
    # delta = int(h - (h * 0.2))		# initial line of code
    delta = int(h - (h * 0.4))			# adjust the bottom % of the cheque image captured for scanning
    # delta = int(h - (h * 0.38))       # skye bank size
    img = img[delta:h, 0:w]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.bitwise_not(gray)
    
    # # -------
    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]

    # # blur = cv2.GaussianBlur(thresh, (3,3), 0)
    # # img = 255 - blur
    # img = thresh 
    # # -------
    
    # We need to do a couple of morphological operations to remove noise around the characters
    # The two operations we use are erosion and dilation
    kernel = np.ones((2, 1), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)

    # out_below = pytesseract.image_to_string(img)        # initial line of code
    # out_below = pytesseract.image_to_string(Image.open(img))
    out_below = pytesseract.image_to_string(img, lang='eng', config='--oem 3 --psm 6')
    searchtext = re.findall(r"[A-Z-]{3,20}\s?", out_below)      # first rank
    # searchtext = re.findall(r"([A-Z]{3,15}-[A-Z]{3,15}\s?|[A-Z]{3,15}\s?)", out_below)  # second rank
    # searchtext = re.findall(r"([A-Z]|-|\s?)", out_below)          # fifth rank
    # searchtext = re.findall(r"[A-Z]-[A-Z]\s?|[A-Z]\s?", out_below)  # third rank
    # searchtext = re.findall(r"[A-Z-]\s?", out_below)              # fourth rank
    # searchtext = re.findall(r"(\w|-|\s?)", out_below)

    if searchtext:        
        return searchtext    

im = " "

import pysftp

# pysftp.CnOpts(knownhosts=None)

cnopts = pysftp.CnOpts()
cnopts.hostkeys = None 

myHostname = '192.168.43.156' # "yourserverdomainorip.com"
# myHostname = '197.210.47.212' # "yourserverdomainorip.com"
# myHostname = '127.0.0.1' # "yourserverdomainorip.com"
myUsername = 'adelekeluqman@yahoo.com' # "root"
myPassword = 'jadun.com' # "12345"
# myPort = 22

# pysftp.Connection(host, username=None, private_key=None, password=None, port=22, private_key_pass=None, ciphers=None, log=False, cnopts=None, default_path=None)
# with pysftp.Connection(host=myHostname, username=myUsername, password=myPassword) as sftp:
# with pysftp.Connection(host=myHostname, username=myUsername, password=myPassword, cnopts=cnopts, port=myPort) as sftp:
with pysftp.Connection(host=myHostname, username=myUsername, password=myPassword, cnopts=cnopts) as sftp:
    print ("Connection succesfully established ... ")
    
    # Switch to a remote directory (based on pysftp.Connection credentials)
    
    # get present working dir of myUsername (adelekeluqman@yahoo.com, which is Algorism in Users dir)
    print ('---pwd---' + sftp.pwd)
    # change working dir into Projects from Algorism in Users dir)
    sftp.cwd('./Projects/ic4_pro_ocr')
    # or
    # sftp.chdir('./Projects')
    
    # source = "D:\\iC4_Pro_Project\\ic4_pro_ocr\\input\\"  # Source path  
    # destinatn = "D:\\iC4_Pro_Project\\ic4_pro_ocr\\processed\\"  # Destination path  
    # exceptn = "D:\\iC4_Pro_Project\\ic4_pro_ocr\\exceptn\\"  # Exception path
    # micrfont = "D:/iC4_Pro_Project/ic4_pro_ocr/micrfont/templateMicr.png"  # MICR font image file path
    # micrpath = "D:\\iC4_Pro_Project\\ic4_pro_ocr\\micrfolder\\"  # MICR digit reference file path    
    # micrfile = " "

    source = "./input/"  # Source path  
    destinatn = "./processed/"  # Destination path  
    exceptn = "./exceptn/"  # Exception path
    micrfont = "./micrfont/templateMicr.png"  # MICR font image file path
    micrpath = "./micrfolder/"  # MICR digit reference file path    
    micrfile = " "

    # for filename2 in os.listdir(micrpath):
    for filename2 in sftp.listdir(micrpath):
        if (filename2.endswith('.png') or filename2.endswith('.jpg')): 
            # micrfile = filename2
            micrfile = micrpath + filename2
            print ('----------' + micrfile + '-----------is micrfile') 

    # im = " "

    # change the current working directory to a newly created one before doing any operations in it
    
    # # os.chdir("D:\\iC4_Pro_Project\\ic4_pro_ocr\\input")
    # sftp.cwd('./input')
    
    # Get the path of current working directory 
    # filepath = os.getcwd() 
    # filepath = sftp.pwd
    filepath = source

    print ('--------' + filepath + '----------is filepath')

    # for filename in os.listdir('.'):      # Loop over all files in the working directory.
    # for filename in os.listdir(filepath):   # Loop over all files in the working directory.
    
    for filename in sftp.listdir(filepath):   # Loop over all files in the working directory.
    # OR
    # directory_structure = sftp.listdir_attr(filepath)
    # for attr in directory_structure:        
        # filename = attr.filename
        
        # if not (filename.endswith('.png') or filename.endswith('.jpg')) \     # initial code with break line format
        # or filename == LOGO_FILENAME:                                         # initial code with break line format
        # if not (filename.endswith('.png') or filename.endswith('.jpg')): 
        if not (filename.endswith('.png') or filename.endswith('.jpg')):    
            print("----------- file moved to exceptn folder 1 -----------")     
            shutil.move(source + filename, exceptn + filename)  # Move the content of source to destination 
            
            posts_ex = db.ic4_ocr_ex
            # posts_ex = db.ic4_ocr_ex_copy
            post_ex_data = {            
                'ticket_image': filename,            
                'extractn_date': datetime.now().date().strftime("%Y%d%m"),
                'extractn_time': datetime.now().time().strftime("%H:%M:%S"),            
                'rejectn_reason': 'incompatible image format'
            }
            result = posts_ex.insert_one(post_ex_data)
            print('One rejected: {0}, {1}'.format(result.inserted_id, 'incompatible image format'))

            continue    # skip non-image files and the logo file itself

        # im = Image.open(filename)     # using "from PIL import Image" module
        # width, height = im.size       # using "from PIL import Image" module
        print ('--------filename is---' + filename)
        im = filepath + filename
        print ('--------im is---' + im)

        #---------------------- check if im is cheque -------------------------------    
        # initialize the list of reference character names, in the same
        # order as they appear in the reference image where the digits
        # their names and:
        # T = Transit (delimit bank branch routing transit #)
        # U = On-us (delimit customer account number)
        # A = Amount (delimit transaction amount)
        # D = Dash (delimit parts of numbers, such as routing or account)
        charNames = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
            "T", "U", "A", "D"]

        # load the reference MICR image from disk, convert it to grayscale,
        # and threshold it, such that the digits appear as *white* on a
        # *black* background  # "./out/output-image.png"
        # ref = cv2.imread(micrpath + micrfile, 0)    
        # ref = cv2.imread("D:\\iC4_Pro_Project\\ic4_pro_ocr\\micrfolder\\micr_e13b_reference.png")
        ref = cv2.imread(micrfile)
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        ref = imutils.resize(ref, width=400)
        ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY_INV |
            cv2.THRESH_OTSU)[1]

        # find contours in the MICR image (i.e,. the outlines of the
        # characters) and sort them from left to right
        refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        refCnts = imutils.grab_contours(refCnts)
        refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]

        # extract the digits and symbols from the list of contours, then
        # initialize a dictionary to map the character name to the ROI
        # refROIs = extract_digits_and_symbols(ref, refCnts,		# initial line of code
        # 	minW=10, minH=20)[0]									# initial line of code
        refROIs = extract_digits_and_symbols(ref, refCnts, minW=10, minH=20)[0]
        chars = {}

        # loop over the reference ROIs
        for (name, roi) in zip(charNames, refROIs):
            # resize the ROI to a fixed size, then update the characters
            # dictionary, mapping the character name to the ROI
            # roi = cv2.resize(roi, (36, 36)) 		# initial line of code
            roi = cv2.resize(roi, (36, 36)) 
            chars[name] = roi

        # initialize a rectangular kernel (wider than it is tall) along with
        # an empty list to store the output of the check OCR
        # rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))	# initial line of code
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))		# better segmented rectangular kernel
        output = []

        # load the input image, grab its dimensions, and apply array slicing
        # to keep only the bottom 20% of the image (that's where the account
        # information is)

        # image = cv2.imread(im, 0)
        # image = cv2.imread(im)
        # image = cv2.imread(filepath + '/' + im)
        # image = cv2.imread(im.encode('utf-8', 'surrogateescape').decode('utf-8', 'surrogateescape'))
        # assert not isinstance(image,type(None)), 'image not found'

        # img is in BGR format if the underlying image is a color image
        # img = cv2.imdecode(np.fromfile(im_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)  --original syntax
        
        # image = cv2.imdecode(np.fromfile(filepath + im, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        # image = cv2.imdecode(np.fromfile('D:/iC4_Pro_Project/OCR/Ticket_API/tmp/Breaking_News.png', dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        image = cv2.imdecode(np.fromfile(im, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        # image = open(filepath + '/' + im, "rb")
        # # image = open(u'D:\\ö\\handschuh.jpg', "rb")
        # bytes = bytearray(image.read())
        # numpyarray = numpy.asarray(bytes, dtype=numpy.uint8)
        # image = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

        print ('-----image below----')
        print (image)

        # image = cv2.imread('D:/iC4_Pro_Project/ic4_pro_ocr/input/sample1c.png', 0)
        (h, w,) = image.shape[:2]
        delta = int(h - (h * 0.2))		# initial line of code
        # delta = int(h - (h * 0.3))	# adjust the bottom % of the cheque image captured for scanning
        bottom = image[delta:h, 0:w]

        # convert the bottom image to grayscale, then apply a blackhat
        # morphological operator to find dark regions against a light
        # background (i.e., the routing and account numbers)
        gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)     # initial code line, throws error: (Invalid number of channels in input image:'VScn::contains(scn)' where 'scn' is 1)
        # gray = bottom
        # gray = cv2.cvtColor(bottom, cv2.CV_8UC1)
        # gray = cv2.cvtColor(bottom, cv2.CV_32SC1)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

        # compute the Scharr gradient of the blackhat image, then scale
        # the rest back into the range [0, 255]
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0,
            ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
        gradX = gradX.astype("uint8")

        # apply a closing operation using the rectangular kernel to help
        # close gaps in between rounting and account digits, then apply
        # Otsu's thresholding method to binarize the image
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]     # initial code line
        # thresh = cv2.threshold(gradX, 0, 255, cv2.CV_8UC1)[1]
        # thresh = cv2.threshold(gradX, 0, 255, cv2.CV_32SC1)[1]

        # remove any pixels that are touching the borders of the image (this
        # simply helps us in the next step when we prune contours)
        thresh = clear_border(thresh)

        # find contours in the thresholded image, then initialize the
        # list of group locations
        # groupCnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        groupCnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ------ code modification on line 164-107, initial line is 104 ------ # 18-05-2020
        # groupCnts = groupCnts[0] if imutils.is_cv2() else groupCnts[1]
        groupCnts = groupCnts[0] #if imutils.is_cv2() else groupCnts[1]
        # OR
        # groupCnts = groupCnts[1] if imutils.is_cv3() else groupCnts[0]
        # -------------------------------------------------------------------- #

        groupLocs = []

        # loop over the group contours
        for (i, c) in enumerate(groupCnts):
            # compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)

            # only accept the contour region as a grouping of characters if
            # the ROI is sufficiently large
            # if w > 50 and h > 15:		# initial line of code
            if w > 30 and h > 9:		# extends coverage of groupable contour region
                groupLocs.append((x, y, w, h))

        # sort the digit locations from left-to-right
        groupLocs = sorted(groupLocs, key=lambda x:x[0])

        # loop over the group locations
        for (gX, gY, gW, gH) in groupLocs:
            # initialize the group output of characters
            groupOutput = []

            # extract the group ROI of characters from the grayscale
            # image, then apply thresholding to segment the digits from
            # the background of the credit card
            group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
            group = cv2.threshold(group, 0, 255,		# initial line of code	-- adjust colour threashold of detectable micr xter
            # group = cv2.threshold(group, 0, 175,
                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            cv2.imshow("Group", group)
            cv2.waitKey(0)

            # find character contours in the group, then sort them from
            # left to right
            charCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            charCnts = imutils.grab_contours(charCnts)
            charCnts = contours.sort_contours(charCnts,
                method="left-to-right")[0]

            # find the characters and symbols in the group
            (rois, locs) = extract_digits_and_symbols(group, charCnts)

            # loop over the ROIs from the group
            for roi in rois:
                # initialize the list of template matching scores and
                # resize the ROI to a fixed size
                scores = []
                # roi = cv2.resize(roi, (36, 36))		# initial line of code
                roi = cv2.resize(roi, (36, 36))

                # loop over the reference character name and corresponding
                # ROI
                for charName in charNames:
                    # apply correlation-based template matching, take the
                    # score, and update the scores list
                    result = cv2.matchTemplate(roi, chars[charName],
                        cv2.TM_CCOEFF)
                    (_, score, _, _) = cv2.minMaxLoc(result)
                    scores.append(score)

                # the classification for the character ROI will be the
                # reference character name with the *largest* template
                # matching score
                # groupOutput.append(charNames[np.argmax(scores)])		# initial line of code
                groupOutput.append(charNames[np.argmax(scores)])

            # draw (padded) bounding box surrounding the group along with
            # the OCR output of the group
            cv2.rectangle(image, (gX - 10, gY + delta - 10),
                (gX + gW + 10, gY + gY + delta), (0, 0, 255), 2)
            cv2.putText(image, "".join(groupOutput),
                (gX - 10, gY + delta - 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.95, (0, 0, 255), 3)

            # add the group output to the overall check OCR output
            output.append("".join(groupOutput))

        # ##############################################################################################################
        # out = [x[1:3] for x in aa]	# sample syntax for extract a substring from an array of strings in python
        out = [x[0:1] for x in output]
        print('-----Get First MICR Xter-----')
        print (out)
        if out :
            if (out[0] == 'U') and (len(out) <= 6) :
                print ('Found U')
                print("----------- enter this line3 -----------")
                
                # using list comprehension + replace() 
                # Replace substring in list of strings 
                res = [sub.replace('U', '') for sub in output]
                res = [sub.replace('T', '') for sub in res] 
                # --------------OR--------------- #
                # # using list comprehension + map() + lambda 
                # # Replace substring in list of strings 
                # res = list(map(lambda st: str.replace(st, "U", ""), output))
                # res = list(map(lambda st: str.replace(st, "T", ""), res)) 

                # insert micr digit into mongodb
                # for micrdigit in res:
                # 	print(micrdigit()) # sample1c.png

                # get account name from cheque image
                accname = get_accname(im)
                searchstring = "".join(str(item) for item in accname)
                # creating substring from start of string 
                # define length upto which substring required 
                sstring = searchstring[:22] 

                posts = db.ic4_ocr
                # posts = db.ic4_ocr_copy
                post_data = {
                    '_id': res[1],                
                    'ticket_image': im,
                    'ticket_name': 'Cheque',
                    'ticket_date': 'null',
                    'account_no': res[2],
                    'account_name': sstring,
                    'amount': 'null',
                    'amount_word': 'null',
                    'cheque_no': res[0],
                    'micr_digit4': res[3],
                    'micr_digit5': res[4],
                    'micr_digit6': res[5],
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
                print('One post: {0}, {1}'.format(result.inserted_id, 'Cheque'))


                # display the output check OCR information to the screen
                print("Check OCR: {}".format(" ".join(res)))

                # display the output Account Name information to the screen
                print("Account Name: {}".format("".join(sstring)))

                cv2.imshow("Check OCR", image)
                cv2.waitKey(0)

                print("----------- enter this line4 -----------")
                shutil.move(source + im, destinatn + im)  # Move successful image from source to destination 
                print("----------- enter this line1 -----------")
                
                continue
        
        
        # found = ['U' in x for x in output]	 # syntax for checking for existence (True/False) of a substring from an array of strings in python
        # print('-----Get found value-----')
        # print(found)
        # if True in found:
        #     print('found True')
        #     
        #     continue    
        # ########################################################################################################################################

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
            # posts_ex = db.ic4_ocr_ex_copy
            post_ex_data = {            
                'ticket_image': filename,            
                'extractn_date': datetime.now().date().strftime("%Y%d%m"),
                'extractn_time': datetime.now().time().strftime("%H:%M:%S"),            
                'rejectn_reason': 'FileNotFoundError'
            }
            result = posts_ex.insert_one(post_ex_data)
            print('One rejected: {0}, {1}'.format(result.inserted_id, 'FileNotFoundError'))
        
            continue
        
        except Exception as e:
            print(e)
            print("----------- file moved to exceptn folder 2 -----------") 
            shutil.move(source + filename, exceptn + filename)  # Move unsuccessful image from source to exception
            
            posts_ex = db.ic4_ocr_ex
            # posts_ex = db.ic4_ocr_ex_copy
            post_ex_data = {            
                'ticket_image': filename,            
                'extractn_date': datetime.now().date().strftime("%Y%d%m"),
                'extractn_time': datetime.now().time().strftime("%H:%M:%S"),            
                'rejectn_reason': e
            }
            result = posts_ex.insert_one(post_ex_data)
            print('One rejected: {0}, {1}'.format(result.inserted_id, e))
            
            continue
        except:
            print("Exception3 raised while extracting")
            print("----------- file moved to exceptn folder 3 -----------") 
            shutil.move(source + filename, exceptn + filename)  # Move unsuccessful image from source to exception
            
            posts_ex = db.ic4_ocr_ex
            # posts_ex = db.ic4_ocr_ex_copy
            post_ex_data = {            
                'ticket_image': filename,            
                'extractn_date': datetime.now().date().strftime("%Y%d%m"),
                'extractn_time': datetime.now().time().strftime("%H:%M:%S"),            
                'rejectn_reason': "Exception3 raised while extracting"
            }
            result = posts_ex.insert_one(post_ex_data)
            print('One rejected: {0}, {1}'.format(result.inserted_id, "Exception3 raised while extracting"))
            
            continue
        finally:
            cv2.waitKey(0)

        # text1 = pytesseract.image_to_string(crop_img1)
        text1 = pytesseract.image_to_string(crop_img1, lang='eng', config='--oem 3 --psm 6')
        searchlist1 = re.findall(r"[A-Z]{3,20}[ ]?", text1)      # first rank
        # searchlist1 = re.findall(r"([A-Z]{3,20}\s?)", text1)      # first rank
        # print(text1)    
        if searchlist1:                
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
            # posts_ex = db.ic4_ocr_ex_copy
            post_ex_data = {            
                'ticket_image': filename,            
                'extractn_date': datetime.now().date().strftime("%Y%d%m"),
                'extractn_time': datetime.now().time().strftime("%H:%M:%S"),            
                'rejectn_reason': "Exception3 raised while extracting"
            }
            result = posts_ex.insert_one(post_ex_data)
            print('One rejected: {0}, {1}'.format(result.inserted_id, "Exception3 raised while extracting"))

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
                # posts_ex = db.ic4_ocr_ex_copy
                post_ex_data = {            
                    'ticket_image': filename,            
                    'extractn_date': datetime.now().date().strftime("%Y%d%m"),
                    'extractn_time': datetime.now().time().strftime("%H:%M:%S"),            
                    'rejectn_reason': "Exception1 raised while extracting ticket_name"
                }
                result = posts_ex.insert_one(post_ex_data)
                print('One rejected: {0}, {1}'.format(result.inserted_id, "Exception1 raised while extracting ticket_name"))

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
            # posts_ex = db.ic4_ocr_ex_copy
            post_ex_data = {            
                'ticket_image': filename,            
                'extractn_date': datetime.now().date().strftime("%Y%d%m"),
                'extractn_time': datetime.now().time().strftime("%H:%M:%S"),            
                'rejectn_reason': "Exception raised while identifying ticket"
            }
            result = posts_ex.insert_one(post_ex_data)
            print('One rejected: {0}, {1}'.format(result.inserted_id, "Exception raised while identifying ticket"))

            continue
        finally:
            cv2.waitKey(0)


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
            # posts_ex = db.ic4_ocr_ex_copy
            post_ex_data = {            
                'ticket_image': filename,            
                'extractn_date': datetime.now().date().strftime("%Y%d%m"),
                'extractn_time': datetime.now().time().strftime("%H:%M:%S"),            
                'rejectn_reason': "Exception4 raised while extracting ticket_id"
            }
            result = posts_ex.insert_one(post_ex_data)
            print('One rejected: {0}, {1}'.format(result.inserted_id, "Exception4 raised while extracting ticket_id"))

            continue

        print(substring2)
        print("----------- file moved to destinatn folder 1 -----------") 
        shutil.move(source + filename, destinatn + filename)  # Move successful image from source to destination 

        # itemlist = [text1,text2]
        # itemlist = [substring1,substring2]
        itemline = [substring1,",",substring2]

        # with open("D:\\iC4_Pro_Project\\ic4_pro_ocr\\outfile.txt", "a") as outfile:
        #     # outfile.write(",".join(itemlist))    
        #     outfile.write("" .join(itemline))
        #     outfile.write("\n")

        
        posts = db.ic4_ocr
        # posts = db.ic4_ocr_copy
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
    
# connection closed automatically at the end of the with-block




    