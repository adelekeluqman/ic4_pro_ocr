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

from src.main import infer_by_web
# import PIL.Image
from datetime import datetime
from pymongo import MongoClient
client = MongoClient()
db = client.ic4pro

# import tensorflow as tf

####################################################################
#               GLOBALS                                            #
####################################################################
# MAX_FEATURES = 1000
# GOOD_MATCH_PERCENT = 0.15
# font = cv2.FONT_HERSHEY_SIMPLEX
# AlignedForm_Colored=[]
# X=0
# Y=0
# W=0
# H=0

# #Load Precompiled Model
# model2 = tf.keras.models.load_model('./Trained-Models/LeNet5_MNIST_ImageGenerator_Custom.model') 

cwd = os.getcwd() 
APP_ROOT = os.path.dirname(os.path.abspath(__file__)) # project abs path

# pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
## OR
pytesseract.pytesseract.tesseract_cmd = os.path.join(cwd, "Tesseract-OCR\\tesseract.exe")
# pytesseract.pytesseract.tesseract_cmd = 'D:/iC4_Pro_Project/towardsdatascience/Tesseract-OCR/tesseract.exe'
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
    # img = cv2.imread("D:/iC4_Pro_Project/towardsdatascience/input/sample1c.png")
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


####################################################################
#               show_wait_destroy() Function for Display           #
####################################################################
def show_wait_destroy(winname, img):
    cv2.imshow(winname, img)
    cv2.moveWindow(winname, 100, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)


####################################################################
#                  HandWrittenDigitRecognition()                   #
#           Function to Find Contours and predict Digits           #
#################################################################### 
def HandWrittenDigitRecognition(image):
    morph = image.copy()
    hi,wi= image.shape
    hi=hi

    
    # smooth the image with alternative closing and opening
    # with an enlarging kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    contours, hierarchy= cv2.findContours(morph,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    show_wait_destroy("morph", morph)

    
    recognizedDigit=""
    ContourPadding=1
    digitsCountours=[]
    
    #Get the Digits Contours Seperated for Recognition
    selected_contour=[]
    
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        x1, y1, w1, h1 = cv2.boundingRect(approx)
        print(" Selected Contour:"+",x="+str(x1)+",y="+str(y1)+",w="+str(w1)+",h="+str(h1))
        if h1>15:
        # if h1>20: 
            selected_contour.append(contour)
            print(" Selected Contour:"+",x="+str(x1)+",y="+str(y1)+",w="+str(w1)+",h="+str(h1))
        else:
            print(" Excluded Contour:"+",x="+str(x1)+",y="+str(y1)+",w="+str(w1)+",h="+str(h1))
        
    
    
    for contour in selected_contour:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        x1, y1, w1, h1 = cv2.boundingRect(approx)
        IsInside=False
        for c in selected_contour:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            x2, y2, w2, h2 = cv2.boundingRect(approx)
            if x1>=x2 and x1<=x2+w2 and y1>y2 and y1<y2+h2:
                IsInside=True
                break    
        print("Contour InInside:"+str(IsInside))
        if IsInside==False:
            # cv2.rectangle(AlignedForm_Colored, (X+x1-ContourPadding, Y+y1-ContourPadding), (X+x1+w1+ContourPadding, Y+y1+h1+ContourPadding), (0, 0, 255), 1);
            cv2.rectangle(Filled_Form_Colored, (X+x1-ContourPadding, Y+y1-ContourPadding), (X+x1+w1+ContourPadding, Y+y1+h1+ContourPadding), (0, 0, 255), 1);
            digitsCountours.append([x1-ContourPadding,y1-ContourPadding,w1+ContourPadding,h1+ContourPadding])
        else:
            # cv2.rectangle(AlignedForm_Colored, (X+x1-ContourPadding, Y+y1-ContourPadding), (X+x1+w1+ContourPadding, Y+y1+h1+ContourPadding), (255, 0, 0), 1);
            cv2.rectangle(Filled_Form_Colored, (X+x1-ContourPadding, Y+y1-ContourPadding), (X+x1+w1+ContourPadding, Y+y1+h1+ContourPadding), (255, 0, 0), 1);
            
    
    #Sort the Digit Countours for Sequential Recognition
    digitsCountours=sorted(digitsCountours, key = lambda x:x[0])        
    
    for c in digitsCountours: 
        x1,y1,w1,h1=c       
        #cv2.rectangle(image, (x1-ContourPadding, y1-ContourPadding), (x1+w1+ContourPadding, y1+h1+ContourPadding), (0, 0, 255), 1);
        digit_array=morph[y1:y1+h1,x1:x1+w1]
        x_pad=int((w1+h1)/4)
        digit_array=np.pad(digit_array, (((x_pad,x_pad),(x_pad,x_pad))), 'constant')
        
        digit_array=cv2.resize(digit_array,(32,32),interpolation = cv2.INTER_AREA)
        
        prediction_array=digit_array.reshape(-1,32,32,1)
        prediction=model2.predict(prediction_array)
        recognizedDigit=recognizedDigit+str(np.argmax(prediction))
        print("Recognized:"+str(np.argmax(prediction))+",x="+str(x1)+",y="+str(y1)+",w="+str(w1)+",h="+str(h1),"x_pad="+str(x_pad))
        # cv2.putText(AlignedForm_Colored,str(np.argmax(prediction)),(X+x1+int(w1/2),Y+y1-ContourPadding-10), font, 1,(0,0,255),2,cv2.LINE_AA)
        cv2.putText(Filled_Form_Colored,str(np.argmax(prediction)),(X+x1+int(w1/2),Y+y1-ContourPadding-10), font, 1,(0,0,255),2,cv2.LINE_AA)

    return recognizedDigit


# @app.route("/upload", methods=["POST"])
def upload(im):
    # folder_name = request.form['uploads']
    target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    # print(request.files.getlist("file"))
    # option = request.form.get('optionsPrediction')
    # print("Selected Option:: {}".format(option))
    # for upload in request.files.getlist("file"):
    #     print(upload)
    #     print("{} is the file name".format(upload.filename))
    # filename = upload.filename
    res_filename = im
    # This is to verify files are supported
    ext = os.path.splitext(res_filename)[1]
    if (ext == ".jpg") or (ext == ".png"):
        print("File supported moving on...")
    else:
        # render_template("Error.html", message="Files uploaded are not supported...")
        print("Error: Files uploaded are not supported...")

    x3,y3,w3,h3=270,530,860,100
    res_filename = res_filename[y3:y3+h3, x3:x3+w3]
    show_wait_destroy2("received_cropped", res_filename)

    savefname = datetime.now().strftime('%Y-%m-%d_%H_%M_%S') + "."+ext
    destination = "/".join([target, savefname])
    print("Accept incoming file:", res_filename)
    print("Save it to:", destination)
    res_filename.save(destination)
    # result = predict_image(destination, option)
    # print("Prediction: ", result)
    # return send_from_directory("images", filename, as_attachment=True)
    # return render_template("complete.html", image_name=savefname, result=result)
    return destination


# os.makedirs('withImage', exist_ok=True)     # create withImage folder intended for storing successfully extracted images, but not yet used for the purpose

source = "D:\\iC4_Pro_Project\\ic4_pro_ocr\\input\\"  # Source path  
destinatn = "D:\\iC4_Pro_Project\\ic4_pro_ocr\\processed\\"  # Destination path  
exceptn = "D:\\iC4_Pro_Project\\ic4_pro_ocr\\exceptn\\"  # Exception path

micrfont = "D:/iC4_Pro_Project/ic4_pro_ocr/micrfont/templateMicr.png"  # MICR font image file path
micrpath = "D:\\iC4_Pro_Project\\ic4_pro_ocr\\micrfolder\\"  # MICR digit reference file path
# micrfile = "D:\\iC4_Pro_Project\\ic4_pro_ocr\\micrfolder\\micr_e13b_reference.png"
micrfile = ""

for filename2 in os.listdir(micrpath):
    if (filename2.endswith('.png') or filename2.endswith('.jpg')): 
        # micrfile = filename2
        micrfile = micrpath + filename2
        print ('----------' + micrfile + '-----------') 

im = ""
path_im = ""
path_im2 = ""

# change the current working directory to a newly created one before doing any operations in it
os.chdir("D:\\iC4_Pro_Project\\ic4_pro_ocr\\input")
# Get the path of current working directory 
filepath = os.getcwd() 
ext = ""

# for filename in os.listdir('.'):      # Loop over all files in the working directory.
for filename in os.listdir(filepath):   # Loop over all files in the working directory.
    # if not (filename.endswith('.png') or filename.endswith('.jpg')) \     # initial code with break line format
    # or filename == LOGO_FILENAME:                                         # initial code with break line format
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
    
    # This is to set type of filename extention
    ext = os.path.splitext(filename)[1]
    
    im = filename    
    path_im = source + filename
    path_im2 = source + filename

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
    image = cv2.imread(im)
    # image = cv2.imread('D:/iC4_Pro_Project/towardsdatascience/input/sample1c.png', 0)
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


    # ################# extracting handwritten field #################
    # show_wait_destroy("im", im)
    print("Receiving image to pass for handwritten : ", path_im);         # modified 30082020
    Filled_Form_Colored = cv2.imread(path_im, cv2.IMREAD_COLOR)
    # Filled_Form_Colored = cv2.imread(path_im[0], cv2.IMREAD_COLOR)  
    show_wait_destroy("path_im_r", Filled_Form_Colored)        # modified 30082020
    print("Received image to pass for handwritten : ", Filled_Form_Colored)
    # print("Aligning images ...")                                  # initial code line
    # # Registered image will be resotred in imReg. 
    # # The estimated homography will be stored in h. 
    # AlignedForm_Colored, h = alignImages(Filled_Form_Colored, imReference)     # initial code line
    ############################################################
    #               Image Pre-Processing                       #
    ############################################################
    #HSV Way
    # Convert BGR to HSV
    # hsv = cv2.cvtColor(AlignedForm_Colored, cv2.COLOR_BGR2HSV)    # initial code line
    hsv = cv2.cvtColor(Filled_Form_Colored, cv2.COLOR_BGR2HSV)      # modified 30082020
    
    # define range of black color in HSV
    # lower_val = np.array([60,45,0],np.uint8)
    lower_val = np.array([60,45,0],np.uint8)
    upper_val = np.array([150,255,255],np.uint8)
    # upper_val = np.array([150,255,255],np.uint8)
    
    # Threshold the HSV image to get only black colors
    mask = cv2.inRange(hsv, lower_val, upper_val)
    # with an enlarging kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    print("Masked image to pass for handwritten : ", mask)
    show_wait_destroy("mask", mask)
    # Bitwise-AND mask and original image
    # res = cv2.bitwise_and(AlignedForm_Colored,AlignedForm_Colored, mask= mask)    # initial code line
    res = cv2.bitwise_and(Filled_Form_Colored,Filled_Form_Colored, mask= mask)      # modified 30082020
    # invert the mask to get black letters on white background
    
    # res2 = cv2.bitwise_not(mask)
    # Processed_Form=cv2.bitwise_not(res2)
    
    res2 = cv2.bitwise_not(mask)
    # Processed_Form=cv2.bitwise_not(res2)
    Processed_Form=res2
    
    ############################################################
    #               Crop By Region and Detect Digits           #
    ############################################################    
    
    ################ SystemError: '<built-in function imread> returned NULL without setting an error' when img = cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE)  # effect changes here - def infer function in main.py
    ################ throws 'ValueError: all input arrays must have the same shape' when use with "img = fnImg" in "def infer(model, fnImg)" function in main.py
    #Account No
    X,Y,W,H=270,530,860,100
    AccountNoImage=Processed_Form[Y:Y+H,X:X+W]    
    print("processed image to pass for handwritten : ", AccountNoImage)
    show_wait_destroy("processed", AccountNoImage)    
    AccountNo=infer_by_web(AccountNoImage, type)    
    print("Account No:",AccountNo)

    # # pil_im = Image.open(AccountNoImage)
    # # print("------------ Opened" + AccountNoImage)
    # savefname = datetime.now().strftime('%Y-%m-%d_%H_%M_%S') + "."+ext
    # # savefname = datetime.now().strftime('%Y-%m-%d_%H_%M_%S') +ext

    # target = os.path.join(APP_ROOT, 'static/')
    # print("------------" + target)
    # if not os.path.isdir(target):
    #     os.mkdir(target)

    # destination = "/".join([target, savefname])
    # print('-------------------' + destination)
    # # pil_im_crop = pil_im.crop((270,530,860,100)) # X,Y,W,H
    # # pil_im_crop = pil_im.crop((270,530,1130,630)) # X,Y,X+W,Y+H
    # # print('------------------- pil_im_crop')
    # # pil_im_crop.save(destination, quality=95)
    # # AccountNoImage.save(destination, quality=95)
    # # np.save(destination, AccountNoImage)
    # cv2.imwrite(destination,AccountNoImage)
    # # OR
    # # pil_im_crop.crop((270,530,860,100)).save(destination, quality=95)

    # AccountNo=infer_by_web(destination, type)
    # print("Account No:",AccountNo)
    # ###################################

    # img3 = cv2.imread(im, cv2.IMREAD_GRAYSCALE)  # read image in grayscale  
    # img3 = cv2.imread(im)  # read image in original color   
    
    # x3,y3,w3,h3=270,530,860,100
    # global crop_img3
    # crop_img3 = img3[y3:y3+h3, x3:x3+w3]

   
    # #Account No
    # # # X,Y,W,H=325,623,745,62
    # X,Y,W,H=270,530,860,100
    # AccountNoImage=Processed_Form[Y:Y+H,X:X+W]
    # # # AccountNoImage=Processed_Form
    # # # AccountNoImage=path_im[Y:Y+H,X:X+W]
    # # # AccountNoImage=path_im
    # print("processed image to pass for handwritten : ", AccountNoImage)
    # show_wait_destroy("processed", AccountNoImage)
    # # # AccountNo=HandWrittenDigitRecognition(AccountNoImage)
    # AccountNo=infer_by_web(AccountNoImage, type)
    
    # # res_path_im = upload(im)
    # # show_wait_destroy("path_im_r", path_im)
    # # AccountNo=infer_by_web(res_path_im, type)

    # print("Account No:",AccountNo)

    # # show_wait_destroy("hw_cropped", crop_img3)
    # # AccountNo=infer_by_web(crop_img3, type)
    # # print("Account No:",AccountNo)
    
    # ###################### ValueError: all input arrays must have the same shape
    # pil_im = Image.open(im)
    # print("------------ Opened" + im)
    # savefname = datetime.now().strftime('%Y-%m-%d_%H_%M_%S') + "."+ext
    # # savefname = datetime.now().strftime('%Y-%m-%d_%H_%M_%S') +ext

    # target = os.path.join(APP_ROOT, 'static/')
    # print("------------" + target)
    # if not os.path.isdir(target):
    #     os.mkdir(target)

    # destination = "/".join([target, savefname])
    # print('-------------------' + destination)
    # # pil_im_crop = pil_im.crop((270,530,860,100)) # X,Y,W,H
    # pil_im_crop = pil_im.crop((270,530,1130,630)) # X,Y,X+W,Y+H
    # print('------------------- pil_im_crop')
    # pil_im_crop.save(destination, quality=95)
    # # OR
    # # pil_im_crop.crop((270,530,860,100)).save(destination, quality=95)

    # AccountNo=infer_by_web(destination, type)
    # print("Account No:",AccountNo)
    # ###############################################################################

    # #Amount Figure
    # # X,Y,W,H=324,707,751,66
    # X,Y,W,H=270,630,860,85
    # AmountFigureImage=Processed_Form[Y:Y+H,X:X+W]
    # # AmountFigureImage=Processed_Form
    # # AmountFigureImage=path_im[Y:Y+H,X:X+W]
    # # AmountFigure=HandWrittenDigitRecognition(AmountFigureImage)
    # # AmountFigure=infer_by_web(AmountFigureImage, type)
    # AmountFigure=infer_by_web(path_im2, type)
    # print("Amount Figure:",AmountFigure) 
    
   
    show_wait_destroy("Filled_Form_Colored",Filled_Form_Colored)

     
    shutil.move(source + filename, destinatn + filename)  # Move successful image from source to destination 
    print("----------- file moved to destinatn folder 1 -----------")
    
    # itemlist = [text1,text2]
    # itemlist = [substring1,substring2]
    itemline = [substring1,",",substring2]

    with open("D:\\iC4_Pro_Project\\ic4_pro_ocr\\outfile.txt", "a") as outfile:
        # outfile.write(",".join(itemlist))    
        outfile.write("" .join(itemline))
        outfile.write("\n")

    
    posts = db.ic4_ocr
    # posts = db.ic4_ocr_copy
    post_data = {
        '_id': substring2,
        'ticket_image': filename,
        'ticket_name': substring1,
        'ticket_date': 'null',
        'account_no': AccountNo,
        'account_name': 'null',
        # 'amount': AmountFigure,
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


    