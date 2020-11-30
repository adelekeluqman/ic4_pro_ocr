import cv2
import pytesseract, re
import numpy as np
import argparse
# from PIL import Image
# import PIL.Image
from pytesseract import image_to_string

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def get_accname():

    # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required = True, help="path to input image")	    	    
    # ap.add_argument("-r", "--reference", required=True, help="path to reference MICR E-13B font")	    	    
    # args = vars(ap.parse_args())

    # convert the image to an inverted binary image
    img = cv2.imread("D:/iC4_Pro_Project/ic4_pro_ocr/input/sample1c.png")
    # img = cv2.imread(image)
    # img = cv2.imread('sample3.jpeg')
    # img = cv2.imread('sample2.jpg')

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
        # print("ACCOUNT NAME:", "".join(str(item) for item in searchtext))
        # print("ACCOUNT NAME:", "".join(searchtext))
        return searchtext    
    
    # with open("./outfile.txt", "w") as outfile:    
    #     outfile.write("".join(str(item) for item in searchtext))    
    #     # outfile.write("".join(searchtext))
