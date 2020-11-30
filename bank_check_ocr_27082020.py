# USAGE
# python bank_check_ocr.py --image example_check.png --reference micr_e13b_reference.png
# python bank_check_ocr.py --image ./cheque/sample1c.png --reference ./cheque/micr_e13b_reference.png

# import the necessary packages
from extractxter import get_accname
from skimage.segmentation import clear_border
from imutils import contours
import imutils
import numpy as np
import argparse
import imutils
import cv2

# pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

import os, sys
from PIL import Image
import shutil

from datetime import datetime
from pymongo import MongoClient
client = MongoClient()
db = client.ic4pro

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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,	help="path to input image")		
ap.add_argument("-r", "--reference", required=True,	help="path to reference MICR E-13B font")	
args = vars(ap.parse_args())

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
# *black* background
ref = cv2.imread(args["reference"])
# ref = cv2.imread(im)
# ref = cv2.imread("D:\\iC4_Pro_Project\\ic4_pro_ocr\\micrfolder\\micr_e13b_reference.png")
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
image = cv2.imread(args["image"])
# image = cv2.imread("D:/iC4_Pro_Project/ic4_pro_ocr/input/sample1c.png")
(h, w,) = image.shape[:2]
delta = int(h - (h * 0.2))		# initial line of code
# delta = int(h - (h * 0.3))	# adjust the bottom % of the cheque image captured for scanning
bottom = image[delta:h, 0:w]

# convert the bottom image to grayscale, then apply a blackhat
# morphological operator to find dark regions against a light
# background (i.e., the routing and account numbers)
gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
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
thresh = cv2.threshold(gradX, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# remove any pixels that are touching the borders of the image (this
# simply helps us in the next step when we prune contours)
thresh = clear_border(thresh)

# find contours in the thresholded image, then initialize the
# list of group locations
groupCnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

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
# # out = [x[1:3] for x in aa]	# sample syntax for extract a substring from an array of strings in python
# out = [x[0:1] for x in output]
# print('-----Get First MICR Xter-----')
# print (out)
# # if 'U' in out :
# if out[0] == 'U' :
# 	print ('Found U')
# else:
# 	print('Not found U')


# found = ['U' in x for x in output]	 # syntax for checking for existence (True/False) of a substring from an array of strings in python
# print('-----Get found value-----')
# print(found)
# if True in found:
# 	print('found True')
# else:
# 	print('found False')
# #########################################################################################################################################

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
accname = get_accname()
searchstring = "".join(str(item) for item in accname)
# creating substring from start of string 
# define length upto which substring required 
sstring = searchstring[:22] 

posts = db.ic4_ocr
# posts = db.ic4_ocr_copy
post_data = {
	'_id': res[1],
	'ticket_image': args["image"],
	# 'ticket_image': 'sample1c.png',
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


# with open("./cheque_outfile.txt", "w") as outfile:
#     # outfile.write("\n".join(str(item) for item in charNames))
# 	# outfile.write("".join(str(item) for item in charNames))
# 	# outfile.write("MICR E-13B digits: \n{}".format(" ".join(res)))
# 	# outfile.write("\n\nAccount Name: \n{}".format(" ".join(accname)))
	
# 	outfile.write("MICR E-13B Digits and Account Name: \n{}".format(" ".join(res)))
# 	outfile.write(" {}".format("".join(sstring)))

# display the output Account Name information to the screen
print("Account Name: {}".format("".join(sstring)))

cv2.imshow("Check OCR", image)
cv2.waitKey(0)