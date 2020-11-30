###########################################################################
#               MAIN MODULE                                               #
###########################################################################
if __name__ == '__main__':
    
    # print("Reading reference image : ", refFilename)              # initial code line
    # imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)       # initial code line
     
    print("Reading image to align : ", imFilename);                 # modified 30082020
    Filled_Form_Colored = cv2.imread(imFilename, cv2.IMREAD_COLOR)  # modified 30082020
       
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
    lower_val = np.array([60,45,0],np.uint8)
    upper_val = np.array([150,255,255],np.uint8)
    
    # Threshold the HSV image to get only black colors
    mask = cv2.inRange(hsv, lower_val, upper_val)
    # with an enlarging kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    show_wait_destroy("mask", mask)
    # Bitwise-AND mask and original image
    # res = cv2.bitwise_and(AlignedForm_Colored,AlignedForm_Colored, mask= mask)    # initial code line
    res = cv2.bitwise_and(Filled_Form_Colored,Filled_Form_Colored, mask= mask)      # modified 30082020
    # invert the mask to get black letters on white background
    res2 = cv2.bitwise_not(mask)
    Processed_Form=cv2.bitwise_not(res2)
    
    
    ############################################################
    #               Crop By Region and Detect Digits           #
    ############################################################    
    
    # #Depositor Name
    # X,Y,W,H=313,941,787,349
    # DepositorNameImage=Processed_Form[Y:Y+H,X:X+W]
    # DepositorName=HandWrittenDigitRecognition(DepositorNameImage)
    # print("Depositor Name:",DepositorName)  

    #Phone No
    # X,Y,W,H=325,980,750,70
    X,Y,W,H=270,970,860,80
    PhoneNoImage=Processed_Form[Y:Y+H,X:X+W]
    PhoneNo=HandWrittenDigitRecognition(PhoneNoImage)
    print("Phone No:",PhoneNo)
    
    # #Voucher No
    # X,Y,W,H=400,245,430,100
    # #X,Y,W,H=180,650,485,150
    # VoucherNoImage=Processed_Form[Y:Y+H,X:X+W]
    # VoucherNo=HandWrittenDigitRecognition(VoucherNoImage)
    # print("Voucher No:",VoucherNo)
     
    # #Account Name
    # X,Y,W,H=280,530,820,60
    # AccountNameImage=Processed_Form[Y:Y+H,X:X+W]
    # AccountName=HandWrittenDigitRecognition(AccountNameImage)
    # print("Account Name:",AccountName) 
    
    #Account No
    # X,Y,W,H=325,623,745,62
    X,Y,W,H=270,580,860,100
    AccountNoImage=Processed_Form[Y:Y+H,X:X+W]
    AccountNo=HandWrittenDigitRecognition(AccountNoImage)
    print("Account No:",AccountNo)
    
    #Amount Figure
    # X,Y,W,H=324,707,751,66
    X,Y,W,H=270,680,860,85
    AmountFigureImage=Processed_Form[Y:Y+H,X:X+W]
    AmountFigure=HandWrittenDigitRecognition(AmountFigureImage)
    print("Amount Figure:",AmountFigure) 
    
    # #Amount Word
    # X,Y,W,H=323,773,777,164
    # AmountWordImage=Processed_Form[Y:Y+H,X:X+W]
    # AmountWord=HandWrittenDigitRecognition(AmountWordImage)
    # print("Amount Word:",AmountWord)
    
    # show_wait_destroy("AlignedForm_Colored",AlignedForm_Colored)      # initial code line
    # now = datetime.now()                                              # initial code line
    # dt_string = now.strftime("%d-%B-%Y_%H-%M-%S")                     # initial code line
    # cv2.imwrite('./Output_'+dt_string+'.jpg',AlignedForm_Colored)     # initial code line

    show_wait_destroy("Filled_Form_Colored",Filled_Form_Colored)        # modified 30082020
    now = datetime.now()                                                # modified 30082020
    dt_string = now.strftime("%d-%B-%Y_%H-%M-%S")                       # modified 30082020
    cv2.imwrite('./Output_'+dt_string+'.jpg',Filled_Form_Colored)       # modified 30082020