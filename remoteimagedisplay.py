import pymongo
import cv2
import os
import time         # for time.sleep(3) usage
import threading    # for event.wait(3) usage

import requests
from PIL import Image

import pysftp
import json

event = threading.Event() # for event.wait(3) usage

# r = requests.get('D:/iC4_Pro_Project/ic4_pro_ocr/input/Breaking_News.png')
# # r = requests.get('https://image.shutterstock.com/image-vector/nigeria-independence-day-60th-logo-600w-1820406143.jpg')

myHostname = ""
myUsername = ""
myPassword = ""
waitTime = 0
ticketmedia = ""

with open("./paramfile.json", "r") as p:
    my_dict = json.load(p)

myHostname = my_dict["host"]
myUsername = my_dict["username"]
myPassword = my_dict["password"]
waitTime = my_dict["waittime"] * 1000
ticketmedia = my_dict["ticketmedia"]
# print(my_dict["x"])

cnopts = pysftp.CnOpts()
cnopts.hostkeys = None 

# myHostname = '192.168.43.156' # "yourserverdomainorip.com"
# myUsername = 'adelekeluqman@yahoo.com' # "root"
# myPassword = 'jadun.com' # "12345"
# myPort = 22

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["ic4pro"]
mycol = mydb["ic4_ocr_ticket"]
# mycol = mydb["ic4_ocr"]

def purgedir(fname):
    for f in os.listdir(fname):
    # if not f.endswith(".bak"):
        # continue
        os.remove(os.path.join(fname, f))


# pysftp.Connection(host, username=None, private_key=None, password=None, port=22, private_key_pass=None, ciphers=None, log=False, cnopts=None, default_path=None)
# with pysftp.Connection(host=myHostname, username=myUsername, password=myPassword) as sftp:
# with pysftp.Connection(host=myHostname, username=myUsername, password=myPassword, cnopts=cnopts, port=myPort) as sftp:
with pysftp.Connection(host=myHostname, username=myUsername, password=myPassword, cnopts=cnopts) as sftp:
    # print ("Connection succesfully established ... ")
          
    # get present working dir of myUsername (adelekeluqman@yahoo.com, which is Algorism in Users dir)
    # print ('---pwd---' + sftp.pwd)

    # change working dir into Projects from Algorism in Users dir)
    sftp.cwd(ticketmedia)
    # sftp.cwd('D:/iC4_Pro_Project/20201112')
    # sftp.cwd('Projects/ic4_pro_ocr/20201112')    
    # or
    # sftp.chdir('./Projects/ic4_pro_ocr/input')

    directory_structure = sftp.listdir_attr()
    local_dir = "D:/iC4_Pro_Project/ic4_pro_ocr/remoteticket/"
    remote_dir = "./"

    # for x in mycol.find({},{ "_id": 0, "ticket_date": 1, "ticket_name": 1 }):
    for x in mycol.find({},{ "_id": 0 }):        
        print(x["ticket_name"])
        # print(sftp.pwd + '/' + x["ticket_name"])
        # print('{0} {1}'.format(x['ticket_date'], x['ticket_name']))

        for attr in directory_structure:    
            if (attr.filename.endswith(".png") or attr.filename.endswith(".jpg")):
                file = attr.filename            
                if x["ticket_name"] == file:                
                    sftp.get(remote_dir + file, local_dir + file)
                    break

        # image = cv2.imread(x["ticket_name"])
        # img = cv2.imread(os.path.join(folder,filename))

        if x["ticket_name"].endswith(".png") or x["ticket_name"].endswith(".jpg"):
            image = cv2.imread(os.path.join(local_dir,x["ticket_name"]))        
            cv2.imshow("Ticket Image", image)            
            # cv2.waitKey(0)            
            # key = cv2.waitKey(3000) #pauses for 3 seconds before fetching next image
            key = cv2.waitKey(waitTime) #pauses for waitTime before fetching next image
            if key == 27: #if ESC is pressed, exit loop
                cv2.destroyAllWindows()
                break
            
            # time.sleep(5) # Sleep for 3 second
            # event.wait(3)   # Sleep for 3 second

            # # This will open the image in your default image viewer.
            # img = Image.open(os.path.join(local_dir,x["ticket_name"]))
            # img.show()                     


    purgedir(local_dir)
    # print ('purged processed')  
     
    # directory_structure = sftp.listdir_attr()
    # local_dir = "D:/iC4_Pro_Project/ic4_pro_ocr/input/"
    # remote_dir = "./"
    
    # for attr in directory_structure:
    #     if attr.filename.endswith(".pdf"):
    #         file = attr.filename
    #         sftp.get(remote_dir + file, local_dir + file)
    #         print("Copied " + file + " to " + local_dir)
    #         # sftp.remove(remote_dir + file)
    
    # for attr in directory_structure:        
    #     file = attr.filename
    #     sftp.get(remote_dir + file, local_dir + file)
    #     # print("Copied " + file + " to " + local_dir)
    #     # print("Copied " + remote_dir + file + " to " + local_dir + file)
    #     # sftp.remove(remote_dir + file)


# # connection closed automatically at the end of the with-block