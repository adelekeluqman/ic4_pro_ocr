# import requests

# r = requests.get('D:/iC4_Pro_Project/ic4_pro_ocr/input/Breaking_News.png')
# # r = requests.get('https://image.shutterstock.com/image-vector/nigeria-independence-day-60th-logo-600w-1820406143.jpg')

# # print (r.content)
# # with open('ng60.jpg', 'wb') as f:
# with open('bn.png', 'wb') as f:
#     f.write(r.content)

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

# ----------- routine 1 ------------

# pysftp.Connection(host, username=None, private_key=None, password=None, port=22, private_key_pass=None, ciphers=None, log=False, cnopts=None, default_path=None)
# with pysftp.Connection(host=myHostname, username=myUsername, password=myPassword) as sftp:
# with pysftp.Connection(host=myHostname, username=myUsername, password=myPassword, cnopts=cnopts, port=myPort) as sftp:

# cinfo = {'host':'hostname', 'username':'me', 'password':'secret', 'port':2222}
# with pysftp.Connection(**cinfo) as sftp:

with pysftp.Connection(host=myHostname, username=myUsername, password=myPassword, cnopts=cnopts) as sftp:
    print ("Connection succesfully established ... ")
    
    # Switch to a remote directory (based on pysftp.Connection credentials)
    
    # get present working dir of myUsername (adelekeluqman@yahoo.com, which is Algorism in Users dir)
    print ('---pwd---' + sftp.pwd)
    # change working dir into Projects from Algorism in Users dir)
    sftp.cwd('./Projects/ic4_pro_ocr/Backup')
    # or
    # sftp.chdir('./Projects')

    # Obtain structure of the remote directory './Projects/ic4_pro_ocr/media'
    directory_structure = sftp.listdir_attr()

    # Print data
    for attr in directory_structure:
        print (attr.filename, attr)
        # print (attr)
    
# connection closed automatically at the end of the with-block

# OR
    # # get present working dir of myUsername (adelekeluqman@yahoo.com, which is Algorism in Users dir)
    # print ('---pwd---' + sftp.pwd)

    # sftp.cwd("Projects/ic4_pro_ocr/input")
    # directory_structure = sftp.listdir_attr()
    # local_dir= "D:/iC4_Pro_Project/OCR/Ticket_API/local/"
    # remote_dir = "./"
    # remote_backup_dir = "./Backup/"

    # print ('-------' + remote_dir)

    # # for attr in directory_structure:
    # #     if attr.filename.endswith(".pdf"):
    # #         file = attr.filename
    # #         sftp.get(remote_dir + file, local_dir + file)
    # #         print("Copied " + file + " to " + local_dir)
    # #         # sftp.remove(remote_dir + file)

    # for attr in directory_structure:        
    #     file = attr.filename
    #     sftp.get(remote_dir + file, local_dir + file)
    #     # print("Copied " + file + " to " + local_dir)
    #     print("Copied " + remote_dir + file + " to " + local_dir + file)
    #     # sftp.remove(remote_dir + file)

# # connection closed automatically at the end of the with-block

# # ----------- routine 2 needs further review ------------

# import pysftp

# cnopts = pysftp.CnOpts()
# cnopts.hostkeys = None 

# def push_file_to_server():
#     s = pysftp.Connection(host='192.168.43.156', username='adelekeluqman@yahoo.com', password='jadun.com', cnopts=cnopts)
#     print ('---pwd---' + s.pwd)
#     s.cwd('./Projects/ic4_pro_ocr/input/')    
#     # remote_path = s.pwd + "/Projects/ic4_pro_ocr/input/"
#     remote_path = s.pwd
#     # remote_path = 'C:/Users/Algorism/Projects/ic4_pro_ocr/input'
#     local_path = "D:/iC4_Pro_Project/OCR/Ticket_API/tmp/"

#     s.put(remote_path, local_path)
#     # s.get(remote_path, local_path)
#     s.close()

# push_file_to_server()