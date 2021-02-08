#!/opt/anaconda3/bin/python
# coding: utf-8
# CREATED BY ALI ABOU TAKA
# This script create a list with all the log files in the directory

def ExpLogFileList(directory,iChos):
########################
#####################################################
    list_of_files = [] 
    file_TDL      = []
    file_ExpL     = []

    if iChos == 'all':
        for filename in sorted(os.listdir(directory)):
            if filename.endswith(".log") and not filename.endswith("TD.log"):
                list_of_files.append(filename)
            if filename.endswith("TD.log"):
                file_TDL.append(filename)
            if filename.endswith(".csv"):
                file_ExpL.append(filename)
    elif iChos == 'input':
        number_of_inputs = int(input("Enter number of Files: "))
#        print('The first input will be the reference')
        for i in range(0, number_of_inputs):
            print('type the name of file no.', i+1)
            filename = input()
            if filename.endswith(".log") and not filename.endswith("TD.log"):
                list_of_files.append(filename)
            if filename.endswith("TD.log"):
                file_TDL.append(filename)
            if filename.endswith(".csv"):
                file_ExpL.append(filename)
#####################################################
#####################################################
