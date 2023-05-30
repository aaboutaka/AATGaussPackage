#!/opt/anaconda3/bin/python
# coding: utf-8
# CREATED BY ALI ABOU TAKA

# Example on how to use the package

import os
import aatgausspackage as aat
####
directory  = os.getcwd()
#listoffiles = aat.Input_or_allLog_FileList(directory)
listoffiles = aat.Input_or_allLog_FileList(directory)

print(listoffiles)
print("")
aat.PlotFC(directory, listoffiles)
