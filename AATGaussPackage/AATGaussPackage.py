#!/opt/anaconda3/bin/python
# coding: utf-8
# CREATED BY ALI ABOU TAKA
# THIS PYTHON PROGRAM  HAS DIFFERENT FUNCTIONS THAT CAN BE USED
# TO EXTRACT DESIRED INFORMATION FOR LOG FILES
# BE CAREFULL WHAT FUNCTION YOU CALL AND WHAT PATH YOU PROVIDE.

import pandas as pd
from sklearn import preprocessing
from scipy.signal import find_peaks
import textwrap
import argparse
import sys
import os
import matplotlib.style as style
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import re
import glob
import warnings
warnings.filterwarnings(action='ignore')

# To normalize the data later
#############################
min_max_scaler = preprocessing.MinMaxScaler()
#############################

# Start the parser
##################
# Parsing arguments
parser = argparse.ArgumentParser(
    prog='Package with different function to apply to Gaussian output files',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent('''\
                                    READ BELOW
                                    ----------                              
                There are the different fucntions in this package.
                --------------------------------------------------

                1) AllFileList:   Generate a list of all files in the
                                  current directory. It takes one argument
                                  only, "the directory".                

                2) LogFileList:   It generates a list of log files in the 
                                  current directory. It takes one argument
                                  only, "the directory".

                3) InputFileList: Generate a list of inputed files in the
                                  current directory provided by the user.
                                  It takes one argumentonly, "the directory".

                4)Input_or_       Generate a list of inputed files or all 
                  allLog_         files in the library. It takes one argument
                  FileList:       only, "the directory".
                                  Accepted switches: -f 

                4) PlotEnrg:      Plot the energy at each SCF vs the number
                                  of SCF cycles. It takes two arguments: the
                                  directory and list of log files
                                  AllFileList function can be used to
                                  generate the list of files, as the function
                                  will only take log files.
                                  Accepted switches: -fn, -ff, -lw, -ms, -fs
                                 -xmn, -xmx, -al, -aw

                5) PlotDIIS:     Plot the DIIS error at each SCF vs the 
                                 number of SCF cycles. It takes two arguments:
                                 the directory and list of log files
                                 AllFileList function can be used to
                                 generate the list of files, as the function
                                 will only take log files.                                  
                                 Accepted switches: -fn, -ff, -lw, -ms, -fs
                                 -xmn, -xmx, -al, -aw                                

                6) PlotRMSDP:    Plot the RMSDP at each SCF vs the number
                                 of SCF cycles. It takes two arguments: the
                                 directory and list of log files
                                 AllFileList function can be used to
                                 generate the list of files, as the function
                                 will only take log files.                                  
                                 Accepted switches: -fn, -ff, -lw, -ms, -fs
                                 -xmn, -xmx, -al,-aw                             

                7) PlotNvirt:    Plot the N_virt metric for pimom methods
                                 vs the number of SCF cycles. It takes three 
                                 arguments: the directory, list of log files,
                                 and a string for either 'Alpha' or 'Beta'.
                                 AllFileList function can be used to
                                 generate the list of files, as the function
                                 will only take log files.                                  
                                 Accepted switches: -fn, -ff, -lw, -ms, -fs
                                 -xmn, -xmx, -al, -aw                              

                8) PlotFC:       Plot FC of log files and exp files.
                                 It takes two arguments: the directory 
                                 and list of files (log + csv file).
                                 The list of files are further categorized 
                                 based on their extension.
                                 AllFileList function can be used to
                                 generate the list of files.
                                 Accepted switches: -fn, -ff, -lw, -ms, -fs, -s,
                                 -ns, -xmx, -al, -u, -esf, -ph
               
               9) MergeDataF:    This Function will merge data files (.dat) into one
                                 csv file.
                                 Accepted switches: -efn

               10) SCFEtopd:     This function grabs SCF Energy of single point calculations 
                                 and create a pd df of log files in a given list.
                                 Accepted switches: -ocsv

               11) TDEtopd:      This function grabs Energy of the last reported excited state
                                 and create a pd df of log files in a given list.
                                 Accepted switches: -ocsv                                      

               12) Plotcsv:      This function plot csv files in a given list.
                                 Accepted switches: -fn, -ff, -a, -lw, -ms, -fs, -al, -sl

               13) PlotFCTrans:  This Function will plot the FC of Photodetachment using the 
                                 reported transitions in Gaussian log file and not the x and y
                                 coordinates.
                                 Accepted switches: -fn, -ff, -lw, -ms, -fs, -s,
                                 -ns, -xmx, -al, -u, -ph, -esf                                 

               14) PlotUVGauss:  This Function will plot the UV spectrum from Gaussian log files 
                                 Accepted switches: -fn, -ff, -lw, -ms, -fs, -s, -ns, -xmx, -al, 
                                 -u, -ph, -esf, -gp, -sd

               If you encounter any problem with using this package,
               please contact Ali Abou Taka at abotaka.ali@gmail.com.
               Check the usage of each argument before using it.
                ------------------------------------------------------
                ''')
)
####################
parser.add_argument("-f", "--files", nargs='?', default="all",
                    help="takes 'input' for files to be specified or 'all' to plot all files. all is the default.")
parser.add_argument("-fn", "--figname", nargs='?', default="OutputFig",
                    help="name of the figure to be saved. OutputFig is the default.")
parser.add_argument("-ff", "--figformat", nargs='?', default='png',
                    help="format of the image to be saved. png is the default.")
parser.add_argument("-ms", "--markersize", nargs='?', type=float, default=5.0,
                    help="Set the size of the marker labels. The default value is 5.0")
parser.add_argument("-lw", "--linewidth", nargs='?', type=float, default=1.0,
                    help="Set the size of the linewidth. The default value is 1.0")
parser.add_argument("-fs", "--fontsize", nargs='?', type=float, default=12.0,
                    help="Set the size of the font. The default value is 12.0")
parser.add_argument("-ls", "--labelsize", nargs='?', type=float, default=12.0,
                    help="Set the size of the x and y labels. The default value is 12.0")
parser.add_argument("-xmn", "--xmin", nargs='?', type=int, default=0,
                    help="Set x axis initial  point. The default value is 0")
parser.add_argument("-xmx", "--xmax", nargs='?', type=int,
                    help="Set x axis end point. The default value is the all the points")
parser.add_argument("-al", "--addlegend",  action="store_true",
                    help="It is a boolean. If used, a legend will be placed under the plot. Without legend is the default.")
parser.add_argument("-sl", "--shiftlegend", nargs='?', type=float, default=-0.2,
                    help="Shift the legend up or down by a specific unit. -0.2 is the default.")
parser.add_argument("-at", "--addtitle",  action="store_true",
                    help="If used, a title will be added to the plot. Without title is the default.")
parser.add_argument("-nor", "--normalize",  action="store_true",
                    help="If used, the data will be normalized. is the default.")
parser.add_argument("-aw", "--axlinewidth", nargs='?', type=float, default=1.0,
                    help="Set the size of the linewidth of the axis. The default value is 1.0")

parser.add_argument("-t", "--temp", nargs='?', default="0K",
                    help="temperature at which the FC is generated at. 0K is the default. Currently supports 0K and 300K")
parser.add_argument("-u", "--unit", nargs='?', default="cm-1",
                    help="energy unit for the x axis.It takes cm-1, nm, or eV. cm-1 is the default.")
parser.add_argument("-esf", "--energyshiftfile", nargs='?', default="EnergyShiftFile.txt",
                    help="name of the file to be save the energy shift needed to align the plots. EnergyShiftFile is the default.")
parser.add_argument("-a", "--alpha", nargs='?', type=float, default=1.0,
                    help="alpha used to adjust transparency. It ranges between 0.0 and 1.0, where 1 is opaque. 1.0 is the default.")
parser.add_argument("-ns", "--noshift",  action="store_true",
                    help="It is a boolean. If used, the data will not be shifted. shifting is the default.")
parser.add_argument("-s", "--scale", nargs='?', type=float, default=1.0,
                    help="to scale the intensity after normalization. The default value is 1.0.")
parser.add_argument("-ph", "--peakheight", nargs='?', type=float, default=0.05,
                    help="to match the plot accprding to the first  peak wsth a specific height. The default value is 0.05.")
parser.add_argument("-gp", "--gridpoints", type=int, default=1000,
                    help="The number of grid points needed to plot the funciton. The default value is 1000.")
parser.add_argument("-sd", "--standarddev", type=float, default=0.4,
                    help="the standard deviation used to broaden the peaks. Should be given in eV units. The default value is 0.2 eV.")

parser.add_argument("-efn", "--expfilename", nargs='?', default="ExpData_Merged.csv",
                    help="takes the name of the experimental file to be generated by the script after combining data files. If one data file is present, the code will still generate a csv formatted file to be used later. ExpData_Merged.csv is the default name.")

parser.add_argument("-ocsv","--outputcsv", nargs='?', default="Combined_SCF_E.csv", help="name of the output csv file. Combined_SCF_E.csv is the default.")

parser.add_argument("-db", "--debug",  type=float, default=1.0,
                    help="Print extra output. Default = 1")

args = parser.parse_args()
#############################
#############################
# Setting Initial parameters
directory = os.getcwd()
iChos = args.files
figname = args.figname
figformat = args.figformat
markersize = args.markersize
linewidth = args.linewidth
axlinewidth = args.axlinewidth
fontsize = args.fontsize
labelsize = args.labelsize
xmn = args.xmin
xmax = args.xmax
add_legend = args.addlegend
add_title =args.addtitle
shift_legend = args.shiftlegend

unit = args.unit
temp = args.temp
alpha_val = args.alpha
noshift = args.noshift
scale = args.scale
EngDifF = args.energyshiftfile
Peakheight = args.peakheight

scale = args.scale
expfilename = args.expfilename

output_csv = args.outputcsv

DEBUG = args.debug
#############################

#############################
if any([iChos == 'all', iChos == 'input']):
    pass
else:
    print("'files' can take either 'input' for giving the input file names next, or 'all' for all the log files")
    sys.exit(1)
#############################


###################################################################
# This function create a list with all the files in the directory #
###################################################################
def AllFileList(directory):
    ###########################
    filenameL = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".log") or filename.endswith(".csv"):
            filenameL.append(filename)
    return filenameL
####################

#######################################################################
# This function create a list with all the log files in the directory #
#######################################################################


def LogFileList(directory):
    ##########################
    list_of_files = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".log"):
            list_of_files.append(filename)
    return list_of_files
########################

################################################
# This Function will merge data files into one #
# Its output is a csv file                     #
################################################


def MergeDataF(directory):
    filenameLExp = []
    mergedExpdf = []
    expfilename = args.expfilename
#   Read all the data files (exp) then form one file out called 'SummedExp.sum'
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".dat"):
            filenameLExp.append(filename)
    for file in filenameLExp:
        file = os.path.join(directory, file)
        expfilename = os.path.join(directory, expfilename)
        if ".dat" in file:
            df= pd.read_csv(file)
            if len(df.columns)==1:
                try:
                    dff = pd.read_csv(file, sep='\t')
                except IndexError:
                    print("the csv file is not tab delimited ",os.path.splitext(os.path.split(file)[1])[0])
                    try:
                        dff = pd.read_csv(file, sep=' ')
                    except IndexError:
                        print(" the csv file is not space delimited ", os.path.splitext(os.path.split(file)[1])[0])
                        print("Check the csv file ", os.path.splitext(os.path.split(file)[1])[0])
                        break
            elif len(df.columns)>=2:
                dff=df

#            df=pd.read_csv(file)
#            dff = dff.iloc[:, [0, 1]]
            dff = dff.iloc[:]
            try:
                dff.columns = ["x", "y"]
            except ValueError:
                print('check the delimiter in file {}. Currently supports "tab","comma", and "space" delimiters '.format(os.path.splitext(os.path.split(file)[1])[0] ))
                break            
            mergedExpdf.append(df)
        dff = pd.concat(mergedExpdf)
        dff = dff.sort_values('x', ascending=True)
        dff.to_csv(expfilename, sep='\t', index=False)

######################################################################
# This functioncreate a list with all the log files in the directory #
# based on input files given by the user                             #
######################################################################


def InputFileList(directory):
    #############################
    file_list = []

    number_of_inputs = int(input("Enter number of Files: "))
    for i in range(0, number_of_inputs):
        print('type the name of file no.', i+1)
        filename = input()
        file_list.append(filename)
    return file_list
####################


def Input_or_allLog_FileList(directory):
    #############################
    file_list = []

    if iChos == 'all':
        for filename in sorted(os.listdir(directory)):
            if filename.endswith(".log") or filename.endswith(".csv") :
                file_list.append(filename)

    elif iChos == 'input':
        number_of_inputs = int(input("Enter number of Files: "))
        for i in range(0, number_of_inputs):
            print('type the name of file no.', i+1)
            filename = input()
            file_list.append(filename)
    return file_list


####################################################
# This Function will grab the energy at each cycle #
####################################################
def PlotEnrg(directory, list_of_files):
    ######################################
    list_of_filesL = []
    x_keys = []
    y_values = []
    markers = ["v", "^", "<", ">", 'D', 'X', 'p', 's']

    for file in list_of_files:
        if "log" in file:
            file = os.path.join(directory, file)
            EnergyL = []
            with open(file, 'r') as f:
                for line in f:
                    if "E= " in line:
                        words = line.split()
                        if (words[0] == "E="):
                            energyval = float(words[1])
                            EnergyL.append(energyval)
            #     print (EnergyL)
                x_keys.append(list(range(len(EnergyL))))
#                     print(x_keys)
                y_values.append(EnergyL)
                list_of_filesL.append(
                    os.path.splitext(os.path.split(file)[1])[0])

            zipped_list = list(zip(x_keys, y_values, markers, list_of_filesL))
#             print(zipped3)
    for x, y, z, fn in zipped_list:
        #         ax.plot(i,j, linestyle = '', marker=next(markers))
        if xmax and xmn:
            plt.xlim(xmn, xmax)
            plt.plot(x, y, label=fn, marker=z,
                     linewidth=linewidth, markersize=markersize)
        elif xmax and not xmn:
            plt.xlim(right=xmax)
            plt.plot(x, y, label=fn, marker=z,
                     linewidth=linewidth, markersize=markersize)
        elif xmn and not xmax:
            print(len(EnergyL))
            plt.xlim(xmn, len(EnergyL))
            plt.plot(x, y, label=fn, marker=z,
                     linewidth=linewidth, markersize=markersize)
        else:
            plt.plot(x, y, label=fn, marker=z,
                     linewidth=linewidth, markersize=markersize)
        if add_legend:
            plt.legend(loc='lower center', bbox_to_anchor=(
                0.5, shift_legend), shadow=True, ncol=3)
        style.use('seaborn-talk')
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.set_xlabel('Number of SCF iterations', fontsize=fontsize)
        ax.set_ylabel('Energy in (a.u.)', fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=labelsize)
        ax.tick_params(axis="y", labelsize=labelsize)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(axlinewidth)
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    # Save the figure
    plt.savefig(os.path.join(directory, figname+'.'+figformat),
                dpi=600, format=figformat, bbox_inches='tight')
    plt.clf()


#######################################################
# This Function will plot the DIIS Error at each cycle#
#######################################################
def PlotDIIS(directory, list_of_files):
    ######################################
    list_of_filesL = []
    x_keys = []
    y_values = []
    markers = ["v", "^", "<", ">", 'D', 'X', 'p', 's']

    for file in list_of_files:
        if "log" in file:
            file = os.path.join(directory, file)
            DIISL = []
            with open(file, 'r') as f:
                for line in f:
                    if "DIIS: error=" in line:
                        line = line.replace("D", "e")
                        words = line.split()
                        DIIS = float(words[2])
                        DIISL.append(DIIS)
#                print(DIISL)
                x_keys.append(list(range(len(DIISL))))
#                     print(x_keys)
                y_values.append(DIISL)
                list_of_filesL.append(
                    os.path.splitext(os.path.split(file)[1])[0])

            zipped_list = list(zip(x_keys, y_values, markers, list_of_filesL))
#             print(zipped3)
    for x, y, z, fn in zipped_list:
        if xmax and xmn:
            plt.xlim(xmn, xmax)
            plt.plot(x, y, label=fn, marker=z,
                     linewidth=linewidth, markersize=markersize)
        elif xmax and not xmn:
            plt.xlim(right=xmax)
            plt.plot(x, y, label=fn, marker=z,
                     linewidth=linewidth, markersize=markersize)
        elif xmn and not xmax:
            plt.xlim(xmn, len(EnergyL))
            plt.plot(x, y, label=fn, marker=z,
                     linewidth=linewidth, markersize=markersize)
        else:
            plt.plot(x, y, label=fn, marker=z,
                     linewidth=linewidth, markersize=markersize)
        if add_legend:
            plt.legend(loc='lower center', bbox_to_anchor=(
                0.5, shift_legend), shadow=True, ncol=3)
        style.use('seaborn-talk')
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.set_xlabel('Number of SCF iterations', fontsize=fontsize)
        ax.set_ylabel('DIIS Error in (a.u.)', fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=labelsize)
        ax.tick_params(axis="y", labelsize=labelsize)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(axlinewidth)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    plt.savefig(os.path.join(directory, figname+'DIIS' + '.' +
                             figformat), dpi=600, format=figformat, bbox_inches='tight')
    plt.clf()

###################################################
# This Function will grab the RMSDP at each cycle #
###################################################


def PlotRMSDP(directory, list_of_files):
    #######################################
    ref = (glob.glob('*ref.log'))
    ref1 = ""
    for ele in ref:
        ref1 += ele
#    list_of_files=[]
    list_of_filesL = []
    x_keys = []
    y_values = []
    markers = ["v", "^", "<", ">", 'D', 'X', 'p', 's']

    for file in list_of_files:
        if "log" in file:
            file = os.path.join(directory, file)
            if file == ref1:
                continue
            RMSDPL = []
            with open(file, 'r') as f:
                for line in f:
                    if "RMSDP" in line:
                        line = line.replace("=", " ")
                        line = line.replace("D", "e")
                        words = line.split()
                        RMSDP = float(words[1])
                        RMSDPL.append(RMSDP)
                x_keys.append(list(range(len(RMSDPL))))
                y_values.append(RMSDPL)
                list_of_filesL.append(
                    os.path.splitext(os.path.split(file)[1])[0])

            zipped_list = list(zip(x_keys, y_values, markers, list_of_filesL))
    for x, y, z, fn in zipped_list:
        if xmax and xmn:
            plt.xlim(xmn, xmax)
            plt.plot(x, y, label=fn, marker=z,
                     linewidth=linewidth, markersize=markersize)
        elif xmax and not xmn:
            plt.xlim(right=xmax)
            plt.plot(x, y, label=fn, marker=z,
                     linewidth=linewidth, markersize=markersize)
        elif xmn and not xmax:
            print(len(EnergyL))
            plt.xlim(xmn, len(EnergyL))
            plt.plot(x, y, label=fn, marker=z,
                     linewidth=linewidth, markersize=markersize)
        else:
            plt.plot(x, y, label=fn, marker=z,
                     linewidth=linewidth, markersize=markersize)
        if add_legend:
            plt.legend(loc='lower center', bbox_to_anchor=(
                0.5, shift_legend), shadow=True, ncol=3)
        style.use('seaborn-talk')
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.set_xlabel('Number of SCF iterations', fontsize=fontsize)
        ax.set_ylabel('RMSDP in (a.u.)', fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=labelsize)
        ax.tick_params(axis="y", labelsize=labelsize)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(axlinewidth)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    plt.savefig(os.path.join(directory, figname+'RMSDP' + '.' +
                             figformat), dpi=600, format=figformat, bbox_inches='tight')
    plt.clf()
####################################################
# This Function will grab the N Virt at each cycle #
####################################################


def PlotNvirt(directory, list_of_files, switch):
    ##############################################
    dicts = {}
    file_keys = []
    maxNvirt_values = []
    list_of_filesL = []
    x_keys = []
    y_values = []
    markers = ["v", "^", "<", ">", 'D', 'X', 'p', 's']

    if switch != 'Alpha' and switch != 'Beta':
        print('This functions is called with the dirctory and the Alpha or Beta switch')
        print('Make sure to use Alpha or Beta')
        sys.exit(1)

    for file in list_of_files:
        if "log" in file:
            file = os.path.join(directory, file)
            NvirtL = []
            with open(file, 'r') as f:
                if switch == 'Alpha':
                    for line in f:
                        if "Sum of the diagonals for virt the Alpha" in line:
                            words = line.split()
                            NvirtAlpha = float(words[9])
                            NvirtL.append(NvirtAlpha)
                    x_keys.append(list(range(len(NvirtL))))
            #                 print(x_keys)
                    y_values.append(NvirtL)
                    list_of_filesL.append(
                        os.path.splitext(os.path.split(file)[1])[0])

                elif switch == 'Beta':
                    for line in f:
                        if "Sum of the diagonals for virt the Beta" in line:
                            words = line.split()
                            NvirtBeta = float(words[9])
                            NvirtL.append(NvirtBeta)
                    x_keys.append(list(range(len(NvirtL))))
            #                 print(x_keys)
                    y_values.append(NvirtL)
                    list_of_filesL.append(
                        os.path.splitext(os.path.split(file)[1])[0])
            zipped_list = list(zip(x_keys, y_values, markers, list_of_filesL))
#             print(zipped_list)
    for x, y, z, fn in zipped_list:
        if xmax and xmn:
            plt.xlim(xmn, xmax)
            plt.plot(x, y, label=fn, marker=z,
                     linewidth=linewidth, markersize=markersize)
        elif xmax and not xmn:
            plt.xlim(right=xmax)
            plt.plot(x, y, label=fn, marker=z,
                     linewidth=linewidth, markersize=markersize)
        elif xmn and not xmax:
            print(len(EnergyL))
            plt.xlim(xmn, len(EnergyL))
            plt.plot(x, y, label=fn, marker=z,
                     linewidth=linewidth, markersize=markersize)
        else:
            plt.plot(x, y, label=fn, marker=z,
                     linewidth=linewidth, markersize=markersize)
        if add_legend:
            plt.legend(loc='lower center', bbox_to_anchor=(
                0.5, shift_legend), shadow=True, ncol=3)
        style.use('seaborn-talk')
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.set_xlabel('Number of SCF iterations', fontsize=fontsize)
        ax.set_ylabel(f'$N_{"{virt}"}^\{switch.lower()}$', fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=labelsize)
        ax.tick_params(axis="y", labelsize=labelsize)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(axlinewidth)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    plt.savefig(os.path.join(directory, figname+'Nvirt_'+str(switch) +
                             '.' + figformat), dpi=600, format=figformat, bbox_inches='tight')
    plt.clf()
########################
########################

###########################################################
# Plot FC of excited calculations log files and exp files #
###########################################################


def PlotFC(directory, list_of_files):
    #######################################
    file_logs = []
    file_TDL = []
    file_ExpL = []
    test = []
    trans00 = []
    temp = args.temp
    xmn = args.xmin
    xmax = args.xmax
    alpha_val = args.alpha
    unit=args.unit
#    fline2 = None
    cmToeV = float(0.000123984)
    cmTonm = float(10000000.0)
    nmTocm = float(10000000.0)
    nmToeV = float(1239.84193)
    cm_unit = "$cm^{-1}$"

#   General printing
    print("USEFUL INFORMATION RELATED TO THE TEMPERATURE AT WHICH THE SPECTRUM IS GENERATED")
    print("--------------------------------------------------------------------------------")
    print()    
    print("Plotting the spectrum at {}.".format(temp))
    print()    

    for filename in list_of_files:
        if ".log" in filename and "TD" not in filename:
            file_logs.append(filename)
        if "TD" in filename:
            file_TDL.append(filename)
        if "csv" in filename:
            file_ExpL.append(filename)
#    print(file_logs)
#    print(file_TDL)
#    print(file_ExpL)
#####################################################
#####################################################
# Prepare the temp to be fed later to the code
    temp = temp.replace(" ", "") # take care of spaces

    try:
        if float(temp) == 0:
            temp =format(float(temp), '#.0f')
            temp = temp+"K"
        else:
            temp =format(float(temp), '#.1f')
            temp = temp+"K"
    #         print(temp)
    except ValueError:
        pass

    if "k" in temp:
        temp = temp.split("k")
        if float(temp[0]) == 0:
            temp =format(float(temp[0]), '#.0f')
            temp = temp+"K"
        else:
            temp =format(float(temp[0]), '#.1f')
            temp = temp+"K"
    if "K" in temp:
        temp = temp.split("K")

        if float(temp[0]) == 0:
            temp =format(float(temp[0]), '#.0f')        
            temp = temp[0]+"K"
        else:
            temp =format(float(temp[0]), '#.1f')
            temp = temp+"K" 

    if "k" not in temp and "K" not in temp:
        if int(temp) == 0:
            temp =format(float(temp), '#.0f')
            temp = temp+"K"
        else:
            temp =format(float(temp), '#.1f')
            temp = temp+"K"     
    

#####################################################
    # Open the file for writing the Energy difference
    if noshift:
        pass
    else:
        openEngDifF = open(EngDifF, "w")

    # Loop through csv files
    for file in file_ExpL:
        if not file_ExpL:
            print('There are no csv files considered')
            break
#        print(file_expL)
        file = os.path.join(directory, file)
        coord = []
        count = len(open(file).readlines())
        if ".csv" in file:
            df= pd.read_csv(file)
            if len(df.columns)==1:            
                try:
                    dff = pd.read_csv(file, sep='\t')
                except IndexError:
                    print("the csv file is not tab delimited ",os.path.splitext(os.path.split(file)[1])[0])
                    try:
                        dff = pd.read_csv(file, sep=' ')
                    except IndexError:
                        print(" the csv file is not space delimited ", os.path.splitext(os.path.split(file)[1])[0])
                        print("Check the csv file ", os.path.splitext(os.path.split(file)[1])[0])
                        break
            elif len(df.columns)>=2:
                dff=df
            dff = dff.iloc[:]
            try:
                dff.columns = ["x", "y"]
            except ValueError:
                print('check the delimiter in file {}. Currently supports "tab","comma", and "space" delimiters '.format(os.path.splitext(os.path.split(file)[1])[0] ))
                sys.exit(1)
            dff = dff.sort_values("x", ascending=True)
            # Next we will find all the peaks in the spectrum
            firstpeak = next(x[0] for x in enumerate(dff['y']) if x[1] > Peakheight)
#            peaks, _ = find_peaks(dff["y"], height=Peakheight)
            p = dff["x"]

            # Convert all the column types to numeric
            dff[["x", "y"]] = dff[["x", "y"]].apply(pd.to_numeric)

            # Find the energy of the first peak
            # In some cases, you may need the second or third peak
            # to align it with the second peak, change the 0 to 1
            MainEng = p.iloc[firstpeak]
            # the Exp data are in nm so we need to convert to cm
            trans00.append(MainEng)

            # Normalize the data
            cols_to_norm = ["y"]
            dff[cols_to_norm] = min_max_scaler.fit_transform(dff[cols_to_norm])
            if DEBUG >= 2:
                print('Normalized Data Frame of',
                      os.path.splitext(os.path.split(file)[1])[0])
                print(dff)
            if DEBUG >= 2:
                print('Normalized and shifted Data Frame of',
                      os.path.splitext(os.path.split(file)[1])[0])
                print(dff)

            if noshift:
                print("Not gonna shift and align the spectra.")
                pass
            else:
                # Compute the energy difference with respect to the first file and
                # shift the x axis
                for i in range(len(trans00)):
                    difference = float(trans00[i])-float(trans00[0])
                dff.x -= difference
                if unit == "cm-1":
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference, 2), unit)
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference, 2), unit, file=openEngDifF)
                elif unit == "eV" or unit == "ev":
                    difference_ev = difference * cmToeV
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference_ev, 3), unit)
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference_ev, 3), unit, file=openEngDifF)
                elif unit == "nm":
                    try:
                        difference_nm = cmTonm / difference
                    except ZeroDivisionError:
                        difference_nm = 0.0
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference_nm, 3), unit)
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference_nm, 3), unit, file=openEngDifF)
                else:
                    print("wrong unit provided. Check the manual for accepted units")
                    sys.exit(1)

            # Convert the energy to the requested unit
            if unit == "cm-1":
                tranE = float(MainEng)
                xlim1 = tranE - 50
#                trans00.append(tranE)
            elif unit == "eV" or unit == "ev":
                dff['x'] = cmToeV * dff['x']
                tranE = cmToeV * float(MainEng)
                xlim1 = tranE - 0.1
#                trans00.append(tranE)
            elif unit == "nm":
                dff['x'] = cmTonm / dff['x']
                tranE = cmTonm / float(MainEng)
                xlim1 = tranE - 0.04
#                trans00.append(MainEng)
            else:
                print("wrong unit provided. Check the manual for accepted units")
                sys.exit(1)
#
            # Plot the data
            plt.plot(dff['x'], dff['y'],
                     label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val, linewidth=linewidth, markersize=markersize)
            plt.ylabel('Intensity', fontsize=fontsize)
            if unit == "cm-1":
                plt.xlabel('E('+cm_unit+')', fontsize=fontsize)
            else:
                plt.xlabel('E('+unit+')', fontsize=fontsize)
            plt.legend(loc='upper center', bbox_to_anchor=(
                0.5, shift_legend), shadow=True, ncol=3)
    if DEBUG >= 2:
        print('List cotaining max peaks in csv files list -- ',
              file_ExpL)
        print(trans00)
#
    # Plot the TD file and set its energy as the reference
    for file in file_TDL:
        if not file_TDL:
            print('There are no TD Gaussian log files considered')
            break
        file = os.path.join(directory, file)
        coord = []
        count = len(open(file).readlines())
        with open(file, 'r') as f:
            fline1 = False
            fline2 = False            
            for line in f:
                if "Intensity: " in line:
                    for i in range(count):
                        nextline = next(f)
                        nextline = nextline.replace("D", "e")
                        nextline = nextline.split()
                        if "Leave" in nextline:
                            break
                        if "Electric" in nextline:
                            break
                        else:
                            coord.append(nextline)
                    # Delete extra points
                    del coord[0]
                    del coord[-7:]
#                   Getting the intensities                    
                if "2nd col." in line:
                    fline = line.split("=")
                    fline1 = fline[1].strip()
                    fline11 = fline[0].split("col")
                if "3rd col." in line:
                    fline = line.split("=")
                    fline2 = fline[1].strip()
                    fline22 = fline[0].split("col")

                # Form a data frame of the data
            df = pd.DataFrame(coord)
            if (DEBUG >= 5):
                print('Data Frame before adding headers of',
                      os.path.splitext(os.path.split(file)[1])[0])
                print(df)

            # Add columns labels
#            print(df)
#            print(len(df.columns))
            if (len(df.columns)) == 0:
                print("No data was found")
                sys.exit(1)                

            elif (len(df.columns)) == 3:
                df.columns = ['x', 'y-0k', 'y-k']
            else:
                df.columns = ['x', 'y-0k']
            dff = df.drop([0])
            if DEBUG >= 4:
                print('Data Frame of', os.path.splitext(
                    os.path.split(file)[1])[0])
                print(dff)

            # Convert all the column types to numeric
            if (len(df.columns)) == 3:
                dff[['x', 'y-0k', 'y-k']] = dff[['x',
                                                 'y-0k', 'y-k']].apply(pd.to_numeric)
            else:
                dff[['x', 'y-0k']] = dff[['x', 'y-0k']].apply(pd.to_numeric)

            # Normalize the data
            if (len(df.columns)) == 3:
                cols_to_norm = ['y-0k', 'y-k']
            else:
                cols_to_norm = ['y-0k']
            dff[cols_to_norm] = min_max_scaler.fit_transform(dff[cols_to_norm])
            if DEBUG >= 4:
                print('Normalized Data Frame of',
                      os.path.splitext(os.path.split(file)[1])[0])
                print(dff)
            # print(dff)
        # Next we will find the first peak in the spectrum
            if temp == fline1:
                firstpeak = next(x[0] for x in enumerate(dff['y-0k']) if x[1] > Peakheight)
#+AAT           
            elif temp == fline2:
                firstpeak = next(x[0] for x in enumerate(dff['y-k']) if x[1] > Peakheight)
            else:
                print()
                print("The temperature provided, {},is not found in the log file, {}.".format(temp,os.path.splitext(os.path.split(file)[1])[0]))
                print("QUITTING")
                print()
                sys.exit(1) 
            if fline2:
                print("Those are the available temperatures {} and {}.".format(fline1,fline2))
            else:
                print("This is the only available temperature, {}.".format(fline1))
#                print("QUITTING")
#                print()
#                sys.exit(1)
#####################################################################            
            p = dff['x']
            # Append the energy of the first peak
            MainEng = p.iloc[firstpeak]
            trans00.append(MainEng)
            # Printing available temps in the log file
            if fline2:
                print()
                print("file",os.path.splitext(os.path.split(file)[1])[0],
                        "has intensity values at {} and {}.".format(fline1,fline2))
                print()
            else:
                print()
                print("file",os.path.splitext(os.path.split(file)[1])[0],
                        "has intensity values at {} only.".format(fline1))
                print()            

            if noshift:
                print("Not gonna shift and align the spectra")
                pass
            else:
                # Compute the energy difference with respect to the first file and
                # shift the x axis
                for i in range(len(trans00)):
                    difference = float(trans00[i])-float(trans00[0])
                dff.x -= difference
                if float(difference) == 0:
                    print("File {} is set to be the reference".format(os.path.splitext(os.path.split(file)[1])[0]))
                    pass
                elif unit == "cm-1":

                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference, 2), unit)
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference, 2), unit, file=openEngDifF)
                elif unit == "eV" or unit == "ev":
                    difference_ev = difference * cmToeV
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference_ev, 3), unit)
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference_ev, 3), unit, file=openEngDifF)
                elif unit == "nm":
                    try:
                        difference_nm = cmTonm / difference
                    except ZeroDivisionError:
                        difference_nm = 0.0
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference_nm, 3), unit)
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference_nm, 3), unit, file=openEngDifF)
                else:
                    print("wrong unit provided. Check the manual for accepted units")
                    sys.exit(1)
#                print(os.path.splitext(os.path.split(file)[1])[0], 'is shifted by', round(difference,2), unit)
                if DEBUG >= 4:
                    print('Normalized and shifted Data Frame of',
                          os.path.splitext(os.path.split(file)[1])[0])
                    print(dff)

            # Scale the data
            if (len(df.columns)) == 3:
                dff['y-0k'] *= scale
                dff['y-k'] *= scale
            else:
                dff['y-0k'] *= scale

            # Convert to correct unit
            if unit == "cm-1":
                pass
            elif unit == "eV" or unit == "ev":
                dff['x'] = cmToeV * dff['x']
            elif unit == "nm":
                dff['x'] = cmTonm / dff['x']
            else:
                print("wrong unit provided. Check the manual for accepted units")
                sys.exit(1)
    #       Prepare  default minimum and maximum x-axis value
            if  unit == "cm-1":
                xmndef = min(dff['x']) - 2000
                xmaxdef = max(dff['x']) + 2000
            elif unit == "eV" or unit == "ev":
                xmndef = min(dff['x']) - 2
                xmaxdef = max(dff['x']) + 2
            elif unit == "nm":
                xmndef = min(dff['x']) - 5
                xmaxdef = max(dff['x']) + 5
#            print(xmndef,xmaxdef)
        # Plot the data        
        if temp == fline1:
            if xmax and xmn:
                plt.xlim(xmn, xmax)
                plt.plot(dff["x"], dff["y-0k"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
            elif xmax and not xmn:
                plt.xlim(xmndef,xmax)
                plt.plot(dff["x"], dff["y-0k"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
            elif xmn and not xmax:
                plt.xlim(xmn, xmaxdef)
                plt.plot(dff["x"], dff["y-0k"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
            else:
                plt.xlim(xmndef,xmaxdef)
                plt.plot(dff["x"], dff["y-0k"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
        else:
            if xmax and xmn:
                plt.xlim(xmn, xmax)
                plt.plot(dff["x"], dff["y-k"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
            elif xmax and not xmn:
                plt.xlim(xmndef,xmax)
                plt.plot(dff["x"], dff["y-k"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
            elif xmn and not xmax:
                plt.xlim(xmn, xmaxdef)
                plt.plot(dff["x"], dff["y-k"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
            else:
                plt.xlim(xmndef,xmaxdef)
                plt.plot(dff["x"], dff["y-k"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
        plt.ylabel('Intensity', fontsize=fontsize)
        if unit == "cm-1":
            plt.xlabel('E('+cm_unit+')', fontsize=fontsize)
        else:
            plt.xlabel('E('+unit+')', fontsize=fontsize)
        if add_legend:
            plt.legend(loc='upper center', bbox_to_anchor=(
                0.5, shift_legend), shadow=True, ncol=3)                     
        if DEBUG >= 3:
            dff.to_csv(os.path.splitext(os.path.split(file)[1])[
                           0]+temp+'.csv', sep=',', index=False, columns=['x', 'y-k'])


#####################################################
#####################################################
    # Plot the rest of the log files
    for file in file_logs:
        if not file_logs:
            print('There are no Gaussian log files considered')
            break
        file = os.path.join(directory, file)
        coord = []
        count = len(open(file).readlines())
        if "log" in file:
            with open(file, 'r') as f:
                fline1 = False
                fline2 = False                
                for line in f:
                    if "Intensity: " in line:
                        for i in range(count):
                            nextline = next(f)
                            nextline = nextline.replace("D", "e")
                            nextline = nextline.split()
                            if "Leave" in nextline:
                                break
                            if "Electric" in nextline:
                                break
                            else:
                                coord.append(nextline)
                        # Delete extra points
                        del coord[0]
                        del coord[-7:]
#                   Getting the intensities
                    if "2nd col." in line:
                        fline = line.split("=")
                        fline1 = fline[1].strip()
                        fline11 = fline[0].split("col")
                    if "3rd col." in line:
                        fline = line.split("=")
                        fline2 = fline[1].strip()
                        fline22 = fline[0].split("col")

                # Form a data frame of the data
                df = pd.DataFrame(coord)
#                print(len(df.columns))
                if DEBUG >= 5:
                    print('Data Frame before adding headers of',
                          os.path.splitext(os.path.split(file)[1])[0])
                    print(df)

                # Add columns labels
                if (len(df.columns)) == 0:
                    print("No data was found")
                    sys.exit(1)
                elif (len(df.columns)) == 3:
                    df.columns = ['x', 'y-0k', 'y-k']
                else:
                    df.columns = ['x', 'y-0k']
                dff = df.drop([0])
                if DEBUG >= 4:
                    print('Data Frame of', os.path.splitext(
                        os.path.split(file)[1])[0])
                    print(dff)

                # Convert all the column types to numeric
                if (len(df.columns)) == 3:
                    dff[['x', 'y-0k', 'y-k']] = dff[['x',
                                                     'y-0k', 'y-k']].apply(pd.to_numeric)
                else:
                    dff[['x', 'y-0k']] = dff[['x', 'y-0k']
                                             ].apply(pd.to_numeric)

                # Normalize the data
                if (len(df.columns)) == 3:
                    cols_to_norm = ['y-0k', 'y-k']
                else:
                    cols_to_norm = ['y-0k']
                dff[cols_to_norm] = min_max_scaler.fit_transform(
                    dff[cols_to_norm])
                if DEBUG >= 4:
                    print('Normalized Data Frame of',
                          os.path.splitext(os.path.split(file)[1])[0])
                    print(dff)
                # Next we will find the first peak in the spectrum
                if temp == fline1:
                    firstpeak = next(x[0] for x in enumerate(dff['y-0k']) if x[1] > Peakheight)
                elif temp == fline2:
                    firstpeak = next(x[0] for x in enumerate(dff['y-k']) if x[1] > Peakheight)
                else:
                    print()
                    print("The temperature provided, {},is not found in the log file {}.".format(temp,os.path.splitext(os.path.split(file)[1])[0]))
#                    print("Temperatures available",fline1,fline2)
                    if fline2:
                        print("Those are the available temperatures {} and {}.".format(fline1,fline2))
                    else:
                        print("This is the only available temperature, {}.".format(fline1))
                    print("QUITTING")
                    print()
                    sys.exit(1)                    
                    if fline2:
                        print("Those are the available temperatures {} and {}.".format(fline1,fline2))
                    else:
                        print("This is the only available temperature, {}.".format(fline1))
#####################################################################
                p = dff['x']
                # Append the energy of the first peak
                MainEng = p.iloc[firstpeak]
                trans00.append(MainEng)
#                print(trans00)
            # Printing available temps in the log file
                if fline2:
                    print()
                    print("file",os.path.splitext(os.path.split(file)[1])[0],
                            "has intensity values at {} and {}.".format(fline1,fline2))
                    print()
                else:
                    print()
                    print("file",os.path.splitext(os.path.split(file)[1])[0],
                            "has intensity values at {} only.".format(fline1))
                    print()

                if noshift:
                    pass
                else:
                    # Compute the energy difference with respect to the first file and
                    # shift the x axis
                    for i in range(len(trans00)):
                        difference = float(trans00[i])-float(trans00[0])
                    dff.x -= difference
                    if float(difference) == 0:
                        print("File {} is set to be the reference".format(os.path.splitext(os.path.split(file)[1])[
                              0]))
                        pass                    
                    elif unit == "cm-1":
                        print(os.path.splitext(os.path.split(file)[1])[
                              0], 'is shifted by', round(difference, 2), unit)
                        print(os.path.splitext(os.path.split(file)[1])[
                              0], 'is shifted by', round(difference, 2), unit, file=openEngDifF)
                    elif unit == "eV" or unit == "ev":
                        difference_ev = difference * cmToeV
                        print(os.path.splitext(os.path.split(file)[1])[
                              0], 'is shifted by', round(difference_ev, 3), unit)
                        print(os.path.splitext(os.path.split(file)[1])[
                              0], 'is shifted by', round(difference_ev, 3), unit, file=openEngDifF)
                    elif unit == "nm":
                        try:
                            difference_nm = cmTonm / difference
                        except ZeroDivisionError:
                            difference_nm = 0.0
                        print(os.path.splitext(os.path.split(file)[1])[
                              0], 'is shifted by', round(difference_nm, 2), unit)
                        print(os.path.splitext(os.path.split(file)[1])[
                              0], 'is shifted by', round(difference_nm, 2), unit, file=openEngDifF)
                    else:
                        print(
                            "wrong unit provided. Check the manual for accepted units")
                        sys.exit(1)

                    if DEBUG >= 3:
                        print('Normalized and shifted Data Frame of',
                              os.path.splitext(os.path.split(file)[1])[0])
                        print(dff)
                # Scale the data
                if (len(df.columns)) == 3:
                    dff['y-0k'] *= scale
                    dff['y-k'] *= scale
                else:
                    dff['y-0k'] *= scale

                # Convert the energy to the requested unit
                if unit == "cm-1":
                    pass
                elif unit == "eV" or unit == "ev":
                    dff['x'] = cmToeV * dff['x']
                elif unit == "nm":
                    dff['x'] = cmTonm / dff['x']
                else:
                    print("wrong unit provided. Check the manual for accepted units")
                    sys.exit(1)


            # Plot the data
    #       Prepare  default minimum and maximum x-axis value
            if  unit == "cm-1":
                xmndef = min(dff['x']) - 2000
                xmaxdef = max(dff['x']) + 2000
            elif unit == "eV" or unit == "ev":
                xmndef = min(dff['x']) - 2
                xmaxdef = max(dff['x']) + 2
            elif unit == "nm":
                xmndef = min(dff['x']) - 5
                xmaxdef = max(dff['x']) + 5
#            print(xmndef,xmaxdef)
#
        if temp == fline1:
            if xmax and xmn:
                plt.xlim(xmn, xmax)
                plt.plot(dff["x"], dff["y-0k"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
            elif xmax and not xmn:
                plt.xlim(xmndef,xmax)
                plt.plot(dff["x"], dff["y-0k"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
            elif xmn and not xmax:
                plt.xlim(xmn, xmaxdef)
                plt.plot(dff["x"], dff["y-0k"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
            else:
                plt.xlim(xmndef,xmaxdef)
                plt.plot(dff["x"], dff["y-0k"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
        else:
            if xmax and xmn:
                plt.xlim(xmn, xmax)
                plt.plot(dff["x"], dff["y-k"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
            elif xmax and not xmn:
                plt.xlim(xmndef,xmax)
                plt.plot(dff["x"], dff["y-k"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
            elif xmn and not xmax:
                plt.xlim(xmn, xmaxdef)
                plt.plot(dff["x"], dff["y-k"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
            else:
                plt.xlim(xmndef,xmaxdef)
                plt.plot(dff["x"], dff["y-k"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
        plt.ylabel('Intensity', fontsize=fontsize)
        if unit == "cm-1":
            plt.xlabel('E('+cm_unit+')', fontsize=fontsize)
        else:
            plt.xlabel('E('+unit+')', fontsize=fontsize)
        if add_legend:
            plt.legend(loc='upper center', bbox_to_anchor=(
                0.5, shift_legend), shadow=True, ncol=3)            

        # Plot the data
#        if temp == '0K':
#            plt.ylabel('Intensity', fontsize=fontsize)
#            if unit == "cm-1":
#                plt.xlabel('E('+cm_unit+')', fontsize=fontsize)
#            else:
#                plt.xlabel('E('+unit+')', fontsize=fontsize)
#            plt.plot(dff['x'], dff['y-0k'],
#                     label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val, linewidth=linewidth, markersize=markersize)
##            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)
#            plt.legend(loc='upper center', bbox_to_anchor=(
#                0.5, shift_legend), shadow=True, ncol=3)
#        else:
#            plt.ylabel('Intensity', fontsize=fontsize)
#            if unit == "cm-1":
#                plt.xlabel('E('+cm_unit+')', fontsize=fontsize)
#            else:
#                plt.xlabel('E('+unit+')', fontsize=fontsize)
#            plt.plot(dff['x'], dff['y-k'],
#                     label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val, linewidth=linewidth, markersize=markersize)
##            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)
#            plt.legend(loc='upper center', bbox_to_anchor=(
#                0.5, shift_legend), shadow=True, ncol=3)
#            plt.legend(loc='upper left', shadow=True, ncol=3)
#####################################################
#####################################################
    # Save the figure
    plt.savefig(figname+'.'+figformat, bbox_inches='tight',
                format=figformat, dpi=600)


###################################################
# This Function will plot the FC of Photodetachment
###################################################

def PlotSpecFC(directory, list_of_files):
    #####################################
    # Setting Initial parameters
    xmn = args.xmin
    xmax = args.xmax
    filenameL = []
    test = []
    x_keys = []
    y_values = []
    trans00 = []
    file_expL = []
    cmToeV = float(1/8065.6)
    cmTonm = float(10000000.0)
    cm_unit = "$cm^{-1}$"

#############################
    if noshift:
        pass
    else:
        openEngDifF = open(EngDifF, "w")

######################################################
#####################################################
# Prepare the lists
###################
    for filename in list_of_files:
        if ".log" in filename:
            filenameL.append(filename)
        if "csv" in filename:
            file_expL.append(filename)

#####################################################
#####################################################
# Set the experimental data as a refernce and plot it
    for file in file_expL:
        #        print(file_expL)
        file = os.path.join(directory, file)
        coord = []
        count = len(open(file).readlines())
        if ".sum" in file or ".txt" in file or ".dat" in file or ".csv" in file:
            df= pd.read_csv(file)
            if len(df.columns)==1:
                try:
                    dff = pd.read_csv(file, sep='\t')
                except IndexError:
                    print("the csv file is not tab delimited ",os.path.splitext(os.path.split(file)[1])[0])
                    try:
                        dff = pd.read_csv(file, sep=' ')
                    except IndexError:
                        print(" the csv file is not space delimited ", os.path.splitext(os.path.split(file)[1])[0])
                        print("Check the csv file ", os.path.splitext(os.path.split(file)[1])[0])
                        break
            elif len(df.columns)>=2:
                dff=df

#            df=pd.read_csv(file)
#            dff = dff.iloc[:, [0, 1]]
            dff = dff.iloc[:]
            try:
                dff.columns = ["x", "y"]
            except ValueError:
                print('check the delimiter in file {}. Currently supports "tab","comma", and "space" delimiters '.format(os.path.splitext(os.path.split(file)[1])[0] ))
                break            
            dff = dff.sort_values("x", ascending=True)
            # Next we will find the first peak in the spectrum
            firstpeak = next(x[0] for x in enumerate(dff['y']) if x[1] > Peakheight)
            p = dff['x']
    #         print('AAT',p.iloc[firstpeak])
            MainEng = p.iloc[firstpeak]
            trans00.append(MainEng)            
#            # Next we will find all the peaks in the spectrum
#            peaks, _ = find_peaks(dff["y"], height=0.4)
#            p = dff["x"]
#
#            # Convert all the column types to numeric
#            dff[["x", "y"]] = dff[["x", "y"]].apply(pd.to_numeric)
#
#            # Find the energy of the first peak
#            MainEng = p.iloc[peaks[0]]

            # Convert the energy to the requested unit
            if unit == "cm-1":
#                xlim1 = float(MainEng) - 50
                trans00.append(MainEng)
            elif unit == "eV" or unit == "ev":
                dff["x"] = dff["x"] * cmToeV
                tranE = float(MainEng) * cmToeV
                xmax = xmax * cmToeV
#                xlim1 = tranE - 0.007
                trans00.append(tranE)
            elif unit == "nm":
                xmax = xmax * cmTonm
                dff["x"] = dff["x"] * cmTonm
                tranE = float(MainEng) * cmTonm
#                xlim1 = tranE - 0.04
                trans00.append(tranE)
            else:
                print("wrong unit provided. Check the manual for accepted units")
                sys.exit(1)

            # Normalize the data
            cols_to_norm = ["y"]
            dff[cols_to_norm] = min_max_scaler.fit_transform(dff[cols_to_norm])
            if DEBUG >= 3:
                print('Normalized Data Frame of',
                      os.path.splitext(os.path.split(file)[1])[0])
                print(dff)

            # Compute the energy difference with respect to the first file and
            # shift the x axis
            for i in range(len(trans00)):
                difference = float(trans00[i])-float(trans00[0])
            dff.x -= difference
            print(os.path.splitext(os.path.split(file)[1])[
                  0], 'is shifted by', round(difference, 3), unit)

            if DEBUG >= 3:
                print('Normalized and shifted Data Frame of',
                      os.path.splitext(os.path.split(file)[1])[0])
                print(dff)
#
            # Plot the data
    #       Prepare  default minimum and maximum x-axis value
            if  unit == "cm-1":
                xmndef = min(dff['x']) - 200
                xmaxdef = max(dff['x']) + 200
            elif unit == "eV" or unit == "ev":
                xmndef = min(dff['x']) - 0.1
                xmaxdef = max(dff['x']) + 0.1
            elif unit == "nm":
                xmndef = min(dff['x']) - 2
                xmaxdef = max(dff['x']) + 2
#            print(xmndef,xmaxdef)
#
            if xmax and xmn:
                plt.xlim(xmn, xmax)
                plt.plot(dff["x"], dff["y"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
            elif xmax and not xmn:
                plt.xlim(xmndef,xmax)
                plt.plot(dff["x"], dff["y"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
            elif xmn and not xmax:
                plt.xlim(xmn, xmaxdef)
                plt.plot(dff["x"], dff["y"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
            else:
                plt.xlim(xmndef,xmaxdef)
                plt.plot(dff["x"], dff["y"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
            
#            if args.xmax == 100000:
#                if unit == "eV" or unit == "ev":
##                    xmax = xlim1 + 2
#                    xmax = xmn + 2
#                elif unit == "nm":
##                    xmax = xlim1 + 4
#                    xmax = xmn + 4
##                plt.xlim(xlim1, float(xmax))
#                plt.xlim(xmn, float(xmax))
#                plt.plot(dff["x"], dff["y"],
#                         label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val, linewidth=linewidth, markersize=markersize)
#                if add_legend:
#                    plt.legend(loc='lower center', bbox_to_anchor=(
#                        0.5, shift_legend), shadow=True, ncol=3)
#                plt.ylabel('Intensity', fontsize=fontsize)
#                if unit == "cm-1":
#                    plt.xlabel('E('+cm_unit+')', fontsize=fontsize)
#                else:
#                    plt.xlabel('E('+unit+')', fontsize=fontsize)
#            else:
#                #                print(dff["x"])
##                plt.xlim(xlim1, xmax)
#                plt.xlim(xmn, float(xmax))
#                plt.plot(dff["x"], dff["y"],
#                         label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val, linewidth=linewidth, markersize=markersize)
            if add_legend:
                plt.legend(loc='lower center', bbox_to_anchor=(
                    0.5, shift_legend), shadow=True, ncol=3)
            plt.ylabel('Intensity', fontsize=fontsize)
            if unit == "cm-1":
                plt.xlabel('E('+cm_unit+')', fontsize=fontsize)
            else:
                plt.xlabel('E('+unit+')', fontsize=fontsize)


###########################################
###########################################
# Plot the log files
    for file in filenameL:
        # print(file)
        file = os.path.join(directory, file)
        coord = []
        count = len(open(file).readlines())

        if ".log" in file:
            with open(file, 'r') as f:
                for line in f:
                    if "Energy of the 0-0 transition" in line:
                        tranE = line[32:43]
                        if unit == "cm-1":
                            trans00.append(float(tranE))
                            try:
                                xlim1
                            except UnboundLocalError:
                                xlim1 = int(float(trans00[0])) - 50
#                                print('xlim= ',xlim1)
                        elif unit == "eV" or unit == "ev":
                            tranE = float(tranE) * cmToeV
#                            xlim1 = int(float(trans00[0])) - 0.007
                            trans00.append(tranE)
                            try:
                                xlim1
                            except UnboundLocalError:
                                xlim1 = float(trans00[0]) - 0.007
#                                print("min trans00 ", min(trans00))
#                                print('xlim= ',xlim1)
                        elif unit == "nm":
                            tranE = float(tranE) * cmTonm
#                            xlim1 = int(float(trans00[0])) - 0.04
                            trans00.append(tranE)
                            try:
                                xlim1
                            except UnboundLocalError:
                                xlim1 = float(trans00[0]) - 0.04                            
#                                print('xlim= ',xlim1)                                
                        else:
                            print(
                                "wrong unit provided. Check the manual for accepted units")
                            sys.exit(1)
                            
                        # append the first energy
#                        trans00.append(tranE)
                    if "Intensity: Molar absorption coefficient" in line:
                        for i in range(count):
                            nextline = next(f)
                            nextline = nextline.replace("D", "e")
                            nextline = nextline.split()
                            if "Leave" in nextline:
                                break
                            if "Electric" in nextline:
                                break
                            else:
                                coord.append(nextline)
                        # Delete extra points
                        del coord[0]
                        del coord[-7:]

                # Form a data frame of the data
                df = pd.DataFrame(coord)
                if DEBUG >= 5:
                    print('Data Frame before adding headers of',
                          os.path.splitext(os.path.split(file)[1])[0])
                    print(df)

                # Add columns labels
                df.columns = ["x", "y"]
                dff = df.drop([0])
                if DEBUG >= 4:
                    print('Data Frame of', os.path.splitext(
                        os.path.split(file)[1])[0])
                    print(dff)

                # Convert all the column types to numeric
                dff[["x", "y"]] = dff[["x", "y"]].apply(pd.to_numeric)
                # Normalize the data
                cols_to_norm = ["y"]
                dff[cols_to_norm] = min_max_scaler.fit_transform(
                    dff[cols_to_norm])
                if DEBUG >= 3:
                    print('Normalized Data Frame of',
                          os.path.splitext(os.path.split(file)[1])[0])
                    print(dff)
                    
#                # Next we will find the first peak in the spectrum
#                firstpeak = next(x[0] for x in enumerate(dff['y']) if x[1] > Peakheight)
#                p = dff['x']
#    #             print('AAT',p.iloc[firstpeak])
#                MainEng = p.iloc[firstpeak]
#                trans00.append(MainEng)                    

                # Scale the data
                dff["y"] *= scale
                # Convert the energy to the requested unit
                if unit == "cm-1":
                    pass
                elif unit == "eV" or unit == "ev":
                    dff["x"] = dff["x"] * cmToeV
                elif unit == "nm":
                    dff["x"] = dff["x"] * cmTonm
                else:
                    print("wrong unit provided. Check the manual for accepted units")
                    sys.exit(1)

                # Shift and align the plots if requested
                if noshift:
                    #                    print(trans00)
                    # Convert the energy to the requested unit
                    if unit == "cm-1":
                        xlim1 = min(trans00) - 50
                    elif unit == "eV" or unit == "ev":
                        xlim1 = min(trans00) - 0.007
                        xmax = xmax * cmToeV
                    elif unit == "nm":
                        xlim1 = min(trans00) - 0.04
                        xmax = xmax * cmTonm

#                    print("min(trans00) = ",min(trans00))
#                    xlim1= min(trans00)
#                    print("This is xlim1 after changing",xlim1)
                else:
                    # Compute the energy difference with respect to the first file and
                    # shift the x axis
                    for i in range(len(trans00)):
                        difference = float(trans00[i])-float(trans00[0])
                    dff.x -= difference
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference, 3), unit)
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference, 3), unit, file=openEngDifF)
                if DEBUG >= 3:
                    print('Normalized and shifted Data Frame of',
                          os.path.splitext(os.path.split(file)[1])[0])
                    print(dff)
#            print(trans00)

            # Plot the data
    #       Prepare  default minimum and maximum x-axis value
            if  unit == "cm-1":
                xmndef = min(dff['x']) - 200
                xmaxdef = max(dff['x']) + 200
            elif unit == "eV" or unit == "ev":
                xmndef = min(dff['x']) - 0.1
                xmaxdef = max(dff['x']) + 0.1
            elif unit == "nm":
                xmndef = min(dff['x']) - 2
                xmaxdef = max(dff['x']) + 2
#            print(xmndef,xmaxdef)                
#                
            if xmax and xmn:
                plt.xlim(xmn, xmax)
                plt.plot(dff["x"], dff["y"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)            
            elif xmax and not xmn:
                plt.xlim(xmndef,xmax)
                plt.plot(dff["x"], dff["y"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)            
            elif xmn and not xmax:
                plt.xlim(xmn, xmaxdef)
                plt.plot(dff["x"], dff["y"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)            
            else:
                plt.xlim(xmndef,xmaxdef)
                plt.plot(dff["x"], dff["y"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
    

#            if args.xmax == 100000:
#                if unit == "eV" or unit == "ev":
#                    xmax = xlim1 + 2
#                elif unit == "nm":
#                    xmax = xlim1 + 4
##                plt.xlim(xlim1, float(xmax))
#                plt.xlim(xmn, float(xmax))
#                plt.plot(dff["x"], dff["y"],
#                         label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val, linewidth=linewidth, markersize=markersize)
#                if add_legend:
#                    plt.legend(loc='lower center', bbox_to_anchor=(
#                        0.5, shift_legend), shadow=True, ncol=3)
#                plt.ylabel('Intensity', fontsize=fontsize)
#                if unit == "cm-1":
#                    plt.xlabel('E('+cm_unit+')', fontsize=fontsize)
#                else:
#                    plt.xlabel('E('+unit+')', fontsize=fontsize)
#
#            else:
##                plt.xlim((xlim1, xmax))
#                plt.xlim(xmn, float(xmax))
#                plt.plot(dff["x"], dff["y"],
#                         label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val, linewidth=linewidth, markersize=markersize)
            if add_legend:
                plt.legend(loc='lower center', bbox_to_anchor=(
                    0.5, shift_legend), shadow=True, ncol=3)
            plt.ylabel('Intensity', fontsize=fontsize)
            if unit == "cm-1":
                plt.xlabel('E('+cm_unit+')', fontsize=fontsize)
            else:
                plt.xlabel('E('+unit+')', fontsize=fontsize)

    plt.savefig(figname+'.'+figformat, bbox_inches='tight',
                format=figformat, dpi=600)

#########################################################
#########################################################
# This code grabs SCF E of single point calculations and 
# create a pd df of log files located in the directory
#########################################################
#########################################################
def SCFEtopd(directory, list_of_files):
    DEBUG     = 1
    EDoneList = []
#####################################################
#####################################################
    output_csv = args.outputcsv
    output_csv = os.path.join(directory, output_csv)
    EDoneList = []
    for file in sorted(list_of_files):
        if not list_of_files:
            print('There are no Gaussian log files considered')
            break
        file = os.path.join(directory, file)
        if "log" in file:
            with open(file, 'r') as f:
                for line in f:
                    if "Done: " in line:
                        line=line.split()
                        EDoneList.append(float(line[4]))
    X_list = list(range(1,len(EDoneList)+1))
    df = pd.DataFrame({'x':X_list , "y":EDoneList})
    if DEBUG == 2:
        print(df)
    df.to_csv(output_csv, index=False)

#########################################################
#########################################################
# This code grabs S of single point calculations and
# create a pd df of log files located in the directory
#########################################################
#########################################################
def TDEtopd(directory, list_of_files):
    DEBUG     = 1
    EDoneList = []
#####################################################
#####################################################
    output_csv = args.outputcsv
    output_csv = os.path.join(directory, output_csv)
    EDoneList = []
    for file in sorted(list_of_files):
        if not list_of_files:
            print('There are no Gaussian log files considered')
            break
        file = os.path.join(directory, file)
        if "log" in file:
            for line in reversed(list(open(file))):
                    if "Total Energy," in line:
                        line=line.split()
                        EDoneList.append(float(line[4]))
    X_list = list(range(1,len(EDoneList)+1))
    df = pd.DataFrame({'x':X_list , "y":EDoneList})
    if DEBUG == 2:
        print(df)
    df.to_csv(output_csv, index=False)
###############################################
###############################################
# This function plots csv files in a given list
###############################################
###############################################
def Plotcsv(directory, list_of_files):
    for file in list_of_files:
        file = os.path.join(directory, file)
        if ".csv" in file:
            dff=pd.read_csv(file)
            dff.columns = ["x", "y"]
            # Plot the data
            plt.plot(dff['x'], dff['y'],
                     label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val, linewidth=linewidth, markersize=markersize)
            plt.ylabel('Energy(au)', fontsize=fontsize)
            plt.xlabel('Scan pnts', fontsize=fontsize)
            if add_title:
                plt.title(figname)
            if add_legend:
                plt.legend(loc='lower center', bbox_to_anchor=(
                    0.5, shift_legend), shadow=True, ncol=3)
    plt.savefig(figname+'.'+figformat, bbox_inches='tight',
                format=figformat, dpi=600)
            #####################################################
#######################################################
#######################################################
# This function plots This Function will plot the FC 
# of Photodetachment using the reported transitions 
# in Gaussian log file and not the x and y coordinates.
######################################################
######################################################
def PlotFCTrans(directory, list_of_files):
    #####################################
    # Setting Initial parameters    
    file_logs = []
    file_TDL = []
    file_ExpL = []
    xmn = args.xmin
    xmax = args.xmax
    filenameL = []
    test = []
    x_keys = []
    y_values = []
    trans00 = []
    file_expL = []
    xval = []
    xvalshift = []
    yval = []
    cmToeV = float(1/8065.6)
    cmTonm = float(10000000.0)
    cm_unit = "$cm^{-1}$"

#############################
    if noshift:
        pass
    else:
        openEngDifF = open(EngDifF, "w")

######################################################
#####################################################
# Prepare the lists
###################
    for filename in list_of_files:
        if ".log" in filename:
            filenameL.append(filename)
        if "csv" in filename:
            file_expL.append(filename)

#####################################################
#####################################################
# Set the experimental data as a refernce and plot it
    for file in file_expL:
        #        print(file_expL)
        file = os.path.join(directory, file)
        coord = []
        count = len(open(file).readlines())
        if ".sum" in file or ".txt" in file or ".dat" in file or ".csv" in file:
            df= pd.read_csv(file)
            if len(df.columns)==1:
                try:
                    dff = pd.read_csv(file, sep='\t')
                except IndexError:
                    print("the csv file is not tab delimited ",os.path.splitext(os.path.split(file)[1])[0])
                    try:
                        dff = pd.read_csv(file, sep=' ')
                    except IndexError:
                        print(" the csv file is not space delimited ", os.path.splitext(os.path.split(file)[1])[0])
                        print("Check the csv file ", os.path.splitext(os.path.split(file)[1])[0])
                        break
            elif len(df.columns)>=2:
                dff=df

#            df=pd.read_csv(file)
#            dff = dff.iloc[:, [0, 1]]
            dff = dff.iloc[:]
            try:
                dff.columns = ["x", "y"]
            except ValueError:
                print('check the delimiter in file {}. Currently supports "tab","comma", and "space" delimiters '.format(os.path.splitext(os.path.split(file)[1])[0] ))
                break
            dff = dff.sort_values("x", ascending=True)
            # Next we will find the first peak in the spectrum
            firstpeak = next(x[0] for x in enumerate(dff['y']) if x[1] > Peakheight)
            p = dff['x']
    #         print('AAT',p.iloc[firstpeak])
            MainEng = p.iloc[firstpeak]
            trans00.append(MainEng)            
#            # Next we will find all the peaks in the spectrum
#            peaks, _ = find_peaks(dff["y"], height=0.4)
#            p = dff["x"]
#
#            # Convert all the column types to numeric
#            dff[["x", "y"]] = dff[["x", "y"]].apply(pd.to_numeric)
#
#            # Find the energy of the first peak
#            MainEng = p.iloc[peaks[0]]

            # Convert the energy to the requested unit
            if unit == "cm-1":
#                xlim1 = float(MainEng) - 50
                trans00.append(MainEng)
            elif unit == "eV" or unit == "ev":
                dff["x"] = dff["x"] * cmToeV
                tranE = float(MainEng) * cmToeV
                xmax = xmax * cmToeV
                trans00.append(tranE)
            elif unit == "nm":
                xmax = xmax * cmTonm
                dff["x"] = dff["x"] * cmTonm
                tranE = float(MainEng) * cmTonm
#                xlim1 = tranE - 0.04
                trans00.append(tranE)
            else:
                print("wrong unit provided. Check the manual for accepted units")
                sys.exit(1)

            # Normalize the data
            cols_to_norm = ["y"]
            dff[cols_to_norm] = min_max_scaler.fit_transform(dff[cols_to_norm])
            if DEBUG >= 3:
                print('Normalized Data Frame of',
                      os.path.splitext(os.path.split(file)[1])[0])
                print(dff)

            # Compute the energy difference with respect to the first file and
            # shift the x axis
            for i in range(len(trans00)):
                difference = float(trans00[i])-float(trans00[0])
            dff.x -= difference
            print(os.path.splitext(os.path.split(file)[1])[
                  0], 'is shifted by', round(difference, 3), unit)

            if DEBUG >= 3:
                print('Normalized and shifted Data Frame of',
                      os.path.splitext(os.path.split(file)[1])[0])
                print(dff)
#
            # Plot the data
            if args.xmax == 100000:
                if unit == "eV" or unit == "ev":
#                    xmax = xlim1 + 2
                    xmax = xmn + 2
                elif unit == "nm":
#                    xmax = xlim1 + 4
                    xmax = xmn + 4
#                plt.xlim(xlim1, float(xmax))
                plt.xlim(xmn, float(xmax))
                plt.plot(dff["x"], dff["y"],
                         label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val, linewidth=linewidth, markersize=markersize)
                if add_legend:
                    plt.legend(loc='lower center', bbox_to_anchor=(
                        0.5, shift_legend), shadow=True, ncol=3)
                plt.ylabel('Intensity', fontsize=fontsize)                
                if unit == "cm-1":
                    plt.xlabel('E('+cm_unit+')', fontsize=fontsize)
                else:
                    plt.xlabel('E('+unit+')', fontsize=fontsize)
            else:
                #                print(dff["x"])
#                plt.xlim(xlim1, xmax)
                plt.xlim(xmn, float(xmax))
                plt.plot(dff["x"], dff["y"],
                         label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val, linewidth=linewidth, markersize=markersize)
                if add_legend:
                    plt.legend(loc='lower center', bbox_to_anchor=(
                        0.5, shift_legend), shadow=True, ncol=3)
                plt.ylabel('Intensity', fontsize=fontsize)
                if unit == "cm-1":
                    plt.xlabel('E('+cm_unit+')', fontsize=fontsize)
                else:
                    plt.xlabel('E('+unit+')', fontsize=fontsize)


###########################################
###########################################
# Plot the log files
    for file in filenameL:
#        print(file)
        file = os.path.join(directory, file)
        coord = []
        count = len(open(file).readlines())
        with open(file, 'r') as f:
            for line in f:
                for j,line in enumerate(f):
                    if " Information on Transitions" in line:
                        for i in range(count-j-1):
                            nextline = next(f)
        #                     nextline = nextline.replace("D", "e")
                            nextline = nextline.split("=")
        #                     nextline = nextline.rstrip()
        #                     print(nextline)
                            if any("Final Spectrum" in w for w in nextline):
    #                             print("Done")
                                break
                            else:
                                coord.append(nextline)
                if DEBUG >= 3:
                    print("Coordinates extracted",coord) 
                for i in coord:
                    for j in i:
                        if "Energy of the 0-0" in j:
    #                     print(j)
                            zsplit = j.split(':')
                            zsplit = zsplit[1].split('cm')
                            AdEng= float(zsplit[0])
    #                     print(AdEng)
                del coord[0:9]
                del coord[-2:]
            for i in range(len(coord)):
                if "Energy" in coord[i][0]:
                    Engval = coord[i][1].split("cm")
                    xval.append(float(Engval[0]))
            
                if "Intensity" in coord[i][0]:
                    Intval = coord[i][1].split("(D")
                    yval.append(float(Intval[0]))
    #                
            for i in xval:
                j = float(i) + AdEng
                xvalshift.append(j)
            # print(xvalshift)
            
            # Convert list to dataframe
            toBePd= list(zip(xvalshift,yval))
            if DEBUG >= 3:
                print('xvalshift',xvalshift)
            dff = pd.DataFrame(toBePd, columns = ['x', 'y'])
            if DEBUG >= 3:
                print('Dataframe with including the 0-0 transition ', dff)
                
            # Normalize the data
            cols_to_norm = ['y']
            dff[cols_to_norm] = min_max_scaler.fit_transform(dff[cols_to_norm])
            if DEBUG >= 4:
                print('Normalized Data Frame of',
                      os.path.splitext(os.path.split(file)[1])[0])
                print(dff)
            if DEBUG >= 3:
                print('After normalize', dff)
            
        # Next we will find the first peak in the spectrum
            firstpeak = next(x[0] for x in enumerate(dff['y']) if x[1] > Peakheight)
            p = dff['x']
    #         print('AAT',p.iloc[firstpeak])
            MainEng = p.iloc[firstpeak]
            trans00.append(MainEng)
    #         print("full list",trans00)
            if noshift:
                print("Not gonna shift and align the spectra")
                pass
            else:
                # Compute the energy difference with respect to the first file and
                # shift the x axis
                for i in range(len(trans00)):
                    difference = float(trans00[i])-float(trans00[0])
    #             print(difference)
                dff.x -= difference
    #             print("After shift",dff)
                if unit == "cm-1":
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference, 2), unit)
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference, 2), unit, file=openEngDifF)
                elif unit == "eV" or unit == "ev":
                    difference_ev = difference * cmToeV
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference_ev, 3), unit)
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference_ev, 3), unit, file=openEngDifF)
                elif unit == "nm":
                    try:
                        difference_nm = cmTonm / difference
                    except ZeroDivisionError:
                        difference_nm = 0.0
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference_nm, 3), unit)
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference_nm, 3), unit, file=openEngDifF)
                else:
                    print("wrong unit provided. Check the manual for accepted units")
                    sys.exit(1)
    #                print(os.path.splitext(os.path.split(file)[1])[0], 'is shifted by', round(difference,2), unit)
                if DEBUG >= 4:
                    print('Normalized and shifted Data Frame of',
                          os.path.splitext(os.path.split(file)[1])[0])
                    print(dff)
            # Scale the data
    #         if (len(dff.columns)) == 3:
    # #             dff['y-0k'] *= scale
    #             dff['y'] *= scale
    #         else:
            dff['y'] *= scale
    
            # Convert to correct unit
            if unit == "cm-1":
                pass
            elif unit == "eV" or unit == "ev":
                dff['x'] = cmToeV * dff['x']
            elif unit == "nm":
                dff['x'] = cmTonm / dff['x']
            else:
                print("wrong unit provided. Check the manual for accepted units")
                sys.exit(1)
    #       Prepare  default minimum and maximum x-axis value
            if  unit == "cm-1":
                xmndef = min(dff['x']) - 2000
                xmaxdef = max(dff['x']) + 2000
            elif unit == "eV" or unit == "ev":
                xmndef = min(dff['x']) - 0.1
                xmaxdef = max(dff['x']) + 0.1
            elif unit == "nm":
                xmndef = min(dff['x']) - 2
                xmaxdef = max(dff['x']) + 2
# plotting                
            if xmax and xmn:
                plt.xlim(float(xmn), xmax)
                markerline, stemline, baseline = plt.stem(dff['x'],dff['y'],linefmt='b',markerfmt='D',basefmt='b',label=os.path.splitext(os.path.split(file)[1])[0])
            elif xmax and not xmn:
                plt.xlim(xmndef,xmax)
                markerline, stemline, baseline = plt.stem(dff['x'],dff['y'],linefmt='b',markerfmt='D',basefmt='b',label=os.path.splitext(os.path.split(file)[1])[0])
            elif xmn and not xmax:
                plt.xlim(float(xmn),xmaxdef)
                markerline, stemline, baseline = plt.stem(dff['x'],dff['y'],linefmt='b',markerfmt='D',basefmt='b',label=os.path.splitext(os.path.split(file)[1])[0])                
            else:
                markerline, stemline, baseline = plt.stem(dff['x'],dff['y'],linefmt='b',markerfmt='D',basefmt='b',label=os.path.splitext(os.path.split(file)[1])[0])
            plt.setp(stemline, linewidth = linewidth)
            plt.setp(markerline, markersize = markersize)
            plt.xlabel('E('+unit+')', fontsize=fontsize)
            plt.ylabel('Intensity', fontsize=fontsize)
            if add_legend:
                plt.legend(loc='upper center', bbox_to_anchor=(
                        0.5, shift_legend), shadow=True, ncol=3)
        # Save the figure
        plt.savefig(figname+'.'+figformat, bbox_inches='tight',
                format=figformat, dpi=600)                
##########################################################################                
def PlotUV(directory, list_of_files):
    #######################################
    file_txt = []
    test = []
    trans00 = []
    temp = args.temp
    xmn = args.xmin
    xmax = args.xmax
    alpha_val = args.alpha
    unit=args.unit
#    fline2 = None
    cmToeV = float(0.000123984)
    cmTonm = float(10000000.0)
    nmTocm = float(10000000.0)
    nmToeV = float(1239.84193)
    cm_unit = "$cm^{-1}$"

#   General printing
    for filename in list_of_files:
        if "txt" in filename:
            file_txt.append(filename)
#    print(file_logs)
#    print(file_TDL)
#    print(file_ExpL)
#####################################################
#####################################################
    # Open the file for writing the Energy difference
    if noshift:
        pass
    else:
        openEngDifF = open(EngDifF, "w")

    # Loop through csv files
    for file in file_txt:
        if not file_txt:
            print('There are no txt files considered')
            break
#        print(file_expL)
        file = os.path.join(directory, file)
        coord = []
        count = len(open(file).readlines())
        count = len(open(file).readlines())
        with open(file, 'r') as f:
            fline1 = False
            fline2 = False
            for line in f:
                if "Spectra" in line:
                 break
            count2= sum(1 for line in f)
        print(count2)
        with open(file, 'r') as f:
            for line in f:
                if "Spectra" in line:
#                 print(line)
                    for i in range(count2-1):
                        nextline = next(f)                   
                        nextline = nextline.split()
                        coord.append(nextline)
          
                # Delete extra points
            del coord[0]
#                   Getting the intensities
#                if "2nd col." in line:
#                    fline = line.split("=")
#                    fline1 = fline[1].strip()
#                    fline11 = fline[0].split("col")
#                if "3rd col." in line:
#                    fline = line.split("=")
#                    fline2 = fline[1].strip()
#                    fline22 = fline[0].split("col")

                # Form a data frame of the data
            df = pd.DataFrame(coord)
            if (DEBUG >= 5):
                print('Data Frame before adding headers of',
                      os.path.splitext(os.path.split(file)[1])[0])
                print(df)

            # Add columns labels
#            print(df)
#            print(len(df.columns))
            if (len(df.columns)) == 0:
                print("No data was found")
                sys.exit(1)

            elif (len(df.columns)) == 3:
                df.columns = ['x', 'y-0k', 'y-k']
            else:
                df.columns = ['x', 'y-0k']
            dff = df.drop([0])
            if DEBUG >= 4:
                print('Data Frame of', os.path.splitext(
                    os.path.split(file)[1])[0])
                print(dff)

            # Convert all the column types to numeric
            if (len(df.columns)) == 3:
                dff[['x', 'y-0k', 'y-k']] = dff[['x',
                                                 'y-0k', 'y-k']].apply(pd.to_numeric)
            else:
                dff[['x', 'y-0k']] = dff[['x', 'y-0k']].apply(pd.to_numeric)

            # Normalize the data
            if (len(df.columns)) == 3:
                cols_to_norm = ['y-0k', 'y-k']
            else:
                cols_to_norm = ['y-0k']
            dff[cols_to_norm] = min_max_scaler.fit_transform(dff[cols_to_norm])
            if DEBUG >= 4:
                print('Normalized Data Frame of',
                      os.path.splitext(os.path.split(file)[1])[0])
                print(dff)
            # print(dff)
        # Next we will find the first peak in the spectrum
#####################################################################
            p = dff['x']
            # Append the energy of the first peak
            MainEng = p.iloc[firstpeak]
            trans00.append(MainEng)
            # Printing available temps in the log file
            if fline2:
                print()
                print("file",os.path.splitext(os.path.split(file)[1])[0],
                        "has intensity values at {} and {}.".format(fline1,fline2))
                print()
            else:
                print()
                print("file",os.path.splitext(os.path.split(file)[1])[0],
                        "has intensity values at {} only.".format(fline1))
                print()

            if noshift:
                print("Not gonna shift and align the spectra")
                pass
            else:
                # Compute the energy difference with respect to the first file and
                # shift the x axis
                for i in range(len(trans00)):
                    difference = float(trans00[i])-float(trans00[0])
                dff.x -= difference
                if float(difference) == 0:
                    print("File {} is set to be the reference".format(os.path.splitext(os.path.split(file)[1])[0]))
                    pass
                elif unit == "cm-1":

                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference, 2), unit)
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference, 2), unit, file=openEngDifF)
                elif unit == "eV" or unit == "ev":
                    difference_ev = difference * cmToeV
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference_ev, 3), unit)
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference_ev, 3), unit, file=openEngDifF)
                elif unit == "nm":
                    try:
                        difference_nm = cmTonm / difference
                    except ZeroDivisionError:
                        difference_nm = 0.0
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference_nm, 3), unit)
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference_nm, 3), unit, file=openEngDifF)
                else:
                    print("wrong unit provided. Check the manual for accepted units")
                    sys.exit(1)
#                print(os.path.splitext(os.path.split(file)[1])[0], 'is shifted by', round(difference,2), unit)
                if DEBUG >= 4:
                    print('Normalized and shifted Data Frame of',
                          os.path.splitext(os.path.split(file)[1])[0])
                    print(dff)

            # Scale the data
            if (len(df.columns)) == 3:
                dff['y-0k'] *= scale
                dff['y-k'] *= scale
            else:
                dff['y-0k'] *= scale

            # Convert to correct unit
            if unit == "cm-1":
                pass
            elif unit == "eV" or unit == "ev":
                dff['x'] = cmToeV * dff['x']
            elif unit == "nm":
                dff['x'] = cmTonm / dff['x']
            else:
                print("wrong unit provided. Check the manual for accepted units")
                sys.exit(1)
    #       Prepare  default minimum and maximum x-axis value
            if  unit == "cm-1":
                xmndef = min(dff['x']) - 2000
                xmaxdef = max(dff['x']) + 2000
            elif unit == "eV" or unit == "ev":
                xmndef = min(dff['x']) - 2
                xmaxdef = max(dff['x']) + 2
            elif unit == "nm":
                xmndef = min(dff['x']) - 5
                xmaxdef = max(dff['x']) + 5
#            print(xmndef,xmaxdef)
        # Plot the data
        if temp == fline1:
            if xmax and xmn:
                plt.xlim(xmn, xmax)
                plt.plot(dff["x"], dff["y-0k"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
            elif xmax and not xmn:
                plt.xlim(xmndef,xmax)
                plt.plot(dff["x"], dff["y-0k"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
            elif xmn and not xmax:
                plt.xlim(xmn, xmaxdef)
                plt.plot(dff["x"], dff["y-0k"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
            else:
                plt.xlim(xmndef,xmaxdef)
                plt.plot(dff["x"], dff["y-0k"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
        else:
            if xmax and xmn:
                plt.xlim(xmn, xmax)
                plt.plot(dff["x"], dff["y-k"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
            elif xmax and not xmn:
                plt.xlim(xmndef,xmax)
                plt.plot(dff["x"], dff["y-k"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
            elif xmn and not xmax:
                plt.xlim(xmn, xmaxdef)
                plt.plot(dff["x"], dff["y-k"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
            else:
                plt.xlim(xmndef,xmaxdef)
                plt.plot(dff["x"], dff["y-k"], label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val,
                         linewidth=linewidth, markersize=markersize)
        plt.ylabel('Abs. (arb.units', fontsize=fontsize)
        if unit == "cm-1":
            plt.xlabel('Wavelength('+cm_unit+')', fontsize=fontsize)
        else:
            plt.xlabel('Wavelength('+unit+')', fontsize=fontsize)
        if add_legend:
            plt.legend(loc='upper center', bbox_to_anchor=(
                0.5, shift_legend), shadow=True, ncol=3)
        if DEBUG >= 3:
            dff.to_csv(os.path.splitext(os.path.split(file)[1])[
                           0]+temp+'.csv', sep=',', index=False, columns=['x', 'y'])


    #       Prepare  default minimum and maximum x-axis value
            if  unit == "cm-1":
                xmndef = min(dff['x']) - 2000
                xmaxdef = max(dff['x']) + 2000
            elif unit == "eV" or unit == "ev":
                xmndef = min(dff['x']) - 2
                xmaxdef = max(dff['x']) + 2
            elif unit == "nm":
                xmndef = min(dff['x']) - 5
                xmaxdef = max(dff['x']) + 5
#            print(xmndef,xmaxdef)
    # Save the figure
    plt.savefig(figname+'.'+figformat, bbox_inches='tight',
                format=figformat, dpi=600)
            #####################################################
#######################################################
#######################################################
# This function plots the UV spectra using the reported 
# excitations in Gaussian log file
######################################################
######################################################
def gaussBand(x, band, strength, stdev):
    "Produces a Gaussian curve"
    bandshape = 1.3062974e8 * (strength / (1e7/stdev))  * np.exp(-(((1.0/x)-(1.0/band))/(1.0/stdev))**2)
    return bandshape

def lorentzBand(x, band, strength, stdev, gamma):
    "Produces a Lorentzian curve"
    bandshape = 1.3062974e8 * (strength / (1e7/stdev)) * ((gamma**2)/((x - band)**2 + gamma**2))
    return bandshape

def PlotUVGauss(directory,list_of_files):
    # Setting Initial parameters
    normalize = args.normalize
    xmn = args.xmin
    xmax = args.xmax
    print(xmn)
    print(xmax)    
    alpha_val = args.alpha
    unit = args.unit
    sd = args.standarddev
    gp = args.gridpoints
    Peakheight = args.peakheight
    cmToeV = float(0.000123984)
    cmTonm = float(10000000.0)
    nmTocm = float(10000000.0)
    nmToeV = float(1239.84193)
    eVTonm = float(1239.84193)
    cm_unit = "$cm^{-1}$"
    e_unit = "$\epsilon$ (L mol$^{-1}$ cm$^{-1}$)"
    x_keys = []
    y_values = []
    trans00 = []
    xval = []
    xvalshift = []
    yval = []
    cmToeV = float(1/8065.6)
    cmTonm = float(10000000.0)
    cm_unit = "$cm^{-1}$"
    filenameL = []
    log_dataframe = []
    file_GL = []
    file_QL =[]
    sd = nmToeV / sd
    for filename in list_of_files:
        if filename.endswith(".log") or filename.endswith(".out"):
            filenameL.append(filename)
    # Make list with disered files
    for filename in filenameL:
        if "log" in filename in filename:
            file_GL.append(filename)
        if "out" in filename:
            file_QL.append(filename)
    print('Gaussina Files considered',file_GL)
    #print('QChem Files Considered',file_QL)
    for file in file_GL:
        print(file)
        file = os.path.join(directory, file)
        waveLen = []
        OscStr = [] 
        count = len(open(file).readlines())
        with open(file) as f:
            for line in f:
                if "Copying SCF densities to generalized" in line:
    #                 print(line)
                    break
            count2= sum(1 for line in f)
    #         print(count2)
        with open(file, 'r') as f:
            for line in f:
                if "Excitation energies and oscillator strengths" in line:
    #                 print(line)
                    for i in range(count2-1):
                        nextline = next(f)
                        nextline = nextline.split()
    #                     print(nextline) 
                        if ("Excited" in nextline):
    #                         print('done')
    #                         break
                            waveLen.append(nextline[4])
                            OscStr.append(nextline[8])

    # Clean Osc Strength List ; remove "f="
            OscStr = [s.replace("f=", "") for s in OscStr]
            OscStr = [float(s) for s in OscStr]
            waveLen = [float(s) for s in waveLen]
            waveLen = [eVTonm/s for s in waveLen]
    #         print(waveLen)
    #         print(OscStr)
            # Convert to correct unit
            if unit == "nm":
                llim=min(waveLen) - 200
                rlim=max(waveLen) + 200
                x = np.linspace(llim,rlim,int(gp)) 
#            elif unit == "eV" or unit == "ev":
#                try:
#                    llim = nmToeV / min(waveLen)
#                except ZeroDivisionError:
#                    llim = 0
##                try:
#                rlim =nmToeV / max(waveLen) + 2
# #               except ZeroDivisionError:
#  #                  rlim = 0
#                x = np.linspace(llim,rlim,int(gp))
#            elif unit == "cm-1":
#                try:
#                    llim = nmTocm / (1/min(waveLen))
#                except ZeroDivisionError:
#                    llim = 0
##                try:
#                rlim = (nmTocm * (1/max(waveLen))) + 500
##                except ZeroDivisionError:
##                    rlim = 0
#                x = np.linspace(llim,rlim,int(gp))
#            else:
#                print("wrong unit provided. Check the manual for accepted units")
#                sys.exit(1)    
            #print('llmit',llim)
            #print('rlimt',rlim)
           # x = np.linspace(int(xmn),int(xmax),int(gp))
            llim=min(waveLen) - 200
            rlim=max(waveLen) + 200           
            x = np.linspace(llim,rlim,int(gp))

            composite = 0
            for count,peak in enumerate(waveLen):
                thispeak = gaussBand(x, peak, OscStr[count], sd)
        #         print(thispeak)
            #    thispeak = lorentzBand(x, peak, f[count], stdev, gamma)
                composite += thispeak
        #     print(x)
        #     print(composite)
        # Convert list to dataframe
            toBePd= list(zip(x,composite))
            df = pd.DataFrame(toBePd, columns = ['x', 'y'])
#            print('AAT1',max(df['x']))
            if (DEBUG >= 5):
                print('Data Frame before adding headers of',
                      os.path.splitext(os.path.split(file)[1])[0])
#            print(df)
            if (len(df.columns)) == 0:
                break
            else:
                df.columns = ['x', 'y']
            dff = df.drop([0])
            if DEBUG >= 4:
                print('Data Frame of', os.path.splitext(
                    os.path.split(file)[1])[0])
                print(dff)

            # Convert all the column types to numeric
            dff[['x', 'y']] = dff[['x', 'y']].apply(pd.to_numeric)

            # Normalize the data
            if normalize:
                cols_to_norm = ['y']
                dff[cols_to_norm] = min_max_scaler.fit_transform(dff[cols_to_norm])
                if DEBUG >= 4:
                    print('Normalized Data Frame of',
                      os.path.splitext(os.path.split(file)[1])[0])
                    print(dff)
#            print('AAT',max(dff['y']))
            # Next we will find the first peak in the spectrum
            try:
                firstpeak = next(x[0] for x in enumerate(dff['y']) if x[1] > Peakheight)
            except StopIteration:
                continue
            p = dff['x']
    #         print('AAT',p.iloc[firstpeak])
            MainEng = p.iloc[firstpeak]
            trans00.append(MainEng)            

            if noshift:
                pass
            else:
                openEngDifF = open(EngDifF, "w")
                # Compute the energy difference with respect to the first file and
                # shift the x axis
                for i in range(len(trans00)):
                    difference = float(trans00[i])-float(trans00[0])
                #print(difference)
                dff.x -= difference
                if unit == "nm":
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference, 2), unit)
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference, 2), unit, file=openEngDifF)
                elif unit == "eV" or unit == "ev":
                    difference_ev = difference/nmToeV
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference_ev, 3), unit)
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference_ev, 3), unit, file=openEngDifF)
                elif unit == "cm-1":
                    try:
                        difference_cm = nmTocm * (1/difference)
                    except ZeroDivisionError:
                        difference_cm = 0.0
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference_cm, 3), unit)
                    print(os.path.splitext(os.path.split(file)[1])[
                          0], 'is shifted by', round(difference_cm, 3), unit, file=openEngDifF)
                else:
                    print("wrong unit provided. Check the manual for accepted units")
                    sys.exit(1)
                if DEBUG >= 4:
                    print('Normalized and shifted Data Frame of',
                          os.path.splitext(os.path.split(file)[1])[0])
                    print(dff)
            # Scale the data
            dff['y'] *= scale

            # Convert to correct unit
            if unit == "nm":
                pass
            elif unit == "eV" or unit == "ev":
                dff['x'] = nmToeV / dff['x']
                unit="eV"
            elif unit == "cm-1":
                dff['x'] = nmTocm * (1/dff['x'])
            else:
                print("wrong unit provided. Check the manual for accepted units")
                sys.exit(1)
        log_dataframe.append(dff)
#        print('here',max(dff['y']))
        #print(trans00[0])
        # Plot the data
        if normalize:
            plt.ylabel('Abs. (arb.units)', fontsize=fontsize)
        else:
            plt.ylabel(e_unit, fontsize=fontsize)
        if unit == "cm-1":
            plt.xlabel('Wavelength('+cm_unit+')', fontsize=fontsize)
        else:
            plt.xlabel('Wavelength('+unit+')', fontsize=fontsize)
#        plt.plot(dff['x'], dff['y'],
#                 label=os.path.splitext(os.path.split(file)[1])[0], alpha=alpha_val, linewidth=linewidth, markersize=markersize)
        if DEBUG >= 3:
            dff.to_csv(os.path.splitext(os.path.split(file)[1])[
                       0]+'.csv', sep=',', index=False, columns=['x', 'y'])
            
    #       Prepare  default minimum and maximum x-axis value
        if  unit == "cm-1":
            xmndef = min(dff['x']) - 2000
            xmaxdef = max(dff['x']) + 2000
        elif unit == "eV" or unit == "ev":
             xmndef = min(dff['x']) - 2
             xmaxdef = max(dff['x']) + 2
        elif unit == "nm":
             xmndef = min(dff['x']) - 5
             xmaxdef = max(dff['x']) + 5
#            print(xmndef,xmaxdef)
#        print('label is ',os.path.splitext(os.path.split(file)[1])[0])
        leglabel= os.path.splitext(os.path.split(file)[1])[0]
        if xmax and xmn:
            plt.xlim(xmn, xmax)
            plt.plot(dff["x"], dff["y"], label=leglabel , alpha=alpha_val,
                    linewidth=linewidth, markersize=markersize)
        elif xmax and not xmn:
            plt.xlim(xmndef,xmax)
            plt.plot(dff["x"], dff["y"], label=leglabel, alpha=alpha_val,
                    linewidth=linewidth, markersize=markersize)
        elif xmn and not xmax:
            plt.xlim(xmn, xmaxdef)
            plt.plot(dff["x"], dff["y"], label=leglabel, alpha=alpha_val,
                    linewidth=linewidth, markersize=markersize)
        else:
            plt.xlim(xmndef,xmaxdef)
            plt.plot(dff["x"], dff["y"], label=leglabel, alpha=alpha_val,
                    linewidth=linewidth, markersize=markersize)
        if add_legend:
            plt.legend(loc='upper center', bbox_to_anchor=(
                .5, shift_legend), shadow=True, ncol=2)            
    # Save the figure
    plt.savefig(figname+'.'+figformat, bbox_inches='tight',
                format=figformat, dpi=600)            
