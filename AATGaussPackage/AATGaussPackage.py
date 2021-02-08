#!/opt/anaconda3/bin/python
# coding: utf-8
# CREATED BY ALI ABOU TAKA
# THIS PYTHON PROGRAM  HAS DIFFERENT FUNCTIONS THAT CAN BE USED
# TO EXTRACT DESIRED INFORMATION FOR LOG FILES
# BE CAREFULL WHAT FUNCTION YOU CALL AND WHAT PATH YOU PROVIDE. 

import warnings
warnings.filterwarnings(action='ignore')
import glob
import re
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.style as style
import os
import argparse
import textwrap
from scipy.signal import find_peaks
from sklearn import preprocessing
import pandas as pd

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

                4) PlotEnrg:      Plot the energy at each SCF vs the number
                                  of SCF cycles. It takes two arguments: the
                                  directory and list of log files
                                  Accepted switches: -fn, -ff, -lw, -ms, -fs
                                 -xmn, -xmx, -al

                5) PlotDIIS:      Plot the DIIS error at each SCF vs the 
                                  number of SCF cycles. It takes two arguments:
                                  the directory and list of log files
                                  Accepted switches: -fn, -ff, -lw, -ms, -fs
                                 -xmn, -xmx, -al                                  

                6) PlotRMSDP:     Plot the RMSDP at each SCF vs the number
                                  of SCF cycles. It takes two arguments: the
                                  directory and list of log files
                                  Accepted switches: -fn, -ff, -lw, -ms, -fs
                                 -xmn, -xmx, -al                                  

                7) PlotNvirt:     Plot the N_virt metric for pimom methods
                                  vs the number of SCF cycles. It takes three 
                                  arguments: the directory, list of log files,
                                  and a string for either 'Alpha' or 'Beta'.
                                  Accepted switches: -fn, -ff, -lw, -ms, -fs
                                 -xmn, -xmx, -al                                  

                8) PlotExFC:      Plot FC of log files and exp files.
                                  It takes two arguments: the directory 
                                  and list of files (log + csv file).
                                  The list of files are further catigorized
                                  based on their extension.
                                  AllFileList function can be used to
                                  generate the list of files.
                                  Accepted switches: -fn, -ff, -lw, -ms, -fs, -s,
                                 -ns, -xmx, -al, -u, -esf, -ph
               
               9) PlotSpecFC:    This Function will plot the FC of Photodetachment.
                                 It takes two arguments: the directory
                                 and list of files (log + csv file).
                                 Accepted switches: -fn, -ff, -lw, -ms, -fs, -s,
                                 -ns, -xmx, -al, -u, -ph, -esf

               10) MergeDataF:   This Function will merge data files (.dat) into one
                                 csv file.
                                 Accepted switches: -efn


               If you encounter any problem with using this package,
               please contact Ali Abou Taka at abotaka.ali@gmail.com.
               Check the usage of each argument before using it.
                ------------------------------------------------------
                ''')
        )
####################
parser.add_argument("-f","--files", nargs='?', default="all", help="takes 'input' for files to be specified or 'all' to plot all files. all is the default.")
parser.add_argument("-fn","--figname", nargs='?', default="OutputFig", help="name of the figure to be saved. OutputFig is the default.")
parser.add_argument("-ff","--figformat", nargs='?', default='jpg', help="format of the image to be saved. jpg is the default.")
parser.add_argument("-ms","--markersize", nargs='?', type=float, default=5.0, help="Set the size of the marker labels. The default value is 5.0")
parser.add_argument("-lw","--linewidth", nargs='?', type=float, default=0.5, help="Set the size of the linewidth. The default value is 0.5")
parser.add_argument("-fs","--fontsize", nargs='?', type=float, default=12.0, help="Set the size of the font. The default value is 12.0")
parser.add_argument("-ls","--labelsize", nargs='?', type=float, default=12.0, help="Set the size of the x and y labels. The default value is 12.0")
parser.add_argument("-xmn","--xmin", nargs='?', type=float, default=0.0, help="Set x axis initial  point. The default value is 0.0")
parser.add_argument("-xmx","--xmax", nargs='?', type=float, default=10000, help="Set x axis end point. The default value is the all the points")
parser.add_argument("-al","--addlegend",  action="store_true", help="It is a boolean. If used, a legend will be placed under the plot. Without legend is the default.")

parser.add_argument("-t","--temp", nargs='?', default="0K", help="temperature at which the FC is generated at. 0K is the default. Currently supports 0K and 300K")
parser.add_argument("-u","--unit", nargs='?', default="cm-1", help="energy unit for the x axis.It takes cm-1, nm, or eV. cm-1 is the default.")
parser.add_argument("-esf","--energyshiftfile", nargs='?', default="EnergyShiftFile.txt", help="name of the file to be save the energy shift needed to align the plots. EnergyShiftFile is the default.")
parser.add_argument("-a","--alpha", nargs='?', type=float, default=1.0, help="alpha used to adjust transparency. It ranges between 0.0 and 1.0, where 1 is opaque. 1.0 is the default.")
parser.add_argument("-ns","--noshift",  action="store_true", help="It is a boolean. If used, the data will not be shifted. shifting is the default.")
parser.add_argument("-s","--scale", nargs='?', type=float, default=1.0, help="to scale the intensity after normalization. The default value is 1.0.")
parser.add_argument("-ph","--peakheight", nargs='?', type=float, default=0.4, help="to match the plot accprding to the first  peak wsth a specific height. The default value is 0.4")


parser.add_argument("-efn", "--expfilename", nargs='?', default="ExpData_Merged.csv", help="takes the name of the experimental file to be generated by the script after combining data files. If one data file is present, the code will still generate a csv formatted file to be used later. ExpData_Merged.csv is the default name.")




args = parser.parse_args()
#############################
#############################
# Setting Initial parameters
directory   = os.getcwd()
iChos       = args.files
figname     = args.figname
figformat   = args.figformat
markersize  = args.markersize
linewidth   = args.linewidth
fontsize    = args.fontsize
labelsize   = args.labelsize
xmin        = args.xmin
xmax        = args.xmax
add_legend  = args.addlegend

unit        = args.unit
temp        = args.temp
alpha_val   = args.alpha
noshift     = args.noshift
scale       = args.scale
EngDifF     = args.energyshiftfile
Peakheight  = args.peakheight

scale       = args.scale
expfilename = args.expfilename
#############################

#############################
if any([iChos == 'all', iChos == 'input']):
    pass
else:
    print("'files' can take either 'input' for giving the input file names next, or 'all' for all the log files")
    sys.exit(0)
#############################


###################################################################
# This function create a list with all the files in the directory #
###################################################################
def AllFileList(directory):
###########################
    filenameL = []
    for filename in sorted(os.listdir(directory)):
        filenameL.append(filename)
    return filenameL
####################

#######################################################################
# This function create a list with all the log files in the directory #
#######################################################################
def LogFileList(directory):
##########################
    list_of_files  = []
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
    filenameLExp= []
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
            df=pd.read_csv(file, sep='\t')
            df.columns = ['x', 'y']
            mergedExpdf.append(df)
            dff=pd.concat(mergedExpdf)
        dff=dff.sort_values('x',ascending=True)
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

####################################################
# This Function will grab the energy at each cycle #
####################################################
def PlotEnrg(directory,list_of_files):
######################################
    DEBUG      = 1
    list_of_filesL = []
    x_keys     = []
    y_values   = []    
    markers=[4,5,6,7,'s','P', '^',"*",'X']   

    for file in list_of_files:
        file=os.path.join(directory,file)
        EnergyL=[]
        with open (file,'r') as f:
            for line in f:
                if "E= " in line:
                    words = line.split()
                    if (words[0] =="E="):
                        energyval=float(words[1])
                        EnergyL.append(energyval)
        #     print (EnergyL)        
            x_keys.append(list(range(len(EnergyL))))
#                 print(x_keys)
            y_values.append(EnergyL)
            list_of_filesL.append(os.path.splitext(os.path.split(file)[1])[0])
        
        zipped_list= list(zip(x_keys,y_values,markers,list_of_filesL)) 
#         print(zipped3)
    for x,y,z,fn in zipped_list:
#         ax.plot(i,j, linestyle = '', marker=next(markers))
        if xmax and xmin:
            plt.xlim(xmin,xmax)
            plt.plot(x,y,label=fn,marker=z,linewidth=linewidth, markersize=markersize)
        elif xmax and not xmin:
            plt.xlim(right=xmax)
            plt.plot(x,y,label=fn,marker=z,linewidth=linewidth, markersize=markersize)
        elif xmin and not xmax:
            print(len(EnergyL))
            plt.xlim(xmin,len(EnergyL))
            plt.plot(x,y,label=fn,marker=z,linewidth=linewidth, markersize=markersize)
        else:
            plt.plot(x,y,label=fn,marker=z,linewidth=linewidth, markersize=markersize)
        if add_legend:
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), shadow=True, ncol=3)            
        style.use('seaborn-talk')
        ax = plt.subplot(111)    
        box = ax.get_position()    
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.set_xlabel('Number of SCF iterations',fontsize=fontsize)
        ax.set_ylabel('Energy in (a.u.)',fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=labelsize)
        ax.tick_params(axis="y", labelsize=labelsize)        
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)        
    # Save the figure
    plt.savefig(os.path.join(directory,figname+'.'+figformat),dpi=600,format=figformat,bbox_inches='tight')
    plt.close()


#######################################################
# This Function will plot the DIIS Error at each cycle#
#######################################################
def PlotDIIS(directory,list_of_files):
######################################    
    list_of_filesL=[]
    x_keys =  []
    y_values= []    
    markers=[4,5,6,7,'s','P', '^',"*",'X']   

    for file in list_of_files:
        file=os.path.join(directory,file)
        DIISL=[]
        with open (file,'r') as f:
            for line in f:
                if "DIIS: error=" in line:
                    line = line.replace("D","e")
                    words = line.split()
                    DIIS = float(words[2])
                    DIISL.append(DIIS) 
#            print(DIISL)
            x_keys.append(list(range(len(DIISL))))
#                 print(x_keys)
            y_values.append(DIISL)
            list_of_filesL.append(os.path.splitext(os.path.split(file)[1])[0])
        
        zipped_list= list(zip(x_keys,y_values,markers,list_of_filesL)) 
#         print(zipped3)
    for x,y,z,fn in zipped_list:
        if xmax and xmin:
            plt.xlim(xmin,xmax)
            plt.plot(x,y,label=fn,marker=z,linewidth=linewidth, markersize=markersize)
        elif xmax and not xmin:
            plt.xlim(right=xmax)
            plt.plot(x,y,label=fn,marker=z,linewidth=linewidth, markersize=markersize)
        elif xmin and not xmax:
            plt.xlim(xmin,len(EnergyL))
            plt.plot(x,y,label=fn,marker=z,linewidth=linewidth, markersize=markersize)
        else:
            plt.plot(x,y,label=fn,marker=z,linewidth=linewidth, markersize=markersize)
        if add_legend:
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), shadow=True, ncol=3)            
        style.use('seaborn-talk')
        ax = plt.subplot(111)    
        box = ax.get_position()    
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.set_xlabel('Number of SCF iterations',fontsize=fontsize)
        ax.set_ylabel('DIIS Error in (a.u.)',fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=labelsize)
        ax.tick_params(axis="y", labelsize=labelsize)        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)        

    plt.savefig(os.path.join(directory,figname+'DIIS'+ '.'+ figformat),dpi=600,format=figformat,bbox_inches='tight')
    plt.close()


###################################################
# This Function will grab the RMSDP at each cycle #
###################################################
def PlotRMSDP(directory,list_of_files):
#######################################
    ref=(glob.glob('*ref.log'))
    ref1 = "" 
    for ele in ref:  
        ref1 += ele
#    list_of_files=[]
    list_of_filesL=[]
    x_keys =  []
    y_values= []    
    markers=[4,5,6,7,'s','P', '^',"*",'X']       

    for file in list_of_files:
        file=os.path.join(directory,file)
        if file == ref1:
            continue    
        RMSDPL=[]
        with open (file,'r') as f:
            for line in f:
                if "RMSDP" in line:
                    line = line.replace("="," ")
                    line = line.replace("D","e")
                    words = line.split()
                    RMSDP = float(words[1])
                    RMSDPL.append(RMSDP) 
            x_keys.append(list(range(len(RMSDPL))))
            y_values.append(RMSDPL)
            list_of_filesL.append(os.path.splitext(os.path.split(file)[1])[0])
            
        zipped_list= list(zip(x_keys,y_values,markers,list_of_filesL)) 
    for x,y,z,fn in zipped_list:
        if xmax and xmin:
            plt.xlim(xmin,xmax)
            plt.plot(x,y,label=fn,marker=z,linewidth=linewidth, markersize=markersize)
        elif xmax and not xmin:
            plt.xlim(right=xmax)
            plt.plot(x,y,label=fn,marker=z,linewidth=linewidth, markersize=markersize)
        elif xmin and not xmax:
            print(len(EnergyL))
            plt.xlim(xmin,len(EnergyL))
            plt.plot(x,y,label=fn,marker=z,linewidth=linewidth, markersize=markersize)
        else:
            plt.plot(x,y,label=fn,marker=z,linewidth=linewidth, markersize=markersize)
        if add_legend:
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), shadow=True, ncol=3)            
        style.use('seaborn-talk')
        ax = plt.subplot(111)    
        box = ax.get_position()    
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.set_xlabel('Number of SCF iterations',fontsize=fontsize)
        ax.set_ylabel('RMSDP in (a.u.)',fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=labelsize)
        ax.tick_params(axis="y", labelsize=labelsize)        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)        

    plt.savefig(os.path.join(directory,figname+'RMSDP'+ '.'+ figformat),dpi=600,format=figformat,bbox_inches='tight')
    plt.close()
####################################################
# This Function will grab the N Virt at each cycle #
####################################################
def PlotNvirt(directory,list_of_files,switch):
##############################################    
    dicts = {}
    file_keys =  []
    maxNvirt_values= []
    list_of_filesL=[]
    x_keys =  []
    y_values= []    
    markers=[4,5,6,7,'s','P', '^',"*",'X']      

    if switch != 'Alpha' and  switch != 'Beta':
        print('This functions is called with the dirctory and the Alpha or Beta switch')
        print('Make sure to use Alpha or Beta')
        sys.exit(1)

    for file in list_of_files:
        file=os.path.join(directory,file)  
        NvirtL=[]
        with open (file,'r') as f:
            if switch== 'Alpha':
                for line in f:
                    if "Sum of the diagonals for virt the Alpha" in line:
                        words = line.split()
                        NvirtAlpha = float(words[9])
                        NvirtL.append(NvirtAlpha)
                x_keys.append(list(range(len(NvirtL))))
        #                 print(x_keys)
                y_values.append(NvirtL)
                list_of_filesL.append(os.path.splitext(os.path.split(file)[1])[0])                        
                        
            elif switch=='Beta':
                for line in f:
                    if "Sum of the diagonals for virt the Beta" in line:
                        words = line.split()
                        NvirtBeta = float(words[9])
                        NvirtL.append(NvirtBeta) 
                x_keys.append(list(range(len(NvirtL))))
        #                 print(x_keys)
                y_values.append(NvirtL)
                list_of_filesL.append(os.path.splitext(os.path.split(file)[1])[0])
        zipped_list= list(zip(x_keys,y_values,markers,list_of_filesL)) 
#         print(zipped_list)
    for x,y,z,fn in zipped_list:
        if xmax and xmin:
            plt.xlim(xmin,xmax)
            plt.plot(x,y,label=fn,marker=z,linewidth=linewidth, markersize=markersize)
        elif xmax and not xmin:
            plt.xlim(right=xmax)
            plt.plot(x,y,label=fn,marker=z,linewidth=linewidth, markersize=markersize)
        elif xmin and not xmax:
            print(len(EnergyL))
            plt.xlim(xmin,len(EnergyL))
            plt.plot(x,y,label=fn,marker=z,linewidth=linewidth, markersize=markersize)
        else:
            plt.plot(x,y,label=fn,marker=z,linewidth=linewidth, markersize=markersize)
        if add_legend:
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), shadow=True, ncol=3)            
        style.use('seaborn-talk')
        ax = plt.subplot(111)    
        box = ax.get_position()    
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.set_xlabel('Number of SCF iterations',fontsize=fontsize)
        ax.set_ylabel(f'$N_{"{virt}"}^\{switch.lower()}$',fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=labelsize)
        ax.tick_params(axis="y", labelsize=labelsize)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)        

    plt.savefig(os.path.join(directory,figname+'Nvirt_'+str(switch)+ '.'+ figformat),dpi=600,format=figformat,bbox_inches='tight')
    plt.close()    
########################
########################

###########################################################
# Plot FC of excited calculations log files and exp files #
###########################################################
def PlotExFC(directory, list_of_files):
#######################################    
    DEBUG     = 1
    file_logs = []
    file_TDL  = []
    file_ExpL = []
    test      = []
    trans00   = []
    alpha_val = args.alpha
    cmToeV    = float(0.000123984)
    cmTonm    = float(10000000.0)
    nmTocm    = float(10000000.0)
    nmToeV    = float(1239.84193)
    cm_unit   = "$cm^{-1}$"

    for filename in list_of_files:
        if ".log" in filename and "TD" not in filename:
            file_logs.append(filename)
        if "TD" in filename:
            file_TDL.append(filename)
        if "csv" in filename:
            file_ExpL.append(filename)           
    print(file_logs)
    print(file_TDL)
    print(file_ExpL)
#####################################################
#####################################################

    # Open the file for writing the Energy difference
    if noshift:
        pass
    else:
        openEngDifF = open(EngDifF, "w")

    # Set the experimental data as a refernce and plot it
    for file in file_ExpL:
#        print(file_expL)
        file = os.path.join(directory, file)
        coord = []
        count = len(open(file).readlines())
        if ".csv" in file:
            dff=pd.read_csv(file)
            dff.columns = ["x", "y"]
            dff = dff.sort_values("x",ascending=True)
            # Next we will find all the peaks in the spectrum
            peaks, _ = find_peaks(dff["y"], height=Peakheight)
            p = dff["x"]

            # Convert all the column types to numeric
            dff[["x", "y"]] = dff[["x", "y"]].apply(pd.to_numeric)

            # Find the energy of the first peak
            # In some cases, you may need the second or third peak
            # to align it with the second peak, change the 0 to 1
            MainEng = p.iloc[peaks[0]]
            # the Exp data are in nm so we need to convert to cm
            trans00.append(nmTocm / MainEng)

            # Normalize the data
            cols_to_norm = ["y"]
            dff[cols_to_norm] = min_max_scaler.fit_transform(dff[cols_to_norm])
            if DEBUG == 3:
                print('Normalized Data Frame of',
                        os.path.splitext(os.path.split(file)[1])[0])
                print(dff)
            if DEBUG == 3:
                print('Normalized and shifted Data Frame of',
                        os.path.splitext(os.path.split(file)[1])[0])
                print(dff)

            # Convert the energy to the requested unit
            if unit == "cm-1":
                dff["x"]= nmTocm / dff["x"]
                tranE = nmTocm / float(MainEng)
                xlim1 = tranE - 50
#                trans00.append(tranE)
            elif unit == "eV" or unit == "ev" :
                dff["x"]=  nmToeV / dff["x"]
                tranE = nmToeV / float(MainEng)
                xlim1 = tranE - 0.1
#                trans00.append(tranE)
            elif unit == "nm":
                xlim1 = float(MainEng) - 0.04
#                trans00.append(MainEng)
            else:
                print("wrong unit provided. Check the manual for accepted units")
                sys.exit(1)
#
            # Plot the data
            plt.plot(dff['x'],dff['y'],
                    label=os.path.splitext(os.path.split(file)[1])[0],alpha= alpha_val,linewidth=linewidth, markersize=markersize)
            plt.ylabel('Intensity', fontsize=12)
            if unit == "cm-1":
                plt.xlabel('E('+cm_unit+')', fontsize=12)
            else:
                plt.xlabel('E('+unit+')', fontsize=12)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=3)

    # Plot the TD file and set its energy as the reference
    for file in file_TDL:
        file = os.path.join(directory, file)
        coord = []
        count = len(open(file).readlines())
        with open(file, 'r') as f:
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

                # Form a data frame of the data
            df = pd.DataFrame(coord)
            if DEBUG == 5:
                print('Data Frame before adding headers of',
                    os.path.splitext(os.path.split(file)))
                print(df)

            # Add columns labels
#            print(df)
#            print(len(df.columns))
            if (len(df.columns)) == 3:
                df.columns = ['x', 'y-0k', 'y-300k']
            else:
                df.columns = ['x', 'y-0k']
            dff = df.drop([0])
            if DEBUG == 4:
                print('Data Frame of', os.path.splitext(
                    os.path.split(file)[1])[0])
                print(dff)

            # Convert all the column types to numeric
            if (len(df.columns)) == 3:
                dff[['x', 'y-0k', 'y-300k']] = dff[['x', 'y-0k', 'y-300k']].apply(pd.to_numeric)
            else:
                dff[['x', 'y-0k']] = dff[['x', 'y-0k']].apply(pd.to_numeric)

            # Normalize the data
            if (len(df.columns)) == 3:
                cols_to_norm = ['y-0k', 'y-300k']
            else:
                cols_to_norm = ['y-0k']
            dff[cols_to_norm] = min_max_scaler.fit_transform(dff[cols_to_norm])
            if DEBUG == 3:
                print('Normalized Data Frame of',
                        os.path.splitext(os.path.split(file)[1])[0])
                print(dff)
            # print(dff)
        # Next we will find the first peak in the spectrum
            if temp == '0K' :
                peaks, _ = find_peaks(dff['y-0k'], height=Peakheight)
            elif temp == '300K' :
                peaks, _ = find_peaks(dff['y-300k'], height=Peakheight)
            else:
                print("Currently supports values 0K and 300K")
                sys.exit(0)
            p = dff['x']
            # Append the energy of the first peak
            MainEng = p.iloc[peaks[0]]
            trans00.append(MainEng)

            if noshift:
                pass
            else:
            # Compute the energy difference with respect to the first file and
            # shift the x axis
                for i in range(len(trans00)):
                    difference = float(trans00[i])-float(trans00[0])
                dff.x -= difference
                if unit == "cm-1":

                    print(os.path.splitext(os.path.split(file)[1])[0], 'is shifted by', round(difference,2), unit)
                    print(os.path.splitext(os.path.split(file)[1])[0], 'is shifted by', round(difference,2), unit, file=openEngDifF)
                elif unit == "eV" or unit == "ev" :
                    difference_ev = difference * cmToeV
                    print(os.path.splitext(os.path.split(file)[1])[0], 'is shifted by', round(difference_ev,3), unit)
                    print(os.path.splitext(os.path.split(file)[1])[0], 'is shifted by', round(difference_ev,3), unit, file=openEngDifF)
                elif unit == "nm":
                    try:
                        difference_nm = cmTonm / difference
                    except ZeroDivisionError:
                        difference_nm = 0.0
                    print(os.path.splitext(os.path.split(file)[1])[0], 'is shifted by', round(difference_nm,3), unit)
                    print(os.path.splitext(os.path.split(file)[1])[0], 'is shifted by', round(difference_nm,3), unit, file=openEngDifF)
                else:
                    print("wrong unit provided. Check the manual for accepted units")
                    sys.exit(1)
#                print(os.path.splitext(os.path.split(file)[1])[0], 'is shifted by', round(difference,2), unit)
                if DEBUG == 3:
                    print('Normalized and shifted Data Frame of',
                            os.path.splitext(os.path.split(file)[1])[0])
                    print(dff)

            #Scale the data
            if (len(df.columns)) == 3:
                dff['y-0k'] *=  scale
                dff['y-300k'] *=  scale
            else:
                dff['y-0k']*=  scale

            # Convert to correct unit
            if unit == "cm-1" :
                pass
            elif unit == "eV" or unit == "ev" :
                dff['x'] = cmToeV * dff['x']
            elif unit == "nm" :
                dff['x'] = cmTonm / dff['x']
            else:
                print("wrong unit provided. Check the manual for accepted units")
                sys.exit(1)


        # Plot the data
        if temp == '0K' :
            plt.ylabel('Intensity', fontsize=12)
            if unit == "cm-1":
                plt.xlabel('E('+cm_unit+')', fontsize=12)
            else:
                plt.xlabel('E('+unit+')', fontsize=12)
            plt.plot(dff['x'],dff['y-0k'],
                    label=os.path.splitext(os.path.split(file)[1])[0],alpha= alpha_val,linewidth=linewidth, markersize=markersize)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=3)
        elif temp == '300K' :
            plt.plot(dff['x'],dff['y-300k'],
                    label=os.path.splitext(os.path.split(file)[1])[0], alpha= alpha_val,linewidth=linewidth, markersize=markersize)
#            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=3)
            plt.ylabel('Intensity', fontsize=12)
            if unit == "cm-1":
                plt.xlabel('E('+cm_unit+')', fontsize=12)
            else:
                plt.xlabel('E('+unit+')', fontsize=12)


#####################################################
#####################################################
    # Plot the rest of the log files
    for file in file_logs:
        file = os.path.join(directory, file)
        coord = []
        count = len(open(file).readlines())
        if "log" in file :
            with open(file, 'r') as f:
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

                # Form a data frame of the data
                df = pd.DataFrame(coord)
#                print(len(df.columns))
                if DEBUG == 5:
                    print('Data Frame before adding headers of',
                          os.path.splitext(os.path.split(file)))
                    print(df)

                # Add columns labels
                if (len(df.columns)) == 3:
                    df.columns = ['x', 'y-0k', 'y-300k']
                else:
                    df.columns = ['x', 'y-0k']
                dff = df.drop([0])
                if DEBUG == 4:
                    print('Data Frame of', os.path.splitext(
                        os.path.split(file)[1])[0])
                    print(dff)

                # Convert all the column types to numeric
                if (len(df.columns)) == 3:
                    dff[['x', 'y-0k', 'y-300k']] = dff[['x', 'y-0k', 'y-300k']].apply(pd.to_numeric)
                else:
                    dff[['x', 'y-0k']] = dff[['x', 'y-0k']].apply(pd.to_numeric)

                # Normalize the data
                if (len(df.columns)) == 3:
                    cols_to_norm = ['y-0k', 'y-300k']
                else:
                    cols_to_norm = ['y-0k']
                dff[cols_to_norm] = min_max_scaler.fit_transform(dff[cols_to_norm])
                if DEBUG == 3:
                    print('Normalized Data Frame of',
                          os.path.splitext(os.path.split(file)[1])[0])
                    print(dff)
                # Next we will find the first peak in the spectrum
                if temp == '0K' :
                    peaks, _ = find_peaks(dff['y-0k'], height=Peakheight)
                elif temp == '300K' :
                    peaks, _ = find_peaks(dff['y-300k'], height=Peakheight)
                else:
                    print("Currently supports values 0K and 300K")
                    sys.exit(0)
                p = dff['x']
                # Append the energy of the first peak
                MainEng = p.iloc[peaks[0]]
                #print(p.iloc[peaks[0]])
                trans00.append(MainEng)
#                print(trans00)

                if noshift:
                    pass
                else:
                # Compute the energy difference with respect to the first file and
                # shift the x axis
                    for i in range(len(trans00)):
                        difference = float(trans00[i])-float(trans00[0])
                    dff.x -= difference
                    if unit == "cm-1":
                        print(os.path.splitext(os.path.split(file)[1])[0], 'is shifted by', round(difference,2), unit)
                        print(os.path.splitext(os.path.split(file)[1])[0], 'is shifted by', round(difference,2), unit, file=openEngDifF)
                    elif unit == "eV" or unit == "ev" :
                        difference_ev = difference * cmToeV
                        print(os.path.splitext(os.path.split(file)[1])[0], 'is shifted by', round(difference_ev,3), unit)
                        print(os.path.splitext(os.path.split(file)[1])[0], 'is shifted by', round(difference_ev,3), unit, file=openEngDifF)
                    elif unit == "nm":
                        try:
                            difference_nm = cmTonm / difference
                        except ZeroDivisionError:
                            difference_nm = 0.0
                        print(os.path.splitext(os.path.split(file)[1])[0], 'is shifted by', round(difference_nm,2), unit)
                        print(os.path.splitext(os.path.split(file)[1])[0], 'is shifted by', round(difference_nm,2), unit, file=openEngDifF)
                    else:
                        print("wrong unit provided. Check the manual for accepted units")
                        sys.exit(1)

                    if DEBUG == 3:
                        print('Normalized and shifted Data Frame of',
                              os.path.splitext(os.path.split(file)[1])[0])
                        print(dff)
                #Scale the data
                if (len(df.columns)) == 3:
                    dff['y-0k'] *=  scale
                    dff['y-300k'] *=  scale
                else:
                    dff['y-0k'] *=  scale

                # Convert the energy to the requested unit
                if unit == "cm-1":
                    pass
                elif unit == "eV" or unit == "ev" :
                    dff['x']= cmToeV * dff['x']
                elif unit == "nm":
                    dff['x'] = cmTonm / dff['x']
                else:
                    print("wrong unit provided. Check the manual for accepted units")
                    sys.exit(1)

        # Plot the data
        if temp == '0K' :
            plt.ylabel('Intensity', fontsize=12)
            if unit == "cm-1":
                plt.xlabel('E('+cm_unit+')', fontsize=12)
            else:
                plt.xlabel('E('+unit+')', fontsize=12)
            plt.plot(dff['x'],dff['y-0k'],
                    label=os.path.splitext(os.path.split(file)[1])[0],alpha= alpha_val,linewidth=linewidth, markersize=markersize)
#            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11), shadow=True, ncol=3)
        elif temp == '300K' :
            plt.ylabel('Intensity', fontsize=12)
            if unit == "cm-1":
                plt.xlabel('E('+cm_unit+')', fontsize=12)
            else:
                plt.xlabel('E('+unit+')', fontsize=12)
            plt.plot(dff['x'],dff['y-300k'],
                    label=os.path.splitext(os.path.split(file)[1])[0], alpha= alpha_val,linewidth=linewidth, markersize=markersize)
#            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11), shadow=True, ncol=3)
#####################################################
#####################################################
    # Save the figure
    plt.savefig(figname+'.'+figformat, bbox_inches='tight', format=figformat, dpi=600)





###################################################
# This Function will plot the FC of Photodetachment
###################################################

def PlotSpecFC(directory, list_of_files):
#####################################    
    # Setting Initial parameters
    DEBUG        = 1
    xmax     = args.xmax
    filenameL    = []  
    test         = []
    x_keys       = []
    y_values     = []
    trans00      = []
    file_expL    = []
    cmToeV = float(1/8065.6)
    cmTonm = float(1/1240)
    cm_unit   = "$cm^{-1}$"  

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
            dff=pd.read_csv(file, sep='\t')
            dff.columns = ["x", "y"]
            dff = dff.sort_values("x",ascending=True)
            # Next we will find all the peaks in the spectrum
            peaks, _ = find_peaks(dff["y"], height=0.4)
            p = dff["x"]

            # Convert all the column types to numeric
            dff[["x", "y"]] = dff[["x", "y"]].apply(pd.to_numeric)

            # Find the energy of the first peak
            MainEng = p.iloc[peaks[0]]

            # Convert the energy to the requested unit
            if unit == "cm-1":
                xlim1 = float(MainEng) - 50
                trans00.append(MainEng)
            elif unit == "eV" or unit == "ev" :
                dff["x"]= dff["x"] * cmToeV
                tranE = float(MainEng) * cmToeV
                xlim1 = tranE - 0.007
                trans00.append(tranE)
            elif unit == "nm":
                dff["x"]= dff["x"] * cmTonm
                tranE = float(MainEng) * cmTonm
                xlim1 = tranE - 0.04
                trans00.append(tranE)
            else:
                print("wrong unit provided. Check the manual for accepted units")
                sys.exit(1)

            # Normalize the data
            cols_to_norm = ["y"]
            dff[cols_to_norm] = min_max_scaler.fit_transform(dff[cols_to_norm])
            if DEBUG == 3:
                print('Normalized Data Frame of',
                        os.path.splitext(os.path.split(file)[1])[0])
                print(dff)

            # Compute the energy difference with respect to the first file and
            # shift the x axis
            for i in range(len(trans00)):
                difference = float(trans00[i])-float(trans00[0])
            dff.x -= difference
            print(os.path.splitext(os.path.split(file)[1])[0], 'is shifted by', round(difference,3), unit)

            if DEBUG == 3:
                print('Normalized and shifted Data Frame of',
                        os.path.splitext(os.path.split(file)[1])[0])
                print(dff)
#
            # Plot the data
            if args.xmax  == 10000:
                if unit == "eV" or unit == "ev":
                    xmax=xlim1 + 1
                elif unit == "nm":
                    xmax=xlim1 + 2
                plt.xlim(xlim1,int(float(xmax)))
                plt.plot(dff["x"],dff["y"],
                        label=os.path.splitext(os.path.split(file)[1])[0],alpha=alpha_val,linewidth=linewidth, markersize=markersize)
                if add_legend:
                    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), shadow=True, ncol=3) 
                plt.ylabel('Intensity', fontsize=12)
                if unit == "cm-1":
                    plt.xlabel('E('+cm_unit+')', fontsize=12)
                else:
                    plt.xlabel('E('+unit+')', fontsize=12)                    
            else:
#                print(dff["x"])
                plt.xlim(xlim1,xmax)
                plt.plot(dff["x"],dff["y"],
                    label=os.path.splitext(os.path.split(file)[1])[0],alpha=alpha_val,linewidth=linewidth, markersize=markersize)
                if add_legend:
                    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), shadow=True, ncol=3) 
                plt.ylabel('Intensity', fontsize=12)
                if unit == "cm-1":
                    plt.xlabel('E('+cm_unit+')', fontsize=12)
                else:
                    plt.xlabel('E('+unit+')', fontsize=12)                    
                

###########################################
###########################################
# Plot the log files
    for file in filenameL:
##        print(file)
        file = os.path.join(directory, file)
        coord = []
        count = len(open(file).readlines())

        if ".log" in file:
            with open(file, 'r') as f:
                for line in f:
                    if "Energy of the 0-0 transition" in line:
                        tranE = line[32:43]
                        if unit == "cm-1":
#                            xlim1 = int(float(trans00[0])) - 50
                            trans00.append(float(tranE))
                        elif unit == "eV" or unit == "ev" :
                            tranE = float(tranE) * cmToeV
#                            xlim1 = int(float(trans00[0])) - 0.007
                            trans00.append(tranE)
                        elif unit == "nm":
                            tranE = float(tranE) * cmTonm
#                            xlim1 = int(float(trans00[0])) - 0.04
                            trans00.append(tranE)
                        else:
                            print("wrong unit provided. Check the manual for accepted units")
                            sys.exit(1)

                        #append the first energy
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
                if DEBUG == 5:
                    print('Data Frame before adding headers of',
                            os.path.splitext(os.path.split(file)))
                    print(df)

                # Add columns labels
                df.columns = ["x", "y"]
                dff = df.drop([0])
                if DEBUG == 4:
                    print('Data Frame of', os.path.splitext(
                        os.path.split(file)[1])[0])
                    print(dff)

                # Convert all the column types to numeric
                dff[["x", "y"]] = dff[["x", "y"]].apply(pd.to_numeric)
                # Normalize the data
                cols_to_norm = ["y"]
                dff[cols_to_norm] = min_max_scaler.fit_transform(dff[cols_to_norm])
                if DEBUG == 3:
                    print('Normalized Data Frame of',
                            os.path.splitext(os.path.split(file)[1])[0])
                    print(dff)

                #Scale the data
                dff["y"] *=  scale
                # Convert the energy to the requested unit
                if unit == "cm-1":
                    pass
                elif unit == "eV" or unit == "ev" :
                    dff["x"]= dff["x"] * cmToeV
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
                    elif unit == "eV" or unit == "ev" :
                        xlim1 = min(trans00) - 0.007
                    elif unit == "nm":
                        xlim1 = min(trans00) - 0.04

#                    xlim1= min(trans00)
#                    print("This is xlim1 after changing",xlim1)
                else:
                # Compute the energy difference with respect to the first file and
                # shift the x axis
                    for i in range(len(trans00)):
                        difference = float(trans00[i])-float(trans00[0])
                    dff.x -= difference
                    print(os.path.splitext(os.path.split(file)[1])[0], 'is shifted by', round(difference,3), unit)
                    print(os.path.splitext(os.path.split(file)[1])[0], 'is shifted by', round(difference,3), unit, file=openEngDifF)
                if DEBUG == 3:
                    print('Normalized and shifted Data Frame of',
                            os.path.splitext(os.path.split(file)[1])[0])
                    print(dff)
#            print(trans00)

            # Plot the data
            if args.xmax   == 10000:
                plt.xlim(xlim1,xmax)
                plt.plot(dff["x"],dff["y"],
                        label=os.path.splitext(os.path.split(file)[1])[0],alpha=alpha_val,linewidth=linewidth, markersize=markersize)
                if add_legend:
                    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), shadow=True, ncol=3)
                plt.ylabel('Intensity', fontsize=12)
                if unit == "cm-1":
                    plt.xlabel('E('+cm_unit+')', fontsize=12)
                else:
                    plt.xlabel('E('+unit+')', fontsize=12)                    

            else:
                plt.xlim((xlim1,xmax))
                plt.plot(dff["x"],dff["y"],
                    label=os.path.splitext(os.path.split(file)[1])[0],alpha=alpha_val,linewidth=linewidth, markersize=markersize)
                if add_legend:
                    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), shadow=True, ncol=3)
                plt.ylabel('Intensity', fontsize=12)
                if unit == "cm-1":
                    plt.xlabel('E('+cm_unit+')', fontsize=12)
                else:
                    plt.xlabel('E('+unit+')', fontsize=12)                    



    plt.savefig(figname+'.'+figformat, bbox_inches='tight', format=figformat, dpi=600)


