# Package with different functions to apply to Gaussian output files
## CREATED BY ALI ABOU TAKA
### Date Modified: Feb 15 2022

                                  READ BELOW
                                    ----------
                These are the different fucntions in this package.
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
