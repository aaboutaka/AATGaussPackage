# CREATED BY ALI ABOU TAKA
Package with different function to apply to Gaussian output files

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

