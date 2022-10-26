#!/usr/bin/env python
#
# file: /home/tup23774/work/spec_analyzer/nedc_spectrum_analyzer.py
#
#------------------------------------------------------------------------------

# import system modules
#
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# import nedc_modules
#
import nedc_cmdl_parser as ncp
import nedc_debug_tools as ndt
import nedc_edf_tools as net
import nedc_file_tools as nft

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# define default argument values
#
ARG_CHAN = "--channel_key"
ARG_ABRV_CHAN = "-c"

ARG_CTIME = "--center_time"
ARG_ABRV_CTIME = "-t"

ARG_WDUR = "--window_duration"
ARG_ABRV_WDUR = "-w"

ARG_MAXF = "--max_freq"
ARG_ABRV_MAXF = "-m"

ARG_MINF = "--min_freq"

# define default argument values
#
DEF_CHANNEL = nft.STRING_EMPTY
DEF_CTIME = float(10.0)
DEF_WDUR = float(2.0)
DEF_MINF = float(0.0)
DEF_MAXF = float(50.0)

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

# declare a global debug object so we can use it in functions
#
dbgl = ndt.Dbgl()

# function: nedc_process_file
#
# arguments:
#  fname: filename to be processed
#  channel_key: channel key
#  center_time: the center time of the samples 
#  window_dur: the duration of the samples to be used to compute the spectrum
# min_freq: the minimum frequency
# max_freq: the maximum frequency
#  fp: file pointer (normally stdout)
#
# return: the signal and the spectrum
#
# This method processes the files given as command line arguments.
#
def nedc_process_file(fname, channel_key, center_time, window_dur,
                      min_freq, max_freq, fp = sys.stdout):

    # declare local variables
    #
    edf = net.Edf()
    

    # read the scaled signal
    #
    if dbgl > ndt.BRIEF:
        fp.write("%s (line: %s) %s: opening (%s)\n" %
                 (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))

    (h, sig) = edf.read_edf(fname, True, True)
    if sig == None:
        fp.write("Error: %s (line: %s) %s: reading (%s) - scaled\n" %
	         (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
        return False

    if dbgl > ndt.NONE:
        fp.write("%s (line: %s) %s: num_channels_total = %ld\n" %
                 (__FILE__, ndt.__LINE__, ndt.__NAME__, len(sig)))

    # assign channel_key to index in dictionary
    #
    channels = {}
    for chan, key in enumerate(sig):
        channels[key] = chan
        
    # get the sample frequency of the channel
    #
    sample_freq = edf.get_sample_frequency(channels[channel_key])

    # get the signal
    #
    sigw = nedc_get_signal(sig, channel_key, center_time,
                           window_dur, sample_freq)
    # compute the spectrum
    #
    spect = nedc_compute_spectrum(min_freq, max_freq, sample_freq, sigw)

    # separate files by a blank line
    #
    fp.write(nft.DELIM_NEWLINE)
    
    # clean up an Edf object
    #
    edf.cleanup();

    # display debugging information
    #
    if dbgl > ndt.BRIEF:
        fp.write("%s (line: %s) %s: done in nedc_spectrum_analyzer\n" %
	         (__FILE__, ndt.__LINE__, ndt.__NAME__))

    # exit gracefully
    #
    return True

# function: nedc_get_signal
#
# arguments:
#   sig: the scaled signal of the sample
#   channel_key: the channel of the samples that will be extracted
#   center_time: the center time of the samples
#   window_dur: the duration of the samples
#   sample_freq: the frequency of the samples
#
# return: a single channel of a signal as a vector
#
def nedc_get_signal(sig, channel_key, center_time, window_dur, sample_freq):
    
    
    # compute the start and end samples and print them
    #
    start_sample = round((center_time - window_dur/2) * sample_freq)
    end_sample = round((center_time + window_dur/2) * sample_freq)
    print("     start sample:       %0.1d" % (start_sample))
    print("     end sample:         %0.1d" % (end_sample))

    if dbgl > ndt.BRIEF:
        print("%s (line: %s) %s: frame range = [%31d, %31d]\n" %
              (__FILE__, ndt.__LINE__, ndt.__NAME__,
               start_sample, end_sample))
        print("\t\t\t\t[%s %31d, %s %31d, %s %31d]\n" %
              ("start sample:", start_sample,
               "end_sample:", end_sample,
                "total:", len(sig[channel_key])))
    
    # loop over all samples that correspond to the center_time and window_dur
    # and print out the corresponding signal
    #
    for i in range(start_sample, end_sample):
        sigw = np.array(sig[channel_key][start_sample:end_sample])
        
        
    # clean up an Edf object
    #
    edf.cleanup();

    # display debugging information
    #
    if dbgl > ndt.BRIEF:
        print("%s (line: %s) %s: done in nedc_get_signal\n" %
              (__FILE__, ndt.__line__, ndt.__NAME__))

    # exit gracefully
    #
    return sigw
    
# function: nedc_compute_spectrum
#
# arguments:
#  sig: the scaled signal of the channel
#  sample_freq = the frequency of the samples in the channel (Hz) 
# return: a graph of the spectrum
#
def nedc_compute_spectrum(min_freq, max_freq, sample_freq, sig):

    # compute the Fourier Transform of the signal and get the magnitude
    #
    N = len(sig)
    N_2 = int(N/2)
    freq = np.zeros(N)
    for i in range(0, N):
        freq[i] = float(i)/float(N) * sample_freq
    i_fmin = int(round(float(N) * min_freq/sample_freq))
    i_fmax = int(round(float(N) * max_freq/sample_freq))
    i_fmin = np.clip(i_fmin, 0, N-1)
    i_fmax = np.clip(i_fmax, 0, N-1)

    fft = np.fft.fft(sig)
    afft = np.array(np.abs(fft))
    dbafft = 20 * np.log10(afft)

    # graph the magnitude of the Fourier Transform
    #
    fig, ax = plt.subplots()
    ax.set(xlabel = "Frequency (Hz)", ylabel = "Amplitude (dB)",
           title = "EEG Signal Spectrum")
    ax.plot(freq[i_fmin:i_fmax], dbafft[i_fmin:i_fmax])
    ax.grid()
    plt.show()

    # exit gracefully
    #
    return True

# function: main
#
def main(argv):

    # declare local variables
    #
    dbgl = ndt.Dbgl()
    edf = net.Edf()

    # create a command line parser
    #
    cmdl = ncp.Cmdl(USAGE_FILE, HELP_FILE)
    cmdl.add_argument("files", type = str, nargs = '*')
    cmdl.add_argument(ARG_ABRV_CHAN, ARG_CHAN, type = str)
    cmdl.add_argument(ARG_ABRV_CTIME, ARG_CTIME, type = float)
    cmdl.add_argument(ARG_ABRV_WDUR, ARG_WDUR, type = float)
    cmdl.add_argument(ARG_ABRV_MAXF, ARG_MAXF, type = float)
    cmdl.add_argument(ARG_MINF, type = float)
    
    # parse the command line
    #
    args = cmdl.parse_args()

    # check the number of arguments
    #
    if len(args.files) == int(0):
        cmdl.print_usage('stdout')

    # get the parameter values
    #
    if args.channel_key is None:
        cmdl.print_usage('stdout')
     
    if args.center_time is None:
        args.center_time = DEF_CTIME
    
    if args.window_duration is None:
        args.window_duration = DEF_WDUR

    if args.min_freq is None:
        args.min_freq = DEF_MINF
        
    if args.max_freq is None:
        args.max_freq = DEF_MAXF

    if dbgl > ndt.NONE:
        print("command line arguments:")
        print(" channel key = %s" % (args.channel_key))
        print(" center_time = %f secs" % (args.center_time))
        print(" window_duration = %f secs" % (args.window_duration))
        print(" min_freq = %f secs" % (args.min_freq))
        print(" max_freq = %f secs" % (args.max_freq))
        print(nft.STRING_EMPTY)

    # display information
    #
    print("     channel:            %s" % (args.channel_key))
    print("     center time:        %0.1f seconds" % (args.center_time))
    print("     window duration:    %0.1f seconds" % (args.window_duration))
    # print("     sample frequency:   %0.1f Hz" % (sample_freq))
    print("     min frequency:      %0.1f Hz" % (args.min_freq))
    print("     max frequency:      %0.1f Hz" % (args.max_freq))

    # display an informational message
    #
    print("beginning argument processing...")
     
    # main processing loop: loop over all input filenames
    #
    num_files_att = int(0)
    num_files_proc = int(0)

    for fname in args.files:
     
        # expand the filename (checking for environment variables)
        #
        ffile = nft.get_fullpath(fname)

        # check if the file exists
        #
        if os.path.exists(ffile) is False:
            print("Error: %s (line: %s) %s: file does not exist (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, ifile))
            sys.exit(os.EX_SOFTWARE)

        # case (1): an edf file
        #
        if (edf.is_edf(fname)):

            # display informational message
            #
            num_files_att += int(1)
            print("%3ld: %s" % (num_files_att, fname))

            if nedc_process_file(fname, args.channel_key,
                                 args.center_time, args.window_duration,
                                 args.min_freq, args.max_freq,
			         sys.stdout) == True:
      	        num_files_proc += int(1)


        # case (2): a list
        #
        else:

            # display debug information
            #
            if dbgl > ndt.NONE:
                print("%s (line: %s) %s: opening list (%s)" %
                      (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))

            # fetch the list
            #
            files = nft.get_flist(ffile)
            if files is None:
                print("Error: %s (line: %s) %s: error opening (%s)" %
                      (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))
                sys.exit(os.EX_SOFTWARE)

            else:

                # loop over all files in the list
                #
                for edf_fname in files:

                    # expand the filename (checking for environment variables)
                    #
                    ffile = nft.get_fullpath(edf_fname)

                    # check if the file exists
                    #
                    if os.path.exists(ffile) is False:
                        print("Error: %s (line: %s) %s: %s (%s)" %
                              (__FILE__, ndt.__LINE__, ndt.__NAME__,
                               "file does not exist", edf_fname))
                        sys.exit(os.EX_SOFTWARE)

                    # display information
                    #
                    num_files_att += int(1)
                    print("%3ld: %s" % (num_files_att, edf_fname))

                    if nedc_process_file(fname, args.channel_key,
                                         args.center_time,
                                         args.window_duration,
                                         args.min_freq, args.max_freq,
                                         sys.stdout) == True:
      	                num_files_proc += int(1);
                    else:
                        print("Error: %s (line: %s) %s: %s (%s)" %
                              (__FILE__, ndt.__LINE__, ndt.__NAME__,
                               "error opening file", edf_fname))
                        sys.exit(os.EX_SOFTWARE)

    # display the results
    #
    print("processed %ld out of %ld files successfully" %
	  (num_files_proc, num_files_att))

    # exit gracefully
    #
    return True

# begin gracefully
#
if __name__ == '__main__':
    main(sys.argv[0:])

#
# end of file
