#!/usr/bin/env python
#
#  Calculates qT1 map using VFA method
#  Assumes 3 AxSPGR images,  FA = 2, 9 , 19
#  Optional B1 field map correction (see b1_norm_calc.py)
#  Optional mean squared error (mse) output
#     Error calculated as follows...
#        err = Si / sin (alpha_i) - exp (-TR/T1) * Si / tan(alpha_i)
#        mse = sum( err^2)/3 for all i (i = FA index)
#    max_mse = 1000
#    max_t1 = 10000 (10s)
#
#  Program autofills voxels with 
#    A) -'ve slope
#    B) T1 < 0 or T1 > max_t1
#  with local inplane average (only using 'good' voxels)
#  MSE for these voxels will be set to max_mse
#
#    AUTHOR - Wayne Lee
#   Created - Sept 9, 2009
#   REVISIONS 
#		Oct 13, 2009 - WL
#			Added autofill for 'error' voxels


from optparse import OptionParser, Option, OptionValueError
import os
from math import *
from numpy import *
from numpy.linalg import inv
import py_minc

""" Calculate qT1 map (based on  Cheng and Wright MRM 55, 566-574, (2006))  """

program_name = 'qT1_vfa.py'

#----------------------------------------------------------------------------  
# SPGR Steady State signal amplitude function 

def t1_SS_fn(Mo, TR, T1, alpha):
#  signal equation for SPGR Steady state signal amplitude
#
#  See: MRM 55, 566-574, (2006)  Cheng and Wright.
#
#  T1       :  T1 recovery time (ms)
#  Mo       :  equilibrium magnetization
#  TR       :  repetition period (ms)
#  alpha    :  excitation angle in degrees

    alpha_rad = alpha / 180.0 * pi
    E = exp( -TR/T1 )
    Si = Mo * sin(alpha_rad) * (1 - E) / (1 - E * cos(alpha_rad) )

    return Si

#----------------------------------------------------------------------------  
# Generate signal / sin(alpha)
def gen_sig_sin(data, alpha_rad):
    sig_sin = data/sin(alpha_rad)
    return sig_sin
    
#----------------------------------------------------------------------------  
# Generate signal / tan(alpha)
def gen_sig_tan(data, alpha_rad):
    sig_tan = data / tan(alpha_rad)
    return sig_tan

#----------------------------------------------------------------------------  
# Calculate T1
#    Slope of Sig_sin vs. Sig_tan gives T1
#    qT1 = -TR / log ( inv(A'A)A'B)
def qT1_calc (TR, A, B):
    return -TR / log( dot( A.transpose(),B) / dot(A.transpose(), A)  )
    
#----------------------------------------------------------------------------  
if __name__ == '__main__':

    usage = "Usage: "+program_name+" filename1.mnc filename2.mnc filename3.mnc output.mnc\n"+\
            "   or  "+program_name+" --help";
    parser = OptionParser(usage)
    parser.add_option("--clobber", action="store_true", dest="clobber",
                        default=0, help="overwrite output file")
    parser.add_option("--FAs", type="float", nargs = 3, dest="FA_deg",
                        default = [2.0, 9.0, 19.0],help="Flip angles [deg] (Default = 2, 9, 19)")
    parser.add_option("--b1_norm", type="string", dest="b1_norm",
                        help="B1_norm map for flip angle error correction")
    parser.add_option("--TR", type="float", dest="TR",
                        default = 3.9,help="TR of Ax SPGR images [ms] (Default = 3.9)")
    parser.add_option("--thresh", type="float", dest="thresh",
                        default = 300.0,help="Simple masking threshold (Default = 300)")
    parser.add_option("--mask", type="string", dest="mask",
                        help="Supplied mask, overrides threshold option")
    parser.add_option("--mse", type="string", dest="mse",
                        help="Optional mean squared error output")
    parser.add_option("--max_T1", type="int", dest="max_T1",
                        default = 10000.0, help="Maximum T1 value [ms] (Default = 10000)")
    parser.add_option("--fill_off", action="store_true", dest="fill_off",
                        default=0, help="Turn off autofill which replaces -'ve slope or -T1 values with inplane local average")
    
    options, args = parser.parse_args()
    if len(args) != 4:
        parser.error("incorrect number of arguments")
        
    FA_file_1, FA_file_2, FA_file_3, output = args
    
    if not options.clobber and os.path.exists(output):
        raise SystemExit, \
        "The --clobber option is needed to overwrite an existing file."

# FAs into an array, converted into radians
    FAs = array(options.FA_deg)/180.0*pi
    
    TR = options.TR
    max_t1 = options.max_T1 # maximum output T1 = 10s
    
# if masking then set threshold to 0 to ignore threshold option
    if not options.mask:
        thresh = options.thresh
    else:
        thresh = 0
        mask_minc = py_minc.ArrayVolume(options.mask)

# loading minc data
    FA_1_minc = py_minc.ArrayVolume(FA_file_1)
    FA_2_minc = py_minc.ArrayVolume(FA_file_2)
    FA_3_minc = py_minc.ArrayVolume(FA_file_3)
# Initializing Output Map, setting output range 0 -> 10,000 ms
    T1_map = py_minc.ArrayVolume(FA_1_minc,copy='definition')

# If error map output then create file / memory for error map, same size as T1_map
    if options.mse:
        mse_map = py_minc.ArrayVolume(FA_1_minc,copy='definition')
        max_mse = 1000
    
# Minc data into numpy array 
    FA_data = array([ FA_1_minc.get_all_values(),
              FA_2_minc.get_all_values(),
              FA_3_minc.get_all_values() ])
    [num_fa, z_res, y_res, x_res ] = FA_data.shape

# Load B1_norm map if option enabled
    if options.b1_norm:
        B1_norm = py_minc.ArrayVolume(options.b1_norm)    
        B1_norm_values = B1_norm.get_all_values()
    
# Calculating computation matrices
    sig_sin_data = zeros(FA_data.shape)
    sig_tan_data = zeros(FA_data.shape)
    for count_fa in arange(3):
        sig_sin_data[count_fa,:,:,:] = gen_sig_sin(FA_data[count_fa,:,:,:], FAs[count_fa])
        sig_tan_data[count_fa,:,:,:] = gen_sig_tan(FA_data[count_fa,:,:,:], FAs[count_fa])
    sig_sin_mean = sig_sin_data.mean(0)
    sig_tan_mean = sig_tan_data.mean(0)
    sig_sin_nomean = sig_sin_data - sig_sin_mean
    sig_tan_nomean = sig_tan_data - sig_tan_mean

# Calculating qT1 map (Initial)
    for count_z in arange(z_res):
        for count_y in arange(y_res):
            for count_x in arange(x_res):
# if an external mask is supplied can adjust threshhold to achieve masking
                if (options.mask and mask_minc.array[count_z, count_y, count_x] == 1) or ( (not options.mask) and (FA_data[0, count_z, count_y, count_x]) > thresh ):
                    A = sig_tan_nomean[:,count_z,count_y,count_x]
                    B = sig_sin_nomean[:,count_z,count_y,count_x]
                    if ((dot( A.transpose(),B) / dot(A.transpose(), A) ) > 0 ):
                        qT1 = qT1_calc(TR, A, B)
# if b1_norm is supplied then calculate adjustment factor
                        if options.b1_norm:
                            X = sig_tan_data[:,count_z, count_y, count_x]
                            Y = sig_sin_data[:,count_z, count_y, count_x]
                            delta_T1_A = -(qT1**2) * exp(TR/qT1) / (TR * 3 * (mean(X**2) - mean(X)**2))
                            delta_T1_sum = 0
                            for count_fa in arange(3):
                                delta_fa = (B1_norm_values[count_z, count_y, count_x] - 1) * FAs[count_fa]
                                delta_T1_sum = delta_T1_sum + \
                                    delta_fa / tan( FAs[count_fa]) * ( Y[count_fa] * ( X[count_fa] - mean(X)) + \
                                    X[count_fa]*(1+ (tan( FAs[count_fa])**2)) * \
                                    (Y[count_fa] - mean(Y) - 2 * exp(-TR/qT1) * ( X[count_fa] - mean(X)) ) )
                            delta_T1 = delta_T1_A * delta_T1_sum
                            qT1 = qT1 + delta_T1
                        if options.mse:
                            err = sig_sin_nomean[:,count_z,count_y,count_x] - \
                                sig_tan_nomean[:,count_z,count_y,count_x] * exp(-TR/qT1)
                            mse_map.array[count_z, count_y, count_x] = sum(err ** 2)/3
                        if ( (qT1 > max_t1) or (qT1 < 0)) :   # T1 value bad, trigger autofill
                            qT1 = -1
                            if options.mse:
                                mse_map.array[count_z, count_y, count_x] = max_mse
                        T1_map.array[count_z, count_y, count_x] = qT1
                    else: # if slope is negative replace values with -1 (triggers autofill)
                        if options.mse:
                            mse_map.array[count_z, count_y, count_x] = max_mse
                        T1_map.array[count_z, count_y, count_x] = -1
                else: # set to 0 if masked out
                    T1_map.array[count_z, count_y, count_x] = 0
                    if options.mse:
                        mse_map.array[count_z, count_y, count_x] = 0

    # if filling gaps (ie replacing -1 T1 values with local average
    num_error = 0 # number of voxels that are 'fixed'
    if not options.fill_off:
       T1_map_orig = py_minc.ArrayVolume(T1_map,copy='all') # create copy of original so as not to double interpret
       for count_z in arange(z_res):
            for count_y in arange(y_res):
                for count_x in arange(x_res):
                    if (T1_map_orig.array[count_z, count_y, count_x] == -1):
                        num_error = num_error + 1
                        average_count = 0
                        qT1_sum = 0
                        for y_range in [-1, 0, 1]:
                            for x_range in [-1, 0, 1]:
                                if ( not ((y_range ==0) and (x_range==0)) ): # ignore center voxel
                                    if (( count_y + y_range > -1) and ( count_y + y_range < y_res) ): # ignore edge voxels
                                        if (( count_x + x_range > -1) and ( count_x + x_range < x_res) ): # ignore edge voxels
                                            if (T1_map_orig.array[count_z, count_y + y_range, count_x + x_range] > 0):
                                                qT1_sum = qT1_sum + T1_map_orig.array[count_z, count_y + y_range, count_x + x_range]
                                                average_count = average_count + 1
                        if (average_count > 0):
                            T1_map.array[count_z, count_y, count_x] = qT1_sum / average_count
                        else:
                            T1_map.array[count_z, count_y, count_x] = 0
                    else:
                        T1_map.array[count_z, count_y, count_x] = T1_map_orig.array[count_z, count_y, count_x]

    print "Number of voxels autofilled = ", num_error
    print max_t1
    T1_map.set_range(0, max_t1)
    T1_map.output(output)
    if options.mse:
        mse_map.set_range(0, max_mse)
        mse_map.output(options.mse)