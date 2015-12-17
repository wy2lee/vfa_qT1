#!/usr/bin/env python
#
#  Calculates B1_norm map from 2 SE images
#  Output is normalized such that 1.0 = no B1 variation
#
#    AUTHOR - Wayne Lee
#   Created - Sept 9, 2009
#   REVISIONS 
#        Oct 13, 2009 - WL
#            Changed to single voxel calculation
#           Voxels with erroneous B1 values are set to 1.0

from optparse import OptionParser, Option, OptionValueError
import os
from math import *
from numpy import *
from numpy.linalg import inv
import py_minc

""" Wayne playing with Python to learn how it works """

program_name = 'b1_norm_calc.py'


#----------------------------------------------------------------------------  
if __name__ == '__main__':

    usage = "Usage: "+program_name+" filename1.mnc filename2.mnc nominal_FA output\n"+\
            "   or  "+program_name+" --help";
    parser = OptionParser(usage)
    parser.add_option("--clobber", action="store_true", dest="clobber",
                        default=0, help="overwrite output file")

    options, args = parser.parse_args()
    if len(args) != 4:
        parser.error("incorrect number of arguments")
        
    SE_file_1, SE_file_2, FA_nom, output = args
    
    if not options.clobber and os.path.exists(output):
        raise SystemExit, \
        "The --clobber option is needed to overwrite an existing file."

# loading minc data
    SE_1_minc = py_minc.ArrayVolume(SE_file_1)
    SE_2_minc = py_minc.ArrayVolume(SE_file_2)
# Initializing Output Map, setting output range -10 -> 10
    B1_norm = py_minc.ArrayVolume(SE_1_minc,copy='definition')
    B1_norm.array = zeros(B1_norm.get_sizes())
    alpha_1_values = zeros(B1_norm.get_sizes())    

    [z_res, y_res, x_res ] = B1_norm.get_sizes()
    
# Minc data into numpy array 
    for count_z in arange(z_res):
        for count_y in arange(y_res):
            for count_x in arange(x_res):
                if (SE_1_minc.array[count_z,count_y,count_x] == 0):   # Check for divide by 0
                    alpha_1_values[count_z,count_y,count_x] = (float(FA_nom)/180.0*pi)
                    # if not divide by 0, then check that arccos will have a valid value
                elif ( (abs((SE_2_minc.array[count_z,count_y,count_x] / SE_1_minc.array[count_z,count_y,count_x])**(1.0/3.0) / 2.0)) > 1 ) :
                    alpha_1_values[count_z,count_y,count_x] = (float(FA_nom)/180.0*pi)
                else:   # Passes error check, solve as normal
                    alpha_1_values[count_z,count_y,count_x] = arccos( (SE_2_minc.array[count_z,count_y,count_x] / SE_1_minc.array[count_z,count_y,count_x])**(1.0/3.0) / 2.0)
                    
    B1_norm.array = abs(alpha_1_values) / (float(FA_nom)/180.0*pi)
    B1_norm.set_range(0.5, 1.5)
    
    B1_norm.output(output)