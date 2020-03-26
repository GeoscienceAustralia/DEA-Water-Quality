#! /usr/bin/env python3
######################################################################
## BOM & CSIRO
##
## DATE:         Feb 2015
## LANGUAGE:     python 2.7.x
##
## PURPOSE:      Test the Above to Below Water converter and the Below to Above water convert
##
## TODO:		 Next step is to wrap the duplicated test code up in a function...
##
##
##
######################################################################
import numpy
import RrsAboveToBelow


# 1) test with a normal set of numbers:

#if our input is 0 0.2 0.4 0.5 0.6 and our coefficients are 0.52 and 1.7 the the output array should be  0, 0.23255814,  0.33333333,  0.3649635 ,  0.38961039
print("Testing with normalish numbers...")
test_below_refl = numpy.array([ 0.        ,  0.23255814,  0.33333333,  0.3649635 ,  0.38961039], dtype=float)
test_above_refl = numpy.array([0., .2, .4, .5, .6], dtype=float)

test_output = RrsAboveToBelow.RrsAboveToBelow(test_above_refl)

if numpy.allclose(test_output,test_below_refl):
    print("+++ Test passed +++")
else:
    print("--- Test failed ---")

print("Input: " + str(test_above_refl))
print("Expected output: " + str(test_below_refl))
print("Actual output: " + str(test_output))

del test_below_refl, test_above_refl, test_output

# 2) test with zeros:
print("Testing with zeros...")
test_below_refl = numpy.array([0., 0., 0., 0., 0.], dtype=float)
test_above_refl = numpy.array([0., 0., 0., 0., 0.], dtype=float)

test_output = RrsAboveToBelow.RrsAboveToBelow(test_above_refl)

if numpy.allclose(test_output,test_below_refl):
    print("+++ Test passed +++")
else:
    print("--- Test failed ---")

print("Input: " + str(test_above_refl))
print("Expected output: " + str(test_below_refl))
print("Actual output: " + str(test_output))


# 3) test with huge numbers:
print("Testing with big numbers...")
test_below_refl = numpy.array([0.587337014, 0.586441473, 0.587636136, 0.587875653, 0.587785811], dtype=float)
test_above_refl = numpy.array([200., 100., 300., 500., 400.], dtype=float)

test_output = RrsAboveToBelow.RrsAboveToBelow(test_above_refl)

if numpy.allclose(test_output,test_below_refl):
    print("+++ Test passed +++")
else:
    print("--- Test failed ---")

print("Input: " + str(test_above_refl))
print("Expected output: " + str(test_below_refl))
print("Actual output: " + str(test_output))




