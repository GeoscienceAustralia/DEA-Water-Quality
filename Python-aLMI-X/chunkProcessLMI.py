#! /usr/bin/env python3

from calc_functions import calcCost, calcReflForward, calcUBackward, calcUForward, calcY
from svd_LMI import svd_LMI
import numpy as np


def chunkProcessLMI(siopSets, input_WL_PIX, g0, g1, components, config, inputIsU=False, outputIsU=False, verbose=False):
    """
    Run the inversion algorithm on a chunk (an array of reflectances:  reflBelowSurf[WL,PIX], or an array of uIOPRatio[WL,PIX]).

    siopSets:  a dict of SIOP sets; keys are the SIOP set names
        each SIOP set is a dict; keys are 'wavelength', 'component', 'a_star', 'bb_star'; values are NumPy arrays
    input_WL_PIX is a NumPy array of input values (reflBelowSurf or uIOPRatio) (dims:  wavelengths, pixels).
        Note that we need to align the input WLs with the SIOP set WLs!
    g0, g1:  parameters for absorption/backscatter model
    components:  list of components to calculate (e.g., ['CHL', 'CDOM', 'NAP'])
    config:  a dict of configuration parameters
    calcU:  if true, then calculate uIOPRatio (the input array is reflBelowSurf); else input array is already uIOPRatio.
        (would it be better to specify the input type, and then do whatever calculations are appropriate?)
    inputIsU:  true if the input is u; false otherwise (input is reflBelowSurf)
    outputIsU:  true if the output is u; false otherwise (output is reflBelowSurf)

    Returns:  a dict of results
    """
    
    ME = 'chunkProcessLMI'
    costType = config['costType']
    minValidConc = config['minValidConc']
    fv = config['outputFillValue']		# fill value of most output arrays
    costThreshhold = config['costThreshhold']		# IDL uses D_R_THRESHOLD (= 100 in par file)
    statsU = (inputIsU or outputIsU)        # if true, stats (cost & deltas) are done with "U"

    siopSetNames = list(siopSets.keys())
    aSiopSet = siopSets[siopSetNames[0]]		# any SIOP set

    chunkShape = input_WL_PIX.shape
    if len(chunkShape) != 2:  raise ValueError("input_WL_PIX must be a 2-D array of reflectances")
    numChunkWLs = chunkShape[0]			# number of WLs in the chunk
    numPixels = chunkShape[1]			# number of pixels in the chunk
    if verbose:
        print(ME + ":  numChunkWLs =", numChunkWLs, "; numPixels =", numPixels)
        print(ME + ":  input_WL_PIX = ...")
        for j in range(input_WL_PIX.shape[1]):		# may be BIG
            print(j, input_WL_PIX[:,j])

    siopWavelength_WL = aSiopSet['wavelength']		# WLs in the SIOP sets
    # print "siopWavelength_WL.shape =", siopWavelength_WL.shape		# DEBUG
    numSiopWLs = len(siopWavelength_WL)			# number of WLs in the SIOP sets
    if numSiopWLs != numChunkWLs:  raise ValueError("numSiopWLs = " + str(numSiopWLs) + ", numChunkWLs = " + str(numChunkWLs) + "; must agree")
    absWater_WL = aSiopSet['a_star']['WATER']
    # print "absWater_WL.shape =", absWater_WL.shape		# DEBUG
    backscatWater_WL = aSiopSet['bb_star']['WATER']

    if not inputIsU:
        if verbose:  print(ME + ":  input_WL_PIX is reflBelowSurf; calculate uIOPRatio")
        reflBelowSurf_WL_PIX = input_WL_PIX
        uIOPRatio_WL_PIX = calcUBackward(reflBelowSurf_WL_PIX, g0, g1)		# convert reflBelowSurf to u for the whole chunk.
        # print "uIOPRatio_WL_PIX.shape =", uIOPRatio_WL_PIX.shape		# DEBUG
        # = u = r_f = the fraction of backscattering coeff to sum of ( backscattering & absorption)
        # i.e., bb / (a + bb), from Brando 2012
        if verbose:  print(ME + ":  uIOPRatio_WL_PIX =", uIOPRatio_WL_PIX.T)		# may be BIG
    else:
        if verbose:  print(ME + ":  input_WL_PIX is uIOPRatio")
        uIOPRatio_WL_PIX = input_WL_PIX

    yIOP_WL_PIX = calcY(uIOPRatio_WL_PIX, absWater_WL, backscatWater_WL)	# calculate the (water-only) Y half of the matrix equation for the whole chunk
    # print "yIOP_WL_PIX.shape =", yIOP_WL_PIX.shape		# DEBUG
    if verbose:  print(ME + ":  yIOP_WL_PIX =", yIOP_WL_PIX.T)		# may be BIG

    numCOMPs = len(components)

    # Initialise these with fill values
    outputType = np.float32		# dtype of most output arrays
    conc_COMP_PIX = np.zeros((numCOMPs, numPixels), dtype=outputType) + fv	# for holding the results from the inversion
    betterCost_PIX = np.zeros((numPixels,), dtype=outputType) + costThreshhold
    bestCost_PIX = np.zeros((numPixels,), dtype=outputType) + fv
    bestSIOPindex_PIX = np.zeros((numPixels,), dtype=int) - 1			# -1 signifies that SVD failed
    bestConc_COMP_PIX = np.zeros((numCOMPs, numPixels), dtype=outputType) + fv
    bestAbsTotal_WL_PIX = np.zeros((numChunkWLs, numPixels), dtype=outputType) + fv
    bestBackscatTotal_WL_PIX = np.zeros((numChunkWLs, numPixels), dtype=outputType) + fv
    bestAbs_WL_COMP_PIX = np.zeros((numChunkWLs, numCOMPs, numPixels), dtype=outputType) + fv
    bestBackscat_WL_COMP_PIX = np.zeros((numChunkWLs, numCOMPs, numPixels), dtype=outputType) + fv
    bestUIOPRatio_WL_PIX = np.zeros((numChunkWLs, numPixels), dtype=outputType) + fv
    bestUIOPRatioPredicted_WL_PIX = np.zeros((numChunkWLs, numPixels), dtype=outputType) + fv
    bestReflBelowSurfPredicted_WL_PIX = np.zeros((numChunkWLs, numPixels), dtype=outputType) + fv
    bestDeltaReflBelowSurf_WL_PIX = np.zeros((numChunkWLs, numPixels), dtype=outputType) + fv
    bestDeltaUIOPRatio_WL_PIX = np.zeros((numChunkWLs, numPixels), dtype=outputType) + fv

    for SIOPindex in range(len(siopSetNames)):
        siopSetName = siopSetNames[SIOPindex]
        if verbose:  print(ME + ":  SIOP_set[" + str(SIOPindex) + "] = '" + siopSetName + "':")
        # Get the parameters for this SIOP set:
        a_star = siopSets[siopSetName]['a_star']
        bb_star = siopSets[siopSetName]['bb_star']

        # TODO:  adjust yIOP_WL_PIX if not calculating all components

        # Calculate the concentrations, by performing the aLMI inversion for each pixel in the chunk:
        for PIX in range(numPixels):		# index of the pixel, in this chunk
            (concInv, residuals, rank, singVals, aMatrix_WL_COMP) = svd_LMI(yIOP_WL_PIX[:,PIX], uIOPRatio_WL_PIX[:,PIX], a_star, bb_star, components)
            conc_COMP_PIX[:,PIX] = concInv
        if verbose:  print(ME + ":  conc_COMP_PIX =", conc_COMP_PIX.T)		# may be BIG

        # Calculate predicted "u IOP ratio", i.e., the "u IOP ratio" predicted by the concentrations (also total abs and backscat):
        forwardResults = calcUForward(conc_COMP_PIX, components, a_star, bb_star)
        uIOPRatioPredicted_WL_PIX = forwardResults['uIOPRatioPredicted']
        if verbose:  print(ME + ":  uIOPRatioPredicted_WL_PIX =", uIOPRatioPredicted_WL_PIX.T)		# may be BIG
        absTotal_WL_PIX = forwardResults['totalAbs']			# absortion at each WL, summed over all components
        backscatTotal_WL_PIX = forwardResults['totalBackscat']		# backscatter at each WL, summed over all components
        abs_WL_COMP_PIX = forwardResults['abs']				# absortion at each WL, for each component
        backscat_WL_COMP_PIX = forwardResults['backscat']		# backscatter at each WL, for each component
        # print "uIOPRatioPredicted_WL_PIX.shape =", uIOPRatioPredicted_WL_PIX.shape		# DEBUG
        # print "absTotal_WL_PIX.shape =", absTotal_WL_PIX.shape		# DEBUG
        # print "backscatTotal_WL_PIX.shape =", backscatTotal_WL_PIX.shape		# DEBUG

        # Calculate the difference between the input and predicted values, and calculate "closure":
        if statsU:
            delta_uIOPRatio_WL_PIX = uIOPRatio_WL_PIX - uIOPRatioPredicted_WL_PIX
            # the closure is measured with a cost function:
            cost_PIX = calcCost(uIOPRatio_WL_PIX, uIOPRatioPredicted_WL_PIX, delta_uIOPRatio_WL_PIX, numSiopWLs, costType)
        else:
            # Calculate "predicted reflectance" from "u IOP ratio":
            reflBelowSurfPredicted_WL_PIX = calcReflForward(uIOPRatioPredicted_WL_PIX, g0, g1)
            # print "reflBelowSurfPredicted_WL_PIX.shape =", reflBelowSurfPredicted_WL_PIX.shape		# DEBUG
            if verbose:  print(ME + ":  reflBelowSurfPredicted_WL_PIX =", reflBelowSurfPredicted_WL_PIX.T)		# may be BIG

            deltaReflBelowSurf_WL_PIX = reflBelowSurf_WL_PIX - reflBelowSurfPredicted_WL_PIX
            # the closure is measured with a cost function:
            cost_PIX = calcCost(reflBelowSurf_WL_PIX, reflBelowSurfPredicted_WL_PIX, deltaReflBelowSurf_WL_PIX, numSiopWLs, costType)

        # print "cost_PIX.shape =", cost_PIX.shape		# DEBUG

        # We consider concentrations to be valid if they are at least (minValidConc):
        # (Note that the IDL code only does this check if the 'do_phys' flag is true (it's hardcoded to true in the setup_status routine);
        # if do_phys is false, then all concentrations are considered valid)
        validConc_PIX = np.all(conc_COMP_PIX >= minValidConc, axis=0)		# locate pixels where all concentrations are valid
        # print "validConc_PIX.shape =", validConc_PIX.shape		# DEBUG

        if verbose:
            # print ME + ":  SIOP_set[" + str(SIOPindex) + "] = '" + siopSetName + "':"
            validConc_NDX = np.nonzero(validConc_PIX)[0]			# indices of pixels where all concentrations are valid
            print(ME + ":  number of pixels where all concentrations are valid =", np.count_nonzero(validConc_PIX))		# DEBUG
            print(ME + ":  <ndx>, cost, conc" + str(components) + ":")
            # print ME + ":  shapes: ", cost_PIX[validConc_NDX].shape, conc_COMP_PIX[:,validConc_NDX].shape
            print(np.vstack((validConc_NDX, cost_PIX[validConc_NDX], conc_COMP_PIX[:,validConc_NDX])).T)		# DEBUG

            invalidConc_PIX = np.any(conc_COMP_PIX < minValidConc, axis=0)		# locate pixels where all concentrations are invalid
            invalidConc_NDX = np.nonzero(invalidConc_PIX)[0]			# indices of pixels where all concentrations are invalid
            num_inversion_failed = np.count_nonzero(invalidConc_PIX)
            print(ME + ":  number of pixels where inversion failed: ", num_inversion_failed)		# DEBUG
            if num_inversion_failed > 0:
                # print "indices of pixels where inversion failed: ", invalidConc_NDX
                print(ME + ":  <ndx>, cost, conc" + str(components) + ":")
                # print ME + ":  shapes: ", cost_PIX[invalidConc_NDX].shape, conc_COMP_PIX[:,invalidConc_NDX].shape
                print(np.vstack((invalidConc_NDX, cost_PIX[invalidConc_NDX], conc_COMP_PIX[:,invalidConc_NDX])).T)		# DEBUG

        # If results are "better", save them:
        whereBetter_PIX = np.logical_and(cost_PIX < betterCost_PIX, validConc_PIX)
        if np.any(whereBetter_PIX):
            betterCost_PIX[whereBetter_PIX] = cost_PIX[whereBetter_PIX]
            bestSIOPindex_PIX[whereBetter_PIX] = SIOPindex
            bestConc_COMP_PIX[:, whereBetter_PIX] = conc_COMP_PIX[:, whereBetter_PIX]
            bestAbsTotal_WL_PIX[:, whereBetter_PIX] = absTotal_WL_PIX[:, whereBetter_PIX]
            bestBackscatTotal_WL_PIX[:, whereBetter_PIX] = backscatTotal_WL_PIX[:, whereBetter_PIX]
            bestAbs_WL_COMP_PIX[:, :, whereBetter_PIX] = abs_WL_COMP_PIX[:, :, whereBetter_PIX]
            bestBackscat_WL_COMP_PIX[:, :, whereBetter_PIX] = backscat_WL_COMP_PIX[:, :, whereBetter_PIX]
            bestUIOPRatio_WL_PIX[:, whereBetter_PIX] = uIOPRatio_WL_PIX[:, whereBetter_PIX]
            bestUIOPRatioPredicted_WL_PIX[:, whereBetter_PIX] = uIOPRatioPredicted_WL_PIX[:, whereBetter_PIX]
            if statsU:
                bestDeltaUIOPRatio_WL_PIX[:, whereBetter_PIX] = delta_uIOPRatio_WL_PIX[:, whereBetter_PIX]
            else:
                bestReflBelowSurfPredicted_WL_PIX[:, whereBetter_PIX] = reflBelowSurfPredicted_WL_PIX[:, whereBetter_PIX]
                bestDeltaReflBelowSurf_WL_PIX[:, whereBetter_PIX] = deltaReflBelowSurf_WL_PIX[:, whereBetter_PIX]

    # end of SIOPindex loop

    whereBest_PIX = betterCost_PIX < costThreshhold
    bestCost_PIX[whereBest_PIX] = betterCost_PIX[whereBest_PIX]

    if verbose:
        # DEBUG:  find pixels with invalid concentrations
        invalidConc_PIX = np.all(bestConc_COMP_PIX < minValidConc, axis=0)		# locate pixels where all concentrations are invalid
        invalidConc_NDX = np.nonzero(invalidConc_PIX)[0]				# indices of pixels where all concentrations are invalid
        print(ME + ":  number of pixels where all inversions failed: ", np.count_nonzero(invalidConc_PIX))		# DEBUG
        print(ME + ":  indices of pixels where all inversions failed: ", invalidConc_NDX)

    result = {}
    result['cost'] = bestCost_PIX
    result['SIOPindex'] = bestSIOPindex_PIX
    result['conc'] = bestConc_COMP_PIX
    result['absTotal'] = bestAbsTotal_WL_PIX
    result['backscatTotal'] = bestBackscatTotal_WL_PIX
    result['abs'] = bestAbs_WL_COMP_PIX
    result['backscat'] = bestBackscat_WL_COMP_PIX
    result['uIOPRatio'] = bestUIOPRatio_WL_PIX
    result['uIOPRatioPredicted'] = bestUIOPRatioPredicted_WL_PIX
    if statsU:
        result['delta_uIOPRatio'] = bestDeltaUIOPRatio_WL_PIX
    else:
        result['reflBelowSurfPredicted'] = bestReflBelowSurfPredicted_WL_PIX
        result['deltaReflBelowSurf'] = bestDeltaReflBelowSurf_WL_PIX

    return result
