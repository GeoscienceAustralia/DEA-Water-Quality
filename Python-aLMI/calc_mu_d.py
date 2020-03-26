#! /usr/bin/env python3

import numpy as np

def calc_mu_d(sun_zen_deg):
    """
    IDL code:
	function compute_mu_d, sun_zen_deg
		 ; and now compute f_mu_d
		 fr_diff=0.0;
		 sun_zen_rad=!dtor*sun_zen_deg
		 n_w=1.333 ; air/water refractive index
		 sun_zen_uw=ASIN(1./n_w*SIN(sun_zen_rad))
		 mu_0=COS(sun_zen_uw)
		 mu_d=0.5*fr_diff+mu_0*(1-fr_diff)
		 return, mu_d
	end

    Returns mu_d
    """

    fr_diff = 0.
    sun_zen_rad = np.radians(sun_zen_deg)
    n_w = 1.333		# air/water refractive index
    sun_zen_uw = np.arcsin(1. / n_w * np.sin(sun_zen_rad))		# radians, in [-pi/2, pi/2] (same as IDL)
    mu_0 = np.cos(sun_zen_uw)
    mu_d = 0.5 * fr_diff + mu_0 * (1. - fr_diff)
    # print "calc_mu_d:  sun_zen_deg = ", sun_zen_deg, "; mu_d = ", mu_d                # DEBUG
    return mu_d

# The '__main__' entry point is only used for testing. 
if __name__ == '__main__':
    import sys

    ME = sys.argv[0]
    if len(sys.argv) != 1:
        print("usage:  " + sys.argv[0])
        sys.exit(1)

    sun_zen_deg = np.linspace(-90., 90., num=19)

    mu_d = calc_mu_d(sun_zen_deg)
    print("Results:")
    print("sun_zen_deg          mu_d")
    print(np.vstack((sun_zen_deg, mu_d)).T)

    print(sys.argv[0] + " done")             # DEBUG
