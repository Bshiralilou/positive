#
from __future__ import print_function
from . import *
from positive.api import *
from positive.plotting import *
from positive.learning import *
from positive.physics import *

#
def rISCO_14067295(a):
    """
    Calculate the ISCO radius of a Kerr BH as a function of the Kerr parameter using eqns. 2.5 and 2.8 from Ori and Thorne, Phys Rev D 62, 24022 (2000)

    Parameters
    ----------
    a : Kerr parameter

    Returns
    -------
    ISCO radius
    """

    import numpy as np
    a = np.array(a)

    # Ref. Eq. (2.5) of Ori, Thorne Phys Rev D 62 124022 (2000)
    z1 = 1.0+(1.0-a**2.0)**(1.0/3)*((1.0+a)**(1.0/3) + (1.0-a)**(1.0/3))
    z2 = np.sqrt(3 * a**2 + z1**2)
    a_sign = np.sign(a)
    return 3+z2 - np.sqrt((3.0-z1)*(3.0+z1+2.0*z2))*a_sign


# Calculate Kerr ISCO Radius
def rKerr_ISCO_Bardeen(j,M=1):
    '''

    Calculate Kerr ISCO Radius

    USAGE:
    rKerr_ISCO_Bardeen(Dimensionless_BH_Spin,BH_Mass)

    ~ londonl@mit.edu

    '''

    #
    from numpy import sign,sqrt

    #
    a = M*j

    #
    p13 = 1.0/3
    jj = j*j

    #
    Z1 = 1 + ( 1-jj )**p13 * ( (1+j)**p13 + (1-j)**p13 )
    Z2 = sqrt( 3*jj + Z1*Z1 )

    #
    rKerr_ISCO = M * (  3 + Z2 - sign(j) * sqrt( (3-Z1)*(3+Z1+2*Z2) )  )

    #
    return rKerr_ISCO


# Calculate Kerr ISCO Angular Frequency
def wKerr_ISCO_Bardeen(j,M=1):
    '''

    Calculate Kerr ISCO Angular Frequency

    USAGE:
    wKerr_ISCO_Bardeen(Dimensionless_BH_Spin,BH_Mass=1)

    ~ londonl@mit.edu

    '''

    #
    from numpy import sin,sign,sqrt

    #
    a = M*j

    # Bardeen et al 2.21, Ori Thorn 2.5
    r = M * rKerr_ISCO_Bardeen(j,M=M) # rISCO_14067295(j)

    # 2.16 of Bardeen et al, ( 2.1 of Ori Thorn misses physics )
    wKerr_ISCO = sqrt(M) / ( r**(3.0/2) + a*sqrt(M) )

    # Return Answer
    return wKerr_ISCO


# Calculate Kerr Light-Ring Radius
def rKerr_LR_Bardeen(j,M=1,branch=0):
    '''

    Calculate Kerr Light-Ring Radius

    USAGE:
    rKerr_LR_Bardeen(Dimensionless_BH_Spin,BH_Mass)

    ~ londonl@mit.edu

    '''

    #
    from numpy import cos,arccos,sqrt
    from positive import acos

    # Bardeen et al 2.18
    rKerr_LR = 2*M*(   1 + cos( (2.0/3)*acos(-j,branch=branch) )   )
    # rKerr_LR = 2*M*(   1 + cos( (2.0/3)*arccos(-j) )   )

    #
    return rKerr_LR


# Calculate Kerr ISCO Angular Frequency
def wKerr_LR_Bardeen(j,M=1,branch=0):
    '''

    Calculate Kerr Light-Ring (aka PH for Photon) Angular Frequency

    USAGE:
    wKerr_LR_Bardeen(Dimensionless_BH_Spin,BH_Mass=1)

    ~ londonl@mit.edu

    '''

    #
    from numpy import sin,sign,sqrt

    #
    a = M*j

    # Bardeen et al 2.18
    r = M * rKerr_LR_Bardeen(j,M=M,branch=branch)

    # 2.16 of Bardeen et al
    wKerr_LR = sqrt(M) / ( r**(3.0/2) + a*sqrt(M) )

    # Return Answer
    return wKerr_LR

