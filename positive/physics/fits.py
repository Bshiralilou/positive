#
from __future__ import print_function
from positive.api import *
from positive.physics.misc import *
from positive.plotting import *
from positive.learning import *
# # > > > > > > > > >  Import adjacent modules  > > > > > > > > > > #
# import positive
# modules = list( basename(f)[:-3] for f in glob.glob(dirname(__file__)+"/*.py") if (not ('__init__.py' in f)) and (not (__file__.split('.')[0] in f)) )
# for module in modules:
#     exec('from .%s import *' % module)
# # > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > #



#
def mass_ratio_convention_sort(m1,m2,chi1,chi2):

    '''
    Function to enforce mass ratio convention m1>m2.

    USAGE:

    m1,m2,chi1,chi2 = mass_ratio_convention_sort(m1,m2,chi1,chi2,format=None)

    INPUTS:

    m1,         1st component mass
    m2,         2nd component mass
    chi1,       1st dimensionless spin
    chi2,       2nd dimensionless spin

    OUTPUTS:

    m1,m2,chi1,chi2

    NOTE that outputs are swapped according to desired convention.

    londonl@mit.edu 2019

    '''

    # Import usefuls
    from numpy import min,max,array,ndarray,ones_like

    # Enforce arrays
    float_input_mass = not isinstance(m1,(ndarray,list,tuple))
    if float_input_mass:
        m1 = array([m1]);     m2 = array([m2])
    float_input_chi = not isinstance(chi1,(ndarray,list,tuple))
    if float_input_chi:
        chi1 = chi1*ones_like(m1); chi2 = chi2*ones_like(m2)

    #
    L = len( m1 )
    if  (L != len(m2)) or (len(chi1)!=len(chi2)) :
        error( 'lengths of input parameters not same' )

    # Prepare for swap / allocate output
    m1_   = array(m2);   m2_   = array(m1)
    chi1_ = array(chi2); chi2_ = array(chi1)

    #
    for k in range(L):

        # Enforce m1 > m2
        if (m1[k] < m2[k]):

            m1_[k] = m2[k]
            m2_[k] = m1[k]

            chi1_[k] = chi2[k]
            chi2_[k] = chi1[k]

    #
    if float_input_mass:
        m1_   = m1_[0];   m2_   = m2_[0]
    if float_input_chi:
        chi1_ = chi1_[0]; chi2_ = chi2_[0]

    #
    return (m1_,m2_,chi1_,chi2_)




# https://arxiv.org/pdf/1406.7295.pdf
def Mf14067295( m1,m2,chi1,chi2,chif=None ):

    import numpy as np

    if np.any(abs(chi1>1)):
      raise ValueError("chi1 has to be in [-1, 1]")
    if np.any(abs(chi2>1)):
      raise ValueError("chi2 has to be in [-1, 1]")


    # Swapping inputs to conform to fit conventions
    # NOTE: See page 2 of https://arxiv.org/pdf/1406.7295.pdf
    m2,m1,chi1,chi2 = mass_ratio_convention_sort(m2,m1,chi1,chi2)
    # if m1>m2:
    #     #
    #     m1_,m2_ = m1,m2
    #     chi1_,chi2_ = chi1,chi2
    #     #
    #     m1,m2 = m2_,m1_
    #     chi1,chi2 = chi2_,chi1_

    # binary parameters
    m = m1+m2
    q = m1/m2
    eta = q/(1.+q)**2.
    delta_m = (m1-m2)/m
    S1 = chi1*m1**2 # spin angular momentum 1
    S2 = chi2*m2**2 # spin angular momentum 2
    S = (S1+S2)/m**2 # symmetric spin (dimensionless -- called \tilde{S} in the paper)
    Delta = (S2/m2-S1/m1)/m # antisymmetric spin (dimensionless -- called tilde{Delta} in the paper

    #
    if chif is None:
        chif = jf14067295(m1, m2, chi1, chi2)
    r_isco = rISCO_14067295(chif)

    # fitting coefficients - Table XI of Healy et al Phys Rev D 90, 104004 (2014)
    # [fourth order fits]
    M0  = 0.951507
    K1  = -0.051379
    K2a = -0.004804
    K2b = -0.054522
    K2c = -0.000022
    K2d = 1.995246
    K3a = 0.007064
    K3b = -0.017599
    K3c = -0.119175
    K3d = 0.025000
    K4a = -0.068981
    K4b = -0.011383
    K4c = -0.002284
    K4d = -0.165658
    K4e = 0.019403
    K4f = 2.980990
    K4g = 0.020250
    K4h = -0.004091
    K4i = 0.078441

    # binding energy at ISCO -- Eq.(2.7) of Ori, Thorne Phys Rev D 62 124022 (2000)
    E_isco = (1. - 2./r_isco + chif/r_isco**1.5)/np.sqrt(1. - 3./r_isco + 2.*chif/r_isco**1.5)

    # final mass -- Eq. (14) of Healy et al Phys Rev D 90, 104004 (2014)
    mf = (4.*eta)**2*(M0 + K1*S + K2a*Delta*delta_m + K2b*S**2 + K2c*Delta**2 + K2d*delta_m**2 \
        + K3a*Delta*S*delta_m + K3b*S*Delta**2 + K3c*S**3 + K3d*S*delta_m**2 \
        + K4a*Delta*S**2*delta_m + K4b*Delta**3*delta_m + K4c*Delta**4 + K4d*S**4 \
        + K4e*Delta**2*S**2 + K4f*delta_m**4 + K4g*Delta*delta_m**3 + K4h*Delta**2*delta_m**2 \
        + K4i*S**2*delta_m**2) + (1+eta*(E_isco+11.))*delta_m**6.

    return mf*m

#
def jf14067295_diff(a_f,eta,delta_m,S,Delta):
    """ Internal function: the final spin is determined by minimizing this function """

    #
    import numpy as np

    # calculate ISCO radius
    r_isco = rISCO_14067295(a_f)

    # angular momentum at ISCO -- Eq.(2.8) of Ori, Thorne Phys Rev D 62 124022 (2000)
    J_isco = (3*np.sqrt(r_isco)-2*a_f)*2./np.sqrt(3*r_isco)

    # fitting coefficients - Table XI of Healy et al Phys Rev D 90, 104004 (2014)
    # [fourth order fits]
    L0  = 0.686710
    L1  = 0.613247
    L2a = -0.145427
    L2b = -0.115689
    L2c = -0.005254
    L2d = 0.801838
    L3a = -0.073839
    L3b = 0.004759
    L3c = -0.078377
    L3d = 1.585809
    L4a = -0.003050
    L4b = -0.002968
    L4c = 0.004364
    L4d = -0.047204
    L4e = -0.053099
    L4f = 0.953458
    L4g = -0.067998
    L4h = 0.001629
    L4i = -0.066693

    a_f_new = (4.*eta)**2.*(L0  +  L1*S +  L2a*Delta*delta_m + L2b*S**2. + L2c*Delta**2 \
        + L2d*delta_m**2. + L3a*Delta*S*delta_m + L3b*S*Delta**2. + L3c*S**3. \
        + L3d*S*delta_m**2. + L4a*Delta*S**2*delta_m + L4b*Delta**3.*delta_m \
        + L4c*Delta**4. + L4d*S**4. + L4e*Delta**2.*S**2. + L4f*delta_m**4 + L4g*Delta*delta_m**3. \
        + L4h*Delta**2.*delta_m**2. + L4i*S**2.*delta_m**2.) \
        + S*(1. + 8.*eta)*delta_m**4. + eta*J_isco*delta_m**6.

    daf = a_f-a_f_new
    return daf*daf

#
def jf14067295(m1, m2, chi1, chi2):
    """
    Calculate the spin of the final BH resulting from the merger of two black holes with non-precessing spins using fit from Healy et al Phys Rev D 90, 104004 (2014)

    Parameters
    ----------
    m1, m2 : component masses
    chi1, chi2 : dimensionless spins of two BHs

    Returns
    -------
    dimensionless final spin, chif
    """
    import numpy as np
    import scipy.optimize as so

    if np.any(abs(chi1>1)):
      raise ValueError("chi1 has to be in [-1, 1]")
    if np.any(abs(chi2>1)):
      raise ValueError("chi2 has to be in [-1, 1]")

    # Vectorize the function if arrays are provided as input
    if np.size(m1) * np.size(m2) * np.size(chi1) * np.size(chi2) > 1:
        return np.vectorize(bbh_final_spin_non_precessing_Healyetal)(m1, m2, chi1, chi2)

    # binary parameters
    m = m1+m2
    q = m1/m2
    eta = q/(1.+q)**2.
    delta_m = (m1-m2)/m

    S1 = chi1*m1**2 # spin angular momentum 1
    S2 = chi2*m2**2 # spin angular momentum 2
    S = (S1+S2)/m**2 # symmetric spin (dimensionless -- called \tilde{S} in the paper)
    Delta = (S2/m2-S1/m1)/m # antisymmetric spin (dimensionless -- called tilde{Delta} in the paper

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # compute the final spin
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    x, cov_x = so.leastsq(jf14067295_diff, 0., args=(eta, delta_m, S, Delta))
    chif = x[0]

    return chif

# Energy Radiated
# https://arxiv.org/abs/1611.00332
# Xisco Jimenez-Forteza, David Keitel, Sascha Husa, Mark Hannam, Sebastian Khan, Michael Purrer
def Erad161100332(m1,m2,chi1,chi2):
    '''
    Final mass fit from: https://arxiv.org/abs/1611.00332
    By Xisco Jimenez-Forteza, David Keitel, Sascha Husa, Mark Hannam, Sebastian Khan, Michael Purrer
    '''
    # Import usefuls
    from numpy import sqrt
    # Test for m1>m2 convention
    m1,m2,chi1,chi2 = mass_ratio_convention_sort(m1,m2,chi1,chi2)
    # if m1<m2:
    #     # Swap everything
    #     m1_,m2_ = m2 ,m1 ;  chi1_,chi2_ =  chi2,chi1
    #     m1,m2   = m1_,m2_;  chi1,chi2   = chi1_,chi2_
    #
    M = m1+m2
    eta = m1*m2/(M*M)
    # Caclulate effective spin
    S = (chi1*m1 + chi2*m2) / M
    # Calculate fitting formula
    E = -0.12282038851475935*(chi1 - chi2)*(1 - 4*eta)**0.5*(1 - 3.499874117528558*eta)*eta**2 +\
        0.014200036099065607*(chi1 - chi2)**2*eta**3 - 0.018737203870440332*(chi1 - chi2)*(1 -\
        5.1830734412467425*eta)*(1 - 4*eta)**0.5*eta*S + (((1 - (2*sqrt(2))/3.)*eta +\
        0.5635376058169301*eta**2 - 0.8661680065959905*eta**3 + 3.181941595301784*eta**4)*(1 +\
        (-0.13084395473958504 - 1.1075070900466686*eta + 5.292389861792881*eta**2)*S + \
        (-0.17762804636455634 + 2.095538044244076*eta**2)*S**2 + (-0.6320190570187046 + \
        5.645908914996172*eta - 12.860272122009997*eta**2)*S**3))/(1 + (-0.9919475320287884 +\
        0.5383449788171806*eta + 3.497637161730149*eta**2)*S)
    # Return answer
    return E

# Remnant mass
# https://arxiv.org/abs/1611.00332
# Xisco Jimenez-Forteza, David Keitel, Sascha Husa, Mark Hannam, Sebastian Khan, Michael Purrer
def Mf161100332(m1,m2,chi1,chi2):
    return m1+m2-Erad161100332(m1,m2,chi1,chi2)

# Remnant Spin
# https://arxiv.org/abs/1611.00332
# Xisco Jimenez-Forteza, David Keitel, Sascha Husa, Mark Hannam, Sebastian Khan, Michael Purrer
def jf161100332(m1,m2,chi1,chi2):
    '''
    Final mass fit from: https://arxiv.org/abs/1611.00332
    By Xisco Jimenez-Forteza, David Keitel, Sascha Husa, Mark Hannam, Sebastian Khan, Michael Purrer
    '''
    # Import usefuls
    from numpy import sqrt
    # Test for m1>m2 convention
    m1,m2,chi1,chi2 = mass_ratio_convention_sort(m1,m2,chi1,chi2)
    # if m1<m2:
    #     # Swap everything
    #     m1_,m2_ = m2 ,m1 ;  chi1_,chi2_ =  chi2,chi1
    #     m1,m2   = m1_,m2_;  chi1,chi2   = chi1_,chi2_
    #
    M = m1+m2
    eta = m1*m2/(M*M)
    # Caclulate effective spin
    S = (chi1*m1 + chi2*m2) / M
    # Calculate fitting formula
    jf = -0.05975750218477118*(chi1 - chi2)**2*eta**3 + 0.2762804043166152*(chi1 - chi2)*(1 -\
         4*eta)**0.5*eta**2*(1 + 11.56198469592321*eta) + (2*sqrt(3)*eta + 19.918074038061683*eta**2 -\
         12.22732181584642*eta**3)/(1 + 7.18860345017744*eta) + chi1*m1**2 + chi2*m2**2 +\
         2.7296903488918436*(chi1 - chi2)*(1 - 4*eta)**0.5*(1 - 3.388285154747212*eta)*eta**3*S + ((0. -\
         0.8561951311936387*eta - 0.07069570626523915*eta**2 + 1.5593312504283474*eta**3)*S + (0. +\
         0.5881660365859452*eta - 2.670431392084654*eta**2 + 5.506319841591054*eta**3)*S**2 + (0. +\
         0.14244324510486703*eta - 1.0643244353754102*eta**2 + 2.3592117077532433*eta**3)*S**3)/(1 +\
         (-0.9142232696116447 + 2.6764257152659883*eta - 15.137168414785732*eta**3)*S)
    # Return answer
    return jf


# Remnant Spin
# https://arxiv.org/abs/1605.01938
# Fabian Hofmann, Enrico Barausse, Luciano Rezzolla
def jf160501938(m1,m2,chi1_vec,chi2_vec,L_vec=None):

    '''
    Remnant Spin
    https://arxiv.org/abs/1605.01938
    Fabian Hofmann, Enrico Barausse, Luciano Rezzolla
    '''

    # Import usefuls
    from numpy import sqrt,dot,array,cos,tan,arccos,arctan,zeros,sign
    from numpy.linalg import norm

    # Handle inputs
    if L_vec is None:
        warning('No initial orbital angular momentum vevtor given. We will assume it is z-ligned.')
        L_vec = array([0,0,1])
    #
    m1 = float(m1); m2 = float(m2)

    # Table 1 (numbers copied from arxiv tex file: https://arxiv.org/format/1605.01938)

    # # bottom block -- doesnt work
    # n_M = 3; n_J = 4
    # k = zeros( (n_M+1,n_J+1) )
    # k[0,1] = 3.39221;   k[0,2] = 4.48865;  k[0,3] = -5.77101
    # k[0,4] = -13.0459;  k[1,0] = 35.1278;  k[1,1] = -72.9336
    # k[1,2] = -86.0036;  k[1,3] = 93.7371;  k[1,4] = 200.975
    # k[2,0] = -146.822;  k[2,1] = 387.184;  k[2,2] = 447.009
    # k[2,3] = -467.383;  k[2,4] = -884.339; k[3,0] = 223.911
    # k[3,1] = -648.502;  k[3,2] = -697.177; k[3,3] = 753.738
    # k[3,4] = 1166.89;   xi = 0.474046
    # k[0,0] = -3.82

    # top block -- works
    n_M = 1; n_J = 2
    k = zeros( (n_M+1,n_J+1) )
    k[0,1] = -1.2019; k[0,2] = -1.20764; k[1,0] = 3.79245; k[1,1] = 1.18385
    k[1,2] = 4.90494; xi = 0.41616
    k[0,0] = -3.82

    # Eq. 5
    p = 1.0/3
    Z1 = lambda a: 1 + (1-a*a)**p * (  (1+a)**p + (1-a)**p  )

    # Eq. 6
    Z2 = lambda a: sqrt( 3*a*a + Z1(a)**2 )

    # Eq. 4
    r_ISCO = lambda a: 3.0 + Z2(a) - sign(a) * sqrt( (3-Z1(a))*(3+Z1(a)+2*Z2(a)) )
    # r_ISCO = lambda a: 3.0 + Z2(a) - ( a / abs(a) ) * sqrt( (3-Z1(a))*(3+Z1(a)+2*Z2(a)) )

    # Eq. 2
    E_ISCO = lambda a: sqrt( 1 - 2.0 / ( 3 * r_ISCO(a) ) )

    # Eq. 3
    p = 0.38490017945975052 # this is 2.0 / (3*sqrt(3))
    L_ISCO = lambda a: p * (  1 + 2*sqrt( 3*r_ISCO(a)-2 )  )

    ## Amplitude of final spin

    # Define low level physical parameters
    L_hat = L_vec / norm(L_vec)
    a1z = dot(L_hat,chi1_vec) # chi1_vec[-1]
    a2z = dot(L_hat,chi2_vec) # chi2_vec[-1]
    a1 = norm(chi1_vec)
    a2 = norm(chi2_vec)
    eta = m1*m2 / (m1+m2)**2
    q = m2/m1 if m2<m1 else m1/m2 # convention as seen above Eq 1

    # Eq. 17
    x1 = (chi1_vec / norm(chi1_vec)) if norm(chi1_vec) else zeros(3)
    x2 = (chi2_vec / norm(chi2_vec)) if norm(chi2_vec) else zeros(3)
    __alpha__ = arccos( dot(x1, x2) )
    __beta__  = arccos( dot(L_hat, x1) )
    __gamma__ = arccos( dot(L_hat, x2) )

    # Eq. 18 for alpha
    eps_alpha = 0
    alpha = 2 * arctan( (1+eps_alpha)     *tan(__alpha__/2) )

    # Eq. 18 for beta
    eps_beta_gamma = 0.024
    beta  = 2 * arctan( (1+eps_beta_gamma)*tan( __beta__/2) )

    # Eq. 18 for gamma
    gamma = 2 * arctan( (1+eps_beta_gamma)*tan(__gamma__/2) )

    # alpha = __alpha__
    # beta = __beta__
    # gamma = __gamma__

    # Eq. 14
    a_tot = (  a1*cos(beta) + a2*cos(gamma)*q  ) / (1.0+q)**2
    a_eff = a_tot + xi*eta*( a1z*cos(beta) + a2z*cos(gamma) )

    # Eq. 13 -- Double sum part
    double_sum_part = 0
    for i in range(n_M+1):
        for j in range(n_J+1):
            double_sum_part += (k[i,j] * eta**(1+i) * a_eff ** j) if k[i,j] else 0

    # Eq. 13
    absl = abs( L_ISCO(a_eff) - 2*a_tot*(E_ISCO(a_eff)-1) + double_sum_part )

    # Eq. 16
    afin = (1.0/(1+q)**2) * sqrt(  a1**2 + a2**2 * q**4 + 2*a1*a2*q*q*cos(alpha) + 2*(a1*cos(beta)+a2*q*q*cos(gamma))*absl*q + absl*absl*q*q  )

    #
    # b = dir()
    # for k in b:
    #     print k+'\t=\t',eval(k)
    return afin



# Fit for spherica-sphoidal harmonic inner-product from Berti et al
def ysprod14081860(j,ll,mm,lmn):
    import positive
    from numpy import loadtxt,array,ndarray
    '''
    Fits for sherical-spheroidal mixing coefficients from arxiv:1408.1860 -- Berti, Klein

    * The refer to mixing coefficients as mu_mll'n'
    * their fits are parameterized by dimensionless BH spin j=a/M
    * They model the real and imaginary parts separately as seen in Eq. 11
    * Fitting coefficients are listed in table I

    NOTE that the input format of this function is designed to be consistent with ysprod(...)
    NOTE that tis function explicitely loads their full fit inpf from an external file

    LondonL@mit.edu 2019
    '''
    tbl = loadtxt(parent(os.path.realpath(__file__))+'/data/berti_swsh_fits.dat')

    def lowlevel(J,__mm__):
        MM,LL,L,N,P1,P2,P3,P4,Q1,Q2,Q3,Q4,_,_,_,_ = tbl.T
        l,m,n = lmn
        flip = False
        if J<0:
            J = abs(J)
            __mm__ *= -1
            flip = True
        k = (ll==LL) & (__mm__==MM) & (l==L) & (n==N)
        if sum(k)!=1:
            error('cannot find fit from Berti+ for (L,M,l,m,n) = (%i,%i,%i,%i,%)'%(L,M,l,m,n))
        lowlevel_ans = (1.0 if ll==l else 0) + P1[k]*J**P2[k] + P3[k]*J**P4[k] + 1j * ( Q1[k]*J**Q2[k] + Q3[k]*J**Q4[k] )
        return (lowlevel_ans if not flip else lowlevel_ans.conj()).conj()

    #
    if isinstance(j,(list,tuple,ndarray)):
        return array( [ lowlevel(j_,mm) for j_ in j ] )
    else:
        return lowlevel(j,mm)


# Fit for spherica-sphoidal harmonic inner-product from Berti et al
def ysprod14081860_from_paper_is_broken(j,L,M,lmn):

    '''
    Fits for sherical-spheroidal mixing coefficients from arxiv:1408.1860 -- Berti, Klein

    * The refer to mixing coefficients as mu_mll'n'
    * their fits are parameterized by dimensionless BH spin j=a/M
    * They model the real and imaginary parts separately as seen in Eq. 11
    * Fitting coefficients are listed in table I

    NOTE that the input format of this function is designed to be consistent with ysprod(...)
    NOTE that the values listed in the table are so large as they are susceptible to severe error due to truncation yielding this function fucked

    LondonL@mit.edu 2019
    '''

    # Warn the user
    warning('This function uses values listed in the paper\'s table. These values are so large as they are susceptible to severe error due to truncation yielding this function fucked.')

    # Unpack spheroidal indeces
    l,m,n = lmn

    # A represenation of their Table I. NOTE that the original tex from the arxiv was used here; find-replace and multi-select was performed for dictionary formatting
    TBL = {}
    TBL[2,2,2,0] = {'p1':-740,'p2':2.889,'p3':-661,'p4':17.129,'q1':1530,'q2':1.219,'q3':-934,'q4':24.992}
    TBL[2,2,2,1] = {'p1':-873,'p2':2.655,'p3':-539,'p4':15.665,'q1':4573,'q2':1.209,'q3':-2801,'q4':25.451}
    TBL[2,2,3,0] = {'p1':14095,'p2':1.112,'p3':4395,'p4':6.144,'q1':1323,'q2':0.854,'q3':-852,'q4':7.042}
    TBL[2,3,2,0] = {'p1':-10351,'p2':1.223,'p3':-5750,'p4':8.705,'q1':-1600,'q2':0.953,'q3':1003,'q4':14.755}
    TBL[-2,2,2,0] = {'p1': -1437,'p2':2.118,'p3':1035,'p4':2.229,'q1':-7015,'q2':1.005,'q3':67,'q4':3.527}
    TBL[-2,2,2,1] = {'p1': -2659,'p2':2.007,'p3':53,'p4':4.245,'q1':-21809,'q2':1.008,'q3':221,'q4':4.248}
    TBL[-2,2,3,0] = {'p1': 14971,'p2':1.048,'p3':-5463,'p4':1.358,'q1':18467,'q2':1.015,'q3':-10753,'q4':1.876}
    TBL[-2,3,2,0] = {'p1': -13475,'p2':1.088,'p3':7963,'p4':1.279,'q1':-1744,'q2':1.011,'q3':516,'q4':1.821}


    if (M,L,l,n) in TBL:

        # Extract relevant data from model dictionary using inputs
        p1 = TBL[M,L,l,n]['p1']
        p2 = TBL[M,L,l,n]['p2']
        p3 = TBL[M,L,l,n]['p3']
        p4 = TBL[M,L,l,n]['p4']
        q1 = TBL[M,L,l,n]['q1']
        q2 = TBL[M,L,l,n]['q2']
        q3 = TBL[M,L,l,n]['q3']
        q4 = TBL[M,L,l,n]['q4']

        #
        delta_llp = 1.0 if L==l else 0

        # Evaluate Equation 11
        if m==M:
            re_mu = delta_llp + (p1 * (j ** (p2*1e5))) + (p3 * (j ** (p4*1e5)))
            im_mu = (q1 * (j ** (q2*1e5))) + (q3 * (j ** (q4*1e5)))
            ans = re_mu + 1j*im_mu
        else:
            ans = 0

    else:

        warning('Berti\'s model does not include (L,M)(l,m,n) = %s%s'%(str((L,M)),str((l,m,n))) )
        ans = j/0

    # Return answer
    return ans


#
def CookZalutskiy14107698(jf,l,m,n):

    '''
    Fir for Kerr s=-2 QNM frequencies from Cook+Zalutskiy arxiv:1410.7698
    '''

    #
    from numpy import sqrt

    #
    eps = 1-jf

    # TABLE I
    TBL1 = {}
    TBL1[2,2] = { 'delta':2.05093, 'a1':2.05084, 'a2':1.64, 'a3':-0.032, 'a4':1.343 }
    TBL1[3,3] = { 'delta':2.79361, 'a1':2.79361, 'a2':2.289, 'a3':-0.0004, 'a4':1.35730 }
    TBL1[4,4] = { 'delta':3.56478, 'a1':3.56478, 'a2':2.79572, 'a3':-0.00020, 'a4':1.34522 }
    TBL1[5,4] = { 'delta':1.07271, 'a1':1.0687, 'a2':-28., 'a3':-3.4, 'a4':-17.90 }
    TBL1[5,5] = { 'delta':4.35761, 'a1':4.35761, 'a2':3.29989, 'a3':-0.000142, 'a4':1.33085 }
    TBL1[6,5] = { 'delta':2.37521, 'a1':2.37515, 'a2':-5.50, 'a3':-0.672, 'a4':-1.206 }
    TBL1[6,6] = { 'delta':5.16594, 'a1':5.16594, 'a2':3.81003, 'a3':-0.000111, 'a4':1.31809 }
    TBL1[7,6] = { 'delta':3.40439, 'a1':3.40438, 'a2':-2.03, 'a3':-0.0109, 'a4':0.2931 }
    TBL1[7,7] = { 'delta':5.98547, 'a1':5.98547, 'a2':4.32747, 'a3':-0.000091, 'a4':1.30757 }
    TBL1[8,7] = { 'delta':4.35924, 'a1':4.35924, 'a2':-0.274, 'a3':-0.0034, 'a4':0.74198 }
    TBL1[8,8] = { 'delta':6.81327, 'a1':6.81327, 'a2':4.85207, 'a3':-0.000078, 'a4':1.29905 }
    TBL1[9,8] = { 'delta':5.28081, 'a1':5.28081, 'a2':0.9429, 'a3':-0.00150, 'a4':0.93670 }
    TBL1[10,8] = { 'delta':2.80128, 'a1':2.80099, 'a2':-22.9, 'a3':-0.43, 'a4':-6.456 }
    TBL1[9,9] = { 'delta':7.64735, 'a1':7.64734, 'a2':5.38314, 'a3':-0.000069, 'a4':1.29217 }
    TBL1[10,9] = { 'delta':6.18436, 'a1':6.18436, 'a2':1.92226, 'a3':-0.00080, 'a4':1.03870 }
    TBL1[11,9] = { 'delta':4.05104, 'a1':4.05101, 'a2':-11.00, 'a3':-0.053, 'a4':-1.568 }
    TBL1[10,10] = { 'delta':8.48628, 'a1':8.48628, 'a2':5.91988, 'a3':-0.000062, 'a4':1.28657 }
    TBL1[11,10] = { 'delta':7.07706, 'a1':7.07706, 'a2':2.7754, 'a3':-0.00048, 'a4':1.09871 }
    TBL1[12,10] = { 'delta':5.14457, 'a1':5.14457, 'a2':-6.269, 'a3':-0.0147, 'a4':-0.2362 }
    TBL1[11,11] = { 'delta':9.32904, 'a1':9.32904, 'a2':6.46155, 'a3':-0.000056, 'a4':1.28198 }
    TBL1[12,11] = { 'delta':7.96274, 'a1':7.96274, 'a2':3.5535, 'a3':-0.00032, 'a4':1.13691 }
    TBL1[12,12] = { 'delta':10.1749, 'a1':10.1749, 'a2':7.00748, 'a3':-0.00005, 'a4':1.27819 }

    # TABLE III
    TBL2 = {}
    TBL2[2,1] = { 'delta': 1j*1.91907,'a1':3.23813,'a2':1.54514,'a3':1.91906,'a4':-0.021,'a5':-0.0109 }
    TBL2[3,1] = { 'delta': 1j*3.17492,'a1':2.11224,'a2':0.710824,'a3':3.17492,'a4':-0.0061,'a5':-0.0022 }
    TBL2[4,1] = { 'delta': 1j*4.26749,'a1':1.82009,'a2':0.000000,'a3':4.26749,'a4':-0.0048,'a5':0.000000 }
    TBL2[3,2] = { 'delta': 1j*1.87115,'a1':7.2471,'a2':3.4744,'a3':1.87108,'a4':-0.12,'a5':-0.061 }
    TBL2[4,2] = { 'delta': 1j*3.47950,'a1':4.19465,'a2':1.30534,'a3':3.47950,'a4':-0.0106,'a5':-0.0040 }
    TBL2[5,2] = { 'delta': 1j*4.72816,'a1':3.61332,'a2':0.882398,'a3':4.72816,'a4':-0.0067,'a5':-0.0018 }
    TBL2[4,3] = { 'delta': 1j*1.37578,'a1':22.66,'a2':12.807,'a3':1.37549,'a4':-1.6,'a5':-0.87 }
    TBL2[5,3] = { 'delta': 1j*3.54313,'a1':6.80319,'a2':2.05358,'a3':3.54312,'a4':-0.024,'a5':-0.0094 }
    TBL2[6,3] = { 'delta': 1j*4.98492,'a1':5.59000,'a2':1.29263,'a3':4.98492,'a4':-0.0103,'a5':-0.0029 }
    TBL2[7,3] = { 'delta': 1j*6.24553,'a1':5.13084,'a2':0.000000,'a3':6.24552,'a4':-0.0077,'a5':0.000000 }
    TBL2[6,4] = { 'delta': 1j*3.38736,'a1':10.6913,'a2':3.26423,'a3':3.38733,'a4':-0.075,'a5':-0.029 }
    TBL2[7,4] = { 'delta': 1j*5.07533,'a1':7.93057,'a2':1.78114,'a3':5.07532,'a4':-0.018,'a5':-0.0053 }
    TBL2[8,4] = { 'delta': 1j*6.47378,'a1':7.07896,'a2':0.000000,'a3':6.47378,'a4':-0.0100,'a5':0.000000 }
    TBL2[7,5] = { 'delta': 1j*2.98127,'a1':18.146,'a2':5.9243,'a3':2.98114,'a4':-0.32,'a5':-0.131 }
    TBL2[8,5] = { 'delta': 1j*5.01168,'a1':10.9114,'a2':2.43320,'a3':5.01167,'a4':-0.033,'a5':-0.0101 }
    TBL2[9,5] = { 'delta': 1j*6.57480,'a1':9.30775,'a2':1.66898,'a3':6.57480,'a4':-0.0152,'a5':-0.0036 }
    TBL2[8,6] = { 'delta': 1j*2.19168,'a1':43.25,'a2':16.68,'a3':2.1906,'a4':-2.9,'a5':-1.45 }
    TBL2[9,6] = { 'delta': 1j*4.78975,'a1':15.0733,'a2':3.41635,'a3':4.78972,'a4':-0.077,'a5':-0.024 }
    TBL2[10,6] = { 'delta': 1j*6.55627,'a1':11.9630,'a2':2.12050,'a3':6.55626,'a4':-0.0252,'a5':-0.0063 }
    TBL2[11,6] = { 'delta': 1j*8.06162,'a1':10.7711,'a2':0.000000,'a3':8.06162,'a4':-0.0153,'a5':0.000000 }
    TBL2[10,7] = { 'delta': 1j*4.38687,'a1':21.7120,'a2':5.1575,'a3':4.38680,'a4':-0.21,'a5':-0.068 }
    TBL2[11,7] = { 'delta': 1j*6.41836,'a1':15.2830,'a2':2.71488,'a3':6.41835,'a4':-0.042,'a5':-0.0108 }
    TBL2[12,7] = { 'delta': 1j*8.07005,'a1':13.2593,'a2':0.000000,'a3':8.07005,'a4':-0.021,'a5':0.000000 }
    TBL2[11,8] = { 'delta': 1j*3.74604,'a1':34.980,'a2':9.159,'a3':3.74569,'a4':-0.91,'a5':-0.32 }
    TBL2[12,8] = { 'delta': 1j*6.15394,'a1':19.6999,'a2':3.56155,'a3':6.15392,'a4':-0.084,'a5':-0.022 }
    TBL2[12,9] = { 'delta': 1j*2.70389,'a1':79.21,'a2':25.64,'a3':2.7015,'a4':-7.2,'a5':-3.20 }

    #
    if (l,m) in TBL1:

        # Unpack table
        a1 = TBL1[l,m]['a1']
        a2 = TBL1[l,m]['a2']
        a3 = TBL1[l,m]['a3']
        a4 = TBL1[l,m]['a4']

        # Evaluate Eq 63
        wr = 0.5*m - a1*sqrt( 0.5*eps ) + (a2 + a3*n)*eps
        wc = -( n+0.5 ) * ( sqrt(0.5*eps) - a4 * eps )

    elif (l,m) in TBL2:

        # Unpack table
        a1 = TBL2[l,m]['a1']
        a2 = TBL2[l,m]['a2']
        a3 = TBL2[l,m]['a3']
        a4 = TBL2[l,m]['a4']
        a5 = TBL2[l,m]['a5']

        # Evaluate Eq 64
        wr = 0.5*m + (a1+a2*n)*eps
        wc = -1.0 * ( a3 + n + 0.5 ) * sqrt( 0.5*eps ) + (a4+a5*n)*eps

    else:

        #
        error('unsupported multipole given (l,m) = %s'%str((l,m)))

    #
    cw = wr + 1j*wc


    #
    return cw


# Berti+'s 2005 fit for QNM frequencies
def Berti0512160(jf,l,m,n):

    '''
    Fit for Kerr -2 QNMs from Berti+ gr-qc/0512160.
    external data sourced from https://pages.jh.edu/~eberti2/ringdown/
    '''

    # Import usefuls
    from numpy import loadtxt,array,ndarray
    import positive

    # Load fit coefficients
    data_path = parent(os.path.realpath(__file__))+'/data/berti_kerrcw_fitcoeffsWEB.dat'
    data = loadtxt(data_path)

    # Unpack: l,m,n,f1,f2,f3,q1,q2,q3
    ll,mm,nn,ff1,ff2,ff3,qq1,qq2,qq3 = data.T

    #
    def lowlevel(JF,M):

        #
        if JF<0:
            M *= -1
            JF *= -1

        #
        k = (ll==l) & (mm==M) & (nn==n)

        #
        if not sum(k):
            error('this model does not include (l,m,n)=(%i,%i,%i)'%(l,m,n))

        #
        f1 = ff1[k]; f2 = ff2[k]; f3 = ff3[k]
        q1 = qq1[k]; q2 = qq2[k]; q3 = qq3[k]
        #
        wr = f1 + f2 * ( 1-JF )**f3
        Q  = q1 + q2 * ( 1-JF )**q3
        # See eqn 2.1 here https://arxiv.org/pdf/gr-qc/0512160.pdf
        wc = wr/(2*Q)
        #
        return wr+1j*wc

    #
    if isinstance(jf,(list,tuple,ndarray)):
        return array( [ lowlevel(j,m) for j in jf ] ).T[0]
    else:
        return lowlevel(jf,m)


# Fits for spin-2 QNM frequencies from arxiv:1810.03550
def cw181003550(jf,l,m,n):
    """Fit for quasi-normal mode frequencies

    Fit for the quasi-normal mode frequencies from
    https://arxiv.org/abs/1810.03550 for Kerr under -2 spin-weighted spheroidal
    harmonics. Fits are provided for the following indices

    |  l  |  m  |  n  |
    | --- | --- | --- |
    |  2  |  2  |  0  |
    |  2  | -2  |  0  |
    |  2  |  2  |  1  |
    |  2  | -2  |  1  |
    |  3  |  2  |  0  |
    |  3  | -2  |  0  |
    |  4  |  4  |  0  |
    |  4  | -4  |  0  |
    |  2  |  1  |  0  |
    |  2  | -1  |  0  |
    |  3  |  3  |  0  |
    |  3  | -3  |  0  |
    |  3  |  3  |  1  |
    |  3  | -3  |  1  |
    |  4  |  3  |  0  |
    |  4  | -3  |  0  |
    |  5  |  5  |  0  |
    |  5  | -5  |  0  |

    Parameters
    ----------
    jf: float
         Dimensionless final spin
    l: int
        Polar index
    m: int
        Azimuthal index
    n: int
        Overtone index

    Returns
    -------
    float
        Evaluation of the fit at `jf` for the requested Kerr -2 QNM
    """

    # Import usefuls
    from numpy import loadtxt,array,ndarray
    from imp import load_source
    import positive

    # Load fit functions from reference module
    module_path = parent(os.path.realpath(__file__))+'/data/ksm2_cw.py'
    cw_module = load_source( '', module_path )

    # Extract dict of fit functions
    fit_dictionary = cw_module.CW

    if (l,m,n) in fit_dictionary:
        #
        ans = fit_dictionary[l,m,n](jf)
    else:
        #
        error('this fit does not apply to (l,m,n)=(%i,%i,%i)'%(l,m,n))

    #
    return ans



# Fits for spin-2 spherical-spheroidal mixing coefficients from arxiv:1810.03550
def ysprod181003550(jf,ll,mm,lmn):
    """Fit for spherical-spheroidal mixing coefficients

    Fit for the spherical-spheroidal mixing coefficients from
    https://arxiv.org/abs/1810.03550 for Kerr under -2 spin-weighted harmonics.
    Fits are provided for the following spherical harmonic and Kerr indices

    | ll  | mm  | lmn       |
    | --- | --- | --------- |
    |  2  |  1  | (2, 1, 0) |
    |  2  |  2  | (2, 2, 0) |
    |  2  |  2  | (2, 2, 1) |
    |  3  |  2  | (2, 2, 0) |
    |  3  |  2  | (2, 2, 1) |
    |  3  |  2  | (3, 2, 0) |
    |  3  |  3  | (3, 3, 0) |
    |  3  |  3  | (3, 3, 1) |
    |  4  |  3  | (3, 3, 0) |
    |  4  |  3  | (4, 3, 0) |
    |  4  |  4  | (4, 4, 0) |
    |  5  |  5  | (5, 5, 0) |

    Parameters
    ----------
    jf: float
         Dimensionless final spin
    ll: int
        Spherical harmonic degree
    mm: int
        Spherical harmonic order
    lmn: list of int
        List of Kerr Indices [l, m, n]

    Returns
    -------
    float
        Evaluation of the fit at `jf` for the requested mixing coefficient
    """

    # Import usefuls
    from numpy import loadtxt,array,ndarray,log,exp
    from imp import load_source
    import positive

    # Extract dict of fit functions
    fit_dictionary = {
        (2, 1, 2, 1, 0): lambda x0: 0.9971577101354098*exp(6.2815088293225569j)  +  6.3541921214411894e-03 * (  143454.0616281430993695*exp(4.5060813273501665j)*(x0) + 354688.4750832330319099*exp(1.7326693927013841j)*(x0*x0) + 240378.2854125543963164*exp(5.1629102848140072j)*(x0*x0*x0) + 6026.9252628974973049*exp(1.8880583980908965j) ) / ( 1.0 +  73780.2159267508977791*exp(1.4129254015451644j)*(x0) + 97493.6172260692255804*exp(4.5395814919256727j)*(x0*x0) + 34814.9777696923483745*exp(1.4206733963391489j)*(x0*x0*x0) ),
        (2, 2, 2, 2, 0): lambda x0: 0.9973343021336419*exp(6.2813148178761091j)  +  7.5336160373579110e-03 * (  14.5923758203105400*exp(5.0600523238823607j)*(x0) + 28.7612830955165180*exp(1.6289978409506747j)*(x0*x0) + 14.5113507529443453*exp(4.6362218859245612j)*(x0*x0*x0) + 1.9624092507664150*exp(3.0112613110519790j) ) / ( 1.0 +  0.8867427721075676*exp(6.2202532788943463j)*(x0) + 1.0019792255065854*exp(3.2737011007062509j)*(x0*x0) + 0.0821482616116315*exp(2.4952790723412233j)*(x0*x0*x0) ),
        (2, 2, 2, 2, 1): lambda x0: 0.9968251006655715*exp(6.2782464977815033j)  +  2.0757805219598607e-02 * (  15.0768877264486765*exp(4.8322813296124032j)*(x0) + 31.1388349304345873*exp(1.5850449725805840j)*(x0*x0) + 15.4486406485143153*exp(4.6727325182881687j)*(x0*x0*x0) + 0.7189749887588365*exp(2.8084047359333271j) ) / ( 1.0 +  0.8059219252903900*exp(0.2579235302312085j)*(x0) + 0.6950198452267902*exp(3.6843441377638189j)*(x0*x0) + 0.3561345138627161*exp(2.8129259367413266j)*(x0*x0*x0) ),
        (3, 2, 2, 2, 0): lambda x0: 0.0205978660711896*exp(0.0474295298012882j)  +  6.9190162168113994e-02 * (  2.7656806568571959*exp(2.1329817160503861j)*(x0) + 3.9562158982293116*exp(4.6530163694526001j)*(x0*x0) + 2.3363698535756998*exp(2.6443646723682295j)*(x0*x0*x0) + 2.3989803671798935*exp(6.2766581209274568j) ) / ( 1.0 +  1.0595398565336638*exp(1.6448612531651137j)*(x0) + 0.9130774341264614*exp(6.0285976636462575j)*(x0*x0) + 0.6946816943215194*exp(3.3327882895176271j)*(x0*x0*x0) ),
        (3, 2, 2, 2, 1): lambda x0: 0.0220303770209119*exp(0.1645224015737339j)  +  7.3233344302596703e-02 * (  24.9322113499648026*exp(1.0180661306127561j)*(x0) + 30.1973220184149511*exp(4.4046911898286494j)*(x0*x0) + 11.2741216816767498*exp(2.9810158869949035j)*(x0*x0*x0) + 2.4373637035655071*exp(6.1958958829162727j) ) / ( 1.0 +  11.3967182799355697*exp(0.8537537359837979j)*(x0) + 10.9150545704939930*exp(2.6609572234291781j)*(x0*x0) + 7.2195768331341306*exp(4.9591911678862166j)*(x0*x0*x0) ),
        (3, 2, 3, 2, 0): lambda x0: 0.9900948849287589*exp(6.2804248687509565j)  +  2.3690082209346527e-02 * (  71893.0284688364044996*exp(1.2395199521970826j)*(x0) + 170547.7535011355357710*exp(5.0370601277314870j)*(x0*x0) + 129473.0632400779868476*exp(2.3589834727049306j)*(x0*x0*x0) + 1935.5063129042880519*exp(4.6680053884319301j) ) / ( 1.0 +  38206.4439771973629831*exp(4.3669921291251104j)*(x0) + 35811.0156501504534390*exp(0.8202376613805515j)*(x0*x0) + 8378.3365919880543515*exp(3.2588556618808240j)*(x0*x0*x0) ),
        (3, 3, 3, 3, 0): lambda x0: 0.9956935091939635*exp(6.2784823412948683j)  +  1.4545599733034410e-02 * (  7.2112377450866258*exp(0.6281060207724505j)*(x0) + 6.5381226130671095*exp(4.6215676022520444j)*(x0*x0) + 4.4510347535912222*exp(2.9227656614904154j)*(x0*x0*x0) + 1.7112774109660507*exp(2.9527107589222190j) ) / ( 1.0 +  1.4974006961584732*exp(4.8103032611120984j)*(x0) + 1.5287573244457504*exp(2.2468641698880383j)*(x0*x0) + 0.5211398892820160*exp(5.6886611584470419j)*(x0*x0*x0) ),
        (3, 3, 3, 3, 1): lambda x0: 0.9947799139299213*exp(6.2687860201480916j)  +  4.0478484180679702e-02 * (  4.4112734035486856*exp(1.2501463097148089j)*(x0) + 11.5876238172707406*exp(0.2795912119606259j)*(x0*x0) + 17.3218141087277253*exp(3.7903687208531087j)*(x0*x0*x0) + 0.6772374108565415*exp(2.5796853216144990j) ) / ( 1.0 +  3.8782075663390487*exp(5.4280212005550617j)*(x0) + 3.4912931569199648*exp(2.5238982607321545j)*(x0*x0) + 1.0367733448133998*exp(6.0498233292055152j)*(x0*x0*x0) ),
        (4, 3, 3, 3, 0): lambda x0: 0.0281121577247875*exp(0.0484878595517481j)  +  8.6382569875194229e-02 * (  12.0872895226366044*exp(0.4722122066619678j)*(x0) + 30.6262626401175417*exp(3.3281158362521670j)*(x0*x0) + 16.3281991298750100*exp(6.1785375072204518j)*(x0*x0*x0) + 2.3603267682389149*exp(6.2662351579714608j) ) / ( 1.0 +  4.9638141584463549*exp(0.4514701750473943j)*(x0) + 6.2552471990602712*exp(3.0584854159700825j)*(x0*x0) + 1.4538151679610098*exp(5.6955388697651710j)*(x0*x0*x0) ),
        (4, 3, 4, 3, 0): lambda x0: 0.9873523634400163*exp(6.2794650976486421j)  +  3.3027803522918557e-02 * (  700838.9044441945152357*exp(1.1066866283952814j)*(x0) + 1843013.1216291214805096*exp(4.8808349224255725j)*(x0*x0) + 1436658.3612134086433798*exp(2.1412051022385667j)*(x0*x0*x0) + 13844.4111501989991666*exp(4.5601015740698632j) ) / ( 1.0 +  356669.1344140542205423*exp(4.1565052819700004j)*(x0) + 327401.1637445305241272*exp(0.6329771391103556j)*(x0*x0) + 88620.8619731830985984*exp(3.2425422751663255j)*(x0*x0*x0) ),
        (4, 4, 4, 4, 0): lambda x0: 0.9947834747976544*exp(6.2775928790314088j)  +  2.4790925424940623e-02 * (  6.5171792863640983*exp(0.7983475243325006j)*(x0) + 7.7748196938039094*exp(4.2485120698165844j)*(x0*x0) + 1.1576799667402324*exp(1.5905452385036978j)*(x0*x0*x0) + 1.2434427129854981*exp(2.9615526265836647j) ) / ( 1.0 +  0.4454803253853161*exp(4.3912182331287690j)*(x0) + 0.5943671978440922*exp(2.5316559344513467j)*(x0*x0) + 0.2474336968120706*exp(5.9707578218053605j)*(x0*x0*x0) ),
        (5, 5, 5, 5, 0): lambda x0: 0.9943365401253647*exp(6.2773480319726191j)  +  3.1259875807999535e-02 * (  6.5508011783841358*exp(0.9339806071173702j)*(x0) + 8.0557862161937326*exp(4.2881187610320346j)*(x0*x0) + 0.9297117055414360*exp(1.0436436392517001j)*(x0*x0*x0) + 1.0903601095842754*exp(2.9711989367542477j) ) / ( 1.0 +  0.2312769649802813*exp(4.9081434508733581j)*(x0) + 0.5495791564905941*exp(2.7762081323140197j)*(x0*x0) + 0.2129999495217834*exp(6.1508375766326919j)*(x0*x0*x0) )
    }

    # Extract kerr indices
    l,m,n = lmn

    # Eval fit
    if (ll,mm,l,m,n) in fit_dictionary:
        #
        beta = 1.0 / ( 2 + ll-abs(mm) )
        kappa = lambda JF: (log( 2 - JF ) / log(3))**(beta)
        return fit_dictionary[ll,mm,l,m,n]( kappa(jf) )
    else:
        #
        error('this fit does not apply to (ll,mm,l,m,n)=(%i,%i,%i)'%(ll,mm,l,m,n))


