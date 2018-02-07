#
from positive import *
from positive.learning import *

#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
# Here are some phenomenological fits used in PhenomD                               #
#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#


# Formula to predict the final spin. Equation 3.6 arXiv:1508.07250
# s is defined around Equation 3.6.
''' Copied from LALSimulation Version '''
def FinalSpin0815_s(eta,s):
    eta = round(eta,8)
    eta2 = eta*eta
    eta3 = eta2*eta
    eta4 = eta3*eta
    s2 = s*s
    s3 = s2*s
    s4 = s3*s
    return 3.4641016151377544*eta - 4.399247300629289*eta2 +\
    9.397292189321194*eta3 - 13.180949901606242*eta4 +\
    (1 - 0.0850917821418767*eta - 5.837029316602263*eta2)*s +\
    (0.1014665242971878*eta - 2.0967746996832157*eta2)*s2 +\
    (-1.3546806617824356*eta + 4.108962025369336*eta2)*s3 +\
    (-0.8676969352555539*eta + 2.064046835273906*eta2)*s4


#Wrapper function for FinalSpin0815_s.
''' Copied from LALSimulation Version '''
def FinalSpin0815(eta,chi1,chi2):
    from numpy import sqrt
    eta = round(eta,8)
    if eta>0.25:
        error('symmetric mass ratio greater than 0.25 input')
    # Convention m1 >= m2
    Seta = sqrt(abs(1.0 - 4.0*float(eta)))
    m1 = 0.5 * (1.0 + Seta)
    m2 = 0.5 * (1.0 - Seta)
    m1s = m1*m1
    m2s = m2*m2
    # s defined around Equation 3.6 arXiv:1508.07250
    s = (m1s * chi1 + m2s * chi2)
    return FinalSpin0815_s(eta, s)


# Formula to predict the total radiated energy. Equation 3.7 and 3.8 arXiv:1508.07250
# Input parameter s defined around Equation 3.7 and 3.8.
def EradRational0815_s(eta,s):
    eta = round(eta,8)
    eta2 = eta*eta
    eta3 = eta2*eta
    eta4 = eta3*eta
    return ((0.055974469826360077*eta + 0.5809510763115132*eta2 - 0.9606726679372312*eta3 + 3.352411249771192*eta4)*\
    (1. + (-0.0030302335878845507 - 2.0066110851351073*eta + 7.7050567802399215*eta2)*s))/(1. + (-0.6714403054720589 \
    - 1.4756929437702908*eta + 7.304676214885011*eta2)*s)


# Wrapper function for EradRational0815_s.
def EradRational0815(eta, chi1, chi2):
    from numpy import sqrt,round
    eta = round(eta,8)
    if eta>0.25:
        error('symmetric mass ratio greater than 0.25 input')
    # Convention m1 >= m2
    Seta = sqrt(1.0 - 4.0*eta)
    m1 = 0.5 * (1.0 + Seta)
    m2 = 0.5 * (1.0 - Seta)
    m1s = m1*m1
    m2s = m2*m2
    # arXiv:1508.07250
    s = (m1s * chi1 + m2s * chi2) / (m1s + m2s)
    return EradRational0815_s(eta,s)


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
    z1 = 1.+(1.-a**2.)**(1./3)*((1.+a)**(1./3) + (1.-a)**(1./3))
    z2 = np.sqrt(3 * a**2 + z1**2)
    a_sign = np.sign(a)
    return 3+z2 - np.sqrt((3.-z1)*(3.+z1+2.*z2))*a_sign


# https://arxiv.org/pdf/1406.7295.pdf
def Mf14067295( m1,m2,chi1,chi2,chif=None ):

    import numpy as np

    if np.any(abs(chi1>1)):
      raise ValueError("chi1 has to be in [-1, 1]")
    if np.any(abs(chi2>1)):
      raise ValueError("chi2 has to be in [-1, 1]")


    # Swapping inputs to conform to fit conventions
    # NOTE: See page 2 of https://arxiv.org/pdf/1406.7295.pdf
    if m1>m2:
        #
        m1_,m2_ = m1,m2
        chi1_,chi2_ = chi1,chi2
        #
        m1,m2 = m2_,m1_
        chi1,chi2 = chi2_,chi1_

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
    # Test for m2>m2 convention
    if m1<m2:
        # Swap everything
        m1_,m2_ = m2 ,m1 ;  chi1_,chi2_ =  chi2,chi1
        m1,m2   = m1_,m2_;  chi1,chi2   = chi1_,chi2_
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
    # Test for m2>m2 convention
    if m1<m2:
        # Swap everything
        m1_,m2_ = m2 ,m1 ;  chi1_,chi2_ =  chi2,chi1
        m1,m2   = m1_,m2_;  chi1,chi2   = chi1_,chi2_
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

# High level function for calculating remant mass and spin
def remnant(m1,m2,chi1,chi2,arxiv=None,verbose=False):
    '''
    High level function for calculating remant mass and spin for nonprecessing BBH systems.

    Available arxiv ids are:
    * 1611.00332 by Jimenez et. al.
    * 1406.7295 by Healy et. al.

    This function automatically imposes m1,m2 conventions.

    spxll'17
    '''

    #
    if arxiv in ('1611.00332',161100332,None):
        if verbose: alert('Using method from arxiv:1611.00332 by Jimenez et. al.')
        Mf = Mf161100332(m1,m2,chi1,chi2)
        jf = jf161100332(m1,m2,chi1,chi2)
    else:
        if verbose:
            alert('Using method from arxiv:1406.7295 by Healy et. al.')
            warning('This method is slow [af]. Please consider using another one.')
        Mf = Mf14067295(m1,m2,chi1,chi2)
        jf = jf14067295(m1,m2,chi1,chi2)

    # Return answer
    ans = (Mf,jf)
    return ans

#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
# Post-Newtonian methods
#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#


# PN estimate for orbital frequency
def pnw0(m1,m2,D=10.0):
    # https://arxiv.org/pdf/1310.1528v4.pdf
    # Equation 228
    # 2nd Reference: arxiv:0710.0614v1
    # NOTE: this outputs orbital frequency
    from numpy import sqrt,zeros,pi,array,sum
    #
    G = 1.0
    c = 1.0
    r = float(D)
    M = float( m1+m2 )
    v = m1*m2/( M**2 )
    gamma = G*M/(r*c*c)     # Eqn. 225
    #
    trm = zeros((4,))
    #
    trm[0] = 1.0
    trm[1] = v - 3.0
    trm[2] = 6 + v*41.0/4.0 + v*v
    trm[3] = -10.0 + v*( -75707.0/840.0 + pi*pi*41.0/64.0 ) + 19.0*0.5*v*v + v*v*v
    #
    w0 = sqrt( (G*M/(r*r*r)) * sum( array([ term*(gamma**k) for k,term in enumerate(trm) ]) ) )

    #
    return w0


#
def mishra( f, m1,m2, X1,X2, lm,    # Intrensic parameters and l,m
            lnhat   = None,         # unit direction of orbital angular momentum
            vmax    = 10,           # maximum power (integer) allwed for V parameter
            leading_order = False,  # Toggle for using only leading order behavior
            verbose = False ):      # toggle for letting the people know
    '''
    PN formulas from "Ready-to-use post-Newtonian gravitational waveforms for binary black holes with non-precessing spins: An update"
    *   https://arxiv.org/pdf/1601.05588.pdf
    *   https://arxiv.org/pdf/0810.5336.pdf
    '''

    # Import usefuls
    from numpy import pi,array,dot,sqrt,log,inf,ones

    # Handle zero values
    f[f==0] = 1e-5 * min(abs(f))

    #
    l,m = lm

    #
    M = m1+m2

    # Normalied mass difference
    delta = (m1-m2)/(m1+m2)

    # Symmetric mass ratio
    eta = float(m1)*m2/(M*M)

    # Frequency parameter (note that this is the same as the paper's \nu)
    V = lambda m: pow(2.0*pi*M*f/m,1.0/3)

    # Here we handle the vmax input using a lambda function
    if vmax is None: vmax = inf
    e = ones(12)        # This identifier will be used to turn off terms
    e[(vmax+1):] = 0    # NOTE that e will effectively be indexed starting
                        # from 1 not 0, so it must have more elements than needed.


    # Handle default behavior for
    lnhat = array([0,0,1]) if lnhat is None else lnhat

    # Symmetric and Anti-symmettric spins
    Xs = 0.5 * ( X1+X2 )
    Xa = 0.5 * ( X1-X2 )

    #
    U = not leading_order

    # Dictionary for FD multipole terms (eq. 12)
    H = {}

    #
    H[2,2] = lambda v: -1 + U*( v**2  * e[2] *  ( (323.0/224.0)-(eta*451.0/168.0) ) \
                          + v**3  * e[3] *  ( -(27.0/8)*delta*dot(Xa,lnhat) + dot(Xs,lnhat)*((-27.0/8)+(eta*11.0/6)) ) \
                          + v**4  * e[4] *  ( (27312085.0/8128512)+(eta*1975055.0/338688) - (105271.0/24192)*eta*eta + dot(Xa,lnhat)**2 * ((113.0/32)-eta*14) + delta*(113.0/16)*dot(Xa,lnhat)*dot(Xs,lnhat) + dot(Xs,lnhat)**2 * ((113.0/32) - (eta/8)) ) )

    #
    H[2,1] = lambda v: -(sqrt(2)/3) * ( v    * delta \
                                      + U*(- v**2 * e[2] * 1.5*( dot(Xa,lnhat)+delta*dot(Xs,lnhat) ) \
                                      + v**3 * e[3] * delta*( (335.0/672)+(eta*117.0/56) ) \
                                      + v**4 * e[4] * ( dot(Xa,lnhat)*(4771.0/1344 - eta*11941.0/336) + delta*dot(Xs,lnhat)*(4771.0/1344 - eta*2549.0/336) + delta*(-1j*0.5-pi-2*1j*log(2)) ) \
                                      ))

    #
    H[3,3] = lambda v: -0.75*sqrt(5.0/7) \
                        *(v           * delta \
                        + U*( v**3 * e[3] * delta * ( -(1945.0/672) + eta*(27.0/8) )\
                        + v**4 * e[4] * ( dot(Xa,lnhat)*( (161.0/24) - eta*(85.0/3) ) + delta*dot(Xs,lnhat)*( (161.0/24) - eta*(17.0/3) ) + delta*(-1j*21.0/5 + pi + 6j*log(3.0/2)) ) \
                        ))

    #
    H[3,2] = lambda v: -(1.0/3)*sqrt(5.0/7) * (\
                        v**2   * e[2] * (1-3*eta) \
                        + U*( v**3 * e[3] * 4*eta*dot(Xs,lnhat) \
                        + v**4 * e[4] * (-10471.0/10080 + eta*12325.0/2016 - eta*eta*589.0/72) \
                        ))

    #
    H[4,4] = lambda v: -(4.0/9)*sqrt(10.0/7) \
                        * ( v**2 * e[2] * (1-3*eta) \
                        +   U*(v**4 * e[4] * (-158383.0/36960 + eta*128221.0/7392 - eta*eta*1063.0/88) \
                        ))

    #
    H[4,3] = lambda v: -(3.0/4)*sqrt(3.0/35) * (
                        v**3 * e[3] * delta*(1-2*eta) \
                        + U*v**4 * e[4] * (5.0/2)*eta*( dot(Xa,lnhat) - delta*dot(Xs,lnhat) )\
                        )

    #
    hlm_amp = M*M*pi * sqrt(eta*2.0/3)*(V(m)**-3.5) * H[l,m]( V(m) )

    #
    return abs(hlm_amp)


# leading order amplitudes in freq fd strain via spa
def lamp_spa(f,eta,lm=(2,2)):
    # freq domain amplitude from leading order in f SPA
    # made using ll-LeadingOrderAmplitudes.nb in PhenomHM repo
    from numpy import pi,sqrt

    #
    warning('This function has a bug related the a MMA bug of unknown origin -- ampliutdes are off by order 1 factors!!')

    # Handle zero values
    f[f==0] = 1e-5 * min(abs(f))

    #
    hf = {}
    #
    hf[2,2] = (sqrt(0.6666666666666666)*sqrt(eta))/(f**1.1666666666666667*pi**0.16666666666666666)
    #
    hf[2,1] = (sqrt(0.6666666666666666)*sqrt(eta - 4*eta**2)*pi**0.16666666666666666)/(3.*f**0.8333333333333334)
    #
    hf[3,3] = (3*sqrt(0.7142857142857143)*sqrt(eta - 4*eta**2)*pi**0.16666666666666666)/(4.*f**0.8333333333333334)
    #
    hf[3,2] = (sqrt(0.47619047619047616)*eta*sqrt(pi))/(3.*sqrt(eta*f)) - (sqrt(0.47619047619047616)*eta**2*sqrt(pi))/sqrt(eta*f)
    #
    hf[3,1] = (sqrt(eta - 4*eta**2)*pi**0.16666666666666666)/(12.*sqrt(21)*f**0.8333333333333334)
    #
    hf[4,4] = (8*sqrt(0.47619047619047616)*eta*sqrt(pi))/(9.*sqrt(eta*f)) - (8*sqrt(0.47619047619047616)*eta**2*sqrt(pi))/(3.*sqrt(eta*f))
    #
    hf[4,3] = (3*sqrt(0.08571428571428572)*sqrt(eta - 4*eta**2)*pi**0.8333333333333334)/(4.*f**0.16666666666666666) - (3*sqrt(0.08571428571428572)*eta*sqrt(eta - 4*eta**2)*pi**0.8333333333333334)/(2.*f**0.16666666666666666)
    #
    hf[4,2] = (sqrt(3.3333333333333335)*eta*sqrt(pi))/(63.*sqrt(eta*f)) - (sqrt(3.3333333333333335)*eta**2*sqrt(pi))/(21.*sqrt(eta*f))
    #
    hf[4,1] = (sqrt(eta - 4*eta**2)*pi**0.8333333333333334)/(84.*sqrt(15)*f**0.16666666666666666) - (eta*sqrt(eta - 4*eta**2)*pi**0.8333333333333334)/(42.*sqrt(15)*f**0.16666666666666666)
    #
    hf[5,5] = (625*sqrt(eta - 4*eta**2)*pi**0.8333333333333334)/(288.*sqrt(11)*f**0.16666666666666666) - (625*eta*sqrt(eta - 4*eta**2)*pi**0.8333333333333334)/(144.*sqrt(11)*f**0.16666666666666666)
    #
    hf[6,6] = 3.6
    #
    return hf[lm]


# Calculate the Center of Mass Energy for a Binary Source
def pn_com_energy(f,m1,m2,X1,X2,L=None):
    '''
    Calculate the Center of Mass Energy for a Binary Source

    Primary Refernce: https://arxiv.org/pdf/0810.5336.pdf
        * Eq. 6.18, C1-C6
    '''

    # Import usefuls
    from numpy import pi,array,dot,ndarray,sqrt

    #~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
    # Validate inputs
    #~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
    if L is None: L = array([0,0,1.0])
    # Handle Spin 1
    if not isinstance(X1,ndarray):
        error('X1 input must be array')
    elif len(X1)<3:
        error('X1 must be array of length 3; the length is %i'%len(X1))
    else:
        X1 = array(X1)
    # Handle Xpin 2
    if not isinstance(X2,ndarray):
        error('X2 input must be array')
    elif len(X2)<3:
        error('X2 must be array of length 3; the length is %i'%len(X2))
    else:
        X2 = array(X2)

    #~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
    # Define low level parameters
    #~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#

    # Total mass
    M = m1+m2
    # Symmetric Mass ratio
    eta = m1*m2 / (M**2)
    delta = sqrt(1-4*eta)
    #
    Xs = 0.5 * ( X1+X2 )
    Xa = 0.5 * ( X1-X2 )
    #
    xs = dot(Xs,L)
    xa = dot(Xa,L)
    # PN frequency parameter
    v = ( 2*pi*M*f ) ** 1.0/3
    #
    Enewt = - 0.5 * M * eta

    # List term coefficients
    e2 = - 0.75 - (1.0/12)*eta
    e3 = (8.0/3 - 4.0/3*eta) * xs + (8.0/3)*delta*xa
    e4 = -27.0/8 + 19.0/8*eta - eta*eta/24 \
         + eta* ( (dot(Xs,Xs)-dot(Xa,Xa))-3*(xs*xs-xa*xa) ) \
         + (0.5-eta)*( dot(Xs,Xs)+dot(Xa,Xa)-3*(xs*xs+xa*xa) ) \
         + delta*( dot(Xs,Xa)-3*( xs*xa ) )
    e5 = xs*(8-eta*121.0/9 + eta*eta*2.0/9) + delta*xa*(8-eta*31.0/9)
    e6 = -675.0/64 + (34445.0/576 - pi*pi*205.0/96)*eta - eta*eta*155.0/96 - 35.0*eta*eta*eta/5184
    e = [e2,e3,e4,e5,e6]

    #
    E = Enewt * v * v * ( 1.0 + sum( [ ek*(v**(k+2)) for k,ek in enumerate(e) ] ) )

    #
    ans = E
    return ans


# Calculate the Center of Mass Energy for a Binary Source
def pn_com_energy_flux(f,m1,m2,X1,X2,L=None):
    '''
    Calculate the Energy Flux for a Binary Source

    Primary Refernce: https://arxiv.org/pdf/0810.5336.pdf
        * Eq. 6.19, C7-C13
    '''

    # Import usefuls
    from numpy import pi,pow,array,dot,ndarray

    #~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
    # Validate inputs
    #~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
    if L is None: L = array([0,0,1.0])
    # Handle Spin 1
    if not isinstance(X1,ndarray):
        error('X1 input must be array')
    elif len(X1)<3:
        error('X1 must be array of length 3; the length is %i'%len(X1))
    else:
        X1 = array(X1)
    # Handle Xpin 2
    if not isinstance(X2,ndarray):
        error('X2 input must be array')
    elif len(X2)<3:
        error('X2 must be array of length 3; the length is %i'%len(X2))
    else:
        X2 = array(X2)

    #~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
    # Define low level parameters
    #~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#

    # Total mass
    M = m1+m2
    # Symmetric Mass ratio
    eta = m1*m2 / (M**2)
    delta = sqrt(1-4*eta)
    #
    Xs = 0.5 * ( X1+X2 )
    Xa = 0.5 * ( X1-X2 )
    #
    xs = dot(Xs,L)
    xa = dot(Xa,L)
    # PN frequency parameter
    v = ( 2*pi*M*f ) ** (1.0/3)
    #
    Fnewt = (32.0/5)*eta*eta

    # List term coefficients
    f2 = -(1247.0/336)-eta*35/12
    f3 = 4*pi - ( (11.0/4-3*eta)*xs + 11.0*delta*xa/4 )
    f4 = -44711.0/9072 + eta*9271.0/504 + eta*eta*65.0/18 + (287.0/96 + eta/24)*xs*xs \
         - dot(Xs,Xs)*(89.0/96 + eta*7/24) + xa*xa*(287.0/96-12*eta) + (4*eta-89.0/96)*dot(Xa,Xa) \
         + delta*287.0*xs*xa/48 - delta*dot(Xs,Xa)*89.0/48
    f5 = -pi*( eta*583.0/24 + 8191.0/672 ) + ( xs*(-59.0/16 + eta*227.0/9 - eta*eta*157.0/9) + delta*xa*(eta*701.0/36 - 59.0/16) )
    f6 = 6643739519.0/69854400 + pi*pi*16.0/3 - 1712.0*GammaE/105 - log(16*v*v)*856.0/105 + ( -134543.0/7776 + 41.0*pi*pi/48 )*eta - eta*eta*94403.0/3024 - eta*eta*eta*775.0/324
    f7 = pi*( -16285/504 + eta*214745.0/1728 + eta*eta*193385.0/3024 )
    f = [f2,f3,f4,f5,f6,f7]

    #
    F = Fnewt * (v**10) * ( 1.0 + sum( [ ek*(v**(k+2)) for k,fk in enumerate(f) ] ) )

    #
    ans = F
    return ans


#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
# Class for TalyorT4 + Spin Post-Newtonian
#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#

#
class pn:

    '''

    High-level class for evaluation of PN waveforms.

    Key references:
    * https://arxiv.org/pdf/0810.5336.pdf

    '''

    # Class constructor
    def __init__( this,             # The current object
                  m1,               # The mass of the larger object
                  m2,               # The mass of the smaller object
                  X1,               # The dimensionless spin of the larger object
                  X2,               # The dimensionless spin of the smaller object
                  wM_min = 0.003,
                  wM_max = 0.18,
                  Lhat = None,      # Unit direction of initial orbital angular momentum
                  verbose=True):    # Be verbose toggle

        # Let the people know that we are here
        if verbose: alert('Now constructing instance of the pn class.','pn')

        # Apply the validative constructor to the inputs and the current object
        this.__validative_constructor__(m1,m2,X1,X2,wM_min,wM_max,Lhat,verbose)

        # Calculate the orbital frequency
        this.__calc_orbital_frequency__()

        # Calculate COM energy
        this.__calc_com_binding_energy__()

        # Calculate system total angular momentum
        this.__calc_total_angular_momentum__()

        # Calculate time domain waveforms
        this.__calc_h_of_t__()

        # Let the people know
        # warning('Note that the calculation of waveforms has not been implemented.','pn')

        #
        return None


    # Calculate all implemented strain time domain waveforms
    def __calc_h_of_t__(this):

        #
        this.h = {}
        #
        for l,m in this.lmlist:
            this.h[l,m] = this.__calc_hlm_of_t__(l,m)

    # Calcilate a single implmented time domain waveform
    def __calc_hlm_of_t__(this,l,m):

        #
        from numpy import pi,log,sqrt,exp

        # Short-hand
        x = this.x
        eta = this.eta

        # l,m  = 2,2
        if (l,m) == (2,2):

            #
            if this.verbose:
                alert('Calculating the (l,m)=(%i,%i) spherical multipole.'%(l,m))

            #
            part2 = 1                                                                + \
                    x     *(-107.0/42 + 55.0/42*eta)                                   + \
                    x**1.5*(2*pi)		                                         + \
                    x**2  *(-2173.0/1512 - 1069.0/216*eta + 2047.0/1512*eta**2)           + \
                    x**2.5*((-107.0/21+34.0/21*eta)*pi - 24*1j*eta)       		 + \
                    x**3  *(27027409.0/646800 + 2.0/3*pi**2                             + \
                              -856.0/105*this.__gamma_E__ - 1712.0/105*log(2) - 428.0/105*log(x)   + \
                              -(278185.0/33264-41.0/96*pi**2)*eta - 20261.0/2772*eta**2      + \
                              114635.0/99792*eta**3 + 428.0*1j*pi/105)
            #
            spin   = x**(1.5) * (-4*this.delta*this.xa/3 + 4/3*(eta-1)*this.xs - 2*eta*x**(0.5)*(this.xa**2 - this.xs**2))
            #
            part1 = sqrt(16.0*pi/5) * 2*eta*this.M*x
            #
            h = part1 * exp(-1j*m*this.phi)*(part2 + spin)

        elif (l,m) == (3,2):

            #
            if this.verbose:
                alert('Calculating the (l,m)=(%i,%i) spherical multipole.'%(l,m))

            #
            part2 =  x     *(1- 3*eta) + \
            x**2.0*(-193.0/90 + 145.0*eta/18 - (73.0*eta**2)/18) + \
            x**2.5*(2*pi*(1-3*eta) - 3*1j + 66*1j*eta/5) + \
            x**3.0*(-1451.0/3960 - 17387.0*eta/3960 + (5557.0*eta**2)/220 - (5341.0*eta**3)/1320)

            part1 = sqrt(16*pi/5) * 2*eta*this.M*x * 1.0/3*sqrt(5.0/7)

            #
            h = exp(-1j*m*this.phi)* ( part1*part2 + 32.0/3*sqrt(pi/7)*(eta**2)*this.xs*(x**2.5) )

        else:
            #
            error( '(l,m) = (%i,%i) not implemented'%(l,m) )

        #
        return h


    # Calculate the orbital frequency of the binary source
    def __calc_orbital_frequency__(this):

        # Import usefuls
        from numpy import mod, array, pi

        # Let the people know
        if this.verbose:
            alert('Calculating evolution of orbital phase using RK4 steps.')

        #
        _wM = this.wM[-1]  # NOTE that M referes to intial system mass
        k = 0
        while _wM < this.wM_max :  # NOTE that M referes to intial system mass

            # NOTE that rk4_step is defined positive.learning
            this.state = rk4_step( this.state, this.t[-1], this.dt, this.__taylort4rhs__ )

            #
            this.t.append( this.t[-1]+this.dt )
            this.phi.append( this.state[0] )
            this.x.append(   this.state[1] )
            _wM = this.state[1] ** 1.5  # NOTE that M referes to intial system mass
            this.dt = 0.00009 * this.dtfac * this.M / ( _wM ** 3 )  # NOTE that M referes to intial system mass
            this.wM.append( _wM )

        # Convert quantities to array
        this.wM = array( this.wM )
        this.x = array( this.x )
        this.phi = array( this.phi )
        this.t = array( this.t )
        # Calculate related quantities
        this.w = this.wM / this.M
        this.fM = this.wM / ( 2*pi )
        this.f = this.w / ( 2*pi )
        this.v = this.wM ** (1.0/3)

        #
        return None


    # Calculate COM binding energy
    def __calc_com_binding_energy__(this):
        #
        alert('Calculating COM binding energy')
        this.E = pn_com_energy(this.f,this.m1,this.m2,this.X1,this.X2,this.Lhat)
        this.remnant['M'] = this.M+this.E


    # Calculate system angular momentum
    def __calc_total_angular_momentum__(this):
        '''
        Non-precessing
        '''

        # Import usefuls
        from numpy import sqrt,pi,log

        #
        if abs(this.x1)+abs(this.x2) > 0:
            warning('This function currently only works with non-spinning systems. See 1310.1528.')

        # Short-hand
        x = this.x
        eta = this.eta

        # Equation 234 of https://arxiv.org/pdf/1310.1528.pdf
        mu = this.eta * this.M
        e4 = -123671.0/5760 + pi*pi*9037.0/1536 + 1792*log(2)/15 + 896*this.__gamma_E__/15 \
             + eta*( pi*pi*3157.0/576 - 498449.0/3456 ) \
             + eta**2 * 301.0/1728 \
             + eta**3 * 77.0/31104
        j4 = -e4*5.0/7 + 64.9/35
        L = ( mu   * this.M / sqrt(x) ) * ( 1 \
            + x    * (1.5 + eta/6)
            + x*x  * ( 27.0/8 - eta*19.0/8 + eta*eta/24 ) \
            + x**3 * ( 135.0/16 + eta*( 41*pi*pi/24 - 6889.0/144 ) + eta*eta*31.0/24 + eta**3 * 7.0/1296 ) \
            + x**4 * ( 2835.0/128 + eta*j4 - 64.0*eta*log(x)/3 ) \
            )

        #
        S1 = this.x1*(this.m1**2)
        S2 = this.x2*(this.m2**2)
        Jz = L + S1 + S2

        # Store the information to the current object
        this.remnant['J'] = Jz


    # Method for calculating the RHS of the TaylorT4 first order ODEs for pn parameter x and frequency
    def __taylort4rhs__(this,state,time,**kwargs):

        # Import usefuls
        from numpy import array, pi, log, array

        # Unpack the state
        phi,x = state
        # * phi, Phase with 2*pi*f = dphi/dt where phi is the GW phase
        # *   x, PN parameter, function of frequency; v = (2*pi*M*f/m)**1/3 = x**0.5 (see e.g. https://arxiv.org/pdf/1601.05588.pdf)

        #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~#
        # Calculate useful parameters from current object
        #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~#

        # Mass ratio
        eta = this.eta
        # Mass difference
        delta = this.delta
        # Total INITIAL mass
        M = this.M
        # Get Euler Gamma
        gamma_E = this.__gamma_E__

        # Spins
        # NOTE that components along angular momentum are used below; see validative constructor
        X1 = this.x1
        X2 = this.x2
        Xs = 0.5 * ( X1+X2 )
        Xa = 0.5 * ( X1-X2 )

        #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~#
        # Calculate PN terms for x RHS
        #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~#

        # Nonspinning terms from 0907.0700 Eqn. 3.6
        Non_SpinningTerms = 1 - (743.0/336 + 11.0/4*eta)*x + 4*pi*x**1.5 \
                           + (34103.0/18144 + 13661.0/2016*eta + 59.0/18*eta**2)*x**2 \
                           + (4159.0/672 + 189.0/8*eta)*pi*x**2.5 + (16447322263.0/139708800 \
                           - 1712.0/105*gamma_E - 56198689.0/217728*eta +  541.0/896*eta**2 \
                           - 5605.0/2592*eta**3 + pi*pi/48*(256+451*eta) \
                           - 856.0/105*log(16*x))*x**3 + (-4415.0/4032 + 358675.0/6048*eta \
                           + 91495.0/1512*eta**2)*pi*x**3.5

        # Spinning terms from 0810.5336 and 0605140v4 using T4 expansion of dx/dt = -F(v)/E'(v)
        S3  = 4.0/3*(2-eta)*Xs + 8.0/3*delta*Xa
        S4  = eta*(2*Xa**2 - 2*Xs**2) + (1.0/2 - eta)*(-2*Xs**2 \
             - 2*Xa**2) + delta*(-2*Xs*Xa)
        S5  = (8 -121*eta/9 + 2*eta**2/9)*Xs + (8-31*eta/9)*delta*Xa
        SF3 = (-11.0/4 + 3*eta)*Xs - 11.0/4*delta*Xa
        SF4 = (33.0/16 -eta/4)*Xs**2 + (33.0/16 - 8*eta)*Xa**2 + 33*delta*Xs*Xa/8
        SF5 = (-59.0/16 + 227.0*eta/9 - 157.0*eta**2/9)*Xs + (-59.0/16 + 701.0*eta/36)*delta*Xa
        SpinningTerms = (-5.0*S3/2 + SF3) *x**1.5 + (-3*S4+ SF4)*x**2.0 \
                        + ( 5.0/672*(239+868*eta)*S3 - 7*S5/2 + (3.0/2 + eta/6)*SF3 \
                        + SF5 )*x**2.5	+ ( (239.0/112 + 31*eta/4)*S4 \
                        + 5*S3/4*(-8*pi+5*S3-2*SF3) + (3/2 + eta/6)*SF4) *x**3.0 \
                        + ( -3*S4*(4*pi+SF3) - 5*S3/18144*(99226+9*eta*(-4377	\
                        + 2966*eta)	+ -54432*S4 + 9072*SF4 ) \
                        + 1.0/288*( 3*(239+868*eta)*S5+4*(891+eta*(-477+11*eta))*SF3\
                        + 48*(9+eta)*SF5))*x**3.5

        #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~#
        # Calculate derivatives
        #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~#

        # Invert the definition of x( f = dphi_dt )
        dphi_dt = x**(1.5) / M
        # e.g. Equation 36 of https://arxiv.org/pdf/0907.0700.pdf
        # NOTE the differing conventions used here vs the ref above
        dx_dt   = (64 * eta / 5*M) * (x**5) * (Non_SpinningTerms+SpinningTerms)
        # Compile the state's derivative
        state_derivative = array( [ dphi_dt, dx_dt ] )

        # Return derivatives
        return state_derivative


    # Validate constructor inputs
    def __validative_constructor__(this,m1,m2,X1,X2,wM_min,wM_max,Lhat,verbose):

        # Import usefuls
        from numpy import ndarray,dot,array,sqrt
        from mpmath import euler as gamma_E

        # Masses must be float
        if not isinstance(m1,(float,int)):
            error('m1 input must be float or double, instead it is %s'%yellow(type(m1).__name__))
        if not isinstance(m2,(float,int)):
            error('m2 input must be float or double, instead it is %s'%yellow(type(m2).__name__))

        # Spins must be iterable of floats
        if len(X1) != 3:
            error( 'Length of X1 must be 3, but it is %i'%len(X1) )
        if len(X2) != 3:
            error( 'Length of X2 must be 3, but it is %i'%len(X2) )

        # Spins must be numpy arrays
        if not isinstance(X1,ndarray):
            X1 = array(X1)
        if not isinstance(X2,ndarray):
            X2 = array(X2)

        # By default it will be assumed that Lhat is in the z direction
        if Lhat is None: Lhat = array([0,0,1.0])

        # Let the people know
        this.verbose = verbose
        if this.verbose:
            alert('Defining the initial binary state based on inputs.')

        # Rescale masses to unit total
        M = float(m1+m2); m1 = float(m1)/M; m2 = float(m2)/M
        if this.verbose: alert('Rescaling masses so that %s'%green('m1+m2=1'))

        # Store core inputs as well as simply derived quantities to the current object
        this.m1 = m1
        this.m2 = m2
        this.M = m1+m2  # Here M referes to intial system mass
        this.eta = m1*m2 / this.M
        this.delta = sqrt( 1-4*this.eta )
        this.X1 = X1
        this.X2 = X2
        this.Lhat = Lhat
        this.x1 = dot(X1,Lhat)
        this.x2 = dot(X2,Lhat)
        this.xs = (this.x1+this.x2)*0.5
        this.xa = (this.x1-this.x2)*0.5
        this.__gamma_E__ = float(gamma_E)
        # Bag for internal system quantities (remnant after radiated, not final)
        this.remnant = {}

        # Define initial binary state based on inputs
        this.t = [0]
        this.phi = [0]
        this.x  = [ wM_min**(2.0/3) ]
        this.wM = [ wM_min ]  # Orbital Frequency; NOTE that M referes to intial system mass
        this.wM_max = wM_max  # Here M referes to intial system mass
        this.wM_min = wM_min  # Here M referes to intial system mass
        this.initial_state = array([this.phi[-1],this.x[-1]])
        this.state = this.initial_state
        this.dtfac = 0.5
        this.dt = 0.00009*this.dtfac*this.M/(wM_min**3)

        # Binding energy
        this.E = [0]

        # Store a list of implemented l,m cases for waveform generation
        this.lmlist = [ (2,2), (3,2) ]
