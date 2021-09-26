#
from __future__ import print_function
from positive.api import *
from positive.plotting import *
from positive.learning import *
# # > > > > > > > > >  Import adjacent modules  > > > > > > > > > > #
# import positive
# modules = list( basename(f)[:-3] for f in glob.glob(dirname(__file__)+"/*.py") if (not ('__init__.py' in f)) and (not (__file__.split('.')[0] in f)) )
# for module in modules:
#     exec('from .%s import *' % module)
# # > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > #


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
            __next_to_leading_32__=False,
            term_array = None,
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
    
    # Allow user to manually toggle terms 
    if term_array:
        e = term_array


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
    from numpy import pi,pow,array,dot,ndarray,log

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
                  sceo = None,        # scentry object for gwylm conversion
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

        # Use strain waveforms to calculate psi4 waveforms
        alert('Calculating psi4 and news from strain.')
        this.__calc_psi4_and_news_of_t__()

        # Make gwylm representation
        if sceo: this.__to_gwylmo__(sceo)

        # Let the people know
        # warning('Note that the calculation of waveforms has not been implemented.','pn')

        #
        return None


    # Convert waveform information into gwylmo
    def __to_gwylmo__(this,
                      sceo): # instance of scentry class from nrutils

        '''
        This function takes in an instance of nrutils' scentry class
        '''

        # Let the people know
        if this.verbose: alert('Making gwylm represenation.')

        # Import useful things
        from nrutils import gwylm,gwf
        from scipy.interpolate import InterpolatedUnivariateSpline as spline
        from copy import deepcopy as copy
        from numpy import arange,linspace,exp,array,angle,unwrap,ones_like,zeros_like

        # Initial gwylm object
        sceo = copy(sceo)
        sceo.config = False
        y = gwylm(sceo,load=False,calcnews=False,calcstrain=False,load_dynamics=False)
        y.__lmlist__ = this.lmlist
        y.__input_lmlist__ = this.lmlist

        # Store useful remnant stuff
        y.remnant = this.remnant
        y.remnant['Mw'] = this.wM
        y.radiated = {}
        y.radiated['time_used'] = this.t
        y.radiated['mask'] = ones_like(this.t,dtype=bool)
        y.remnant['mask'] = y.radiated['mask']
        y.remnant['X'] = array( [zeros_like(this.t),zeros_like(this.t),this.remnant['J']/(this.remnant['M']**2)] ).T

        # Make equispace strain pn td and store to y
        alert('Interpolating time domain waveforms for equispacing.')
        dt = 0.5
        t = arange( min(this.t), max(this.t), dt )
        y.t = t
        for l,m in this.lmlist:

            #
            t_ = this.t

            # Store Strain
            hlm_ = this.h[l,m]
            amp_ = abs(hlm_); phi_ = unwrap(angle(hlm_))
            amp = spline(t_,amp_)(t)
            phi = spline(t_,phi_)(t)
            hlm = amp * exp( -1j*phi )
            wfarr = array([ t, hlm.real, hlm.imag ]).T
            y.hlm.append(  gwf( wfarr,l=l,m=m,kind='$rh_{%i%i}/M$'%(l,m) )  )

            # Store Psi4
            ylm_ = this.psi4[l,m]
            amp_ = abs(ylm_); phi_ = unwrap(angle(ylm_))
            amp = spline(t_,amp_)(t)
            phi = spline(t_,phi_)(t)
            ylm = amp * exp( -1j*phi )
            wfarr = array([ t, ylm.real, ylm.imag ]).T
            y.ylm.append(  gwf( wfarr,l=l,m=m,kind='$rM\psi_{%i%i}$'%(l,m) )  )

        #
        y.__curate__()
        y.pad( len(y.t)+500 )

        # Store the gwylmo represenation
        this.pn_gwylmo = y

    # Calculate all implemented strain time domain waveforms
    def __calc_h_of_t__(this):

        #
        this.h = {}
        #
        for l,m in this.lmlist:
            this.h[l,m] = this.__calc_hlm_of_t__(l,m)
        # Calculte m<0 multipoles using symmetry relationship
        alert('Calculating m<0 multipoles using symmetry relation.')
        for l,m in this.lmlist:
            this.h[l,-m] = (-1)**l * this.h[l,m].conj()
        #
        alert('Updating lmlist to inlcude m<0 multipoles.')
        this.lmlist = list(this.h.keys())


    # Use previously calculated strain waveforms to calculate psi4
    def __calc_psi4_and_news_of_t__(this):

        #
        this.news = {}
        this.psi4 = {}
        for l,m in this.h:
            h = this.h[l,m]
            this.news[l,m] = spline_diff( this.t, h, n=1 )
            this.psi4[l,m] = spline_diff( this.t, h, n=2 )


    # Calcilate a single implmented time domain waveform
    def __calc_hlm_of_t__(this,l,m):

        #
        from numpy import pi,log,sqrt,exp

        # Short-hand
        x = this.x
        eta = this.eta

        # Let the people know
        if this.verbose:
            alert('Calculating the (l,m)=(%i,%i) spherical multipole.'%(l,m))

        #
        if (l,m) == (2,2):

            #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
            # (l,m) = (2,2)
            #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#

            #
            part2 = 1 + \
                    x      * (-107.0/42 + 55.0/42*eta) + \
                    x**1.5 * (2*pi) + \
                    x**2   * (-2173.0/1512 - 1069.0/216*eta + 2047.0/1512*eta**2) + \
                    x**2.5 * ((-107.0/21+34.0/21*eta)*pi - 24*1j*eta) + \
                    x**3   * (27027409.0/646800 + 2.0/3*pi**2 + \
                              -856.0/105*this.__gamma_E__ - 1712.0/105*log(2) - 428.0/105*log(x) + \
                              -(278185.0/33264-41.0/96*pi**2)*eta - 20261.0/2772*eta**2 + \
                              114635.0/99792*eta**3 + 428.0*1j*pi/105)
            #
            spin   = x**(1.5) * (-4*this.delta*this.xa/3 + 4/3*(eta-1)*this.xs - 2*eta*x**(0.5)*(this.xa**2 - this.xs**2))
            #
            part1 = sqrt(16.0*pi/5) * 2*eta*this.M*x
            #
            h = part1 * exp(-1j*m*this.phi)*(part2 + spin)

        elif (l,m) == (2,1):

            #
            part2 = (x**0.5 + \
        	       + x**1.5 * (-17.0/28+5*eta/7) + \
        	       + x**2.0 * (pi - 1.0j/2 - 2*1j*log(2)) + \
        	       + x**2.5 * (-43.0/126 - 509*eta/126 + 79.0*eta**2 / 168)) + \
        	       + x**3.0 * (-17.0*pi/28 + 3.0*pi*eta/14 + 1j*(17.0/56 + \
        			eta*(-995.0/84 - 3.0*log(2)/7) + 17.0*log(2)/14))
            #
            part1 = sqrt(16.0*pi/5) * 2*eta*this.M*x * 1j/3*this.delta
            #
            h = part1 * exp(-1j*m*this.phi) * part2 + 4*1j*sqrt(pi/5)*exp(-1j*m*this.phi) * x**2 * eta*(this.xa+this.delta*this.xs)

        elif (l,m) == (2,0):

            #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
            # (l,m) = (2,0)
            #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#

            #
            part2 = 1.0
            part1 = sqrt(16.0*pi/5) * 2*eta*this.M*x * (-5.0/(14*sqrt(6)))
            #
            h = part1*part2

        elif (l,m) == (3,3):

            #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
            # (l,m) = (3,3)
            #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#

            #
            part2 =  x**0.5	 					+ \
                	+x**1.5 * (-4 + 2*eta)					+ \
                	+x**2.0 * (3*pi - 21*1j/5 + 6*1j*log(1.5))		+ \
                	+x**2.5 * (123.0/110 - 1838.0*eta/165 + (887.0*eta**2)/330)	+ \
                	+x**3.0 * (-12*pi + 9.0*pi*eta/2 + 1j*(84.0/5 -24*log(1.5)   + \
                	eta*(-48103.0/1215+9*log(1.5))))
            #
            part1 = sqrt(16.0*pi/5) * 2*eta*this.M*x * (-3.0/4*1j*this.delta*sqrt(15.0/14))
            #
            h = part1 * exp(-1j*m*this.phi) * part2

        elif (l,m) == (3,2):

            #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
            # (l,m) = (3,2)
            #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#

            #
            part2 =  x     *(1- 3*eta) + \
            x**2.0*(-193.0/90 + 145.0*eta/18 - (73.0*eta**2)/18) + \
            x**2.5*(2*pi*(1-3*eta) - 3*1j + 66*1j*eta/5) + \
            x**3.0*(-1451.0/3960 - 17387.0*eta/3960 + (5557.0*eta**2)/220 - (5341.0*eta**3)/1320)

            part1 = sqrt(16*pi/5) * 2*eta*this.M*x * 1.0/3*sqrt(5.0/7)

            #
            h = exp(-1j*m*this.phi)* ( part1*part2 + 32.0/3*sqrt(pi/7)*(eta**2)*this.xs*(x**2.5) )

        elif (l,m) == (3,1):

            #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
            # (l,m) = (3,1)
            #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#

            #
            part2 =  x**0.5 + \
            	     x**1.5 * (-8.0/3 - 2*eta/3) + \
            	     x**2.0 * (pi - 7.0*1j/5 - 2.0*1j*log(2)) + \
            	     x**2.5 * (607.0/198 - 136.0*eta/99 + 247.0*eta**2/198) + \
            	     x**3.0 * ( -8.0*pi/3 - 7.0*pi*eta/6 + 1j*(56.0/15 + 16*log(2)/3 + \
            		 eta*(-1.0/15 + 7.0*log(2)/3)))
            #
            part1 =  sqrt(16.0*pi/5) * 2*eta*this.M*x * 1j*this.delta/(12*sqrt(14))
            #
            h = part1 * exp(-1j*m*this.phi) * part2

        elif (l,m) == (3,0):

            #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
            # (l,m) = (3,0)
            #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
            part2 = 1
            part1 = sqrt(16.0*pi/5) * 2*eta*this.M * x * (-2.0/5*1j*sqrt(6.0/7)*x**2.5*eta)
            h = part1*part2

        elif (l,m) == (4,4):

            #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
            # (l,m) = (4,4)
            #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
            part2 =  x    *(1 - 3*eta)							+ \
                	 x**2.0*(-593.0/110 + 1273.0/66*eta - 175*eta**2/22)			+ \
                	 x**2.5*(4*pi*(1-3*eta) - 42*1j/5 + 1193*1j*eta/40 + 8*1j*(1-3*eta)*log(2)) + \
                	 x**3.0*(1068671.0/200200 - 1088119*eta/28600 +146879*eta**2/2340 + \
                		  -226097*eta**3/17160)
            part1 = sqrt(16.0*pi/5) * 2*eta*this.M*x * (-8.0/9*sqrt(5.0/7));
            h = part1 * exp(-1j*m*this.phi) * part2

        elif (l,m) == (4,3):

            #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
            # (l,m) = (4,3)
            #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
            part2 = x**1.5 * (1 - 2*eta)                                        + \
                	x**2.5 * (-39.0/11 + 1267*eta/132 - 131*eta**2/33)			+ \
                	x**3.0 * (3*pi - 6*pi*eta + 1j*(-32/5 + eta*(16301.0/810 - 12*log(1.5)) + 6*log(1.5)))
            part1 = sqrt(16.0*pi/5) * 2*eta*this.M*x * (-9*1j*this.delta/(4*sqrt(70)))
            h = part1 * exp(-1j*m*this.phi) * part2

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
        # *   x, PN parameter, function of frequency; v = (2*pi*M*f/m)**1.0/3 = x**0.5 (see e.g. https://arxiv.org/pdf/1601.05588.pdf)

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
            alert('Defining the initial binary state based on inputs.','pn')

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
        this.lmlist = [ (2,2), (2,1), (2,0), (3,3), (3,2), (3,1), (3,0), (4,4) ]



