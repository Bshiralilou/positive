#
from __future__ import print_function
from . import *
from positive.api import *
from positive.plotting import *
from positive.learning import *
# > > > > > > > > >  Import adjacent modules  > > > > > > > > > > #
import positive
modules = list( basename(f)[:-3] for f in glob.glob(dirname(__file__)+"/*.py") if (not ('__init__.py' in f)) and (not (__file__.split('.')[0] in f)) )
for module in modules:
    exec('from .%s import *' % module)
# > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > #



# Compute inner-products between cos(theta) and cos(theta)**2 and SWSHs
def swsh_prod_cos(s,l,lp,m,u_power):

    '''
    Compute the inner product
    < sYlm(l) | u^u_power | sYlm(lp) >

    USAGE
    ---
    ans = swsh_clebsh_gordan_prods(k,left_shift,u_power,right_shift)

    NOTES
    ---
    * Here, k is the polar index more commonly denoted l.
    * The experssions below were generated in Mathematica
    * u_power not in (1,2) causes this function to throw an error
    * related selection rules are manually implemented

    londonl@mit.edu 2020
    '''


    # Import usefuls
    from scipy import sqrt

    # Define a delta and do a precompute
    delta = lambda a,b: 1 if a==b else 0
    sqrt_factor = sqrt((2*l+1.0)/(2*lp+1.0))

    #
    if u_power==1:

        # Equation 3.9b of Teukolsky 1973 ApJ 185 649P
        ans = sqrt_factor * clebsh_gordan_wrapper(l,1,m,0,lp,m) * clebsh_gordan_wrapper(l,1,-s,0,lp,-s)

    elif u_power==2:

        # Equation 3.9a of Teukolsky 1973 ApJ 185 649P
        by3 = 1.0/3
        twoby3 = 2.0*by3
        ans = by3*delta(l,lp)  +  twoby3 * sqrt_factor * clebsh_gordan_wrapper(l,2,m,0,lp,m) * clebsh_gordan_wrapper(l,2,-s,0,lp,-s)

    else:

        #
        error('The inner-product between SWSHs and cos(theta)^%i is not handled by this function. The u_power input must be 1 or 2.'%u_power)

    # Return answer
    return ans


# Compute the innder product: < sYlm(k+left_shift) | u^u_power | k+right_shift >
def swsh_clebsh_gordan_prods(k,m,s,left_shift,u_power,right_shift):
    '''
    Compute the inner product
    < sYlm(k+left_shift) | u^u_power | k+right_shift >

    USAGE
    ---
    ans = swsh_clebsh_gordan_prods(k,left_shift,u_power,right_shift)

    NOTES
    ---
    * Here, k is the polar index more commonly denoted l.
    * The experssions below were generated in Mathematica
    * u_power not in (1,2) causes this function to throw an error
    * related selection rules are manually implemented

    londonl@mit.edu 2020
    '''

    #
    from scipy import sqrt,floor,angle,pi,sign

    #
    (k,m,s) = [ float(x) for x in (k,m,s) ]

    #
    Q ={    (0,1,-1):(sqrt(k - m)*sqrt(k + m)*sqrt(k - s)*sqrt(k + s))/(k*sqrt(-1 + 4*k**2)), \
            (0,1,1):sqrt((3 + 4*k*(2 + k))*(1 + k - m)*(1 + k + m)*(1 + k - s)*(1 + k + s))/((1 + k)*(1 + 2*k)*(3 + 2*k)), \
            (0,2,-1):(-2*sqrt(-1 + 4*k**2)*sqrt(k - m)*m*sqrt(k + m)*sqrt(k - s)*s*sqrt(k + s))/(k - 5*k**3 + 4*k**5), \
            (0,2,1):(-2*m*s*sqrt((3 + 4*k*(2 + k))*(1 + k - m)*(1 + k + m)*(1 + k - s)*(1 + k + s)))/(k*(1 + k)*(2 + k)*(1 + 2*k)*(3 + 2*k)), \
            (0,2,-2):((-1)**floor((pi - angle(-3 + 2*k) - angle(-1 + k - m) + angle(-1 + k + m) + angle(-1 + k - s) - angle(-1 + k + s))/(2*pi))*sqrt(-1 + k - m)*sqrt(k - m)*sqrt(-1 + k + m)*sqrt(k + m)*sqrt(-1 + k - s)*sqrt(k - s)*sqrt(-1 + k + s)*sqrt(k + s))/(sqrt(-3 + 2*k)*sqrt(1 + 2*k)*(k - 3*k**2 + 2*k**3)), \
            (0,2,2):sqrt(((1 + k - m)*(2 + k - m)*(1 + k + m)*(2 + k + m)*(1 + k - s)*(2 + k - s)*(1 + k + s)*(2 + k + s))/((1 + 2*k)*(5 + 2*k)))/((1 + k)*(2 + k)*(3 + 2*k)), \
            (0,1,0):-((m*s)/(k + k**2)), \
            (-2,1,-2):(m*s)/((2 - 3*k + k**2)*sign(3 - 2*k)), \
            (-1,1,-1):(m*s)/(k - k**2), \
            (-1,1,-2):((-1)**floor((pi - angle(3 + 4*(-2 + k)*k) - angle(-1 + k - m) + angle(-1 + k + m) + angle(-1 + k - s) - angle(-1 + k + s))/(2*pi))*sqrt(3 + 4*(-2 + k)*k)*sqrt(-1 + k - m)*sqrt(-1 + k + m)*sqrt(-1 + k - s)*sqrt(-1 + k + s))/((-1 + k)*(-3 + 2*k)*(-1 + 2*k)), \
            (-2,1,-1):((-1)**floor((pi - angle(3 + 4*(-2 + k)*k) + angle(-1 + k - m) - angle(-1 + k + m) - angle(-1 + k - s) + angle(-1 + k + s))/(2*pi))*sqrt(3 + 4*(-2 + k)*k)*sqrt(-1 + k - m)*sqrt(-1 + k + m)*sqrt(-1 + k - s)*sqrt(-1 + k + s))/((-1 + k)*(-3 + 2*k)*(-1 + 2*k)), \
            (1,1,1):-((m*s)/(2 + 3*k + k**2)), \
            (1,1,2):sqrt(((2 + k - m)*(2 + k + m)*(2 + k - s)*(2 + k + s))/((3 + 2*k)*(5 + 2*k)))/(2 + k), \
            (2,1,1):sqrt(((2 + k - m)*(2 + k + m)*(2 + k - s)*(2 + k + s))/((3 + 2*k)*(5 + 2*k)))/(2 + k), \
            (2,1,2):-((m*s)/(6 + 5*k + k**2)), \
            (0,2,0):1.0/3 + (2*(k + k**2 - 3*m**2)*(k + k**2 - 3*s**2))/(3*k*(1 + k)*(-1 + 2*k)*(3 + 2*k)), \
            (0,2,2):sqrt(((1 + k - m)*(2 + k - m)*(1 + k + m)*(2 + k + m)*(1 + k - s)*(2 + k - s)*(1 + k + s)*(2 + k + s))/((1 + 2*k)*(5 + 2*k)))/((1 + k)*(2 + k)*(3 + 2*k)), \
            (-2,2,-2):1.0/3 - (2*(2 + (-3 + k)*k - 3*m**2)*(2 + (-3 + k)*k - 3*s**2))/(3*(-2 + k)*(-1 + k)*(-5 + 2*k)*(-1 + 2*k)*sign(3 - 2*k)), \
            (-1,2,-2):(2*(-1)**floor((pi - angle(3 + 4*(-2 + k)*k) - angle(1 - k + m) + angle(-1 + k + m) + angle(1 - k + s) - angle(-1 + k + s))/(2*pi))*sqrt(3 + 4*(-2 + k)*k)*m*sqrt(1 - k + m)*sqrt(-1 + k + m)*s*sqrt(1 - k + s)*sqrt(-1 + k + s))/((-2 + k)*(-1 + k)*k*(-3 + 2*k)*(-1 + 2*k)), \
            (-2,2,-1):(2*(-1)**floor((pi - angle(3 + 4*(-2 + k)*k) + angle(1 - k + m) - angle(-1 + k + m) - angle(1 - k + s) + angle(-1 + k + s))/(2*pi))*sqrt(3 + 4*(-2 + k)*k)*m*sqrt(1 - k + m)*sqrt(-1 + k + m)*s*sqrt(1 - k + s)*sqrt(-1 + k + s))/((-2 + k)*(-1 + k)*k*(-3 + 2*k)*(-1 + 2*k)), \
            (-1,2,-1):((-1 + k)*k*(-1 + 2*(-1 + k)*k - 2*m**2) + 2*(k - k**2 + 3*m**2)*s**2)/(k*(3 + k + 4*(-2 + k)*k**2)), \
            (1,2,1):1.0/3 + (2*(2 + k*(3 + k) - 3*m**2)*(2 + k*(3 + k) - 3*s**2))/(3*(1 + k)*(2 + k)*(1 + 2*k)*(5 + 2*k)), \
            (1,2,2):(-2*m*s*sqrt(((2 + k - m)*(2 + k + m)*(2 + k - s)*(2 + k + s))/((3 + 2*k)*(5 + 2*k))))/((1 + k)*(2 + k)*(3 + k)), \
            (2,2,1):(-2*m*s*sqrt(((2 + k - m)*(2 + k + m)*(2 + k - s)*(2 + k + s))/((3 + 2*k)*(5 + 2*k))))/((1 + k)*(2 + k)*(3 + k))
        }

    #
    if not ( u_power in [1,2] ):
       error('This function handles inner-products between two spin wieghted spherical harmonics and cos(theta) or cos(theta)^2. You have entrered %s, which would correspond to cos(theta)^%s'%(red(str(u_power)),red(str(u_power))))

    #
    z = (left_shift,u_power,right_shift)

    #
    is_handled = (z in Q)
    lp_exists = ((z[0]+k)>=abs(s)) and ((z[-1]+k)>=abs(s))
    is_valid = lp_exists and is_handled

    #
    ans = Q[z] if is_valid else 0

    # Return answer // implicitely implement selection rules
    if z in Q:
       return Q[z]
    else:
       error('make sure that you should not be using the sympy version of this function: swsh_prod_cos')
       return 0


# Function for the calculation of a biorthogonal spheroidal harmonic subset
def calc_adjoint_slm_subset(a,m,n,p,s,lrange,theta=None,full_output=False):
    
    '''
    DESCRIPTION
    ---
    Function for the calculation of a biorthogonal spheroidal harmonic subset
    hainv fixed values of overtone and partiy indeces n and p, but varuing 
    values of the legendre index l.
    
    USAGE
    ---
    output = calc_adjoint_slm_subset(a,m,n,p,s,lrange,theta=None,full_output=False)
    
    INPUTS
    ---
    aw,             The oblateness 
    m,              The azimuthal eigenvalue (aka the polar index for the associated
                    legendre problem)
    n,              The overtone label. NOTE that we use the convention where 
                    n starts
                    at zero.
    s,              The spin weight (the original iteration of this function 
                    only handles |s|=2)
    lrange,         Values of the legendre indeces to consider. 
    theta,          OPTIONAL. The polar angle. If not given, a default stencil will 
                    be generated and output.
    full_output,    OPTIONAL. Toggle for output a dictionary of various data products
                    rather than standard minimal output.
                    
    OUTPUT
    ---
    
    IF full_output=True THEN output is a dictionary with fields inhereited from ysprod_matrix along with additional fields:
    
        spheroidal_of_theta_dict,           a dictionary with keys (l,m,n,p) and values
                                            given by spheroidal harmonics in theta 
    
        adjoint_spheroidal_of_theta_dict,   a dictionary with keys (l,m,n,p) and values
                                            given by adjoint spheroidal harmonics in theta 
                                            
    IF full_output=False THEN output is 
    
        theta, adjoint_spheroidal_of_theta_dict, spheroidal_of_theta_dict
    
    AUTHOR
    ---
    londonl@mit, pilondon2@gmail.com, 2021
    
    '''
    
    #
    foo = aslm_helper( a,m,n,p,s,lrange,theta=theta,full_output=full_output )
    
    #
    return foo



# Function to validate inputs to adjoint spheroidal harmonic type functions
# NOTE that this function and related ones use the NR convention for labeling the QNMs
def validate_aslm_inputs(a,m,n,p,s,lrange):
    
    '''
    Function to test whether theta input to a spheroidal harmonic method is appropriate
    '''
    
    # Import usefuls 
    from positive import lim
    from numpy import ndarray,pi,double,complex
    
    # Check indices
    if not isinstance(s,int):
        error('spin wieght, s, must be int, but %g found'%s)
    
    # Check for sign convention on spin
    if a<0:
        error('This object uses the convention that a>0. To select the retrograde QNM branch, set p=-1')
    
    # Check for acceptable p values
    if not (p in [-1,1]):
        error('p must be +1 (for prograde) or -1 (for retrograde), instead it is %s'%str(p))
        
    # Check for extremal or nearly extremal cases
    if abs(a)>1:
        error('Kerr parameter must be non-extremal')
    if abs(a)>(1-1e-3):
        warning('You have selected a nearly extremal spin. Please take significant care to ensure that results make sense.')
    
    # Check for consistent definition of l values
    for l in lrange:
        if not isinstance(l,int):
            error('legendre, l, index must be int, but %g found'%l)
        if not isinstance(l,int):
            error('azimuthal, m, index must be int, but %g found'%m)
        if l<abs(s):
            error('l must be >= |s| due to the structure of Teukolsk\'s angular equation')
        if abs(m)>l:
            error('|m|>l and it should not be du to the structure of Teukolsk\'s angular equation')
    


# Helper function for adjoint spheroidal harmoinc calculator
def aslm_helper( a,m,n,p,s,lrange,theta=None,full_output=False ):
    
    '''
    Helper function for the calculation of adjoint spheroidal harmonics. Sets of these functions must be computed simulteneously at present.
    '''
    
    # Import usefuls 
    # ---
    from numpy import linspace,pi,zeros_like
    
    # Validate inputs
    # ---
    validate_aslm_inputs(a,m,n,p,s,lrange)
    
    # Handle theta input 
    # ---
    if theta is None:
        zero = 1e-6
        num_theta = 2**9
        theta = linspace(zero,pi-zero,num_theta)
    
    # NOTE that the use must apply the phi dependence externally via exp( 1j * m * phi )
    # ---
    phi = 0
    
    # Get all of the relevant info
    # ---
    foo = ysprod_matrix(a,m,n,p,s,lrange,verbose=False,spectral=True,full_output=True)
    
    # Prepare dictionaries for spheroidal and adjoint spheroidal arrays
    aS = {}
    S  = {}
    
    # Compute spheroidal and adjoint spheroidal harmonics as a sum over their spherical harmonic multipole moments.
    # ---
    for l in lrange:
        
        #
        aS_vector = foo['adj_spheroidal_vector_dict'][ l,m,n,p ]
        S_vector  = foo['spheroidal_vector_dict'][     l,m,n,p ]
        
        #
        Sl  = zeros_like(theta,dtype=complex)
        aSl = zeros_like(theta,dtype=complex)
        
        for k,lk in enumerate(lrange):
            #
            Y = sYlm(s,lk,m,theta,phi,leaver=True)
            Sl  +=  S_vector[k] * Y
            aSl += aS_vector[k] * Y
            
        # Store in dictionaries for ease of access
        S[l,m,n,p]  =  Sl
        aS[l,m,n,p] = aSl
    
    # Either output all data, or only the minimal products
    if full_output:
        #
        foo['theta']                            = theta
        foo[        'spheroidal_of_theta_dict'] =  S
        foo['adjoint_spheroidal_of_theta_dict'] = aS
        #
        return foo
    else:
        #
        return (theta,aS,S)



#
def aslmcg( a, s, l, m, n, p, theta, phi, kerr=True, lmin=None, lmax=None, span=6, force_norm=True, lrange=None ):
    '''
    Compute adjoint spheroidal harmonic functoin using the spherical harmonic clensch-gordan method. By default, kerr adjoint functions are computed.
    londonl@mit.edu 2020
    '''
    
    #
    error('please use calc_adjoint_slm_subset')
    
    #
    from numpy.linalg import inv
    from numpy import zeros,array,double,ndarray,dot,sqrt,zeros_like
    
    # Return standard adjoint if requested -- NOTE the keyword here is confusing and should be changed!
    if not kerr:
        aw = a * leaver( a, l, m, n )[0]
        S,A = slmcg( aw, s, l, m, theta, phi )
        return S.conj(), A.conj()
        
    #
    if not isinstance(phi,(float,int,double)):
        error('phi must be number; zero makes sense as the functions phi dependence is exp(1j*m*phi), and so can be added externally')
        
    # Otherwise proceed to computeation of Kerr "adjoint" function(s)
    
    #
    if l is not 2:
        error('This function currently works when l=2 is given, but it will generate and output harmonics for all l<=lmax.')
    
    # Handle input format
    if isinstance(a,(list,ndarray)):
        if len(a)>1:
            error('first input as iterable not handled; fun function on each element')
        else:
            a = a[0]
            
    # if lmin in None, set it relative to l
    if lmin is None: lmin = max(l-span,max(abs(s),abs(m)))
    # if lmax in None, set it relative to l
    if lmax is None: lmax = l+span

    #
    if lrange is None:
        lrange = range(lmin,lmax+1)  # range of ell
    else:
        lrange = list(lrange)
        lmin,lmax = lim(lrange)
    
    # Define index range and perform precomputations
    num_ell = len(lrange)
    
    #
    beta,A,qnmo_dict = ysprod_matrix(a,m,n,p,s,lrange,spectral=True,full_output=True)
        
    # Compute the inverse conjugate matrix
    X, Z = beta, inv(beta)
    nu = Z.conj()
    
    # Compute the related operator's matrix rep (a "heterogeneous adjoint")
    L_ddag = dot( Z, dot( A, X ) ).conj()
    
    # Construct space of spherical harmonics to use 
    Yspace = array( [ sYlm(s,llj,m,theta,phi,leaver=True) for llj in lrange ] )
    
    # Compute adjoint functions 
    # NOTE that this line can be slow if many (thousand) values of theta are used
    aSspace = dot( nu, Yspace )
    # Compute regular spheroidal function
    Sspace = dot(beta.T,Yspace)
    
    # # Enforce normalization
    # # NOTE that this is very optional as harmonics are arlready normalized to high
    # # accuracy by construction. However, the stencil in theta may cause minor departures
    # if force_norm:
    #     norm = lambda x: x/sqrt( prod(x,x,theta) )
    #     for k,llk in enumerate(lrange):
    #         aSspace[k,:] = norm( aSspace[k,:] )
    #         Yspace[k,:]  = norm(  Yspace[k,:] )
    #         Sspace[k,:]  = norm(  Sspace[k,:] )
    
    # Create maps between ell and harmonics
    foo,bar,sun = {},{},{}
    for k,llk in enumerate(lrange):
        foo[ llk ] = aSspace[k,:]
        bar[ llk ] =  Yspace[k,:]
        sun[ llk ] =  Sspace[k,:]
        
    # Package output
    ans = {}
    ans['Ylm'] = bar
    ans['AdjSlm'] = foo
    ans['Slm'] = sun
    ans['lnspace'] = lrange 
    ans['Yspace'] = Yspace
    ans['aSspace'] = aSspace
    ans['YSGramian'] = beta
    ans['overtone_index'] = n
    ans['matrix_op'] = L_ddag
    ans['ZAX'] = (Z,A,X)
    
    # Return output
    return ans

  
       
#
def slmcg_helper( aw, s, l, m, lmin=None, lmax=None, span=6, case=None, lrange=None ):
    '''
    Compute matrix elements of spheroidal differential operator in spherical harmonic basis
    londonl@mit.edu 2020 
    '''        
    
    # Import usefuls
    from scipy.linalg import eig,inv
    from numpy import array,zeros,zeros_like,exp,double
    from numpy import ones,arange,ndarray,complex128

    # Preliminaries
    # ------------------------------ #

    # Handle input format
    if isinstance(aw,(list,ndarray)):
        if len(aw)>1:
            error('first input as iterable not handled; fun function on each element')
        else:
            aw = aw[0]
              
    # if lmin in None, set it relative to l
    if lmin is None: lmin = max(l-span,max(abs(s),abs(m)))
    # if lmax in None, set it relative to l
    if lmax is None: lmax = l+span
    
    #
    if lrange is None:
        lrange = range(lmin,lmax+1)  # range of ell
    else:
        lrange = list(lrange)
        if max(lrange) < (l+3) :
            warning( 'the provided range of \ell values is insufficient to properly resolve a harmonic with ell=%i. Please use a maximum \ell value of at least l+3=%i'%(l,l+3) )
        if min(lrange) < max( abs(s), l-3 ) :
            error( 'min value in lrange must be greater than l-3 = %i'%max( abs(s), l-3 ) ) 
        lmin,lmax = lim(lrange)
    
    # Define index range and perform precomputations
    num_ell = len(lrange)
    aw2 = aw*aw; c1 = -2*aw*s; c2 = aw2
    
    #
    if case==1:
        c2 = 0    

    # Main bits
    # ------------------------------ #

    # Eigenvalue for non-spinning solution
    A0 = lambda ll: (ll-s)*(1+ll+s)

    # Make lambdas to reduce duplicate code
    # TODO: determine which clebsch gordan method is faster
    # # Possibly the faster option, but throws warnings which must be investigated
    # c1_term = lambda llj,llk: c1*swsh_clebsh_gordan_prods(llj,m,s,0,1,llk-llj)
    # c2_term = lambda llj,llk: c2*swsh_clebsh_gordan_prods(llj,m,s,0,2,llk-llj)
    
    # Safer option, likely
    c1_term = lambda llj,llk: c1*swsh_prod_cos(s,llj,llk,m,1)
    c2_term = lambda llj,llk: c2*swsh_prod_cos(s,llj,llk,m,2)

    # Pre-allocate and then fill the coefficient matrix
    Q = zeros((num_ell,num_ell),dtype=complex128)
    for j,lj in enumerate(lrange):
        for k,lk in enumerate(lrange):

            # Populate the coefficient matrix
            if   lk == lj-2:
                Q[j,k] = c2_term(lj,lk)
            elif lk == lj-1:
                Q[j,k] = c1_term(lj,lk)+c2_term(lj,lk)
            elif lk == lj+0:
                Q[j,k] = c1_term(lj,lk)+c2_term(lj,lk)-A0(lj)
            elif lk == lj+1:
                Q[j,k] = c1_term(lj,lk)+c2_term(lj,lk)
            elif lk == lj+2:
                Q[j,k] = c2_term(lj,lk)
            else:
                Q[j,k] = 0

    # Use scipy to find the eigenvalues and eigenvectors,
    # aka the spheroidal eigenvalues, and lists of spherical-spheroidal inner-products
    vals,vecs = eig(Q)  
    
    #
    return Q,vals,vecs,lrange  


# Compute spheroidal harmonic eigenvalue using clebsh-gordan coefficients
def slmcg_eigenvalue(aw, s, l, m, lmin=None, lmax=None, span=8):

    '''
    Use Clebsh-Gordan coefficients to calculate spheroidal harmonic eigenvalue
    londonl@mit.edu 2015+2020

    '''

    # Import usefuls
    from scipy.linalg import eig,inv
    from sympy.physics.quantum import cg
    from numpy import array,zeros,zeros_like,exp,double
    from numpy import ones,arange,ndarray,complex128

    # Preliminaries
    # ------------------------------ #

    # Handle input format
    if isinstance(aw,(list,ndarray)):
        if len(aw)>1:
            error('first input as iterable not handled; for nor, use function on each element')
        else:
            None
            aw = aw[0]

    # Main bits
    # ------------------------------ #
    
    if aw:
        
        # Use helper function to calculate matrx elements
        _,vals,_,lrange = slmcg_helper( aw, s, l, m, lmin=lmin, lmax=lmax, span=span )

        #
        dex_map = { ll:lrange.index(ll) for ll in lrange }
        sep_consts = vals
        
        # Extract separation constant. Account for sign convention
        A = -sep_consts[ dex_map[l] ]
        
    #
    else:
        
        A = (l-s)*(l+s+1)
    
    #
    return A

        
# Compute spheroidal harmonics with clebsh-gordan coefficients and matrix method
def slmcg( aw, s, l, m, theta, phi, lmin=None, lmax=None, span=6, full_output=False ):

    '''
    Use Clebsh-Gordan coefficients to calculate spheroidal harmonics
    londonl@mit.edu 2015+2020

    '''

    # Import usefuls
    from scipy.linalg import eig,inv
    from sympy.physics.quantum import cg
    from numpy import array,zeros,zeros_like,exp,double
    from numpy import ones,arange,ndarray,complex128

    # Preliminaries
    # ------------------------------ #

    # Handle input format
    if isinstance(aw,(list,ndarray)):
        if len(aw)>1:
            error('first input as iterable not handled; fun function on each element')
        else:
            aw = aw[0]

    # Main bits
    # ------------------------------ #
    
    # Use helper function to calculate matrx elements
    Q,vals,vecs,lrange = slmcg_helper( aw, s, l, m, lmin=lmin, lmax=lmax, span=span )

    #
    dex_map = { ll:lrange.index(ll) for ll in lrange }
    sep_consts = vals
    ysprod_array = vecs

    # Extract spherical-spheroidal inner-products of interest
    ysprod_vec = ysprod_array[ :,dex_map[l] ]

    # Compute Spheroidal Harmonic
    S = zeros_like(theta,dtype=complex128)
    for k,lp in enumerate(lrange):
        S += sYlm(s,lp,m,theta,0,leaver=True) * ysprod_vec[k]
    S *= exp(1j*m*phi)

    # Extract separation constant. Account for sign convention
    A = -sep_consts[ dex_map[l] ]

    # Package output
    # ------------------------------ #

    # Initialize answer
    if full_output:
        ans = {}
        # Spheroidal harmonic and separation constant
        ans['standard_output'] = (S,A)
        # All separation constants
        ans['sep_consts'] = sep_consts
        # Array and l specific ysprods
        ans['ysprod_array'] = ysprod_array
        ans['ysprod_vec'] = ysprod_vec
        # Coefficient matrix
        ans['coeff_array'] = Q
        # store space of l values considered
        ans['lrange'] = lrange
        # map between l and index
        ans['dex_map'] = dex_map
    else:
        # Only output harmonic and sep const
        ans = S,A

    # Return asnwer
    return ans




# A function for calculating spheroidal harmonics in the spherical harmonic basis
def slmy(aw,l,m,theta,phi,s=-2,tol=None,verbose=False,output_iterations=False,sc=None,leaver=True,test=True):
    
    '''
    Calculate spheroidal harmonic using ansatz:
            u = cos(theta)
            S_j(u) = exp( -aw_j * u ) Sum( a[k]Y_k(u) )
    See case london==-4 in leaver_ahelper for recursive formula.
    NOTE that this method does not use NR conventions.
    londonl@mit.edu/2020
    '''
    
    # Import usefuls
    from numpy import pi,sort,sqrt
    
    # Validate input 
    # ---
    validate_slm_inputs(aw,theta,s,l,m)
    
        
    # NOTE that london=-4 seems to have the most consistent behavior for all \ell and m
    if sc is None:
        sc = sc_leaver( dtyp(aw), l, m, s, verbose=False,adjoint=False, london=-4)[0]
    
    #
    def __main__(aw,l,m,theta,phi,s=-2,tol=None,verbose=False,output_iterations=False,sc=None):
        
        #
        k1,k2,alpha,beta,gamma,scale_fun_u,u2v_map,theta2u_map = leaver_ahelper( l,m,s,aw,sc, london=-4, verbose=verbose )
        
        #
        zero = 1e-10
        theta[ theta==0 ] = zero 
        theta[ theta==pi] = pi-zero
        
        # Variable map for theta
        u = theta2u_map(theta)

        # Precompute the series precactors
        a = slm_sequence(aw,l,m,s=s,sc=sc)
        
        # Compute the theta dependence using the precomputed coefficients
        Y = 0
        for k in sort(list(a.keys())):
            Yk = sYlm(s,k,m,theta,phi,leaver=leaver)
            dY = a[k]*Yk
            Y += dY

        # together now
        S = scale_fun_u(u) * Y 
        
        #
        return S
        
    # NOTE that we use conjugate symmetry for m<0 only becuase the various numerical algorithms are not fully stable for negative m: some values of m<0 are OK, others cause NANs. More understanding possible, but not high priority. 
    if m>=0:
        S = __main__(aw,l,m,theta,phi,s,tol,verbose,output_iterations,sc)
    else:
        S = __main__(-aw.conj(),l,-m,theta,phi,s,tol,verbose,output_iterations,sc.conj()).conj()[::-1]
        
    # normalize
    S = S / sqrt( prod(S,S,theta) )
        
    #
    if test:
        # Perform the test
        test_slm(S,sc,aw,l,m,s,theta,tol=1e-6,verbose=verbose)
    
    #
    return S
    


# Low level function for calculating spheroidal harmonics
def slm_helper( aw, l, m, theta, phi, s, sc=None, tol=None, verbose=False,london=False, full_output = False,conjugate=False ):
    
    '''
    SLM_HELPER
    ---
    LOW LEVEL function for evaluating spheroidal harmonics for a give oblateness.
    
    USAGE
    ---
    foo = slm_helper( aw, l, m, theta, phi, s, sc=None, tol=None, verbose=False,london=False, full_output = False )
    
    IF full output, then foo is a dictionary of information inluding field Slm containing the harmonic 
    
    ELSE foo is a tuple of the spheroidal harmonic and its eigenvalue (Slm,Alm)
    
    IF full output, then foo is a dictionary with the following fields (more may be present as this functions is updated):
    
    * Slm,    The spheroidal harmonic. NOT normalized.
    * Itr,    Iterations of the sphoidal harmonic.
    * Err,    The change in prefactor between iterations.
    * Alm,    The spheroidal harmonic eigenvalue
    
    AUTHOR
    ---
    londonl@mit, pilondon2@gmail.com, 2021
    
    '''
    
    # 
    from positive import red
    from positive import leaver as lvr
    from positive import rgb,lim,leaver_workfunction,cyan,alert,pylim,sYlm,error,internal_ssprod
    from numpy import complex256, cos, ones, mean, isinf, pi, exp, array, ndarray, unwrap, angle, linalg, sqrt, linspace, sin, float128
    from scipy.integrate import trapz
    from numpy import complex128 as dtyp
        
    # Define separation constant if not input
    if sc is None:
        #sc = slmcg_eigenvalue( dtyp(aw), s, l, m)
        sc = sc_leaver( dtyp(aw), l, m, s, verbose=False, adjoint=False, london=london)[0]
        
    # Sanity check separation constant
    sc2 = sc_leaver( aw, l, m, s, verbose=False,adjoint=False, london=london, tol=tol)[0]
    if abs(sc2-sc)>1e-3:
        print('aw  = '+str(aw))
        print('sc_input  = '+str(sc))
        print('sc_leaver = '+str(sc2))
        print('sc_london = '+str(sc_london( aw,l,m,s )[0]))
        print('aw, s, l, m = ',aw, s, l, m)
        print('slmcg_eigenvalue = '+str( slmcg_eigenvalue( aw, s, l, m ) ) )
        print('err = '+str(abs(sc2-sc)))
        warning('input separation constant not consistent with angular constraint, so we will use a different one to give you an answer that converges.')
        sc = sc2

    # ------------------------------------------------ #
    # Angular parameter functions
    # ------------------------------------------------ #

    # Retrieve desired information from central location
    k1,k2,alpha,beta,gamma,scale_fun_u,u2v_map,theta2u_map = leaver_ahelper( l,m,s,aw,sc, london=london, verbose=verbose )

    # ------------------------------------------------ #
    # Calculate the angular eigenfunction
    # ------------------------------------------------ #

    # Variable map for theta
    u = theta2u_map(theta)
    # Calculate the variable used for the series solution
    v = u2v_map( u )

    # the non-sum part of eq 18
    X = ones(u.shape,dtype=complex256)
    X = X * scale_fun_u(u)

    # initial series values
    a0 = 1.0 # a choice, setting the norm of Slm

    a1 = -a0*beta(0)/alpha(0)

    C = 1.0
    C = C*((-1)**(max(-m,-s)))*((-1)**l)
    
    # the sum part
    done = False
    Y = a0*ones(u.shape,dtype=complex256)
    Y = Y + a1*v
    k = 1
    kmax = 5e3
    err,yy = [],[]
    et2=1e-8 if tol is None else tol
    max_a = max(abs(array([a0,a1])))
    v_pow_k = v
    while not done:
        k += 1
        j = k-1
        a2 = -1.0*( beta(j)*a1 + gamma(j)*a0 ) / alpha(j)
        v_pow_k = v_pow_k*v
        dY = a2*v_pow_k
        Y += dY
        xx = max(abs( dY ))

        #
        if full_output:
            yy.append( C*array(Y)*X*exp(1j*m*phi) )
            err.append( xx )

        k_is_too_large = k>kmax
        done = (k>=l) and ( (xx<et2 and k>30) or k_is_too_large )
        done = done or xx<et2
        a0 = a1
        a1 = a2

    # together now
    S = X*Y*exp(1j*m*phi)

    # Use same sign convention as spherical harmonics
    # e.g http://en.wikipedia.org/wiki/Spin-weighted_spherical_harmonics#Calculating
    S = C * S

    # Warn if the series did not appear to converge
    if k_is_too_large:
        print(l,m,s,sc,aw)
        warning('The while-loop exited becuase too many iterations have passed. The series may not converge for given inputs. This may be cuased by the use of an inapproprate eigenvalue.')

    #
    if conjugate:
        S = S.conj()
        
    #
    if full_output:
        foo['Slm'] = S 
        foo['Iterant_Slm'] = yy 
        foo['Iterant_Error'] = err 
        foo['Alm'] = sc
        return foo
    else:
        return (S,sc)



# High level wrapper for slm_helper that calculates spheroidal harmonics
def slm( aw, l, m, theta, phi, s, sc=None, tol=None, verbose=False,london=False, full_output = False, test=True ):
    
    '''
    Function to cumpute normalized spheroidal harmonic for input values of 
    
    aw,     The oblateness 
    l,      The polar index (aka the legendre index for the spheroidal problem)
    m,      The azimuthal eigenvalue (aka the polar index for the associated legendre problem)
    theta,  The series of spherical polar angles desired. NOTE that is input values do not sufficiently cover [0,pi], then an error will be thrown. 
    phi,    The SINGLE azimuthal angle desired for the spheroidal function 
    s,      The spin weight of the harmonic 
    
    USAGE (See also doc for slm_helper)
    ---
    S = slm( aw, l, m, theta, phi, s, sc=None, tol=None, verbose=False,london=False, full_output = False )
    
    NOTES
    ---
    * IF test, the spheroidal harmonic is input into the related differential equation. If the differential equation is not solved, then a warning is raised. 
    * IF full_output, then the output is a dictionary of various informations. In this instance, function evaluation is slowed due to additional information at play.
    
    AUTHOR
    ---
    londonl@mit, pilondon2@gmail.com, 2021
    
    '''
    
    # Import usefuls 
    from numpy import pi
    
    # Validate input 
    # ---
    validate_slm_inputs(aw,theta,s,l,m)
    
    
    # Impose symmetry relation to map positive m functions to negative ones. 
    # NOTE that this is optimal as the series solutions used by the helper function are not always stable for m<0.
    # ---
    if m>=0:
        foo = slm_helper( aw, l, m, theta, phi, s, sc=sc, tol=None, verbose=verbose )
    else:
        foo = slm_helper(-aw.conj(), l,-m, pi-theta, phi, s, sc=sc.conj(), tol=None, verbose=verbose, conjugate=True  )
        
    #
    if test:
        # Extract harmonic and its eigenvalue
        S = foo[0] if not full_output else foo['Slm']
        A = foo[1] if not full_output else foo['Alm']
        # Perform the test
        test_slm(S,A,aw,l,m,s,theta,tol=1e-6,verbose=verbose)
    
        
    # NOTE that 
    # * IF output_iterations, then foo is a dictionary of S,Itr,Err
    # * ELSE, foo is S, where S is the spheroidal harmonic
    # ---
    return foo

# Function to validate inputs to spheroidal harmonic type functions
def validate_slm_inputs(aw,theta,s,l,m):
    
    '''
    Function to test whether theta input to a spheroidal harmonic method is appropriate
    '''
    
    # Import usefuls 
    from positive import lim
    from numpy import ndarray,pi,double,complex,int64
    
    # Check oblateness 
    if not isinstance(aw,(double,complex)):
        error('oblateness parameter aw must be double or complex typed')
    
    # Check indices
    if not isinstance(s,(int64,int)):
        error('spin wieght, s, must be int, but %g found'%s)
    if not isinstance(l,(int64,int)):
        error('legendre, l, index must be int, but %g found'%l)
    if not isinstance(l,(int64,int)):
        error('azimuthal, m, index must be int, but %g found'%m)
    
    # Verify that theta is array 
    if not isinstance(theta,ndarray):
        error('theta input must be numpy array')
    # Verify that theta starts and ends on [0,pi]
    zero = 1e-8
    theta_min,theta_max = lim(theta)
    if (theta_min<0) or (theta_max<0):
        error('theta must be on [0,pi] but negative value found')
    if abs(theta_max-theta[-1])>zero:
        error('theta must monotonically increasing but its last value is not its max')
    if abs(theta_min-theta[0])>zero:
        error('theta must monotonically increasing but its first value is not its min')
    if theta_max>pi:
        error('theta must be on [0,pi] but its last value is greater than pi')
    if abs(theta[0])>zero:
        error('theta does not start close enough to zero (as defined internally here by %1.2e)'%zero)
    if abs(theta[-1]-pi)>zero:
        error('|theta[-1]-pi|>zero (where zero is defined internally here by %1.2e)'%zero)
    # Verify that there are enough points in theta for reasobale answers 
    len_theta = len(theta)
    if len_theta<180:
        error('There are less than 180 points in theta. This is deemed to not be enough for precise and accurate results.')
    if len_theta<256:
        warning('There are less than 256 points in theta. This may reduce precision of results. At least 256 points are recommended.')



#
def slm_sequence_backwards(aw,l,m,s=-2,sc=None,verbose=False,span=10,__COMPUTE_RADIAL_SEQUENCE__=False,__a__=None,__cw__=None):
    
    '''
    unstable for large l values and so should be used in conjunction with slm_sequence_forwards which is also unstable but in the other direction 
    
    The relevant spheroidal harmonic ansatz is:
            u = cos(theta)
            S_j(u) = exp( -aw_j * u ) Sum( a[k]Y_k(u) )
    See case london==-4 in leaver_ahelper for recursive formula.
    '''

    #
    from positive import red
    from positive import leaver as lvr
    from positive import rgb,lim,leaver_workfunction,cyan,alert,pylim,sYlm,error,internal_ssprod
    from numpy import complex256, cos, ones, mean, isinf, pi, exp, array, ndarray, unwrap, angle, linalg, sqrt, linspace, sin, float128, inf, isnan, argmax
    from scipy.integrate import trapz
    from numpy import complex128 as dtyp
    
    # NOTE that london=-4 seems to have the most consistent behavior for all \ell and m
    if sc is None:
        sc = slmcg_eigenvalue( dtyp(aw), s, l, m)
        # sc = sc_leaver( dtyp(aw), l, m, s, verbose=verbose,adjoint=False, london=-4)[0]
        
        
    #
    if __COMPUTE_RADIAL_SEQUENCE__:
        if not isinstance(__a__,(float,int)):
            error('You have requested that we compute the radial sequence, but you have not provided a FLOAT or INT value of the BH spin in the __a__ keyword.')
        if not isinstance(__cw__,complex):
            error('You have requested that we compute the radial sequence, but you have not provided a COMPLEX value of the QNM frequency in the __cw__ keyword.')
            
            
    #
    if not __COMPUTE_RADIAL_SEQUENCE__:
        # NOTE that london=-4 here is not only correct but required for this method
        k1,k2,alpha,beta,gamma,scale_fun_u,u2v_map,theta2u_map = leaver_ahelper( l,m,s,aw,sc, london=-4, verbose=verbose )
    else:
        # Change convention from M=1 to M=1.2 so that leaver's equations may be used
        # leaver_a,leaver_cw,leaver_M = rlm_change_convention(__a__,__cw__,1)
        leaver_a,leaver_cw,leaver_M = __a__,__cw__,0.5
        # NOTE that a*cw is invariant under convention change above
        aw = leaver_a*leaver_cw
        # Collect information needed for recursion relations
        k1,k2,alpha,beta,gamma,r_exp_scale = leaver_rhelper( l,m,s,leaver_a,leaver_cw,sc, london=False, verbose=verbose )
    
    #
    a = {} 
    
    #
    tol = 1e-10
    
    #
    k_min = 0
    
    if not __COMPUTE_RADIAL_SEQUENCE__:
        k_max = l+span
        k_ell = l-max(abs(s),abs(m))
    else:
        k_max = span-1
        k_ell = 0
    
    # Initialize sequence
    k = k_max
    
    #
    if __COMPUTE_RADIAL_SEQUENCE__:
        a[ k + 0 ]     = tol
        a[ k - 1 ] = - a[k] * beta(k) / gamma(k) 
    else:
        a[ k + 0 ]     = (tol if (aw or __COMPUTE_RADIAL_SEQUENCE__) else 0) if k!=k_ell else 1
        a[ k - 1 ] = - a[k] * beta(k) / gamma(k) if (aw or __COMPUTE_RADIAL_SEQUENCE__) else 0
    
    #
    done = False 
    while not done:
        
        #
        k = k - 1
        
        #
        if aw:
        
            #
            a[k-1] = -( a[k+1]*alpha(k) + a[k]*beta(k) ) / gamma(k)
            
            #
            avals = array(list(a.values()))
            for p in a.keys():
                a[p] /= (avals[argmax(abs(avals))] if k>=k_ell else a[k_ell])
                
        else:
            
            #
            if (k-1)==k_ell:
                a[k-1] = 1
            else:
                a[k-1] = 0
            
        #
        done = (k-1) == k_min
    
    #
    b = {}
    kref = max(abs(m),abs(s)) if (not __COMPUTE_RADIAL_SEQUENCE__) else 0
    for key in a:
        b[ key+kref ] = a[key]
        
    #
    return b

#
def slm_sequence_forwards(aw,l,m,s=-2,sc=None,verbose=False,span=10,__COMPUTE_RADIAL_SEQUENCE__=False,__a__=None,__cw__=None):
    
    '''
    unstable for large l values and so should be used in conjunction with slm_sequence_backwards which is also unstable but in the other direction 
    
    The relevant spheroidal harmonic ansatz is:
            u = cos(theta)
            S_j(u) = exp( -aw_j * u ) Sum( a[k]Y_k(u) )
    See case london==-4 in leaver_ahelper for recursive formula.
    
    '''

    #
    from positive import red
    from positive import leaver as lvr
    from positive import rgb,lim,leaver_workfunction,cyan,alert,pylim,sYlm,error,internal_ssprod
    from numpy import complex256, cos, ones, mean, isinf, pi, exp, array, ndarray, unwrap, angle, linalg, sqrt, linspace, sin, float128, inf, isnan, argmax
    from scipy.integrate import trapz
    from numpy import complex128 as dtyp
    
    # NOTE that london=-4 seems to have the most consistent behavior for all \ell and m
    if sc is None:
        sc = slmcg_eigenvalue( dtyp(aw), s, l, m)
        # sc = sc_leaver( dtyp(aw), l, m, s, verbose=verbose,adjoint=False, london=-4)[0]
        
    #
    if __COMPUTE_RADIAL_SEQUENCE__:
        if not isinstance(__a__,(float,int)):
            error('You have requested that we compute the radial sequence, but you have not provided a FLOAT or INT value of the BH spin in the __a__ keyword.')
        if not isinstance(__cw__,complex):
            error('You have requested that we compute the radial sequence, but you have not provided a COMPLEX value of the QNM frequency in the __cw__ keyword.')
        
    #
    if not __COMPUTE_RADIAL_SEQUENCE__:
        # NOTE that london=-4 here is not only correct but required for this method
        k1,k2,alpha,beta,gamma,scale_fun_u,u2v_map,theta2u_map = leaver_ahelper( l,m,s,aw,sc, london=-4, verbose=verbose )
    else:
        # Change convention from M=1 to M=1.2 so that leaver's equations may be used
        leaver_a,leaver_cw,leaver_M = __a__,__cw__,0.5
        # NOTE that a*cw is invariant under convention change above
        aw = leaver_a*leaver_cw
        # Collect information needed for recursion relations
        k1,k2,alpha,beta,gamma,r_exp_scale = leaver_rhelper( l,m,s,leaver_a,leaver_cw,sc, london=False, verbose=verbose )
    
    #
    a = {} 
    
    #
    tol = 1e-10
    
    #
    k_min = 0

    if not __COMPUTE_RADIAL_SEQUENCE__:
        k_max = l+span
        k_ell = l-max(abs(s), abs(m))
    else:
        k_max = span-1
        k_ell = 0
    
    # Initialize sequence
    k = 0
    if __COMPUTE_RADIAL_SEQUENCE__:
        a[ k + 0 ] = 1.0 
        a[ k + 1 ] = - a[k] * beta(k) / alpha(k) 
    else:
        a[ k + 0 ] = 1.0 if aw else (1 if k==k_ell else 0)
        a[ k + 1 ] = - a[k] * beta(k) / alpha(k) if aw else (1 if (k+1)==k_ell else 0)
    
    #
    done = False 
    while not done:
        
        #
        k = k + 1
        
        #
        if aw:
            
            #
            a[k+1] = -( a[k]*beta(k) + a[k-1]*gamma(k) ) / alpha(k)
            
            #
            avals = array(list(a.values()))
            for p in a:
                a[p] /= (avals[argmax(abs(avals))] if k<=k_ell else a[k_ell])
                
        else:
            
            #
            if (k+1)==k_ell:
                a[k+1] = 1
            else:
                a[k+1] = 0
            
        #
        done = (k+1) == k_max
    
    #
    b = {}
    kref = max(abs(m),abs(s)) if (not __COMPUTE_RADIAL_SEQUENCE__) else 0
    for key in a:
        b[ key+kref ] = a[key]
        
    #
    return b

#
def slm_sequence(aw,l,m,s=-2,sc=None,verbose=False,span=10):
    
    #
    from numpy import sort 
    
    #
    a = slm_sequence_forwards( aw,l,m,s=s,sc=sc,verbose=verbose,span=span)
    b = slm_sequence_backwards(aw,l,m,s=s,sc=sc,verbose=verbose,span=span)
    
    #
    c = {} 
    keys = sort(list(a.keys())) 
    for key in keys: 
        if key <= l :
            c[key] = a[key]
        else:
            c[key] = b[key]
    
    #
    return c



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
''' Calculate set of spheroidal harmonic duals '''
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
def slm_dual_set( jf, l, m, n, theta, phi, s=-2, lmax=8, lmin=2, aw=None, verbose=False, tol=None, conjugate_expansion=True ):
    '''
    Construct set of dual-spheroidals
    '''
    
    error('please use calc_adjoint_slm_subset')

    # Import usefuls
    from numpy import array,pi,arange,linalg,dot,conj,zeros,double

    # Warn if l is large
    if lmax>8: warning('Input of lmax>8 found. The output of this function is increasingly inaccurate for lmax>8 due to numerical errors at the default value of the tol keyword input.')
        
    #
    if not isinstance(phi,(float,int,double)):
        error('phi must be number; zero makes sense as the functions phi dependence is exp(1j*m*phi), and so can be added externally')

    #
    if lmin<abs(m): lmin = max(abs(m),abs(s))

    # -------------------------------------- #
    # Construct a space of spheroidals as a starting basis
    # -------------------------------------- #
    lnspace = []; nmin=0; nmax=0
    lspace = arange( lmin,lmax+1 )
    nspace = arange( nmin,nmax+1 )
    for j,ll in enumerate(lspace):
        for k,nn in enumerate(nspace):
            lnspace.append( (ll,nn) )
    Sspace = []
    for ln in lnspace:
        ll,nn = ln
        Sspace.append( slm( jf, ll, m, nn, theta, phi, s=s, verbose=verbose, aw=aw, use_nr_convention=False, tol=tol ) )
    Sspace = array(Sspace)

    # Handle sanity-check option for whether to expand in spheroidals or their complex conjugates
    if conjugate_expansion:

        ##########################################
        # Expand in spheroidal conjugates        #
        ##########################################

        # -------------------------------------- #
        # Construct Gram matrix for spheroidals
        # -------------------------------------- #
        u = zeros( (len(lnspace),len(lnspace)), dtype=complex )
        for j,ln1 in enumerate(lnspace):
            for k,ln2 in enumerate(lnspace):
                s1 = Sspace[j,:]
                s2 = Sspace[k,:]
                # Compute the normalized inner-product
                u[j,k] = ssprod( None, s1, s2.conj(), theta=theta, aw=None, use_nr_convention=False )
        # -------------------------------------- #
        # Invert and conjugate
        # -------------------------------------- #
        v = linalg.pinv(u)

        # -------------------------------------- #
        # Use v to project dual functions out of regular ones
        # -------------------------------------- #
        aSspace = dot(v,Sspace.conj())

    else:

        #########################################
        # Expand in regular spheroidals         #
        #########################################

        # -------------------------------------- #
        # Construct Gram matrix for spheroidals
        # -------------------------------------- #
        u = zeros( (len(lnspace),len(lnspace)), dtype=complex )
        for j,ln1 in enumerate(lnspace):
            for k,ln2 in enumerate(lnspace):
                s1 = Sspace[j,:]
                s2 = Sspace[k,:]
                u[j,k] = ssprod(jf, s1, s2, theta=theta, aw=aw, use_nr_convention=False )
        # -------------------------------------- #
        # Invert and conjugate
        # -------------------------------------- #
        v = conj(linalg.pinv(u))

        # -------------------------------------- #
        # Use v to project dual functions out of regular ones
        # -------------------------------------- #
        aSspace = dot(v,Sspace.conj())

    #
    foo,bar = {},{}
    for k,(l,n) in enumerate(lnspace):
        foo[ (l,n) if len(nspace)>1 else l ] = aSspace[k,:]
        bar[ (l,n) if len(nspace)>1 else l ] =  Sspace[k,:]
    #
    ans = {}
    ans['Slm'] = bar
    ans['AdjSlm'] = foo
    ans['lnspace'] = lnspace
    ans['Sspace'] = Sspace
    ans['aSspace'] = aSspace
    ans['SGramian'] = u
    return ans



def depreciated_slm_dual_set_slow( jf, l, m, n, theta, phi, s=-2, lmax=8, lmin=2, aw=None, verbose=False ):
    '''
    Construct set of dual-spheroidals
    Things that could spped this function up:
    * calculate Sspace first, and then use it to directly calculate gramian rather than calling ssprod
    '''
    # Import usefuls
    from numpy import array,pi,arange,linalg,dot,conj,zeros
    #error('This function is still being written')
    # -------------------------------------- #
    # Construct Gram matrix for spheroidals
    # -------------------------------------- #
    lnspace = []; nmin=0; nmax=0
    lspace = arange( lmin,lmax+1 )
    nspace = arange( nmin,nmax+1 )
    for j,ll in enumerate(lspace):
        for k,nn in enumerate(nspace):
            lnspace.append( (ll,nn) )
    u = zeros( (len(lnspace),len(lnspace)), dtype=complex )
    for j,ln1 in enumerate(lnspace):
        for k,ln2 in enumerate(lnspace):
            l1,n1 = ln1
            l2,n2 = ln2
            z1 = (l1,m,n1)
            z2 = (l2,m,n2)
            u[j,k] = ssprod(jf, z1, z2, N=2**10, aw=aw, use_nr_convention=False)
    # -------------------------------------- #
    # Invert and conjugate
    # -------------------------------------- #
    v = conj(linalg.pinv(u))
    # -------------------------------------- #
    # Construct a space of spheroidals as a starting basis
    # -------------------------------------- #
    Sspace = []
    for ln in lnspace:
        ll,nn = ln
        Sspace.append( slm( jf, ll, m, nn, theta, phi, s=s, verbose=verbose, aw=aw, use_nr_convention=False ) )
    Sspace = array(Sspace)
    # -------------------------------------- #
    # Use v to project dual functions out of regular ones
    # -------------------------------------- #
    aSspace = dot(v,Sspace)
    #
    foo,bar = {},{}
    for k,(l,n) in enumerate(lnspace):
        foo[ (l,n) if len(nspace)>1 else l ] = aSspace[k,:]
        bar[ (l,n) if len(nspace)>1 else l ] =  Sspace[k,:]
    #
    ans = {}
    ans['Slm'] = bar
    ans['AdjSlm'] = foo
    ans['lnspace'] = lnspace
    ans['Sspace'] = Sspace
    ans['aSspace'] = aSspace
    ans['SGramian'] = u
    return ans



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
''' Spheroidal Harmonic angular function via leaver's sum FOR m>0 '''
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
def __slpm_legacy__( jf,               # Dimentionless spin parameter
          l,
          m,
          n,
          theta,            # Polar spherical harmonic angle
          phi,              # Azimuthal spherical harmonic angle
          s       = -2,     # Spin weight
          __rescale__ = True, # Internal usage only: Recover scaling of spherical harmonics in zero spin limit
          norm = True,     # If true, normalize the waveform
          output_iterations = False,
          __aw_sc__ = None, # optional frequency and separation parameter pair
          london = False,   # toggle for which series solution to use (they are all largely equivalent)
          aw = None,        # when not none, the separation constant will be found automatically
          adjoint=False,
          tol=None,
          sc=None,
          verbose = False ):# Be verbose

    #
    from positive import red
    from positive import leaver as lvr
    from positive import rgb,lim,leaver_workfunction,cyan,alert,pylim,sYlm,error,internal_ssprod
    from numpy import complex256, cos, ones, mean, isinf, pi, exp, array, ndarray, unwrap, angle, linalg, sqrt, linspace, sin, float128
    # from scipy.misc import factorial as f
    from scipy.integrate import trapz

    #
    if m<0: error('this low level routine is only valid for m>0; use slm() for general m cases')

    # Ensure that thet is iterable
    if not isinstance(theta,ndarray):
        theta = array([theta])

    #
    if __aw_sc__ is None:

        if aw is None:

            # Validate the spin input
            if isinstance(jf,int): jf = float(jf)

            # Use tabulated cw and sc values from the core package
            cw,sc = lvr( jf, l, m, n, s=s, __legacy__ = True )

            # Validate the QNM frequency and separation constant used
            lvrtol=1e-4
            lvrwrk = linalg.norm( leaver_workfunction(jf,l,m,[cw.real,cw.imag,sc.real,sc.imag],s=s, london=1) )
            if lvrwrk>lvrtol:
                msg = 'There is a problem in '+cyan('kerr.core.leaver')+'. The values output are not consistent with leaver''s characteristic equations within %f.\n%s\n# The mode is (jf,l,m,n)=(%f,%i,%i,%i)\n# The leaver_workfunction value is %s\n%s\n'%(lvrtol,'#'*40,jf,l,m,n,red(str(lvrwrk)),'#'*40)
                error(msg,'slm')
            # If verbose, check the consisitency of the values used
            if verbose:
                msg = 'Checking consistency of QNM frequncy and separation constant used against Leaver''s constraint equations:\n\t*  '+cyan('leaver_workfunction(jf=%1.4f,l,m,[cw,sc]) = %s'%(jf,lvrwrk))+'\n\t*  cw = %s\n\t*  sc = %s'%(cw,sc)
                alert(msg,'slm')

            # Define dimensionless deformation parameter
            aw = complex256( jf*cw )

        else:

            #
            from numpy import complex128 as dtyp
            if sc is None:
                #sc = slmcg_eigenvalue( dtyp(aw), s, l, m)
                sc = sc_leaver( dtyp(aw), l, m, s, verbose=verbose, adjoint=False, london=london)[0]

    else:

        # Use input values
        aw,sc = __aw_sc__


    #
    from numpy import complex128 as dtyp
    sc2 = sc_leaver( dtyp(aw), l, m, s, verbose=verbose,adjoint=False, london=london, tol=tol)[0]
    #sc2 = slmcg_eigenvalue( dtyp(aw), s, l, m)
    if abs(sc2-sc)>1e-3:
        print('aw  = '+str(aw))
        print('sc_input  = '+str(sc))
        print('sc_leaver = '+str(sc2))
        print('sc_london = '+str(sc_london( aw,l,m,s )[0]))
        print('err = '+str(abs(sc2-sc)))
        warning('input separation constant not consistent with angular constraint, so we will use a different one to give you an answer that converges.')
        sc = sc2

    # ------------------------------------------------ #
    # Angular parameter functions
    # ------------------------------------------------ #

    # Retrieve desired information from central location
    k1,k2,alpha,beta,gamma,scale_fun_u,u2v_map,theta2u_map = leaver_ahelper( l,m,s,aw,sc, london=london, verbose=verbose )

    # ------------------------------------------------ #
    # Calculate the angular eighenfunction
    # ------------------------------------------------ #

    # Variable map for theta
    u = theta2u_map(theta)
    # Calculate the variable used for the series solution
    v = u2v_map( u )

    # the non-sum part of eq 18
    X = ones(u.shape,dtype=complex256)
    X = X * scale_fun_u(u)

    # initial series values
    a0 = 1.0 # a choice, setting the norm of Slm

    a1 = -a0*beta(0)/alpha(0)

    C = 1.0
    C = C*((-1)**(max(-m,-s)))*((-1)**l)

    # Apply normalization
    if norm:
        if __aw_sc__: error('this function is not configured to output normalized harmonics when manually specifying the deformation parameter and separation consant')
        z = (l,m,n)
        if norm == -1:
            # Scale such that spherical counterpart is normalized
            C /= ysprod(jf,l,m,z,s,london=london)
        else:
            C /= sqrt( internal_ssprod(jf,z,z,s,london=london,aw=aw) )


    # the sum part
    done = False
    Y = a0*ones(u.shape,dtype=complex256)
    Y = Y + a1*v
    k = 1
    kmax = 5e3
    err,yy = [],[]
    et2=1e-8 if tol is None else tol
    max_a = max(abs(array([a0,a1])))
    v_pow_k = v
    while not done:
        k += 1
        j = k-1
        a2 = -1.0*( beta(j)*a1 + gamma(j)*a0 ) / alpha(j)
        v_pow_k = v_pow_k*v
        dY = a2*v_pow_k
        Y += dY
        xx = max(abs( dY ))

        #
        if output_iterations:
            yy.append( C*array(Y)*X*exp(1j*m*phi) )
            err.append( xx )

        k_is_too_large = k>kmax
        done = (k>=l) and ( (xx<et2 and k>30) or k_is_too_large )
        done = done or xx<et2
        a0 = a1
        a1 = a2

    # together now
    S = X*Y*exp(1j*m*phi)

    # Use same sign convention as spherical harmonics
    # e.g http://en.wikipedia.org/wiki/Spin-weighted_spherical_harmonics#Calculating
    S = C * S

    # Warn if the series did not appear to converge
    if k_is_too_large:
        warning('The while-loop exited becuase too many iterations have passed. The series may not converge for given inputs.')

    #
    return S,yy,err



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
''' Spheroidal Harmonic angular function via leaver's sum '''
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
def __slm_legacy__(  jf,               # Dimentionless spin parameter
          l,
          m,
          n,
          theta,            # Polar spherical harmonic angle
          phi,              # Azimuthal spherical harmonic angle
          s       = -2,     # Spin weight
          plot    = False,  # Toggel for plotting
          __rescale__ = True, # Internal usage only: Recover scaling of spherical harmonics in zero spin limit
          norm = True,     # If true, normalize the waveform
          ax = None,        # axes handles for plotting to; must be length 1(single theta) or 2(many theta)
          __aw_sc__ = None,
          aw = None,        # when not none, the separation constant will be found automatically
          london = False,
          use_nr_convention = True, # Toggle whether to use NR convention for multipoles
          tol = None,
          sc=None,
          verbose = False ):# Be verbose

    # Setup plotting backend
    if plot:
        import matplotlib as mpl
        mpl.rcParams['lines.linewidth'] = 0.8
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.size'] = 16
        mpl.rcParams['axes.labelsize'] = 16

    #
    from positive import red
    from positive import leaver as lvr
    from positive import rgb,lim,leaver_workfunction,cyan,alert,pylim,sYlm,error,internal_ssprod
    from numpy import complex256,cos,ones,mean,isinf,pi,exp,array,ndarray,unwrap,angle,linalg,sqrt,linspace,sin,float128
    from matplotlib.pyplot import subplot,gca,xlabel,ylabel,xlim,ylim,title,figure,sca
    from matplotlib.pyplot import plot as plot_
    # from scipy.misc import factorial as f
    from scipy.integrate import trapz
    
    #
    if london==-4:
        warning('london=-4 is incompoatible with this function. We will set london=True')
        london=True

    #
    if m<0:
        if use_nr_convention:
            alert('Cowboy')
            S,yy,err = __slpm_legacy__( jf, l, -m, n, pi-theta, pi+phi, s=s, __rescale__=__rescale__, norm=norm, output_iterations=plot, verbose=verbose, __aw_sc__=__aw_sc__, london=london, aw=aw, tol=tol )
            S = ((-1)**(l+m)) * S.conj()
            if aw is None: warning('NR convention being used for m<0 multipole. This results in a spheroidal function that does not satisfy Teukolsky\'s equation with the given labeling. To disable this warning, use keyword input use_nr_convention=False.')
        else:
            aw = -aw
            m = -m
            s = -s
            S,yy,err = __slpm_legacy__( jf, l, m, n, theta, phi, s=s, __rescale__=__rescale__, norm=norm, output_iterations=plot, verbose=verbose, __aw_sc__=__aw_sc__, london=london, aw=aw, tol=tol, sc=sc )
    else:
        S,yy,err = __slpm_legacy__( jf, l, m, n, theta, phi, s=s, __rescale__=__rescale__, norm=norm, output_iterations=plot, verbose=verbose, __aw_sc__=__aw_sc__, london=london, aw=aw, tol=tol, sc=sc )

    #
    if plot:
        def ploterr():
            plot_( err, '-ob',mfc='w',mec='b' )
            gca().set_yscale("log", nonposy='clip')
            title( '$l = %i$, $m = %i$, $jf = %s$'%(l,m,'%1.4f'%jf if jf is not None else 'n/a') )
            ylabel('Error Estimate')
            xlabel('Iteration #')
        if isinstance(theta,(float,int)):
            if ax is None:
                figure(figsize=3*array([3,3]))
            else:
                sca(ax[0])
            ploterr()
        elif isinstance(theta,ndarray):
            if ax is None:
                figure(figsize=3*array([6,2.6]))
                subplot(1,2,1)
            else:
                sca(ax[0])
            ploterr()
            if ax is not None:
                sca(ax[-1])
            else:
                subplot(1,2,2)
            clr = rgb( max(len(yy),1),reverse=True )
            for k,y in enumerate(yy):
                plot_(theta,abs(y),color=clr[k],alpha=float(k+1)/len(yy))
            plot_(theta,abs(S),'--k')
            pylim(theta,abs(S))
            fs=20
            xlabel(r'$\theta$',size=fs)
            ylabel(r'$|S_{%i%i%i}(\theta,\phi)|$'%(l,m,n),size=fs )
            title(r'$\phi=%1.4f$'%phi)

    #
    if len(S) is 1:
        return S[0]

    #
    return S
