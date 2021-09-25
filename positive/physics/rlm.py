#
from __future__ import print_function
from . import *
from positive.api import *
from positive.plotting import *
from positive.learning import *
from positive.physics import *



#
def rlm_change_convention(a,cw,M):
    '''
    Change form M=1 to M=1/2 convention so that Leaver's solutions may be used.
    '''
    
    #
    if M!= 1.0: error('This fucntion those calling it use mass scaling convention M=1.')
    
    #
    scale = 1.0 / (2 * M)
    leaver_cw = cw / scale 
    leaver_a  = a  * scale 
    leaver_M  = M  * scale 
    
    #
    return leaver_a,leaver_cw,leaver_M

#
def rlm_leading_order_scale(x,M,a,w,m,s):
    
    #
    from numpy import exp,sqrt
    
    #
    delta = sqrt( M*M - a*a )
    
    #
    rp = M + delta
    rm = M - delta
    
    #
    horizon_lo_scale = x**(-s + ((1j*1.0/2)*(a*m - 2*M*rp*w))*1.0/delta)
    infinity_lo_scale = exp((1j*w*(-rp + rm*x))*1.0/(-1 + x)) * (-1 + x)**(1 + 2*s - (2*1j)*M*w)
    
    #
    lo_scale = exp((1j*w*(-rp + rm*x))*1.0/(-1 + x))*(-1 + x)**(1 + 2*s - (2*1j)*M*w)*x**(-s - ((1j*1.0/2)*(-(a*m) + 2*M*rp*w))*1.0/delta)
    
    #
    return lo_scale, horizon_lo_scale, infinity_lo_scale
    

#
def rlm_sequence_backwards(a,cw,sc,l,m,s=-2,verbose=False,span=50,london=False):
    '''
    Function for calulating recursive sequence elements for Leaver's representation of the Teukolsky function (ie solutions to Teukolsky's radial equation).
    '''
    
    # 
    b = slm_sequence_backwards(None,l,m,s=s,sc=sc,verbose=verbose,span=span,__COMPUTE_RADIAL_SEQUENCE__=True,__a__=a,__cw__=cw)
    b_ = { k: b[k]/b[0] for k in b }
    
    #
    return b_
#
def rlm_sequence_forwards(a,cw,sc,l,m,s=-2,verbose=False,span=50,london=False):
    '''
    Function for calulating recursive sequence elements for Leaver's representation of the Teukolsky function (ie solutions to Teukolsky's radial equation).
    '''
    
    
    # 
    b = slm_sequence_forwards(None,l,m,s=s,sc=sc,verbose=verbose,span=span,__COMPUTE_RADIAL_SEQUENCE__=True,__a__=a,__cw__=cw)
    
    b_ = { k: b[k]/b[0] for k in b }
    
    #
    return b_
    

# Helper function for calculating solutions to Teukolsky's raial equation using Leaver's ansatz
def rlm_helper( geometric_a,geometric_cw,sc, l, m, x, s, geometric_M=1, tol=None, verbose=False,london=False, full_output = False,conjugate=False,pre_solution=None ):
    
    '''
    RLM_HELPER
    ---
    LOW LEVEL function for evaluating tuekolsky radial function for a give oblateness.
    
    USAGE
    ---
    foo = rlm_helper( a,cw,sc, l, m, x, s, tol=None, verbose=False,london=False, full_output = False )
    
    NOTE that x = (r-rp) / (r-rm) where rp and rm are Kerr outer and inner radii and r is Boyer-Lidquist r
    
    IF full output, then foo is a dictionary of information inluding field Slm containing the harmonic 
    
    ELSE foo is a tuple of the radial spheroidal harmonic and its eigenvalue (Slm,Alm)
    
    IF full output, then foo is a dictionary with the following fields (more may be present as this functions is updated):
    
    * Rlm,    The radial spheroidal harmonic. NOT normalized.
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
    from numpy import complex256, cos, ones, mean, isinf, pi, exp, array, ndarray, unwrap, angle, linalg, sqrt, linspace, sin, float128, zeros_like, sort, ones_like
    from scipy.integrate import trapz
    from numpy import complex128 as dtyp

    # ------------------------------------------------ #
    # Calculate the radial eigenfunction
    # aka the Teukolsky function
    # ------------------------------------------------ #
    
    #
    a,cw,M = rlm_change_convention(geometric_a,geometric_cw,geometric_M)
    
    #
    aw = a*cw
    
    #
    if M!=0.5:
        error('M must be 0.5 ie Leavers convention')
    
    
    # # Precompute useful quantities for the overall prefactor
    # # NOTE that this prefactor encodes the QNM radial boundary conditions
    # # --
    # # M = 0.5 # NOTE that a and cw must be defined uner this convention for consistency
    # # a = a/2
    # # cw *= 2
    # b = sqrt(1-4*a*a) # sqrt( M*M - a*a )
    # rp = (1 + b)*0.5 
    # rm = (1 - b)*0.5
    # sp = (-(a*m) + rp*cw)*1.0/b
    # k0 = -s - 1j*sp
    # k1 = -1 - s + 1j*cw + 1j*sp
    # r = (-rp + rm*x)*1.0/(-1 + x)
    # # Compute the prefactor. This is Leaver's prefactor but in x=(r-rp)/(r-rm) coordinates
    # if pre_solution is None:
    #     #
    #     pre_solution = ((r-rp)**k0) * ((r-rm)**k1) * exp(1j*r*cw)
    #     # pre_solution = x**k0 * (x-1)**k1 * exp(-1j*cw*r)
    
    #
    if pre_solution is None:
        pre_solution,_,_ = rlm_leading_order_scale(x,M,a,cw,m,s)

    # the non-sum part 
    X = pre_solution*ones_like(x)
    
    #
    Y = zeros_like(x,dtype=complex)
    
    # NOTE that this should be generalized to look for poor convergence. This is currently not done becuase the recursion relation is solved backwards, thus enforcing convergence. However, it is possible that "span=200" (i.e. 200 terms) is not sufficient for accuracy.
    b = rlm_sequence_backwards(a,cw,sc,l,m,s=s,span=200)
    
    #
    kspace = sort(list(b.keys()))
    last_pow_x = ones_like(x)
    for k in kspace:
        Y += last_pow_x * b[k]
        last_pow_x *= x

    # together now
    R = X*Y
        
    #
    if full_output:
        foo['Rlm'] = R 
        foo['Iterant_Rlm'] = yy 
        foo['Iterant_Error'] = err 
        foo['Alm'] = sc
        return foo
    else:
        return R

