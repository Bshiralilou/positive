#
from __future__ import print_function
from . import *
from positive.api import *
from positive.plotting import *
from positive.learning import *
from positive.physics import *



#
def tkangular( S, theta, acw, m, s=-2, separation_constant=None,adjoint=False,flip_phase_convention=False  ):
    '''
    Apply Teukolsy's angular operator to input. NOTE that ell does not appear explicitely in the equation. It is instead encapsulated by the separation constant.
    '''
    #
    from numpy import sin, cos, isnan, diff,sign,pi
    #
    mask = (theta!=0) & (theta!=pi)
    if separation_constant is None:
        separation_constant = sc_leaver( acw, 2, m, s,verbose=False,adjoint=False)[0]
    A = separation_constant
    #
    sn = sin(theta)
    cs = cos(theta)
    
    #
    u = 1
    if flip_phase_convention:
        u = -1

    dS = spline_diff(theta,S)
    D2S = spline_diff( theta, sn * dS ) / sn
    #
    mscs = m+s*cs
    acwcs = acw*cs
    VS = (  -mscs*mscs/(sn*sn) + s + acwcs*( acwcs - 2*s ) + A  ) * S

    #
    ans = (D2S + VS)[mask]

    #
    return ans



#
def tkradial( R, x, M, a, cw, m, s, separation_constant,flip_phase_convention=False  ):
    '''
    Apply Teukolsy's angular operator to input. NOTE that ell does not appear explicitely in the equation. It is instead encapsulated by the separation constant.
    '''
    
    #
    from numpy import sin, cos, isnan, diff,sign,pi, sqrt
    
    #
    if M!=0.5:
        error('this function only works in the M=1/2 convention, but M is %f'%M)
    if M<a:
        error('BH spin cannot be greater than its mass')
    
    #
    Alm = separation_constant
    
    #
    D0R = R
    D1R = spline_diff(x,R, n=1)
    D2R = spline_diff(x, R, n=2)
    
    #
    b = sqrt(M*M - a*a)
    rp = M+b
    rm = M-b
    
    ####
    
    P0 = -Alm - a**2*cw**2 - ((2*1j)*cw*s*(rp - rm*x))*1.0/(-1 + x) + (-4*a*cw*m*M*rp + m**2*rm*rp + cw**2*rm**2*rp**2 + 2*cw**2*rm*rp**3 + cw**2*rp**4 - (2*1j)*a*m*M*s + (2*1j)*a*m*rp*s + (2*1j)*cw*M*rm*rp*s - (2*1j)*cw*M*rp**2*s + 4*a*cw*m*M*rm*x + 12*a*cw*m*M*rp*x - 4*m**2*rm*rp*x - 8*cw**2*rm**2*rp**2*x - 8*cw**2*rm*rp**3*x + (8*1j)*a*m*M*s*x - (2*1j)*a*m*rm*s*x - (6*1j)*a*m*rp*s*x - (4*1j)*cw*M*rm*rp*s*x + (4*1j)*cw*M*rp**2*s*x - 12*a*cw*m*M*rm*x**2 - 12*a*cw*m*M*rp*x**2 + 6*m**2*rm*rp*x**2 + 2*cw**2*rm**3*rp*x**2 + 20*cw**2*rm**2*rp**2*x**2 + 2*cw**2*rm*rp**3*x**2 - (12*1j)*a*m*M*s*x**2 + (6*1j)*a*m*rm*s*x**2 - (2*1j)*cw*M*rm**2*s*x**2 + (6*1j)*a*m*rp*s*x**2 + (4*1j)*cw*M*rm*rp*s*x**2 - (2*1j)*cw*M*rp**2*s*x**2 + 12*a*cw*m*M*rm*x**3 + 4*a*cw*m*M*rp*x**3 - 4*m**2*rm*rp*x**3 - 8*cw**2*rm**3*rp*x**3 - 8*cw**2*rm**2*rp**2*x**3 + (8*1j)*a*m*M*s*x**3 - (6*1j)*a*m*rm*s*x**3 + (4*1j)*cw*M*rm**2*s*x**3 - (2*1j)*a*m*rp*s*x**3 - (4*1j)*cw*M*rm*rp*s*x**3 - 4*a*cw*m*M*rm*x**4 + cw**2*rm**4*x**4 + m**2*rm*rp*x**4 + 2*cw**2*rm**3*rp*x**4 + cw**2*rm**2*rp**2*x**4 - (2*1j)*a*m*M*s*x**4 + (2*1j)*a*m*rm*s*x**4 - (2*1j)*cw*M*rm**2*s*x**4 + (2*1j)*cw*M*rm*rp*s*x**4)*1.0/((rm - rp)**2*(-1 + x)**2*x)
    
    P1 = (-2*(-1 + x)*(M - rp + M*s - rp*s - M*x + rp*x - M*s*x + rm*s*x))*1.0/(rm - rp)
    
    P2 = (-1 + x)**2*x
    
    ####
    
    # #
    # P0 = -Alm - rm*rp*cw**2 + ((2*1j)*s*cw*(-rp + rm*x))*1.0/(-1 + x) + ((-1 + x)**2*(m**2*rm*rp + (2*a*m*cw*(rp - rm*x))*1.0/(-1 + x) + cw**2*(rm*rp + (rp - rm*x)**2*1.0/(-1 + x)**2)**2 + (1j*s*(a*m*(-1 + x)*(1 - 2*rp - x + 2*rm*x) + (rm - rp)*cw*(rp - rm*x**2)))*1.0/(-1 + x)**2))*1.0/((rm - rp)**2*x)
    
    # #
    # P1 = ((-1 + x)*(-1 + 2*rp*(1 + s - x) + x + s*(-1 + x - 2*rm*x)))*1.0/(rm - rp)
    
    # #
    # P2 = (-1 + x)**2*x
    
    
    ####
    
    # #
    # b = sqrt(1 - 4*a*a)
    # rp = 0.5*(1+b)
    # rm = 0.5*(1-b)

    # #
    # x[x == 1] = 1-1e-8
    # x[x == 0] = 1e-8
    
    # #
    # P0 = ((-1j - 2*cw)*(4*a**2*cw + cw*(-2 + x) - 2*a*m*(-1 + x)) - b*(1 + Alm - (2*1j)*cw - 4*cw**2 + a**2*cw**2 + 2*a*cw*m + s + (1j + 2*cw)*(cw + 1j*(1 + s))*x))*1.0/b
    
    # #
    # P1 = ((-1j)*(cw - 2*a*m*(-1 + x)**2 + 8*a**2*cw*x + cw*(-4 + x)*x + 1j*b*(-1 + x)*(-1 + 1j*cw + s + (3 - (3*1j)*cw + s)*x)))*1.0/b
    
    # #
    # P2 = (-1 + x)**2*x
    
    
    ####

    #
    ans = P0*D0R + P1*D1R + P2*D2R

    #
    return ans




#
def tkradialr( R, x, M, a, cw, m, s, separation_constant,flip_phase_convention=False  ):
    '''
    Apply Teukolsy's angular operator to input. NOTE that ell does not appear explicitely in the equation. It is instead encapsulated by the separation constant.
    '''
    
    #
    from numpy import sin, cos, isnan, diff,sign,pi, sqrt
    
    #
    if M!=0.5:
        error('this function only works in the M=1/2 convention, but M is %f'%M)
    if M<a:
        error('BH spin cannot be greater than its mass')
    
    #
    Alm = separation_constant
    
    #
    x[x == 1] = 1-1e-8
    x[x == 0] = 1e-8

    #
    b = sqrt(1 - 4*a*a)
    rp = 0.5*(1+b)
    rm = 0.5*(1-b)
    w = cw
    
    #
    r = (-rp + rm*x)*1.0/(-1 + x)
    
    #
    D0R = R
    D1R = spline_diff(r,R, n=1)
    D2R = spline_diff(r, R, n=2)
    
    #
    P0 = -Alm + (2*1j)*r*s*w - rm*rp*w**2 + (m**2*rm*rp - 2*a*m*r*w + (r**2 + rm*rp)**2*w**2 + 1j*s*(a*m*(-1 + 2*r) - r**2*w + rm*rp*w))*1.0/((r - rm)*(r - rp))
    
    #
    P1 = (-1 + 2*r)*(1 + s)
    
    #
    P2 = (r - rm)*(r - rp)

    #
    ans = P0*D0R + P1*D1R + P2*D2R

    #
    return ans



#
def tkradial_regularized( G, x, M, a, cw, m, s, separation_constant,flip_phase_convention=False  ):
    '''
    Apply Teukolsy's angular operator to input -- the operator here is in fractioanl radial coordinates and has had the singular exponents applied to the presolution such that the potential is well behaved. NOTE that ell does not appear explicitely in the equation. It is instead encapsulated by the separation constant.
    '''
    
    #
    from numpy import sin, cos, isnan, diff,sign,pi, sqrt
    
    #
    if M!=0.5:
        error('this function only works in the M=1/2 convention, but M is %f'%M)
    if M<a:
        error('BH spin cannot be greater than its mass')
    
    #
    Alm = separation_constant
    
    #
    D0G = G
    D1G = spline_diff(x,G, n=1)
    D2G = spline_diff(x, G, n=2)
    
    #
    b = sqrt(1 - 4*a*a)

    #
    x[x == 1] = 1-1e-8
    x[x == 0] = 1e-8
    
    #
    P0 = ((-1j - 2*cw)*(4*a**2*cw + cw*(-2 + x) - 2*a*m*(-1 + x)) - b*(1 + Alm + s - (2*1j)*cw + 2*a*m*cw - 4*cw**2 + a**2*cw**2 + (1j*(1 + s) + cw)*(1j + 2*cw)*x))*1.0/b
    
    #
    P1 = ((-1j)*(cw - 2*a*m*(-1 + x)**2 + 8*a**2*cw*x + cw*(-4 + x)*x + 1j*b*(-1 + x)*(-1 + s + 1j*cw + (3 + s - (3*1j)*cw)*x)))*1.0/b
    
    #
    P2 = (-1 + x)**2*x

    #
    ans = P0*D0G + P1*D1G + P2*D2G

    #
    return ans


#
def tkradial_r( R, r, M, a, cw, m, s, separation_constant,flip_phase_convention=False, convert_x = False  ):
    '''
    Apply Teukolsy's angular operator to input. NOTE that ell does not appear explicitely in the equation. It is instead encapsulated by the separation constant.
    '''
    
    #
    from numpy import sin, cos, isnan, diff,sign,pi, sqrt
    
    #
    if M!=0.5:
        error('this function only works in the M=1/2 convention, but M is %f'%M)
    
    #
    Alm = separation_constant
    
    #
    b = sqrt(1 - 4*a*a)
    rp = M*(1+b)
    rm = M*(1-b)
    
    #
    if convert_x:
        x = r
        r = (-rp + rm*x)*1.0/(-1 + x)
        #print(M,a,lim(r))

    #
    D0R = R
    D1R = spline_diff(r, R, n=1)
    D2R = spline_diff(r, R, n=2)

    #
    # r[x == 1] = 1-1e-8
    r[x == 0] = 1e-8
    
    #
    P0 = -Alm - (cw**2)*rm*rp + (2*1j)*cw*r*s + (-2*a*cw*m*r + m**2*rm*rp + (cw**2)*(
        r**2 + rm*rp)**2 + 1j*(-(cw*r**2) + a*m*(-1 + 2*r) + cw*rm*rp)*s)*1.0/((r - rm)*(r - rp))
    
    #
    P1 = (-1 + 2*r)*(1 + s)
    
    #
    P2 = (r - rm)*(r - rp)

    #
    ans = P0*D0R + P1*D1R + P2*D2R

    #
    return ans

