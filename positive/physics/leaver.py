#
from __future__ import print_function
from . import *
from positive.physics import *
from positive.api import *
from positive.plotting import *
from positive.learning import *
# > > > > > > > > >  Import adjacent modules  > > > > > > > > > > #
import positive
modules = list( basename(f)[:-3] for f in glob.glob(dirname(__file__)+"/*.py") if (not ('__init__.py' in f)) and (not (__file__.split('.')[0] in f)) )
for module in modules:
    exec('from .%s import *' % module)
# > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > #


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
'''Functions for calculating QNM freuqncies parameterized by BH spin'''
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #




def leaver_dev( a, l, m, n, s, M=1.0, verbose=False, solve=False ):

    # Import usefuls
    from numpy import ndarray,array

    # Validate inputs


    # construct string for loading data

    # Define function to handle single spin and mass values
    def helper( a, l, m, n, s, M, verbose, solve ):

        # If requested, use loaded data as guess for full solver

        # Return answer for single spin value
        return (cw,sc)

    #
    if isinstance(jf,(tuple,list,ndarray)):
        #
        cw,sc = array( [ helper(a_, l, m, n, s, M, verbose, solve) for a_ in a ] )[:,:,0].T
        return cw,sc
    else:
        #
        return helper( a, l, m, n, s, M, verbose, solve )


'''
Method to load tabulated QNM data, interpolate and then output for input final spin
'''
def leaver( jf,                     # Dimensionless BH Spin
            l,                      # Polar Index
            m,                      # Azimuthal index
            n =  0,                 # Overtone Number
            p = None,               # Parity Number for explicit selection of prograde (p=1) or retrograde (p=-1) solutions.
            s = -2,                 # Spin weight
            Mf = 1.0,               # BH mass. NOTE that the default value of 1 is consistent with the tabulated data. (Geometric units ~ M_bare / M_ADM )
            use_nr_convention=False, # 
            full_output=False,
            __legacy__ = False,
            refine=False,
            __CHECK__=True,
            verbose = False ):      # Toggle to be verbose

    #
    from numpy import ndarray,array
    
    #
    if __legacy__:
        alert('Using legacy version of leaver_helper.',verbose=verbose)
        helper = __leaver_helper_legacy__
    else:
        helper = __leaver_helper__

    #
    if isinstance(jf,(tuple,list,ndarray)):
        #
        if full_output:
            cw,sc,aw = array( [ helper(jf_, l, m, n, p , s, Mf, verbose=verbose,use_nr_convention=use_nr_convention,refine=refine,full_output=True,__CHECK__=__CHECK__) for jf_ in jf ] ).T
            return cw,sc
        else:
            cw,sc = array( [ helper(jf_, l, m, n, p , s, Mf, verbose=verbose,use_nr_convention=use_nr_convention,refine=refine,__CHECK__=__CHECK__) for jf_ in jf ] ).T
            return cw,sc

    else:
        #
        return helper(jf, l, m, n, p , s, Mf, verbose=verbose,use_nr_convention=use_nr_convention,refine=refine,full_output=full_output,__CHECK__=__CHECK__)



def __leaver_helper__( jf, l, m, n =  0, p = None, s = -2, Mf = 1.0, verbose = False,use_nr_convention=False,full_output=False,refine=False,__CHECK__=True):


    # Import useful things
    import os,positive
    from scipy.interpolate import InterpolatedUnivariateSpline as spline
    from numpy import loadtxt,exp,sign,abs,ndarray,array,complex128,complex256
    from numpy.linalg import norm

    # Validate jf input: case of int given, make float. NOTE that there is further validation below.
    REVERT_TO_FLOAT = False
    if isinstance(jf,(int,float)): 
        jf = [float(jf)]
        REVERT_TO_FLOAT = True
    if not isinstance(jf,ndarray): jf = array(jf)
    # Valudate s input
    if abs(s) != 2: raise ValueError('This function currently handles on cases with |s|=2, but s=%i was given.'%s)
    # Validate l input
    # Validate m input
    
    #
    if not use_nr_convention:
        if not ((p is None) or (p is 0)):
            error('When not using the NR convention, p must remain None. Instead p is %i.'%p)
    else:
        if p is None:
            error('When using NR convention, p must be either 1 or -1.')

    # #%%%%%%%%%%%%%%%%%%%%%%%%%# NEGATIVE SPIN HANDLING #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    # # Define a parity value to be used later:
    # # NOTE that is p>0, then the corotating branch will be loaded, else if p<0, then the counter-rotating solution branch will be loaded.
    # if p is None:
    #     p = sign(jf) + ( jf==0 )
    # # NOTE that the norm of the spin input will be used to interpolate the data as the numerical data was mapped according to jf>=0
    # # Given l,m,n,sign(jf) create a RELATIVE file string from which to load the data
    
    # # ENFORCE positive spin convention
    # if jf<0:
    #     error('we will use the spin>0 convention. to select a retrograde mode, please set p=-1')
    
    cmd = positive.parent( positive.__path__[0] )  + 'positive/'
    #********************************************************************************#
    if use_nr_convention:
        if (p < 0) and (m!=0):
            m_label = 'mm%i'%abs(m)
        else:
            m_label = 'm%i'%abs(m)
    else:
        if jf<0:
            m *= -1
        if m < 0:
            m_label = 'mm%i'%abs(m)
        else:
            m_label = 'm%i'%abs(m)
    # m_label = 'm%i'%abs(m) if (p>=0) or (abs(m)==0) else 'mm%i'%abs(m)
    #********************************************************************************#
    data_location = os.path.join( cmd,'data/kerr/l%i/n%il%i%s.dat' % (l,n,l,m_label) )
    if use_nr_convention:
        alert(magenta('Using NR convention ')+'for organizing solution space and setting the sign of the QNM freuency imaginary part (via nrutils convention).',verbose=verbose)
    else:
        alert(magenta('NOT using NR convention ')+'for organizing solution space and setting the sign of the QNM freuency imaginary part.',verbose=verbose)
    alert('Loading: %s'%cyan(data_location),verbose=verbose)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    # Validate data location
    if not os.path.isfile(data_location): raise ValueError('The OS reports that "%s" is not a file or does not exist. Either the input QNM data is out of bounds (not currently stored in this repo), or there was an input error by the user.' % green(data_location) )

    # Load the QNM data
    data = loadtxt( data_location )

    # Extract spin, frequencies and separation constants
    JF = data[:,0]
    CW = data[:,1] + 1j*data[:,2] 
    CS = data[:,3] + 1j*data[:,4] 

    # Validate the jf input
    njf = abs(jf) # NOTE that the calculations were done using the jf>=0 convention
    if min(njf)<min(JF) or max(njf)>max(JF):
        warning('The input value of |jf|=%1.4f is outside the domain of numerical values [%1.4f,%1.4f]. Note that the tabulated values were computed on jf>0.' % (njf,min(JF),max(JF)) )

    # Here we rescale to a unit mass. This is needed because leaver's convention was used to perform the initial calculations.
    M_leaver = 0.5
    CW *= M_leaver

    # Interpolate/Extrapolate to estimate outputs
    cw = spline( JF, CW.real )(njf) + 1j*spline( JF, CW.imag )( njf )
    cs = spline( JF, CS.real )(njf) + 1j*spline( JF, CS.imag )( njf )
    
    # Use external function to calculate separation constant based on a*cw 
    # NOTE that sc_london is used for speed; interpolatoin is faster but less accurate
    # cs,_,_,_ = sc_london( njf*cw,l,m,s,nowarn=True); cs = array([cs])
    # cs = array([slmcg_eigenvalue( njf*cw, s, l, m)],dtype=complex128)

    # If needed, use symmetry relationships to get correct output.
    def qnmflip(CW,CS):
        return -cw.conj(),cs.conj()

    # Handle positive spin weight
    if s>0:
        cs = cs + 2*abs(s)
        cw = cw.conj()
        cs = cs.conj()
        
    #
    if m<0:
        
        # NOTE that this is needed when NOT using the NR convention
        cw,cs = qnmflip(cw,cs)
    
    #
    if REVERT_TO_FLOAT:
        cw,cs = cw[0],cs[0]
        
    #
    use_nrutils_sign_convention = use_nr_convention
    if use_nrutils_sign_convention:
        cw = cw.conj()
        cs = cs.conj()
    
    # Validate the QNM frequency and separation constant found
    lvrtol=1e-6
    # NOTE that leaver_workfunction evals Leaver's angular and radial constraint functions. Only the angular problem is invariant under conjugation of the QNM ferquency and separation constant. Thus we treat it's test differently below.
    if __CHECK__:
        lvrwrk_vector = leaver_workfunction(jf,l,m,[cw.real,cw.imag,cs.real,cs.imag],s=s, london=1, use_nr_convention = use_nrutils_sign_convention)
        lvrwrk = norm( lvrwrk_vector )
      
        #
        refine = refine or (lvrwrk>lvrtol)
        if refine:
            #
            print_method = warning if (lvrwrk>lvrtol) else alert
            print_method('Refining results becuase' + ( 'we have been aske to by the user.' if (lvrwrk<lvrtol) else 'the interpolated values are below the internally set accuracy standard.'  ), verbose=verbose  )
            #
            guess = array([cw.real,cw.imag,cs.real,cs.imag],dtype=float)
            refined_cw,refined_cs,refined_lvrwrk,retry = lvrsolve(jf,l,m,guess,tol=1e-8,s=s, use_nr_convention = use_nrutils_sign_convention)
            #
            cw = refined_cw
            cs = refined_cs
            lvrwrk = refined_lvrwrk
            
        #
        if lvrwrk>lvrtol:
            alert( (l,m,n,p) )
            alert(cw)
            alert(cs)
            alert( lvrwrk_vector )
            msg = 'There is a bug. The values output are not consistent with Leaver\'s characteristic equations within %f.\n%s\n# The mode is (jf,l,m,n,p)=(%f,%i,%i,%i,%s)\n# The leaver_workfunction value is %s\n%s\n'%(lvrtol,'#'*40,jf,l,m,n,p,red(str(lvrwrk)),'#'*40)
            error(msg,'slm')
        else:
            alert(blue('Check Passed:')+'Frequency and separation const. %s with (l,m)=(%i,%i). Zero is approx %s.'% (bold(blue('satisfy Leaver\'s equations')),l,m,magenta('%1.2e'%(lvrwrk))),verbose=verbose )
        
    # Here we scale the frequency by the BH mass according to the optional Mf input
    CW = cw/Mf
    CS = cs    # NOTE that not scaling is needed for the separation constant

    #
    if full_output:
        aw = jf*cw
        return CW,CS,aw 
    else:
        return CW,CS



def __leaver_helper_legacy__( jf, l, m, n =  0, p = None, s = -2, Mf = 1.0, verbose = False,use_nr_convention=None,full_output=None,refine=False,__CHECK__=None):


    # Import useful things
    import os,positive
    from scipy.interpolate import InterpolatedUnivariateSpline as spline
    from numpy import loadtxt,exp,sign,abs,ndarray,array,complex128,complex256
    from numpy.linalg import norm

    # Validate jf input: case of int given, make float. NOTE that there is further validation below.
    REVERT_TO_FLOAT = False
    if isinstance(jf,(int,float)): 
        jf = [float(jf)]
        REVERT_TO_FLOAT = True
    if not isinstance(jf,ndarray): jf = array(jf)
    # Valudate s input
    if abs(s) != 2: raise ValueError('This function currently handles on cases with |s|=2, but s=%i was given.'%s)
    # Validate l input
    # Validate m input

    #%%%%%%%%%%%%%%%%%%%%%%%%%# NEGATIVE SPIN HANDLING #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    # Define a parity value to be used later:
    # NOTE that is p>0, then the corotating branch will be loaded, else if p<0, then the counter-rotating solution branch will be loaded.
    if p is None:
        p = sign(jf) + ( jf==0 )
    # NOTE that the norm of the spin input will be used to interpolate the data as the numerical data was mapped according to jf>=0
    # Given l,m,n,sign(jf) create a RELATIVE file string from which to load the data
    cmd = positive.parent( positive.__path__[0] )  + 'positive/'
    #********************************************************************************#
    m_label = 'm%i'%abs(m) if (p>=0) or (abs(m)==0) else 'mm%i'%abs(m)
    #********************************************************************************#
    data_location = '%s/data/kerr/l%i/n%il%i%s.dat' % (cmd,l,n,l,m_label)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    # Validate data location
    if not os.path.isfile(data_location): raise ValueError('The OS reports that "%s" is not a file or does not exist. Either the input QNM data is out of bounds (not currently stored in this repo), or there was an input error by the user.' % green(data_location) )

    # Load the QNM data
    data = loadtxt( data_location )

    # Extract spin, frequencies and separation constants
    JF = data[:,0]
    CW = data[:,1] - 1j*data[:,2] # NOTE: The minus sign here sets a phase convention
                          # where exp(+1j*cw*t) rather than exp(-1j*cw*t)
    CS = data[:,3] - 1j*data[:,4] # NOTE: There is a minus sign here to be consistent with the line above

    # Validate the jf input
    njf = abs(jf) # NOTE that the calculations were done using the jf>=0 convention
    if min(njf)<min(JF) or max(njf)>max(JF):
        warning('The input value of |jf|=%1.4f is outside the domain of numerical values [%1.4f,%1.4f]. Note that the tabulated values were computed on jf>0.' % (njf,min(JF),max(JF)) )

    # Here we rescale to a unit mass. This is needed because leaver's convention was used to perform the initial calculations.
    M_leaver = 0.5
    CW *= M_leaver

    # Interpolate/Extrapolate to estimate outputs
    cw = spline( JF, CW.real )(njf) + 1j*spline( JF, CW.imag )( njf )
    cs = spline( JF, CS.real )(njf) + 1j*spline( JF, CS.imag )( njf )
    
    # Use external function to calculate separation constant based on a*cw 
    # NOTE that sc_london is used for speed; interpolatoin is faster but less accurate
    # cs,_,_,_ = sc_london( njf*cw,l,m,s,nowarn=True); cs = array([cs])
    # cs = array([slmcg_eigenvalue( njf*cw, s, l, m)],dtype=complex128)

    # If needed, use symmetry relationships to get correct output.
    def qnmflip(CW,CS):
        return -cw.conj(),cs.conj()
    if m<0:
        cw,cs =  qnmflip(cw,cs)
    if p<0:
        cw,cs =  qnmflip(cw,cs)

    # NOTE that the signs must be flipped one last time so that output is
    # directly consistent with the argmin of leaver's equations at the requested spin values
    cw = cw.conj()
    cs = cs.conj()

    # Here we scale the frequency by the BH mass according to the optional Mf input
    cw /= Mf

    # Handle positive spin weight
    if s>0:
        cs = cs - 2*abs(s)
        
    #
    if REVERT_TO_FLOAT:
        cw,cs = cw[0],cs[0]

    #
    return cw,cs





# Try solving the 4D equation near a single guess value [ cw.real cw.imag sc.real sc.imag ]
def lvrsolve(jf,l,m,guess,tol=1e-8,s=-2, use_nr_convention=False):

    '''
    Low-level function for numerically finding the root of leaver's equations
    '''

    # Import Maths
    from numpy import log,exp,linalg,array
    from scipy.optimize import root,fmin,minimize
    from positive import alert,red,warning,leaver_workfunction

    # Try using root
    # Define the intermediate work function to be used for this iteration
    fun = lambda STATE: log( 1.0 + abs(array(leaver_workfunction( jf,l,m, STATE, s=s, use_nr_convention=use_nr_convention ))) )
    X  = root( fun, guess, tol=tol, method='lm' )
    cw1,sc1 = X.x[0]+1j*X.x[1], X.x[2]+1j*X.x[3]
    __lvrfmin1__ = linalg.norm(array( exp(X.fun)-1.0 ))
    retry1 = ( 'not making good progress' in X.message.lower() ) or ( 'error' in X.message.lower() )


    # Try using fmin
    # Define the intermediate work function to be used for this iteration
    fun = lambda STATE: log(linalg.norm(  leaver_workfunction( jf,l,m, STATE, s=s, use_nr_convention=use_nr_convention )  ))
    X  = fmin( fun, guess, disp=False, full_output=True, ftol=tol )
    cw2,sc2 = X[0][0]+1j*X[0][1], X[0][2]+1j*X[0][3]
    __lvrfmin2__ = exp(X[1])
    retry2 = __lvrfmin2__ > 1e-3

    # Use the solution that converged the fastest to avoid solutions that have wandered significantly from the initial guess OR use the solution with the smallest fmin
    if __lvrfmin1__ < __lvrfmin2__ : # use the fmin value for convenience
        cw,sc,retry = cw1,sc1,retry1
        __lvrfmin__ = __lvrfmin1__
    else:
        cw,sc,retry = cw2,sc2,retry2
        __lvrfmin__ = __lvrfmin2__

    # Don't retry if fval is small
    if __lvrfmin__ > 1e-3:
        retry = True
        alert(red('Retrying because the trial fmin value is greater than 1e-3.'))

    # Don't retry if fval is small
    if retry and (__lvrfmin__ < 1e-4):
        retry = False
        alert(red('Not retrying becuase the fmin value is low.'))

    # Return the solution
    return cw,sc,__lvrfmin__,retry


# Extrapolative guess for gravitational perturbations. This function is best used within leaver_needle which solves leaver's equations for a range of spin values.
def leaver_extrap_guess( j, cw, sc, l, m, tol = 1e-3, d2j = 1e-6, step_sign = 1, verbose=False, plot=False, spline_order=3, monotomic_steps=False, boundary_spin=None, s=-2, adjoint=False ):

    '''
    Extrapolative guess for gravitational perturbations. This function is best used within leaver_needle which solves leaver's equations for a range of spin values.

    londonl@mit.edu 2019
    '''

    #
    from numpy import complex128, polyfit, linalg, ndarray, array, linspace, polyval, diff, sign, ones_like, linalg, hstack, sqrt
    if plot: from matplotlib.pyplot import figure,show,plot,xlim
    from scipy.interpolate import InterpolatedUnivariateSpline as spline

    #
    if not ( step_sign in (-1,1) ):
        error('step_sign input must be either -1 or 1 as it determins the direction of changes in spin')

    #
    if not isinstance(j,(list,ndarray,tuple)):
        j = array([j])
    if not isinstance(cw,(list,ndarray,tuple)):
        cw_ = ones_like(j,dtype=complex)
        cw_[-1] = cw; cw = cw_
    if not isinstance(sc,(list,ndarray,tuple)):
        sc_ = ones_like(j,dtype=complex)
        sc_[-1] = sc; sc = sc_

    #
    current_j = j[-1]
    initial_solution = [ cw[-1].real, cw[-1].imag, sc[-1].real, sc[-1].imag ]

    #
    lvrwrk = lambda J,STATE: linalg.norm(  leaver_workfunction( J,l,abs(m),STATE,s=s, adjoint=adjoint )  )

    # Make sure that starting piont satisfies the tolerance
    current_err = best_err = lvrwrk( current_j, initial_solution )

    # print current_err
    if current_err>tol:
        print(j)
        print(current_j)
        print(initial_solution)
        print(current_err)
        warning('Current solution does not satisfy the input tolerance value.')

    # Determine the polynomial order to use based on the total number of points
    nn = len(j)
    order = min( nn-1, spline_order ) # NOTE that 4 and 5 don't work as well, especially near extremal values of spin; 3may also have problems
    place = -order-1

    xx = array(j)[place:]
    yy = array(cw)[place:]
    zz = array(sc)[place:]

    alert('order is: %i'%order)
    if order>0:

        ## NOTE that coupling the splines can sometimes cause unhandled nans
        # yrspl = spline( xx, yy.real, k=order )
        # ycspl = spline( yy.real, yy.imag, k=order )
        # zrspl = spline( xx, zz.real, k=order )
        # zcspl = spline( zz.real, zz.imag, k=order )
        # yrspl_fun = yrspl
        # ycspl_fun = lambda J: ycspl( yrspl(J) )
        # zrspl_fun = zrspl
        # zcspl_fun = lambda J: zcspl( zrspl(J) )

        yrspl_fun = spline( xx, yy.real, k=order )
        ycspl_fun = spline( xx, yy.imag, k=order )
        zrspl_fun = spline( xx, zz.real, k=order )
        zcspl_fun = spline( xx, zz.imag, k=order )

    else:
        yrspl_fun = lambda J: cw[0].real*ones_like(J)
        ycspl_fun = lambda J: cw[0].imag*ones_like(J)
        zrspl_fun = lambda J: sc[0].real*ones_like(J)
        zcspl_fun = lambda J: sc[0].imag*ones_like(J)


    if plot:
        figure()
        plot( xx, yy.real, 'ob' )
        dxs = abs(diff(lim(xx)))/4 if abs(diff(lim(xx)))!=0 else 1e-4
        xs = linspace( min(xx)-dxs,max(xx)+dxs )
        plot( xs, yrspl_fun(xs), 'r' )
        xlim(lim(xs))
        if verbose:
            print(lim(xs))
            print(lim(yrspl_fun(xs)))
        show()

    guess_fit = lambda J: [ yrspl_fun(J), ycspl_fun(J), zrspl_fun(J), zcspl_fun(J) ]

    #
    k = -1; kmax = 500
    done = False
    exit_code = 0
    near_bounary = False
    best_j = current_j = j[-1]
    mind2j = 1e-7
    max_dj = sqrt(5)/42
    starting_j = j[-1]
    best_guess = guess_fit(current_j)
    if verbose: print('>> k,starting_j,starting_err = ',k,current_j,current_err)
    while not done:

        #
        k+=1
        current_j += d2j*step_sign
        if boundary_spin:
            if (boundary_spin-current_j)*step_sign < 0:
                alert('We\'re quite close to the specified boundary, so we will reduce the internal step size as to not exceed the boundary.')
                current_j -= d2j*step_sign
                print('** current_j = ',current_j)
                print('** boundary_spin = ',boundary_spin)
                new_d2j = max( min( d2j/21.0, abs( ( boundary_spin-current_j ) /21.0) ), 1e-6 )
                if new_d2j == 1e-6:
                    warning('Min value of d2j reached')
                print('** new_d2j = ',new_d2j)
                current_j = current_j + new_d2j*step_sign
                print('** new_current_j = ',current_j)
                print('** old_tol = ',tol)
                tol *= 0.01
                if tol<1e-7:
                    tol = 1e-7
                    warning('Min value of tol reached')
                print('** new_tol = ',tol)
                d2j = new_d2j
                if not near_bounary:
                    near_bounary = True
                    kmax = 4*kmax


        #
        # if d2j<mind2j: d2j = mind2j
        current_guess = guess_fit(current_j)

        #
        current_err = lvrwrk( current_j, current_guess )

        #
        if verbose: print('* k,best_j,best_err = ',k,best_j,best_err)

        #
        tolerance_is_exceeded = (current_err>tol)
        max_dj_exceeded = abs(current_j-starting_j)>max_dj

        # #
        # if (len(j)>3) and monotomic_steps:
        #     stepsize_may_increase = (abs(current_j-j[-1])+d2j) > abs(j[-1]-j[-2])
        # else:
        #     stepsize_may_increase = False

        best_j = current_j
        best_guess = current_guess
        best_err = current_err

        #
        if tolerance_is_exceeded: # or stepsize_may_increase:
            print('* k,best_j,best_err,tol,d2j = ',k,best_j,best_err,tol,d2j)
            if k>0:
                done = True
                alert('Tolerance exceeded. Exiting.')
                exit_code = 0
            else:
                warning('Tolerance exceeded on first iteration. Shrinking intermediate step size.')
                d2j = d2j/200
                if d2j<1e-20:
                    error('d2j<1e-20 -- something is wrong with the underlying equations setup')
                current_j = j[-1]
                k = -1
                best_guess = initial_solution
        else:
            if (k==kmax) and (not near_bounary):
                done = True
                warning('Max iterations exceeded. Exiting.')
                exit_code = 1

        if max_dj_exceeded:
            alert('Exiting because max_dj=%f has been exceeded'%max_dj)
            done = True
            exit_code = 0

        if abs(current_j-boundary_spin)<1e-10:
            alert('We are close enough to the boundary to stop.')
            print('$$ start_spin = ',j[-1])
            print('$$ boundary_spin = ',boundary_spin)
            print('$$ current_j = ',current_j)
            k = kmax
            done = True
            exit_code = -1

    #
    return best_j, best_guess, exit_code



# Solve leaver's equations between two spin values given a solution at a starting point
def leaver_needle( initial_spin, final_spin, l, m, initial_solution, tol=1e-3, initial_d2spin=1e-3, plot = False,verbose=False, use_feedback=True, spline_order=3, s=-2, adjoint=False ):

    '''
    Given an initial location and realted solution in the frequency-separation constant space,
    find the solution to leaver's equations between inut BH spin values for the specified l,m multipole.

    londonl@mit.edu 2019
    '''

    # Import usefuls
    from numpy import sign,array,diff,argmin,argsort,hstack

    # Determin the direction of requested changes in spin
    step_sign = sign( final_spin - initial_spin )

    # Unpack the initial solution.
    # NOTE that whether this is a solution of leaver's equations is tested within leaver_extrap_guess
    cwr,cwc,scr,scc = initial_solution
    initial_cw = cwr+1j*cwc
    initial_sc = scr+1j*scc

    # Initialize outputs
    j,cw,sc,err,retry = [initial_spin],[initial_cw],[initial_sc],[0],[False]

    #
    done = False
    k = 0
    internal_res = 24
    current_j = initial_spin
    d2j = initial_d2spin # Initial value of internal step size
    monotomic_steps = False
    while not done :

        #
        current_j,current_guess,exit_code = leaver_extrap_guess( j, cw, sc, l, m, tol=tol, d2j=d2j, step_sign=step_sign, verbose=False, plot=plot, spline_order=spline_order, boundary_spin=final_spin,s=s, adjoint=adjoint )
        if (current_j == j[-1]) and (exit_code!=1):
            # If there has been no spin, then divinde the extrap step size by internal_res.
            # Here we use internal_res as a resolution heuristic.
            d2j/=internal_res
            alert('current_j == j[-1]')
            if d2j<1e-9:
                done = True
                warning('Exiting becuase d2j is too small.')
        else:

            done = step_sign*(final_spin-current_j) < 1e-10

            j.append( current_j )
            # Set the dynamic step size based on previous step sizes
            # Here we use internal_res as a resolution heuristic.
            d2j = abs(j[-1]-j[-2])/internal_res
            if verbose: print('d2j = ',d2j)
            current_retry = True
            tol2 = 1.0e-8
            k2 = 0
            while current_retry:
                k2 += 1
                current_cw,current_sc,current_err,current_retry = lvrsolve(current_j,l,m,current_guess,s=s,tol=tol2/k2**2, adjoint=adjoint)
                if k2>6:
                    current_retry = False
                    warning('Exiting lvrsolve loop becuase a solution could not be found quickly enough.')
            if verbose: print(k,current_j,current_cw,current_sc,current_err,current_retry)
            cw.append( current_cw )
            sc.append( current_sc )
            err.append( current_err )
            retry.append( current_retry )

        #
        if d2j==0:
            warning('Exiting because the computer thinks d2j is zero')
            break


    # Convert lists to arrays with increasing spin
    j,cw,sc,err,retry = [ array( v if step_sign==1 else v[::-1] ) for v in [j,cw,sc,err,retry] ]

    #
    return j,cw,sc,err,retry

#
def greedy_leaver_needle( j,cw,sc,err,retry, l, m, plot = False, verbose=False, spline_order=3, s=-2, adjoint=False ):

    #
    from positive import spline,leaver_needle,findpeaks
    from numpy import array ,argmax,argsort,linspace,linalg,median,log, exp,hstack,diff,sort,pi,sin

    # #
    # j,cw,sc,err,retry = leaver_needle( initial_spin, final_spin, l,m, initial_solution, tol=tol, verbose=verbose, plot=plot, spline_order=spline_order )

    # ------------------------------------------------------------ #
    # Calculate the error of the resulting spline model between the boundaries
    # ------------------------------------------------------------ #
    nums = 501
    alert('Calculating the error of the resulting spline model between the boundaries',verbose=verbose)
    lvrwrk = lambda J,STATE: linalg.norm(  leaver_workfunction( J,l,abs(m),STATE,s=s, adjoint=adjoint )  )
    # js = linspace(min(j),max(j),nums)
    js =  sin( linspace(0,pi/2,nums) )*(max(j)-min(j)) + min(j)
    cwrspl = spline(j,cw.real,k=2)
    cwcspl = spline(j,cw.imag,k=2)
    scrspl = spline(j,sc.real,k=2)
    sccspl = spline(j,sc.imag,k=2)
    statespl = lambda J: [ cwrspl(J), cwcspl(J), scrspl(J), sccspl(J) ]
    errs = array( [ lvrwrk(J,statespl(J)) for J in js ] )
    pks,locs = findpeaks( log(errs) )
    tols = exp( median( pks ) )

    #
    pks,locs = findpeaks( log(errs) )
    tols = exp( median( pks ) )

    #
    alert('Using greedy process to refine solution',header=True)
    from matplotlib.pyplot import plot,yscale,axvline,axhline,show,figure,figaspect,subplot

    #
    # j,cw,sc,err,retry = [ list(v) for v in [j,cw,sc,err,retry] ]
    done = ((max(errs)/tols) < 10) or (max(errs)<1e-4)
    print(done, 'max(errs) = ',max(errs),' tols = ',tols)
    if done:
        alert('The data seems to have no significant errors due to interpolation. Exiting.')
    while not done:

        kmax = argmax( errs )

        k_right = find( j>js[kmax] )[0]
        k_left = find( j<js[kmax] )[-1]

        jr = j[k_right]
        jl = j[k_left]


        plot( js, errs )
        yscale('log')
        axvline( js[kmax], color='r' )
        plot( j, err, 'or', mfc='none' )
        axvline( jr, ls=':', color='k' )
        axvline( jl, ls='--', color='k' )
        axhline(tols,color='g')
        show()

        initial_spin = jl
        final_spin = jr
        initial_solution = [ cw[k_left].real, cw[k_left].imag, sc[k_left].real, sc[k_left].imag ]

        j_,cw_,sc_,err_,retry_ = leaver_needle( initial_spin, final_spin, l,abs(m), initial_solution, tol=tols, verbose=verbose, spline_order=spline_order,s=s, adjoint=adjoint )

        j,cw,sc,err,retry = [ hstack([u,v]) for u,v in [(j,j_),(cw,cw_),(sc,sc_),(err,err_),(retry,retry_)] ]

        #
        sortmask = argsort(j)
        j,cw,sc,err,retry = [ v[sortmask] for v in (j,cw,sc,err,retry) ]
        uniquemask = hstack( [array([True]),diff(j)!=0] )
        j,cw,sc,err,retry = [ v[uniquemask] for v in (j,cw,sc,err,retry) ]

        #
        alert('Calculating the error of the resulting spline model between the boundaries',verbose=verbose)
        lvrwrk = lambda J,STATE: linalg.norm(  leaver_workfunction( J,l,abs(m),STATE,s=s, adjoint=adjoint )  )
        js = linspace(min(j),max(j),2e2)
        js = hstack([j,js])
        js = array(sort(js))
        cwrspl = spline(j,cw.real,k=spline_order)
        cwcspl = spline(j,cw.imag,k=spline_order)
        scrspl = spline(j,sc.real,k=spline_order)
        sccspl = spline(j,sc.imag,k=spline_order)
        statespl = lambda J: [ cwrspl(J), cwcspl(J), scrspl(J), sccspl(J) ]
        #
        errs = array( [ lvrwrk(J,statespl(J)) for J in js ] )

        #
        done = max(errs)<tols

        # current_j = js[k]
        # current_state = statespl( current_j )
        # current_cw,current_sc,current_err,current_retry = lvrsolve(current_j,l,m,current_state)
        #
        # j,cw,sc,err,retry = [ list(v) for v in [j,cw,sc,err,retry] ]
        # j.append( current_j )
        # cw.append( current_cw )
        # sc.append( current_sc )
        # err.append( current_err )
        # retry.append( current_retry )
        #
        # sort_mask = argsort(j)
        # j,cw,sc,err,retry = [ array(v) for v in [j,cw,sc,err,retry] ]
        # j,cw,sc,err,retry = [ v[sort_mask] for v in [j,cw,sc,err,retry] ]
        #
        # cwrspl = spline(j,cw.real,k=2)
        # cwcspl = spline(j,cw.imag,k=2)
        # scrspl = spline(j,sc.real,k=2)
        # sccspl = spline(j,sc.imag,k=2)
        # statespl = lambda J: [ cwrspl(J), cwcspl(J), scrspl(J), sccspl(J) ]
        #
        # errs = array( [ lvrwrk(J,statespl(J)) for J in js ] )
        #
        # figure( figsize=2*figaspect(0.5) )
        # subplot(1,2,1)
        # plot( js, errs )
        # axvline( js[k], color='r' )
        # axhline(tol,color='b',ls='--')
        # axhline(tols,color='g',ls='-')
        # yscale('log')
        # subplot(1,2,2)
        # plot( js, cwrspl(js) )
        # axvline( js[k], color='r' )
        #
        # done = max(errs)<=(tols)
        # print('max(errs) = ',max(errs))
        # print('tols = ',tols)
        # show()


    #
    return j,cw,sc,err,retry


#
def teukolsky_angular_adjoint_rule(aw,Alm,allow_warning=True):
    '''
    The adjoint of teukolskys angular equation is simply its complex conjugate
    '''
    from numpy import conj

    warning('The angular ajoint should only be envoked when there is no interest in the radial problem. If this is indeed the setting in which we wish to use the angular adjoint, then please turn adjoint off via "adjoint=False", and manually conjugate the output of angular related quantities (ie frequency and separation constant). Applying radial and angular adjoint options concurrently happens to be redundant.')
    return ( conj(aw), conj(Alm) )

#
def teukolsky_radial_adjoint_rule(s,w,Alm):
    '''
    The adjoint of teukolskys radial equation is ( s->-s,A->conj(A)+2s )
    '''
    from numpy import conj
    return ( -s, w, 2*s+conj(Alm) )


# Define function that returns the recursion coefficients as functions of an integer index
def leaver_mixed_ahelper( l,m,s,awj,awk,Bjk,london=1,verbose=False,adjoint=False ):
    '''
    Let L(awj) be the spheroidal angular operator without the eigenvalue.
    Let Sk be the eigenvector of L(awk)
    Here we store the recursion functions needed to solve:
    ( L(awj) + Bjk ) Sk == 0
    Where Bjk is a complex valued constant.
    The implication here is that while L(awk)Sk = -Ak as is handled in leaver_ahelper,
    Sk is also an aigenvector of L(awj)
    '''

    error('This functionality is based on an incorrect premise. Do not use related functions and options.')

    # Import usefuls
    from numpy import exp,sqrt

    # If the adjoint equation is of interest, apply the adjoint rule
    if adjoint:
        awj,Alm = teukolsky_angular_adjoint_rule(awj,Alm)

    # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    # ANGULAR
    # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    if london==1:

        '''
        S(u) = exp( a w u ) (1+u)^k1 (1-u)^k2 Sum( a[k](1+u)^k )
        '''

        # Use abs of singular exponents AND write recursion functions so that the
        # appropriate physical solution is always used
        k1 = 0.5*abs(m-s)
        k2 = 0.5*abs(m+s)
        # As output from Mathematica
        a_alpha = lambda k:	-2*(1 + k)*(1 + k + 2*k1)
        a_beta  = lambda k:	-Bjk + k + k1 - 2*awk*(1 + 2*k + 2*k1) + k2 + (k + k1 + k2)**2 - s - (awj + s)**2
        a_gamma = lambda k:  2*(awk*(-awk + k + k1 + k2) + awj*(awj + s))
        # Exponential pre scale for angular function evaluation
        scale_fun_u = lambda COSTH: exp( awk * COSTH )
        # Define how the exopansion variable relates to u=cos(theta)
        u2v_map = lambda U: 1+U

    elif london==-1:

        '''
        S(u) = exp( - a w u ) (1+u)^k1 (1-u)^k2 Sum( a[k](1+u)^k )
        '''

        # Use abs of singular exponents AND write recursion functions so that the
        # appropriate physical solution is always used
        k1 = 0.5*abs(m-s)
        k2 = 0.5*abs(m+s)
        # As output from Mathematica
        a_alpha = lambda k:	2*(1 + k)*(1 + k + 2*k2)
        a_beta  = lambda k:	-Bjk + k + k1 + k2 + (k + k1 + k2)**2 - 2*awk*(1 + 2*k + 2*k2) - (awj - s)**2 - s
        a_gamma = lambda k: -2*(awj**2 + awk*(-awk + k + k1 + k2) - awj*s)
        # Exponential pre scale for angular function evaluation
        scale_fun_u = lambda COSTH: exp( -awk * COSTH )
        # Define how the exopansion variable relates to u=cos(theta)
        u2v_map = lambda U: U-1

    else:

        error('Unknown input option. Must be -1 or 1 corresponding to the sign of the exponent in the desired solution form.')

    # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    # Package for output
    # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    if (k1<0) or (k2<0):
        print( 'k1 = '+str(k1))
        print( 'k2 = '+str(k2))
        error('negative singular exponent!')

    # Construct answer
    ans = (k1,k2,a_alpha,a_beta,a_gamma,scale_fun_u,u2v_map)

    # Return answer
    return ans


# Define function that returns the recursion coefficients as functions of an integer index
def leaver_ahelper( l,m,s,aw,Alm,london=False,verbose=False,adjoint=False ):
    '''
    Note that we will diver from Leaver's solution by handling the angular singular exponents differently than Leaver has. To use leaver's solution set london=False.
    '''

    # Import usefuls
    from numpy import exp,sqrt,ndarray,cos,array

    # # Determine if the user wishes to consider the mixed problem
    # mixed = isinstance(aw,(tuple,list,ndarray))
    # if mixed:
    #     mixed = 2==len(aw)
    #     if len(aw)>2: error('Iterable aw found, but it is not of length 2 as requirede to consider the mixed eigenvalue problem.')
    # # Interface with leaver_mixed_ahelper
    # if mixed:
    #     # Unpack aw values. Note that awj lives in the operator, awk lives in the eigenfunction
    #     awj,awk = aw
    #     if london==False: london=1
    #     alert('Retrieving recurion functions for mixed the problem',verbose=verbose)
    #     return leaver_mixed_ahelper( l,m,s,awj,awk,Alm,london=london,verbose=verbose,adjoint=adjoint )

    # If the adjoint equation is of interest, apply the adjoint rule
    if adjoint:
        aw,Alm = teukolsky_angular_adjoint_rule(aw,Alm)

    # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    # ANGULAR
    # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    if london:

        if london==1:

            '''
            S(u) = exp( a w u ) (1+u)^k1 (1-u)^k2 Sum( a[k](1+u)^k )
            '''

            # Use abs of singular exponents AND write recursion functions so that the
            # appropriate physical solution is always used
            k1 = 0.5*abs(m-s)
            k2 = 0.5*abs(m+s)
            # Use Leaver's form for the recurion functions
            a_alpha = lambda k:	-2*(1 + k)*(1 + k + 2*k1)
            a_beta  = lambda k:	-Alm - aw**2 - 2*aw*(1 + 2*k + 2*k1 + s) + (k + k1 + k2 - s)*(1 + k + k1 + k2 + s)
            a_gamma = lambda k:  2.0*aw*( k + k1+k2 + s )
            # Define how the exopansion variable relates to u=cos(theta)
            u2v_map = lambda U: 1+U
            # Define starting variable transformation variable
            theta2u_map = lambda TH: cos(TH)
            #
            scale_fun_u = lambda U: (1+U)**k1 * (1-U)**k2 * exp( aw * U )

        elif london==4:

            '''
            Solution to angular equation if (a*w)^2 term in potential is removed
            '''
            # alert('Not using (aw)^2 term in potential')
            # Use abs of singular exponents AND write recursion functions so that the
            # appropriate physical solution is always used
            k1 = 0.5*abs(m-s)
            k2 = 0.5*abs(m+s)
            # Use Leaver's form for the recurion functions
            a_alpha = lambda k:	-2*(1 + k)*(1 + k + 2*k1)
            a_beta  = lambda k:	-Alm - 2*aw*(1 + 2*k + 2*k1 + s) + (k + k1 + k2 - s)*(1 + k + k1 + k2 + s)
            a_gamma = lambda k: 2*aw*(-aw + k + k1 + k2 + s)
            # Define how the exopansion variable relates to u=cos(theta)
            u2v_map = lambda U: 1+U
            # Define starting variable transformation variable
            theta2u_map = lambda TH: cos(TH)
            #
            scale_fun_u = lambda U: (1+U)**k1 * (1-U)**k2 * exp( aw * U )

        elif london==3:

            '''
            Solution to adjoint equation where w is replaced with 1j*d/dt
            S(u) = exp( a w u ) (1+u)^k1 (1-u)^k2 Sum( a[k](1+u)^k )
            '''

            # Use abs of singular exponents AND write recursion functions so that the
            # appropriate physical solution is always used
            k1 = 0.5*abs(m-s)
            k2 = 0.5*abs(m+s)
            # Use Leaver's form for the recurion functions
            a_alpha = lambda k:	-2*(1 + k)*(1 + k + 2*k1)
            a_beta  = lambda k:	-Alm - aw**2 - 2*aw*(1 + 2*k + 2*k1 - s) + (k + k1 + k2 - s)*(1 + k + k1 + k2 + s)
            a_gamma = lambda k: 2*aw*(k + k1 + k2 - s)
            # Define how the exopansion variable relates to u=cos(theta)
            u2v_map = lambda U: 1+U
            # Define starting variable transformation variable
            theta2u_map = lambda TH: cos(TH)
            #
            scale_fun_u = lambda U: (1+U)**k1 * (1-U)**k2 * exp( aw * U )

        elif london==-1:

            '''
            S(u) = exp( - a w u ) (1+u)^k1 (1-u)^k2 Sum( a[k](u-1)^k )
            '''

            # Use abs of singular exponents AND write recursion functions so that the
            # appropriate physical solution is always used
            k1 = 0.5*abs(m-s)
            k2 = 0.5*abs(m+s)
            # Use Leaver's form for the recurion functions
            a_alpha = lambda k:	2*(1 + k)*(1 + k + 2*k2)
            a_beta  = lambda k:	-Alm - aw**2 - 2*aw*(1 + 2*k + 2*k2 - s) + (k + k1 + k2 - s)*(1 + k + k1 + k2 + s)
            a_gamma = lambda k: -2*aw*(k + k1 + k2 - s)
            # Define how the exopansion variable relates to u=cos(theta)
            u2v_map = lambda U: U-1
            # Define starting variable transformation variable
            theta2u_map = lambda TH: cos(TH)
            #
            scale_fun_u = lambda U: (1+U)**k1 * (1-U)**k2 * exp( -aw * U )

        elif london==2:

            '''
            u = aw*cos(theta)
            S(u) = exp( u ) (aw+u)^k1 (aw-u)^k2 Sum( a[k](aw+u)^k )
            '''

            # Use abs of singular exponents AND write recursion functions so that the
            # appropriate physical solution is always used
            k1 = 0.5*abs(m-s)
            k2 = 0.5*abs(m+s)
            # Use Leaver's form for the recurion functions
            a_alpha = lambda k:	-2*aw*(1 + k)*(1 + k + 2*k1)
            a_beta  = lambda k:	-Alm - aw**2 - 2*aw*(1 + 2*k + 2*k1 + s) + (k + k1 + k2 - s)*(1 + k + k1 + k2 + s)
            a_gamma = lambda k: 2*(k + k1 + k2 + s)
            # Define starting variable transformation variable
            theta2u_map = lambda TH: aw*cos(TH)
            # Define how the exopansion variable relates to u=cos(theta)
            u2v_map = lambda U: U+aw
            #
            scale_fun_u = lambda U: (array(aw+U,dtype=complex))**k1 * (array(aw-U,dtype=complex))**k2 * exp( U )

        elif london==-2:

            '''
            u = aw*cos(theta)
            S(u) = exp( -u ) (aw+u)^k1 (aw-u)^k2 Sum( a[k](-aw+u)^k )
            '''

            # Use abs of singular exponents AND write recursion functions so that the
            # appropriate physical solution is always used
            k1 = 0.5*abs(m-s)
            k2 = 0.5*abs(m+s)
            # Use Leaver's form for the recurion functions
            a_alpha = lambda k:	2*aw*(1 + k)*(1 + k + 2*k2)
            a_beta  = lambda k:	-Alm - aw**2 - 2*aw*(1 + 2*k + 2*k2 - s) + (k + k1 + k2 - s)*(1 + k + k1 + k2 + s)
            a_gamma = lambda k: -2*(k + k1 + k2 - s)
            # Define starting variable transformation variable
            theta2u_map = lambda TH: aw*cos(TH)
            # Define how the exopansion variable relates to u=cos(theta)
            u2v_map = lambda U: U-aw
            #
            scale_fun_u = lambda U: (array(aw+U,dtype=complex))**k1 * (array(aw-U,dtype=complex))**k2 * exp( -U )
            
        elif london==-4:
            
            '''
            u = cos(theta)
            S_j(u) = exp( -aw_j * u ) Sum( a[k]Y_k(u) )
            '''
            
            #
            k1 = 1.0 # NOTE that k1 and k2 are NOT used in this encarnation
            k2 = 1.0 # NOTE that k1 and k2 are NOT used in this encarnation
            kref = max(abs(m),abs(s))
            #
            a_alpha = lambda k: (2*aw*(1 + k + kref - s)*sqrt(((1 + k + kref - m)*(1 + k + kref + m)*(1 + k + kref - s)*(1 + k + kref + s))*1.0/(3 + 4*(k + kref)*(2 + k + kref))))*1.0/(1 + k + kref)
            #
            a_beta  = lambda k: Alm + aw**2 - k - kref - (k + kref)**2 + s + s**2 + (2*aw*m*s**2)*1.0/(k + kref + (k + kref)**2)
            #
            a_gamma = lambda k: (-2*aw*sqrt((k + kref - m)*(k + kref + m)*(k + kref - s)*(k + kref + s)))*1.0/((k + kref)*sqrt((1 + 2*(k + kref))*1.0/(-1 + 2*(k + kref)))) + (2*aw*(-1 + k + kref - s)*sqrt(((k + kref - m)*(k + kref + m)*(k + kref - s)*(k + kref + s))*1.0/(-1 + 4*(k + kref)**2)))*1.0/(k + kref)
            # Define starting variable transformation variable
            theta2u_map = lambda TH: cos(TH)
            # Define how the exopansion variable relates to u=cos(theta)
            u2v_map = lambda U: U
            #
            scale_fun_u = lambda U: exp( aw * U )

        else:

            error('Unknown input option.')

    else:

        '''
        S(u) = exp( a w u ) (1+u)^k1 (1-u)^k2 Sum( a[k](1+u)^k )
        '''

        # Use abs of singular exponents AND write recursion functions so that the
        # appropriate physical solution is always used
        k1 = 0.5*abs(m-s)
        k2 = 0.5*abs(m+s)
        # Use Leaver's form for the recurion functions
        a_alpha = lambda k:	-2.0 * (k+1.0) * (k+2.0*k1+1.0)
        a_beta  = lambda k:	k*(k-1.0) \
                            + 2.0*k*( k1+k2+1.0-2.0*aw ) \
                            - ( 2.0*aw*(2.0*k1+s+1.0)-(k1+k2)*(k1+k2+1) ) \
                            - ( aw*aw + s*(s+1.0) + Alm )
        a_gamma = lambda k:   2.0*aw*( k + k1+k2 + s )
        # Define how the exopansion variable relates to u=cos(theta)
        u2v_map = lambda U: 1+U
        # Define starting variable transformation variable
        theta2u_map = lambda TH: cos(TH)
        #
        scale_fun_u = lambda U: (1+U)**k1 * (1-U)**k2 * exp( aw * U )

    # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    # Package for output
    # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    if (k1<0) or (k2<0):
        print( 'k1 = '+str(k1))
        print( 'k2 = '+str(k2))
        error('negative singular exponent!')

    # Construct answer
    ans = (k1,k2,a_alpha,a_beta,a_gamma,scale_fun_u,u2v_map,theta2u_map)

    # Return answer
    return ans


#
def leaver_rhelper( l,m,s,a,w,Alm, london=False, verbose=False, adjoint=False ):

    # Import usefuls
    from numpy import sqrt, exp

    #
    if adjoint:
        s,w,Alm = teukolsky_radial_adjoint_rule(s,w,Alm)

    # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    # RADIAL
    # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    london=False # set london false here becuase we observe leaver's version to be faster. Both versions are correct.
    if london:
        # There is only one solution that satisfies the boundary
        # conditions at the event horizon and infinity.
        p1 = p2 = 1
        b  = sqrt(1.0-4.0*a*a)
        r_alpha = lambda k: 1 + k**2 - s + k*(2 - s - 1j*w) + ((2*1j)*a*m + k*((2*1j)*a*m - 1j*w) - 1j*w)/b - 1j*w
        r_beta  = lambda k: lambda k: -1 - Alm - 2*k**2 - s + k*(-2 + (4*1j)*w) + (2*1j)*w - 2*a*m*w + (4 - a**2)*w**2 + ((-2*1j)*a*m + (2*1j)*w - (4*1j)*a**2*w - 4*a*m*w + (4 - 8*a**2)*w**2 + k*((-4*1j)*a*m + (4*1j - (8*1j)*a**2)*w))/b
        r_gamma = lambda k: k**2 + k*(s - (3*1j)*w) - (2*1j)*s*w - 2*w**2 + (k*((2*1j)*a*m - 1j*w) + 4*a*m*w - 2*w**2)/b
        # Exponential pre scale for radial function evaluation
        r_exp_scale = lambda r: exp( 1j*w * r )
    else:
        # There is only one solution that satisfies the boundary
        # conditions at the event horizon and infinity.
        p1 = p2 = 1
        # Precompute usefuls
        b  = sqrt(1.0-4.0*a*a)
        c_param = 0.5*w - a*m
        #
        c0    =         1.0 - s - 1.0j*w - (2.0j/b) * c_param
        c1    =         -4.0 + 2.0j*w*(2.0+b) + (4.0j/b) * c_param
        c2    =         s + 3.0 - 3.0j*w - (2.0j/b) * c_param
        c3    =         w*w*(4.0+2.0*b-a*a) - 2.0*a*m*w - s - 1.0 \
                        + (2.0+b)*1j*w - Alm + ((4.0*w+2.0j)/b) * c_param
        c4    =         s + 1.0 - 2.0*w*w - (2.0*s+3.0)*1j*w - ((4.0*w+2.0*1j)/b)*c_param
        # Define recursion functions
        r_alpha = lambda k:	k*k + (c0+1)*k + c0
        r_beta  = lambda k:   -2.0*k*k + (c1+2.0)*k + c3
        r_gamma = lambda k:	k*k + (c2-3.0)*k + c4 - c2 + 2.0
        # Exponential pre scale for radial function evaluation
        r_exp_scale = lambda r: exp( 1j*w * r )

    return p1,p2,r_alpha,r_beta,r_gamma,r_exp_scale


# Define function that returns the recursion coefficients as functions of an integer index
def leaver_helper( l,m,s,a,w,Alm, london=True, verbose=False, adjoint=False ):
    '''
    Note that we will diver from Leaver's solution by handling the angular singular exponents differently than Leaver has. To use leaver's solution set london=False.
    '''

    # Import usefuls
    from numpy import exp,sqrt

    # Predefine useful quantities
    aw = a*w

    # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    # ANGULAR
    # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    # Use lower level module
    k1,k2,a_alpha,a_beta,a_gamma,scale_fun_u,u2v_map,theta2u_map = leaver_ahelper( l,m,s,a*w,Alm, london=london, verbose=verbose, adjoint=adjoint )

    # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    # RADIAL
    # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    # Use lower level module
    p1,p2,r_alpha,r_beta,r_gamma,r_exp_scale = leaver_rhelper( l,m,s,a,w,Alm, london=london, verbose=verbose, adjoint=adjoint )

    # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    # Package for output
    # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #

    # Construct answer
    ans = { 'angulars':(k1,k2,a_alpha,a_beta,a_gamma,scale_fun_u,u2v_map,theta2u_map),
            'radials': (p1,p2,r_alpha,r_beta,r_gamma,r_exp_scale) }

    # Return answer
    return ans


# Equation 27 of Leaver '86
# Characteristic Eqn for Spheroidal Radial Functions
def leaver27( a, l, m, w, Alm, s=-2.0, vec=False, mpm=False, adjoint=False, tol=1e-10,london=True,verbose=False,use_nr_convention=False, **kwargs ):

    '''
    Equation 27 of Leaver '86
    Characteristic Eqn for Spheroidal Radial Functions
    - LLondon
    '''

    from numpy import complex256 as dtyp

    # Enforce Data Type
    a = dtyp(a)
    w = dtyp(w)
    Alm = dtyp(Alm)
    
    # NOTE that if the NR convention is being used for the frequencies, then they need to be conjugated here to correspond to compatible convention for phase
    if use_nr_convention:
        w = w.conj()
        Alm = Alm.conj()

    # alert('s=%i'%s)
    # if s != 2:
    #     error('wrong spin')

    #
    pmax = 5e2

    #
    if mpm:
        from mpmath import sqrt
    else:
        from numpy import sqrt

    # global c0, c1, c2, c3, c4, alpha, beta, gamma, l_min

    #
    l_min = l-max(abs(m),abs(s)) # VERY IMPORTANT


    # ------------------------------------------------ #
    # Radial parameter defs
    # ------------------------------------------------ #
    _,_,alpha,beta,gamma,_ = leaver_rhelper( l,m,s,a,w,Alm, london=london, verbose=verbose, adjoint=adjoint )

    #
    v = 1.0
    for p in range(l_min+1):
        v = beta(p) - ( alpha(p-1.0)*gamma(p) / v )

    #
    aa = lambda p:   -alpha(p-1.0+l_min)*gamma(p+l_min)
    bb = lambda p:   beta(p+l_min)
    u,state = lentz(aa,bb,tol)
    u = beta(l_min) - u

    #
    x = v-u
    if vec:
        x = [x.real,x.imag]

    #
    return x


# Equation 21 of Leaver '86
# Characteristic Eqn for Spheroidal Angular Functions
def leaver21( a, l, m, w, Alm, s=-2.0, vec=False, adjoint=False,tol=1e-10,london=True,verbose=False, **kwargs ):
    '''
    Equation 21 of Leaver '86
    Characteristic Eqn for Spheroidal Angular Functions
    - LLondon
    '''

    #
    pmax = 5e2

    #
    l_min = l-max(abs(m),abs(s)) # VERY IMPORTANT

    '''# NOTE: Here we do NOT pass the adjoint keyword, as it we will adjuncticate(?) in leaver27 for the radial equation, and it would be redundant to adjuncticate(??) twice. '''
    k1,k2,alpha,beta,gamma,_,_,_ = leaver_ahelper( l,m,s,a*w,Alm, london=london, verbose=verbose, adjoint=False )

    #
    v = 1.0
    for p in range(l_min+1):
        v = beta(p) - (alpha(p-1.0)*gamma(p) / v)

    #
    aa = lambda p: -alpha(p-1.0+l_min)*gamma(p+l_min)
    bb = lambda p: beta(p+l_min)
    u,state = lentz(aa,bb,tol)
    u = beta(l_min) - u

    #
    x = v-u
    if vec:
        x = [x.real,x.imag]

    #
    return x



#
def leaver_2D_workfunction( j, l, m, cw, s, tol=1e-10 ):

    # Import Maths
    from numpy import log,exp,linalg,array
    from scipy.optimize import root,fmin,minimize
    from positive import alert,red,warning,leaver_workfunction

    error('function must be rewritten to evaluate the angular constraint given possible frequency values, and using sc_leaver to estimate the separation constant (with a guess?)')
    # Try using fmin
    # Define the intermediate work function to be used for this iteration
    fun = lambda X: log(linalg.norm(  leaver21( jf,l,m, cw, X[0]+1j*X[1], s=s )  ))
    foo  = fmin( fun, guess, disp=False, full_output=True, ftol=tol )
    sc = foo[0][0]+1j*foo[0][1]
    __lvrfmin2__ = exp(foo[1])

    # given A(cw), evaluate leaver27

    # return output of leaver27
    return None



# Work function for QNM solver
def leaver_workfunction( j, l, m, state, s=-2, mpm=False, tol=1e-10, use21=True, use27=True, london=None, use_nr_convention=False ):
    '''
    work_function_to_zero = leaver( state )

    state = [ complex_w complex_eigenval ]
    '''

    #
    from numpy import complex128,array,double
    if mpm:
        import mpmath
        mpmath.mp.dps = 8
        dtyp = mpmath.mpc
    else:
        from numpy import complex256 as dtyp


    # alert('s=%i'%s)

    # Unpack inputs
    a = dtyp(j)/2.0                 # Change from M=1 to M=1/2 mass convention

    #
    complex_w = 2.0*dtyp(state[0])  # Change from M=1 to M=1/2 mass convention
    ceigenval = dtyp(state[1])
    

    #
    if len(state) == 4:
        complex_w = 2 * (dtyp(state[0])+1.0j*dtyp(state[1]))
        ceigenval = dtyp(state[2]) + 1.0j*dtyp(state[3])

    # concat list outputs
    #print adjoint

    # x = leaver21(a,l,m,complex_w,ceigenval,vec=True,s=s,mpm=mpm,tol=tol) +  leaver27(a,l,m,complex_w,ceigenval,vec=True,s=s,mpm=mpm,tol=tol)

    x = []
    if use21:
        x += leaver21(a,l,m,complex_w,ceigenval,vec=True,s=s,mpm=mpm,tol=tol,london=london)
    if use27:
        x += leaver27(a,l,m,complex_w,ceigenval,vec=True,s=s,mpm=mpm,tol=tol,london=london,use_nr_convention=use_nr_convention)
    if not x:
        error('use21 or/and use27 must be true')


    #
    x = [ float(e) for e in x ]

    #
    return x




# Given a separation constant, find a frequency*spin such that the spheroidal series expansion converges
def aw_leaver( Alm, l, m, s,tol=1e-9, london=True, verbose=False, guess=None, awj=None, awk=None, adjoint=False  ):
    '''
    Given a separation constant, find a frequency*spin such that the spheroidal series expansion converges
    '''

    # Import Maths
    from numpy import log,exp,linalg,array
    from scipy.optimize import root,fmin,minimize
    from positive import alert,red,warning,leaver_workfunction
    from numpy import complex128 as dtyp

    #
    l_min = l-max(abs(m),abs(s)) # VERY IMPORTANT

    #
    if awj and awk:
        error('When specifying spin-frequencies for the mixed problem, only one of the two frequencies must specified.')
    if awj:
        internal_leaver_ahelper = lambda AWK: leaver_ahelper( l,m,s,[awj,AWK],Alm+s, london=london, verbose=verbose, adjoint=adjoint )
    elif awk:
        internal_leaver_ahelper = lambda AWJ: leaver_ahelper( l,m,s,[AWJ,awk],Alm+s, london=london, verbose=verbose, adjoint=adjoint )
    else:
        internal_leaver_ahelper = lambda AW: leaver_ahelper( l,m,s,AW,Alm+s, london=london, verbose=verbose, adjoint=adjoint )

    #
    def action(aw):
        _,_,alpha,beta,gamma,_,_ = internal_leaver_ahelper(aw)
        v = 1.0
        for p in range(l_min+1):
            v = beta(p) - (alpha(p-1.0)*gamma(p) / v)
        aa = lambda p: -alpha(p-1.0+l_min)*gamma(p+l_min)
        bb = lambda p: beta(p+l_min)
        u,state = lentz(aa,bb,tol)
        u = beta(l_min) - u
        x = v-u
        alert('err = '+str(abs(x)),verbose=verbose)
        x = array([x.real,x.imag],dtype=float).ravel()
        return x

    if verbose: print('')

    # Try using root
    # Define the intermediate work function to be used for this iteration
    indirect_action = lambda STATE: action(STATE[0]+1j*STATE[1])
    # indirect_action = lambda STATE: log( 1.0 + abs( array(  action(STATE[0]+1j*STATE[1])  ) ) )
    aw_guess = 0.5 + 1j * 0.01 if guess is None else guess
    guess = [aw_guess.real,aw_guess.imag]
    foo  = root( indirect_action, guess, tol=tol )
    aw = foo.x[0]+1j*foo.x[1]
    fmin = foo.fun
    foo.action = indirect_action
    retry = ( 'not making good progress' in foo.message.lower() ) or ( 'error' in foo.message.lower() )

    # # Try using fmin
    # # Define the intermediate work function to be used for this iteration
    # indirect_action = lambda STATE: linalg.norm(action(STATE[0]+1j*STATE[1]))**2
    # Alm_guess = scberti(aw,l,m,s)
    # guess = [Alm_guess.real,Alm_guess.imag]
    # foo  = fmin( indirect_action, guess, disp=False, full_output=True, ftol=tol )
    # Alm = foo[0][0]+1j*foo[0][1]
    # fmin = foo[1]
    # retry = fmin>1e-3

    # alert('err = '+str(fmin))
    if retry:
        warning('retry!')

    return (aw,fmin,retry,foo)



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
''' Implement Berti's approximation for the separation constants '''
# NOTE that Beuer et all 1977 also did it
# NOTE Relevant references:
# * Primary: arxiv:0511111v4
# * Proc. R. Soc. Lond. A-1977-Breuer-71-86
# * E_Seidel_1989_Class._Quantum_Grav._6_012
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
def scberti(acw, l,m,s=-2,adjoint=False,verbose=True,nowarn=False):

    '''
    Estimate the Shpheroidal Harmonic separation constant using results of a general perturbative expansion. Primary reference: arxiv:0511111v4 Equation 2.3
    '''

    #
    from numpy import zeros,array,sum

    # If the adjoint equation is of interest, apply the adjoint rule
    if adjoint:
        acw,_ = teukolsky_angular_adjoint_rule(acw,0)

    #
    # from positive import warning
    # if not nowarn:
    #     warning('Please use "slmcg_eigenvalue" for exact calulation.')

    # NOTE that the input here is acw = jf*complex_w
    f = zeros((6,),dtype='complex128')

    #
    l,m,s = float(l),float(m),float(s)

    f[0] = l*(l+1) - s*(s+1)
    f[1] = - 2.0 * m * s*s / ( l*(l+1) )

    hapb = max( abs(m), abs(s) )
    hamb = m*s/hapb
    h = lambda ll: (ll*ll - hapb*hapb) * (ll*ll-hamb*hamb) * (ll*ll-s*s) / ( 2*(l-0.5)*ll*ll*ll*(ll-0.5) )

    f[2] = h(l+1) - h(l) - 1
    f[3] = 2*h(l)*m*s*s/((l-1)*l*l*(l+1)) - 2*h(l+1)*m*s*s/(l*(l+1)*(l+1)*(l+2))
    f[4] = m*m*s*s*s*s*( 4*h(l+1)/(l*l*(l+1)*(l+1)*(l+1)*(l+1)*(l+2)*(l+2)) \
                        - 4*h(l)/((l-1)*(l-1)*l*l*l*l*(l+1)*(l+1)) ) \
                        - (l+2)*h(l+1)*h(l+2)/(2*(l+1)*(2*l+3)) \
                        + h(l+1)*h(l+1)/(2*l+2) + h(l)*h(l+1)/(2*l*l+2*l) - h(l)*h(l)/(2*l) \
                        + (l-1)*h(l-1)*h(l)/(4*l*l-2*l)

    '''
    # NOTE that this term diverges for l=2, and the same is true for the paper's f[6]
    f[5] = m*m*m*s*s*s*s*s*s*( 8.0*h(l)/(l*l*l*l*l*l*(l+1)*(l+1)*(l+1)*(l-1)*(l-1)*(l-1)) \
                             - 8.0*h(l+1)/(l*l*l*(l+1)*(l+1)*(l+1)*(l+1)*(l+1)*(l+1)*(l+2)*(l+2)*(l+2)) ) \
              + m*s*s*h(l) * (-h(l+1)*(7.0*l*l+7*l+4)/(l*l*l*(l+2)*(l+1)*(l+1)*(l+1)*(l-1)) \
                              -h(l-1)*(3.0*l-4)/(l*l*l*(l+1)*(2*l-1)*(l-2)) ) \
              + m*s*s*( (3.0*l+7)*h(l+1)*h(l+2)/(l*(l+1)*(l+1)*(l+1)*(l+3)*(2*l+3)) \
                                 -(3.0*h(l+1)*h(l+1)/(l*(l+1)*(l+1)*(l+1)*(l+2)) + 3.0*h(l)*h(l)/(l*l*l*(l-1)*(l+1)) ) )
    '''

    # Calcualate the series sum, and return output
    return sum( array([ f[k] * acw**k for k in range(len(f)) ]) )



# Compute perturbed Kerr separation constant given a frequency
def sc_london( aw, l, m, s,tol=1e-12, s_included=False, verbose=False, adjoint=False, __CHECK__=True,london=-4,guess = None,nowarn=False  ):
    '''
    Given (aw, l, m, s), compute and return separation constant. This method uses a three-term recursion relaition obtained by applying the following ansatz to the spheroidal problem:
    
            u = cos(theta)
            S_j(u) = exp( -aw_j * u ) Sum( a[k]Y_k(u) )

    NOTE that this method is equivalent to using sc_leaver(...,london=-4,...).

    londonl@mit.edu Nov 2020
    '''

    # Import Maths
    from numpy import log,exp,linalg,array
    from scipy.optimize import root,fmin,minimize
    from positive import alert,red,warning,leaver_workfunction
    from numpy import complex128 as dtyp

    #
    if not nowarn:
        warning('Please use "slmcg_eigenvalue" for preferred output.')

    #
    p_min = l-max(abs(m),abs(s)) # VERY IMPORTANT

    #
    def action(Alm):
        _,_,alpha,beta,gamma,_,_,_ = leaver_ahelper( l,m,s,aw,Alm, london=-4, verbose=verbose, adjoint=False )
        v = 1.0
        for p in range(p_min+1):
            v = beta(p) - (alpha(p-1.0)*gamma(p) / v)
        aa = lambda p: -alpha(p-1.0+p_min)*gamma(p+p_min)
        bb = lambda p: beta(p+p_min)
        u,state = lentz(aa,bb,tol)
        u = beta(p_min) - u
        x = v-u
        alert('err = '+str(abs(x)),verbose=verbose)
        x = array([x.real,x.imag],dtype=float).ravel()
        return x

    if verbose: print('')

    # Try using root
    # Define the intermediate work function to be used for this iteration
    indirect_action = lambda STATE: action(STATE[0]+1j*STATE[1])
    # indirect_action = lambda STATE: log( 1.0 + abs( array(  action(STATE[0]+1j*STATE[1])  ) ) )
    aw_guess = aw if isinstance(aw,(float,complex)) else aw
    Alm_guess = scberti(aw_guess,l,m,s,adjoint=False,nowarn=nowarn)
    guess = [Alm_guess.real,Alm_guess.imag] if guess is None else guess
    foo  = root( indirect_action, guess, tol=tol )
    Alm = foo.x[0]+1j*foo.x[1]
    fmin = indirect_action( foo.fun )
    retry = ( 'not making good progress' in foo.message.lower() ) or ( 'error' in foo.message.lower() )

    # 
    if s_included:
        Alm = Alm + s

    # Impose check on equivalence class
    if __CHECK__:
        if (l==abs(s)+3):
            (Alm_,fmin_,retry_,foo_) = sc_london( -aw, l, -m, s,tol=tol, london=london, s_included=s_included, verbose=verbose, adjoint=adjoint, __CHECK__=False,nowarn=nowarn  )
            if linalg.norm(fmin_)<linalg.norm(fmin):
                (Alm,fmin,retry,foo) = (Alm_,fmin_,retry_,foo_)
            if verbose:
                warning('Strange behavior has been noticed for this equivalence class of l and s. We have checked to verify that the optimal root is used here.')

    # 
    if retry:
        warning('retry! needed in sc_london')

    return (Alm,fmin,retry,foo)


# Compute perturbed Kerr separation constant given a frequency
def sc_leaver( aw, l, m, s,tol=1e-10, london=-4, s_included=False, verbose=False, adjoint=False, __CHECK__=True,guess=None  ):
    '''
    Given (aw, l, m, s), compute and return separation constant.
    '''

    #
    from positive import warning
    #warning('Please use "slmcg_eigenvalue" for preferred output.')

    # Import Maths
    from numpy import log,exp,linalg,array
    from scipy.optimize import root,fmin,minimize
    from positive import alert,red,warning,leaver_workfunction
    from numpy import complex128 as dtyp

    #
    p_min = l-max(abs(m),abs(s)) # VERY IMPORTANT

    #
    def action(Alm):
        _,_,alpha,beta,gamma,_,_,_ = leaver_ahelper( l,m,s,aw,Alm, london=london, verbose=verbose, adjoint=False )
        v = 1.0
        for p in range(p_min+1):
            v = beta(p) - (alpha(p-1.0)*gamma(p) / v)
            if v==0: break
        aa = lambda p: -alpha(p-1.0+p_min)*gamma(p+p_min)
        bb = lambda p: beta(p+p_min)
        u,state = lentz(aa,bb,tol)
        u = beta(p_min) - u
        x = v-u
        alert('err = '+str(abs(x)),verbose=verbose)
        x = array([x.real,x.imag],dtype=float).ravel()
        return x

    if verbose: print('')

    # Try using root
    # Define the intermediate work function to be used for this iteration
    indirect_action = lambda STATE: action(STATE[0]+1j*STATE[1])
    # indirect_action = lambda STATE: log( 1.0 + abs( array(  action(STATE[0]+1j*STATE[1])  ) ) )
    aw_guess = aw
    Alm_guess = scberti(aw_guess,l,m,s,adjoint=False) if guess is None else guess
    guess = [Alm_guess.real,Alm_guess.imag]
    foo  = root( indirect_action, guess, tol=tol )
    Alm = foo.x[0]+1j*foo.x[1]
    fmin = indirect_action( foo.fun )
    retry = ( 'not making good progress' in foo.message.lower() ) or ( 'error' in foo.message.lower() )

    # # Try using fmin
    # # Define the intermediate work function to be used for this iteration
    # indirect_action = lambda STATE: linalg.norm(action(STATE[0]+1j*STATE[1]))**2
    # Alm_guess = scberti(aw,l,m,s)
    # guess = [Alm_guess.real,Alm_guess.imag]
    # foo  = fmin( indirect_action, guess, disp=False, full_output=True, ftol=tol )
    # Alm = foo[0][0]+1j*foo[0][1]
    # fmin = foo[1]
    # retry = fmin>1e-3

    # Given the structure of the spheroidal harmonic differential equation for perturbed Kerr, we have a choice to include a factor of s in the potential, or in the eigenvalue. There's good reason to consider it a part of the latter as s->-s has the action of leaving the eigenvalue unchanged.
    if s_included:
        Alm = Alm + s

    # Impose check on equivalence class
    if __CHECK__:
        if (l==abs(s)+3):
            (Alm_,fmin_,retry_,foo_) = sc_leaver( -aw, l, -m, s,tol=tol, london=london, s_included=s_included, verbose=verbose, adjoint=adjoint, __CHECK__=False  )
            if linalg.norm(fmin_)<linalg.norm(fmin):
                (Alm,fmin,retry,foo) = (Alm_,fmin_,retry_,foo_)
            if verbose:
                warning('Strange behavior has been noticed for this equivalence class of l and s. We have checked to verify that the optimal root is used here.')

    # alert('err = '+str(fmin))
    if retry:
        warning('retry! needed in sc_leaver')

    return (Alm,fmin,retry,foo)




#
class leaver_solve_workflow:

    '''
    Workflow class for solving and saving data from leaver's equationsself.
    '''

    #
    def __init__( this, initial_spin, final_spin, l, m, tol=1e-3, verbose=False, basedir=None, box_xywh=None, max_overtone=6, output=True, plot=True, initial_box_res=81, spline_order=3, initialize_only=False, s=-2, adjoint=False ):

        #
        this.__validate_inputs__(initial_spin, final_spin, l, m, tol, verbose, basedir, box_xywh, max_overtone, output, plot, initial_box_res, spline_order, s, adjoint)


        # ------------------------------------------------------------ #
        # Define a box in cw space over which to calculate QNM solutions
        # ------------------------------------------------------------ #
        alert('Initializing leaver box',verbose=this.verbose,header=True)
        this.__initialize_leaver_box__()

        # ------------------------------------------------------------ #
        # Map the QNM solution space at an initial spin value
        # ------------------------------------------------------------ #
        this.leaver_box.map( this.initial_spin )

        # ------------------------------------------------------------ #
        # Let's see what's in the box
        # ------------------------------------------------------------ #
        alert('The following QNMs have been found in the box:',header=True)
        this.starting_solutions = { k:this.leaver_box.data[k] for k in sorted(this.leaver_box.data.keys(), key = lambda x: x[1], reverse=True ) if k[2]<this.max_overtone }
        # Note that it is here that we enforce the max_overtone input
        this.sorted_mode_list = sorted(this.starting_solutions.keys(), key = lambda x: -float(x[-1])/(x[2]+1), reverse=not True )
        for k in this.sorted_mode_list:
            print('(l,m,n,x,p) = %s'%(str(k)))

        # ------------------------------------------------------------ #
        # Plot the QNM solution space at an initial spin value
        # ------------------------------------------------------------ #
        alert('Plotting 2D start frame',verbose=this.verbose,header=True)
        if this.plot: this.__plot2Dframe__()

        #
        if not initialize_only:
            alert('Threading QNM solutions',verbose=this.verbose,header=True)
            this.solve_all_modes()

        #
        alert('Done!',verbose=this.verbose,header=True)

    #
    def solve_all_modes(this):

        for z in this.sorted_mode_list:
            (l,m,n,x,p) = z
            if True:# m<0:

                alert('Working: (l,m,n,x,p) = %s'%str(z),header=True)
                this.solve_mode(*z)

                # ------------------------------------------------------------ #
                # Ploting
                # ------------------------------------------------------------ #
                if this.plot:
                    # Plot interpolation error
                    this.__plotModeSplineError__(z)
                    # Plot frequency and separation constant
                    this.__plotCWSC__(z)


    #
    def __plot2Dframe__(this,save=None):
        from matplotlib.pyplot import savefig,plot,xlim,ylim,xlabel,ylabel,title,figure,gca,subplot,close,gcf,figaspect
        alert('Saving 2D solution space frame at initial spin',verbose=this.verbose)
        this.leaver_box.plot(showlabel=True)
        frm_prefix = ('l%im%i_start' % (this.l,this.m)).replace('-','m')
        frm_fname = ('%s.png' % (frm_prefix))
        if save is None: save = this.output
        if save:
            savefig( this.frm_outdir+frm_fname )
            close(gcf())

    #
    def __plotModeSplineError__(this,z,save=None):

        #
        from matplotlib.pyplot import savefig, plot, xlim, ylim, xlabel, ylabel, title, figure, gca, subplot, close, gcf, figaspect, axhline, yscale, legend
        l,m,n,x,p = z
        figure( figsize=1.2*figaspect(0.618) )
        plot( this.results[z]['js'],this.results[z]['errs'] )
        axhline(this.tol,color='orange',ls='--',label='tol = %g'%this.tol)
        yscale('log'); xlabel( '$j$' )
        ylabel( r'$\epsilon$' ); legend()
        title(r'$(\ell,m,n)=(%i,%i,%i)$'%(l,m,n))
        fig_fname = ('l%im%in%i_epsilon.pdf' % (l,m,n)).replace('-','m')
        if save is None: save = this.output
        if save:
            savefig( this.outdir+fig_fname,pad_inches=.2, bbox_inches='tight' )
            close(gcf())

    #
    def __plotCWSC__(this,z,save=None):

        #
        from matplotlib.pyplot import savefig,plot,xlim,ylim,xlabel,ylabel,title,figure,gca,subplot,close,gcf,figaspect,legend,grid
        l,m,n,x,p = z

        #
        j = this.results[z]['j']
        cw = this.results[z]['cw']
        sc = this.results[z]['sc']
        js = this.results[z]['js']
        cwrspl = this.results[z]['cwrs']
        cwcspl = this.results[z]['cwcs']
        scrspl = this.results[z]['scrs']
        sccspl = this.results[z]['sccs']

        grey = '0.9'
        n = z[2]
        figure( figsize=3*figaspect(0.618) )
        subplot(2,2,1)
        plot( j, cw.real,'o',label='Numerical Data' )
        plot( js, cwrspl(js), 'r',label='Spline, k=%i'%this.spline_order )
        legend()
        xlabel('$j$'); ylabel(r'$\mathrm{re}\; \tilde{\omega}_{%i%i%i}$'%(l,m,n))
        grid(color=grey, linestyle='-')
        subplot(2,2,2)
        plot( j, cw.imag,'o' )
        plot( js, cwcspl(js), 'r' )
        xlabel('$j$'); ylabel(r'$\mathrm{im}\; \tilde{\omega}_{%i%i%i}$'%(l,m,n))
        grid(color=grey, linestyle='-')
        subplot(2,2,3)
        plot( j, sc.real,'o' )
        plot( js, scrspl(js), 'r' )
        xlabel('$j$'); ylabel(r'$\mathrm{re}\; \tilde{A}_{%i%i%i}$'%(l,m,n))
        grid(color=grey, linestyle='-')
        subplot(2,2,4)
        plot( j, sc.imag,'o' )
        plot( js, sccspl(js), 'r' )
        xlabel('$j$'); ylabel(r'$\mathrm{im}\; \tilde{A}_{%i%i%i}$'%(l,m,n))
        grid(color=grey, linestyle='-')
        fig_fname = ('l%im%in%i_results.pdf' % (l,m,n)).replace('-','m')
        if save is None: save = this.output
        if save:
            savefig( this.outdir+fig_fname,pad_inches=.2, bbox_inches='tight' )
            close(gcf())

    #
    def __initialize_leaver_box__(this,box_xywh=None):

        #
        if box_xywh is None:
            box_xywh = this.box_xywh
        # Extract box parameters
        x,y,wid,hig = this.box_xywh
        # Define the cwbox object
        this.leaver_box = cwbox( this.l,this.m,x,y,wid,hig,res=this.initial_box_res,maxn=this.max_overtone,verbose=this.verbose,s=this.s, adjoint=this.adjoint )
        # Shorthand
        a = this.leaver_box

        #
        return None

    #
    def solve_mode(this,l,m,n,x,p):

        #
        if this.plot: from matplotlib.pyplot import savefig,plot,xlim,ylim,xlabel,ylabel,title,figure,gca,subplot,close,gcf,figaspect
        from numpy import linspace,complex128,array,log,savetxt,vstack,pi,mod,cos,linalg,sign,exp,hstack,argsort,diff
        from positive import spline,leaver_needle
        import dill
        import pickle

        #
        z = (l,m,n,x,p)
        this.results[z] = {}

        #
        if not ( z in this.leaver_box.data ):
            error('Input mode indeces not in this.leaver_box')

        # ------------------------------------------------------------ #
        # Thread the solution from cwbox through parameter space
        # ------------------------------------------------------------ #
        alert('Threading the solution from cwbox through parameter space',verbose=this.verbose)
        solution_cw,solution_sc = this.leaver_box.data[z]['cw'][-1],this.leaver_box.data[z]['sc'][-1]
        print('>> ',this.leaver_box.data[z]['lvrfmin'][-1])
        # forwards
        initial_solution = [ solution_cw.real, solution_cw.imag, solution_sc.real, solution_sc.imag ]
        print('** ',leaver_workfunction( this.initial_spin, l, abs(m), initial_solution, s=this.s, adjoint=this.adjoint ))
        j,cw,sc,err,retry = leaver_needle( this.initial_spin, this.final_spin, l,abs(m), initial_solution, tol=this.tol/( n+1 + 2*(1-p) ), verbose=this.verbose, spline_order=this.spline_order, s=this.s, adjoint=this.adjoint )

        # backwards
        alert('Now evaluating leaver_needle backwards!',header=True)
        initial_solution = [ cw[-1].real, cw[-1].imag, sc[-1].real, sc[-1].imag ]
        j_,cw_,sc_,err_,retry_ = leaver_needle( this.final_spin, this.initial_spin, l,abs(m), initial_solution, tol=this.tol/( n+1 + 2*(1-p) ), verbose=this.verbose, spline_order=this.spline_order, s=this.s,initial_d2spin=abs(j[-1]-j[-2])/5, adjoint=this.adjoint )
        #
        j,cw,sc,err,retry = [ hstack([u,v]) for u,v in [(j,j_),(cw,cw_),(sc,sc_),(err,err_),(retry,retry_)] ]
        sortmask = argsort(j)
        j,cw,sc,err,retry = [ v[sortmask] for v in (j,cw,sc,err,retry) ]
        uniquemask = hstack( [array([True]),diff(j)!=0] )
        j,cw,sc,err,retry = [ v[uniquemask] for v in (j,cw,sc,err,retry) ]

        #
        this.results[z]['j'],this.results[z]['cw'],this.results[z]['sc'],this.results[z]['err'],this.results[z]['retry'] = j,cw,sc,err,retry

        # ------------------------------------------------------------ #
        # Calculate the error of the resulting spline model between the boundaries
        # ------------------------------------------------------------ #
        alert('Calculating the error of the resulting spline model between the boundaries',verbose=this.verbose)
        lvrwrk = lambda J,STATE: linalg.norm(  leaver_workfunction( J,l,abs(m),STATE,s=this.s, adjoint=this.adjoint )  )
        js = linspace(min(j),max(j),1e3)

        cwrspl = spline(j,cw.real,k=this.spline_order)
        cwcspl = spline(j,cw.imag,k=this.spline_order)
        scrspl = spline(j,sc.real,k=this.spline_order)
        sccspl = spline(j,sc.imag,k=this.spline_order)
        statespl = lambda J: [ cwrspl(J), cwcspl(J), scrspl(J), sccspl(J) ]

        # Store spline related quantities
        this.results[z]['errs'] = array( [ lvrwrk(J,statespl(J)) for J in js ] )
        this.results[z]['js'] = js
        this.results[z]['cwrs'] = cwrspl
        this.results[z]['cwcs'] = cwcspl
        this.results[z]['scrs'] = scrspl
        this.results[z]['sccs'] = sccspl
        this.results[z]['statespl'] = statespl

        # ------------------------------------------------------------ #
        # Save raw data and splines
        # ------------------------------------------------------------ #
        if this.output:
            data_array = vstack(  [ j,
                                    cw.real,
                                    cw.imag,
                                    sc.real,
                                    sc.imag,
                                    err ]  ).T

            fname = this.outdir+('l%im%1.0fn%i.txt'%z[:3]).replace('-','m')
            savetxt( fname, data_array, fmt='%18.12e', delimiter='\t\t', header=r's=%i Kerr QNM: [ jf reMw imMw reA imA error ], 2016/2019, londonl@mit, https://github.com/llondon6/'%this.s )
            # Save the current object
            alert('Saving the current object using pickle',verbose=this.verbose)
            with open( fname.replace('.txt','_splines.pickle') , 'wb') as object_file:
                pickle.dump( {'cwreal':cwrspl,'cwimag':cwcspl,'screal':scrspl,'scimag':sccspl} , object_file, pickle.HIGHEST_PROTOCOL )


    #
    def __validate_inputs__( this, initial_spin, final_spin, l, m, tol, verbose, basedir, box_xywh, max_overtone, output, plot, initial_box_res, spline_order, s, adjoint ):

        # Save inputs as properties of the current object
        alert('Found inputs:',verbose=verbose)
        for k in dir():
            if k != 'this':
                this.__dict__[k] = eval(k)
                if verbose: print('  ->  %s = %s'%( k, blue(str(eval(k))) ))

        # Import usefuls
        from os.path import join,expanduser
        from numpy import pi
        from positive import mkdir

        # Name directories for file IO
        alert('Making output directories ...',verbose=this.verbose)
        if this.basedir is None: this.basedir = ''
        this.basedir = expanduser(this.basedir)
        this.outdir = ( join( this.basedir,'./s%il%im%i/' % (s,l,abs(m))) ).replace('-','m')
        if this.outdir[-1] is not '/': this.outdir += '/'
        this.frm_outdir = this.outdir+'frames/'
        this.s = s
        # Make directories if needed
        mkdir(this.outdir,rm = False,verbose=this.verbose)  # Make the directory if it doesnt already exist and remove the directory if it already exists. Dont remove if there is checkpint data.
        mkdir(this.frm_outdir,verbose=this.verbose)

        #
        if this.output: this.plot = True

        #
        this.spline_order = spline_order

        #
        this.results = {}

        #
        if this.box_xywh is None:
            x = 0
            y = -0.5
            wid = 1.2*(m if abs(m)>0 else 1.0)
            hig = pi-1
            this.box_xywh = [x,y,wid,hig]
            alert('Using default parameter space box for starting spin.',verbose=this.verbose)
            alert('this.box_xywh = %s'%(str(this.box_xywh)),verbose=this.verbose)






# ---------------------- #
'''Class for boxes in complex frequency space'''
# The routines of this class assist in the solving and classification of
# QNM solutions
# ---------------------- #
class cwbox:
    # ************************************************************* #
    # This is a class to fascilitate the solving of leaver's equations varying
    # the real and comlex frequency components, and optimizing over the separation constants.
    # ************************************************************* #
    def __init__(this,
                 l,m,               # QNM indeces
                 cwr,               # Center coordinate of real part
                 cwc,               # Center coordinate of imag part
                 wid,               # box width
                 hig,               # box height
                 res = 50,          # Number of gridpoints in each dimension
                 parent = None,     # Parent of current object
                 sc = None,         # optiional holder for separatino constant
                 verbose = False,   # be verbose
                 maxn = None,       # Overtones with n>maxn will be actively ignored. NOTE that by convention n>=0.
                 smallboxes = True, # Toggle for using small boxes for new solutions
                 s = -2,            # Spin weight
                 adjoint = False,
                 **kwargs ):
        #
        from numpy import array,complex128,meshgrid,float128
        #
        this.verbose,this.res = verbose,res
        # Store QNM ideces
        this.l,this.m,this.s = l,m,s
        # Set box params
        this.width,this.height = None,None
        this.setboxprops(cwr,cwc,wid,hig,res,sc=sc)
        # Initial a list of children: if a box contains multiple solutions, then it is split according to each solutions location
        this.children = [this]
        # Point the object to its parent
        this.parent = parent
        #
        this.__jf__ = []
        # temp grid of separation constants
        this.__scgrid__ = []
        # current value of scalarized work-function
        this.__lvrfmin__ = None
        # Dictionary for high-level data: the data of all of this object's children is collected here
        this.data = {}
        this.dataformat = '{ ... (l,m,n,tail_flag) : { "jf":[...],"cw":[...],"sc":[...],"lvrfmin":[...] } ... }'
        # Dictionary for low-level data: If this object is fundamental, then its data will be stored here in the same format as above
        this.__data__ = {}
        # QNM label: (l,m,n,t), NOTE that "t" is 0 if the QNM is not a power-law tail and 1 otherwise
        this.__label__ = ()
        # Counter for the number of times map hass benn called on this object
        this.mapcount = 0
        # Default value for temporary separation constant
        this.__sc__ = 4.0
        # Maximum overtone label allowed.  NOTE that by convention n>=0.
        this.__maxn__ = maxn
        #
        this.adjoint = adjoint
        #
        this.__removeme__ = False
        #
        this.__smallboxes__ = smallboxes

    # Set box params & separation constant center
    def setboxprops(this,cwr,cwc,wid,hig,res,sc=None,data=None,pec=None):
        # import maths and other
        from numpy import complex128,float128,array,linspace
        import matplotlib.patches as patches
        # set props for box geometry
        this.center = array([cwr,cwc])
        this.__cw__ = cwr + 1j*cwc          # Store cw for convinience

        # Boxes may only shrink. NOTE that this is usefull as some poetntial solutions, or unwanted solutions may be reomved, and we want to avoid finding them again. NOTE that this would be nice to implement, but it currently brakes the root finding.
        this.width,this.height  = float128( abs(wid) ),float128( abs(hig) )
        # if (this.width is None) or (this.height is None):
        #     this.width,this.height  = float128( abs(wid) ),float128( abs(hig) )
        # else:
        #     this.width,this.height  = min(float128( abs(wid) ),this.width),min(this.height,float128( abs(hig) ))

        this.limit  = array([this.center[0]-this.width/2.0,       # real min
                             this.center[0]+this.width/2.0,       # real max
                             this.center[1]-this.height/2.0,      # imag min
                             this.center[1]+this.height/2.0])     # imag max
        this.wr_range = linspace( this.limit[0], this.limit[1], res )
        this.wc_range = linspace( this.limit[2], this.limit[3], res )
        # Set patch object for plotting. NOTE the negative sign exists here per convention
        if None is pec: pec = 'k'
        this.patch = patches.Rectangle( (min(this.limit[0:2]), min(-this.limit[2:4]) ), this.width, this.height, fill=False, edgecolor=pec, alpha=0.4, linestyle='dotted' )
        # set holder for separation constant value
        if sc is not None:
            this.__sc__ = sc
        # Initiate the data holder for this box. The data holder will contain lists of spin, official cw and sc values
        if data is not None:
            this.data=data

    # Map the potential solutions in this box
    def map(this,jf):

        # Import useful things
        from positive.maths import localmins # finds local minima of a 2D array
        from positive import alert,green,yellow,cyan,bold,magenta,blue
        from numpy import array,delete,ones

        #%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
        # Add the input jf to the list of jf values. NOTE that this is not the primary recommended list for referencing jf. Please use the "data" field instead.
        this.__jf__.append(jf)
        #%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#

        #

        if this.verbose:
            if this.parent is None:
                alert('\n\n# '+'--'*40+' #\n'+blue(bold('Attempting to map qnm solutions for: jf = %1.8f'%(jf)))+'\n# '+'--'*40+' #\n','map')
            else:
                print('\n# '+'..'*40+' #\n'+blue('jf = %1.8f,  label = %s'%(jf,this.__label__))+'\n# '+'..'*40+' #')

        # Map solutions using discrete grid
        if this.isfundamental():
            # Brute-force calculate solutions to leaver's equations
            if this.verbose: alert('Solvinq Leaver''s Eqns over grid','map')
            this.__x__,this.__scgrid__ = this.lvrgridsolve(jf)
            # Use a local-min finder to estimate the qnm locations for the grid of work function values, x
            if this.verbose: alert('Searching for local minima. Ignoring mins on boundaries.','map')
            this.__localmin__ = localmins(this.__x__,edge_ignore=True)
            if this.verbose: alert('Number of local minima found: %s.'%magenta('%i'%(len(array(this.__localmin__)[0]))),'map')
            # If needed, split the box into sub-boxes: Give the current box children!
            this.splitcenter() # NOTE that if there is only one lcal min, then no split takes place
            # So far QNM solutions have been estimates mthat have discretization error. Now, we wish to refine the
            # solutions using optimization.
            if this.verbose: alert('Refining QNM solution locations using a hybrid strategy.','map')
            this.refine(jf)
        else:
            # Map solutions for all children
            for child in [ k for k in this.children if this is not k ]:
                child.map(jf)

        # Collect QNM solution data for this BH spin. NOTE that only non-fundamental objects are curated
        if this.verbose: alert('Collecting final QNM solution information ...','map')
        this.curate(jf)

        # Remove duplicate solutions
        this.validatechildren()

        #
        if this.verbose: alert('Mapping of Kerr QNM with (l,m)=(%i,%i) within box now complete for this box.' % (this.l,this.m ) ,'map')

        # Some book-keeping on the number of times this object has been mapped
        this.mapcount += 1


    # For the given bh spin, collect all QNM frequencies and separation constants within the current box
    # NOTE that the outputs are coincident lists
    def curate(this,jf):

        #
        from numpy import arange,array,sign

        #
        children = this.collectchildren()
        cwlist,sclist = [ child.__cw__ for child in children ],[ child.__sc__ for child in children ]
        if this.isfundamental():
            cwlist.append( this.__cw__ )
            sclist.append( this.__sc__ )

        # sort the output lists by the imaginary part of the cw values
        sbn = lambda k: abs( cwlist[k].imag ) # Sort By Overtone(N)
        space = arange( len(cwlist) )
        map_  = sorted( space, key=sbn )
        std_cwlist = array( [ cwlist[k] for k in map_ ] )
        std_sclist = array( [ sclist[k] for k in map_ ] )

        # ---------------------------------------------------------- #
        # Separate positive, zero and negative frequency solutions
        # ---------------------------------------------------------- #

        # Solutions with frequencies less than this value will be considered to be power-laws
        pltol = 0.01
        # Frequencies
        if jf == 0: jf = 1e-20 # NOTE that here we ensure that 0 jf is positive only for BOOK-KEEPING purposes
        sorted_cw_pos = list(  std_cwlist[ (sign(std_cwlist.real) == sign(jf)) * (abs(std_cwlist.real)>pltol) ]  )
        sorted_cw_neg = list(  std_cwlist[ (sign(std_cwlist.real) ==-sign(jf)) * (abs(std_cwlist.real)>pltol) ]  )
        sorted_cw_zro = list(  std_cwlist[ abs(std_cwlist.real)<=pltol ]  )

        # Create a dictionary between (cw,sc) and child objects
        A,B = {},{}
        for child in children:
            A[child] = ( child.__cw__, child.__sc__ )
            B[ A[child] ] = child
        #
        def inferlabel( cwsc ):
            cw,sc = cwsc[0],cwsc[1]
            ll = this.l
            if abs(cw.real)<pltol :
                # power-law decay
                tt = 1
                nn = sorted_cw_zro.index( cw )
                mm = this.m
                pp = 0
            else:
                tt = 0
                if sign(jf)==sign(cw.real):
                    # prograde
                    mm = this.m
                    nn = sorted_cw_pos.index( cw )
                    pp = 1
                else:
                    # retrograde
                    mm = -1 * this.m
                    nn = sorted_cw_neg.index( cw )
                    pp = -1
            #
            return (ll,mm,nn,tt,pp)

        # ---------------------------------------------------------- #
        # Create a dictionary to keep track of potential solutions
        # ---------------------------------------------------------- #
        label = {}
        invlabel = {}
        for child in children:
            cwsc = ( child.__cw__, child.__sc__ )
            child.state = [ child.__cw__.real, child.__cw__.imag,  child.__sc__.real, child.__sc__.imag ]
            __label__ = inferlabel( cwsc )
            label[child] = __label__
            invlabel[__label__] = child
            child.__label__ = label[child]

        #
        this.labelmap = label
        this.inverse_labelmap = invlabel
        '''
        IMPORTANT: Here it is assumed that the solutions will change in a continuous manner, and that after the first mapping, no new solutions are of interest, unless a box-split occurs.
        '''

        # Store the high-level data product
        for child in children:
            L = this.labelmap[child]
            if not L in this.data:
                this.data[ L ] = {}
                this.data[ L ][ 'jf' ] = [jf]
                this.data[ L ][ 'cw' ] = [ child.__cw__ ]
                this.data[ L ][ 'sc' ] = [ child.__sc__ ]
                this.data[ L ][ 'lvrfmin' ] = [ child.__lvrfmin__ ]
            else:
                this.data[ L ][ 'jf' ].append(jf)
                this.data[ L ][ 'cw' ].append(child.__cw__)
                this.data[ L ][ 'sc' ].append(child.__sc__)
                this.data[ L ][ 'lvrfmin' ].append(child.__lvrfmin__)
            # Store the information to this child also
            child.__data__['jf'] = this.data[ L ][ 'jf' ]
            child.__data__['cw'] = this.data[ L ][ 'cw' ]
            child.__data__['sc'] = this.data[ L ][ 'sc' ]
            child.__data__['lvrfmin'] = this.data[ L ][ 'lvrfmin' ]


    # Refine the box center using fminsearch
    def refine(this,jf):
        # Import useful things
        from numpy import complex128,array,linalg,log,exp,abs
        from scipy.optimize import fmin,root,fmin_tnc,fmin_slsqp
        from positive.physics import leaver_workfunction,scberti
        from positive import alert,say,magenta,bold,green,cyan,yellow
        from positive import localmins # finds local minima of a 2D array

        #
        if this.isfundamental():
            # use the box center for refined minimization
            CW = complex128( this.center[0] + 1j*this.center[1] )
            # SC = this.__sc__
            SC = scberti( CW*jf, this.l, this.m, s=this.s )
            state = [ CW.real,CW.imag, SC.real,SC.imag ]

            #
            retrycount,maxretrycount,done = -1,1,False
            while done is False:

                #
                retrycount += 1

                #
                if retrycount==0:
                    alert(cyan('* Constructing guess using scberti-grid or extrap.'),'refine')
                    state = this.guess(jf,gridguess=state)
                else:
                    alert(cyan('* Constructing guess using 4D-grid or extrap.'),'refine')
                    state = this.guess(jf)

                # Solve leaver's equations using a hybrid strategy
                cw,sc,this.__lvrfmin__,retry = this.lvrsolve(jf,state)

                # If the root finder had some trouble, then mark this box with a warning (for plotting)
                done = (not retry) or (retrycount>=maxretrycount)
                #
                if retry:

                    newres = 2*this.res

                    if this.verbose:
                        msg = yellow( 'The current function value is %s. Retrying root finding for %ind time with higher resolution pre-grid, and brute-force 4D.'%(this.__lvrfmin__, retrycount+2) )
                        alert(msg,'refine')
                        # say('Retrying.','refine')

                    # Increase the resolution of the box
                    this.setboxprops(this.__cw__.real,this.__cw__.imag,this.width,this.height,newres,sc=this.__sc__)
                    # NOTE that the commented out code below is depreciated by the use of guess() above.
                    # # Brute force solve again
                    # this.__x__,this.__scgrid__ = this.lvrgridsolve(jf,fullopt=True)
                    # # Use the first local min as a guess
                    # this.__localmin__ = localmins(this.__x__,edge_ignore=True)
                    # state = this.grids2states()[0]

            # if this.verbose: print X.message+' The final function value is %s'%(this.__lvrfmin__)
            if this.verbose: print('The final function value is '+green(bold('%s'%(this.__lvrfmin__))))

            if this.verbose:
                print( '\n\t Geuss   cw: %s' % CW)
                print( '\t Optimal cw: %s' % cw)
                print( '\t Approx  sc: %s' % scberti( CW*jf, this.l, this.m ))
                print( '\t Geuss   sc: %s' % (state[2]+1j*state[3]))
                print( '\t Optimal sc: %s\n' % sc)

            # Set the core properties of the new box
            this.setboxprops( cw.real, cw.imag, this.width,this.height,this.res,sc=sc )

            # Rescale this object's boxes based on new centers
            this.parent.sensescale()

        else:
            #
            for child in [ k for k in this.children if this is not k ]:
                child.refine(jf)


    # Determine if the current object has more than itself as a child
    def isfundamental(this):
        return len(this.children) is 1


    # ************************************************************* #
    # Determin whether to split this box into sub-boxes (i.e. children)
    # and if needed, split
    # ************************************************************* #
    def splitcenter(this):
        from numpy import array,zeros,linalg,inf,mean,amax,amin,sqrt
        from positive import magenta,bold,alert,error,red,warning,yellow
        mins =  this.__localmin__
        num_solutions = len(array(mins)[0])
        if num_solutions > 1: # Split the box
            # for each min
            for k in range(len(mins[0])):

                # construct the center location
                kr = mins[1][k]; wr = this.wr_range[ kr ]
                kc = mins[0][k]; wc = this.wc_range[ kc ]
                sc = this.__scgrid__[kr,kc]
                # Determine the resolution of the new box
                res = int( max( 20, 1.5*float(this.res)/num_solutions ) )
                # Create the new child. NOTE that the child's dimensions will be set below using a standard method.
                child = cwbox( this.l,this.m,wr,wc,0,0, res, parent=this, sc=sc, verbose=this.verbose,s=this.s )
                # Add the new box to the current box's child list
                this.children.append( child )

            # NOTE that here we set the box dimensions of all children using the relative distances between them
            this.sensescale()

            # Now redefine the box size to contain all children
            # NOTE that this step exists only to ensure that the box always contains all of its children's centers
            children = this.collectchildren()
            wr = array( [ child.center[0] for child in children ] )
            wc = array( [ child.center[1] for child in children ] )
            width = amax(wr)-amin(wr)
            height = amax(wc)-amin(wc)
            cwr = mean(wr)
            cwc = mean(wc)
            this.setboxprops( cwr,cwc,width,height,this.res,sc=sc )

        elif num_solutions == 1:
            # construcut the center location
            k = 0 # there should be only one local min
            kr = mins[1][k]
            kc = mins[0][k]
            wr = this.wr_range[ kr ]
            wc = this.wc_range[ kc ]
            # retrieve associated separation constant
            sc  = this.__scgrid__[kr,kc]
            # Recenter the box on the current min
            this.setboxprops(wr,wc,this.width,this.height,this.res,sc=sc)
        else:
            #
            if len(this.__jf__)>3:
                alert('Invalid number of local minima found: %s.'% (magenta(bold('%s'%num_solutions))), 'splitcenter' )
                # Use the extrapolated values as a guess?
                alert(yellow('Now trying to use extrapolation, wrather than grid guess, to center the current box.'),'splitcenter')
                #
                guess = this.guess(this.__jf__[-1],gridguess=[1.0,1.0,4.0,1.0])
                wr,wc,cr,cc = guess[0],guess[1],guess[2],guess[3]
                sc = cr+1j*cc
                # Recenter the box on the current min
                this.setboxprops(wr,wc,this.width,this.height,this.res,sc=sc)
            else:
                warning('Invalid number of local minima found: %s. This box will be removed. NOTE that this may not be what you want, and further inspection may be warranted.'% (magenta(bold('%s'%num_solutions))), 'splitcenter' )
                this.__removeme__ = True


    # Validate children: Remove duplicates
    def validatechildren(this):
        #
        from numpy import linalg,array
        from positive import alert,yellow,cyan,blue,magenta
        tol = 1e-6

        #
        if not this.isfundamental():

            #
            children = this.collectchildren()
            initial_count = len(children)

            # Remove identical twins
            for a,tom in enumerate( children ):
                for b,tim in enumerate( children ):
                    if b>a:
                        if linalg.norm(array(tom.center)-array(tim.center)) < tol:
                            if this.verbose:
                                msg = 'Removing overtone '+yellow('%s'%list(tim.__label__))+' becuase it has a twin.'
                                alert(msg,'validatechildren')
                            tim.parent.children.remove(tim)
                            del tim
                            break

            # Remove overtones over the max label
            if this.__maxn__ is not None:
                for k,child in enumerate(this.collectchildren()):
                    if child.__label__[2] > this.__maxn__:
                        if this.verbose:
                            msg = 'Removing overtone '+yellow('%s'%list(child.__label__))+' becuase its label is higher than the allowed value specified.'
                            alert(msg,'validatechildren')
                        this.labelmap.pop( child.__label__ , None)
                        child.parent.children.remove(child)
                        del child

            # Remove all boxes marked for deletion
            for child in this.collectchildren():
                if child.__removeme__:
                    this.labelmap.pop( child.__label__, None )
                    child.parent.children.remove( child )
                    del child

            #
            final_count = len( this.collectchildren() )
            #
            if this.verbose:
                if final_count != initial_count:
                    alert( yellow('%i children have been removed, and %i remain.') % (-final_count+initial_count,final_count) ,'validatechildren')
                else:
                    alert( 'All children have been deemed valid.', 'validatechildren' )


    # Method for collecting all fundamental children
    def collectchildren(this,children=None):
        #
        if children is None:
            children = []
        #
        if this.isfundamental():
            children.append(this)
        else:
            for child in [ k for k in this.children if k is not this ]:
                children += child.collectchildren()
        #
        return children


    # Method to plot solutions
    def plot(this,fig=None,show=False,showlabel=False):
        #
        from numpy import array,amin,amax,sign
        from matplotlib.pyplot import plot,xlim,ylim,xlabel,ylabel,title,figure,gca,text
        from matplotlib.pyplot import show as show_

        #
        children = this.collectchildren()
        wr = array( [ child.center[0] for child in children ] )
        wc =-array( [ child.center[1] for child in children ] )
        wr_min,wr_max = amin(wr),amax(wr)
        wc_min,wc_max = amin(wc),amax(wc)

        padscale = 0.15
        padr,padc = 1.5*padscale*(wr_max-wr_min), padscale*(wc_max-wc_min)
        wr_min -= padr; wr_max += padr
        wc_min -= padc; wc_max += padc
        #
        if fig is None:
            # fig = figure( figsize=12*array((wr_max-wr_min, wc_max-wc_min))/(wr_max-wr_min), dpi=200, facecolor='w', edgecolor='k' )
            fig = figure( figsize=12.0*array((4.5, 3))/4.0, dpi=200, facecolor='w', edgecolor='k' )
        #
        xlim( [wr_min,wr_max] )
        ylim( [wc_min,wc_max] )
        ax = gca()
        #
        for child in children:
            plot( child.center[0],-child.center[1], '+k', ms=10 )
            ax.add_patch( child.patch )
            if showlabel:
                text( child.center[0]+sign(child.center[0])*child.width/2,-(child.center[1]+child.height/2),
                      '$(%i,%i,%i,%i,%i)$'%(this.labelmap[child]),
                      ha=('right' if sign(child.center[0])<0 else 'left' ),
                      fontsize=10,
                      alpha=0.9 )
        #
        xlabel(r'$\mathrm{re}\;\tilde\omega_{%i%i}$'%(this.l,this.m))
        ylabel(r'-$\mathrm{im}\;\tilde\omega_{%i%i}$'%(this.l,this.m))
        title(r'$j_f = %1.6f$'%this.__jf__[-1],fontsize=18)
        #
        if show: show_()

    # ************************************************************* #
    # Solve leaver's equations in a given box=[wr_range,wc_range]
    # NOTE that the box is a list, not an array
    # ************************************************************* #
    def lvrgridsolve(this,jf=0,fullopt=False):
        # Import maths
        from numpy import linalg,complex128,ones,array
        from positive.physics import scberti
        from positive.physics import leaver_workfunction
        from scipy.optimize import fmin,root
        import sys

        # Pre-allocate an array that will hold work function values
        x = ones(  ( this.wc_range.size,this.wr_range.size )  )
        # Pre-allocate an array that will hold sep const vals
        scgrid = ones(  ( this.wc_range.size,this.wc_range.size ), dtype=complex128  )
        # Solve over the grid
        for i,wr in enumerate( this.wr_range ):
            for j,wc in enumerate( this.wc_range ):
                # Costruct the complex frequency for this i and j
                cw = complex128( wr+1j*wc )

                # # Define the intermediate work function to be used for this iteration
                # fun = lambda SC: linalg.norm( array(leaver_workfunction( jf,this.l,this.m, [cw.real,cw.imag,SC[0],SC[1]] )) )
                # # For this complex frequency, optimize over separation constant using initial guess
                # SC0_= scberti( cw*jf, this.l, this.m ) # Use Berti's analytic prediction as a guess
                # SC0 = [SC0_.real,SC0_.imag]
                # X  = fmin( fun, SC0, disp=False, full_output=True, maxiter=1 )
                # # Store work function value
                # x[j][i] = X[1]
                # # Store sep const vals
                # scgrid[j][i] = X[0][0] + 1j*X[0][1]

                if fullopt is False:

                    # Define the intermediate work function to be used for this iteration
                    fun = lambda SC: linalg.norm( array(leaver_workfunction( jf,this.l,this.m, [cw.real,cw.imag,SC[0],SC[1]], s=this.s, adjoint=this.adjoint )) )
                    # For this complex frequency, optimize over separation constant using initial guess
                    SC0_= sc_leaver( cw*jf, this.l, this.m, s=this.s, adjoint=False )[0]
                    SC0 = [SC0_.real,SC0_.imag]
                    # Store work function value
                    x[j][i] = fun(SC0)
                    # Store sep const vals
                    scgrid[j][i] = SC0_

                else:

                    SC0_= sc_leaver( cw*jf, this.l, this.m, s=this.s, adjoint=False )[0]
                    SC0 = [SC0_.real,SC0_.imag,0,0]
                    #cfun = lambda Y: [ Y[0]+abs(Y[3]), Y[1]+abs(Y[2]) ]
                    fun = lambda SC:leaver_workfunction( jf,this.l,this.m, [cw.real,cw.imag,SC[0],SC[1]], s=this.s, adjoint=this.adjoint )
                    X  = root( fun, SC0 )
                    scgrid[j][i] = X.x[0]+1j*X.x[1]
                    x[j][i] = linalg.norm( array(X.fun) )


            if this.verbose:
                sys.stdout.flush()
                print('.',end='')

        if this.verbose: print('Done.')
        # return work function values AND the optimal separation constants
        return x,scgrid


    # Convert output of localmin to a state vector for minimization
    def grids2states(this):

        #
        from numpy import complex128
        state = []

        #
        for k in range( len(this.__localmin__[0]) ):
            #
            kr,kc = this.__localmin__[1][k], this.__localmin__[0][k]
            cw = complex128( this.wr_range[kr] + 1j*this.wc_range[kc] )
            sc = complex128( this.__scgrid__[kr,kc] )
            #
            state.append( [cw.real,cw.imag,sc.real,sc.imag] )

        #
        return state


    # Get guess either from local min, or from extrapolation of past data
    def guess(this,jf,gridguess=None):
        #
        from positive.physics import leaver_workfunction
        from positive import alert,magenta,apolyfit
        from positive import localmins
        from numpy import array,linalg,arange,complex128,allclose,nan
        from scipy.interpolate import InterpolatedUnivariateSpline as spline
        # Get a guess from the localmin
        if gridguess is None:
            this.__x__,this.__scgrid__ = this.lvrgridsolve(jf,fullopt=True)
            this.__localmin__ = localmins(this.__x__,edge_ignore=True)
            states = this.grids2states()
            if len(states):
                guess1 = states[0]
            else:
                error('The grid is empty.')
        else:
            guess1 = gridguess
        # Get a guess from extrapolation ( performed in curate() )
        guess2 = [ v for v in guess1 ]
        if this.mapcount > 3:
            # if there are three map points, try to use polynomial fitting to determine the state at the current jf value
            nn = len(this.__data__['jf'])
            order = min(2,nn)
            #
            xx = array(this.__data__['jf'])[-4:]
            #
            yy = array(this.__data__['cw'])[-4:]
            yr = apolyfit( xx, yy.real, order )(jf)
            yc = apolyfit( yy.real, yy.imag, order )(yr)
            cw = complex128( yr + 1j*yc )
            #
            zz = array(this.__data__['sc'])[-4:]
            zr = apolyfit( xx, zz.real, order  )(jf)
            zc = apolyfit( zz.real, zz.imag, order  )(zr)
            sc = complex128( zr + 1j*zc )
            #
            guess2 = [ cw.real, cw.imag, sc.real, sc.imag ]
        # Determine the best guess
        if not ( allclose(guess1,guess2) ):
            x1 = linalg.norm( leaver_workfunction( jf,this.l,this.m, guess1, s=this.s, adjoint=this.adjoint ) )
            x2 = linalg.norm( leaver_workfunction( jf,this.l,this.m, guess2, s=this.s, adjoint=this.adjoint ) )
            alert(magenta('The function value at guess from grid is:   %s'%x1),'guess')
            alert(magenta('The function value at guess from extrap is: %s'%x2),'guess')
            if x2 is nan:
                x2 = 100.0*x1
            if x1<x2:
                guess = guess1
                alert(magenta('Using the guess from the grid.'),'guess')
            else:
                guess = guess2
                alert(magenta('Using the guess from extrapolation.'),'guess')
        else:
            x1 = linalg.norm( leaver_workfunction( jf,this.l,this.m, guess1, s=this.s, adjoint=this.adjoint ) )
            guess = guess1
            alert(magenta('The function value at guess from grid is %s'%x1),'guess')

        # Recenter the box on the current guess
        wr,wc = guess[0],guess[1]
        sc = guess[2]+1j*guess[3]
        this.setboxprops(wr,wc,this.width,this.height,this.res,sc=sc)

        # Return the guess solution
        return guess


    # Determine whether the current box contains a complex frequency given an iterable whose first two entries are the real and imag part of the complex frequency
    def contains(this,guess):
        #
        cwrmin = min( this.limit[:2] )
        cwrmax = max( this.limit[:2] )
        cwcmin = min( this.limit[2:] )
        cwcmax = max( this.limit[2:] )
        #
        isin  = True
        isin = isin and ( guess[0]<cwrmax )
        isin = isin and ( guess[0]>cwrmin )
        isin = isin and ( guess[1]<cwcmax )
        isin = isin and ( guess[1]>cwcmin )
        #
        return isin


    # Try solving the 4D equation near a single guess value [ cw.real cw.imag sc.real sc.imag ]
    def lvrsolve(this,jf,guess,tol=1e-8):

        # Import Maths
        from numpy import log,exp,linalg,array
        from scipy.optimize import root,fmin,minimize
        from positive.physics import leaver_workfunction
        from positive import alert,red,warning

        # Try using root
        # Define the intermediate work function to be used for this iteration
        fun = lambda STATE: log( 1.0 + abs(array(leaver_workfunction( jf,this.l,this.m, STATE, s=this.s ))) )
        X  = root( fun, guess, tol=tol )
        cw1,sc1 = X.x[0]+1j*X.x[1], X.x[2]+1j*X.x[3]
        __lvrfmin1__ = linalg.norm(array( exp(X.fun)-1.0 ))
        retry1 = ( 'not making good progress' in X.message.lower() ) or ( 'error' in X.message.lower() )


        # Try using fmin
        # Define the intermediate work function to be used for this iteration
        fun = lambda STATE: log(linalg.norm(  leaver_workfunction( jf,this.l,this.m, STATE, s=this.s )  ))
        X  = fmin( fun, guess, disp=False, full_output=True, ftol=tol )
        cw2,sc2 = X[0][0]+1j*X[0][1], X[0][2]+1j*X[0][3]
        __lvrfmin2__ = exp(X[1])
        retry2 = this.__lvrfmin__ > 1e-3

        # Use the solution that converged the fastest to avoid solutions that have wandered significantly from the initial guess OR use the solution with the smallest fmin
        if __lvrfmin1__ < __lvrfmin2__ : # use the fmin value for convenience
            cw,sc,retry = cw1,sc1,retry1
            __lvrfmin__ = __lvrfmin1__
        else:
            cw,sc,retry = cw2,sc2,retry2
            __lvrfmin__ = __lvrfmin2__


        # Always retry if the solution is outside of the box
        if not this.contains( [cw.real,cw.imag] ):
            warning('The proposed solution is outside of the box, and may now not correspond to the correct label.')
            # retry = True
            # alert(red('Retrying because the trial solution is outside of the box.'),'lvrsolve')

        # Don't retry if fval is small
        if __lvrfmin__ > 1e-3:
            retry = True
            alert(red('Retrying because the trial fmin value is greater than 1e-3.'),'lvrsolve')

        # Don't retry if fval is small
        if retry and (__lvrfmin__ < 1e-4):
            retry = False
            alert(red('Not retrying becuase the fmin value is low.'),'lvrsolve')

        # Return the solution
        return cw,sc,__lvrfmin__,retry

    # Give a solution for the current spin, fix the solution location, but vary the spin such that the current solution has an error of tol. This change in spin for tol may be useful in dynamically stepping through spin values.
    def gauge( this, jf, solution, tol=1e-4 ):

        # Import Maths
        from numpy import log,exp,linalg,array
        from scipy.optimize import root,fmin,minimize
        from positive.physics import leaver_workfunction
        from positive import alert,red,warning,error

        #
        fun = lambda JF: linalg.norm(  leaver_workfunction( JF,this.l,this.m, solution, s=this.s, adjoint=this.adjoint )  )

        f0 = fun( jf )
        djf = 1e-6
        done = False
        _jf = jf; kmax = 2e3
        k = 0; rtol = 1e-2 * tol
        while not done:
            #
            k+=1
            #
            _jf += djf
            #
            f = fun( _jf )
            delta = (f-f0) - tol
            #
            we_have_gone_too_far = delta > rtol
            we_must_go_further = delta < 0
            #
            if we_have_gone_too_far:
                # go back a step, and half djf
                _jf -= djf
                djf /= 2.0
                done = False
            elif we_must_go_further:
                done = False
                djf *= 1.055
            else:
                done = True

            #
            if k>kmax: error('This process has not converged.')

        #
        ans = _jf
        return ans



    # Given a box's children, resize the boxes relative to child locations: no boxes overlap
    def sensescale(this):

        #
        from numpy import array,inf,linalg,sqrt
        from positive import alert

        #
        children = this.collectchildren()

        # Let my people know.
        if this.verbose:
            alert('Sensing the scale of the current object\'s sub-boxes.','sensescale')

        # Determine the distance between this min, and its closest neighbor
        scalar = sqrt(2) if (not this.__smallboxes__) else 2.0*sqrt(2.0)
        for tom in children:

            d = inf
            for jerry in [ kid for kid in children if kid is not tom ]:

                r = array(tom.center)
                r_= array(jerry.center)
                d_= linalg.norm(r_-r)
                if d_ < d:
                    d = d_

            # Use the smallest distance found to determine a box size
            s = d/scalar
            width = s; height = s; res = int( max( 20, 1.5*float(this.res)/len(children) ) ) if (len(children)>1) else this.res

            # Define the new box size for this child
            tom.setboxprops( tom.center[0], tom.center[1], width, height, res )

    #
    def pop(this):

        def pop(a):
            a = a[:-1]

        this.__jf__.pop()
        for k in this.__data__:
            pop( this.__data__[ k ][ 'jf' ]      )
            pop( this.__data__[ k ][ 'cw' ]      )
            pop( this.__data__[ k ][ 'sc' ]      )
            pop( this.__data__[ k ][ 'lvrfmin' ] )

