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



# Evaluate spheroidal multipole moments (spheroical projections also output)
def eval_spheroidal_moments( a, M, spheroidal_amplitudes_dict, times=None, verbose=False ):
    
    
    '''
    Evaluate spheroidal multipole moments (spheroical projections also output).
    '''
    
    #
    from numpy import ndarray,sum,array,zeros_like
    
    # Validate inputs
    # ---
    
    # Dictionary formatting 
    mref = None
    for k in spheroidal_amplitudes_dict:
        
        #
        if not isinstance(k,(tuple,list,ndarray)):
            error('keys of amplitude dict must be iterable containing ell m n p, where n and p are overtone and parity label for spheroidal momements (eg QNMs)')
        elif len(k) != 4:
            error('keys of ampltides dict must of length 4 and contain ell m n p, where n and p are overtone and parity label for spheroidal momements (eg QNMs)')    
        else:
            for index in k:
                if not isinstance(index,int):
                    error('QNM index not int: keys of ampltides dict must of length 4 and contain ell m n p, where n and p are overtone and parity label for spheroidal momements (eg QNMs)')
                    
        #
        if mref is None: 
            _,mref,_,_ = k
            
        #
        if mref != k[1]:
            error('m-pole mismatch: all values of m in keys of spheroidal_amplitudes_dict must be equal. The user should consider only sets of like m when using this method. WE ARE BORG. YOU WILL BE ASSIMILATED.')
        
    # Amplitudes must be consistent or single number
    test_amplitude = spheroidal_amplitudes_dict[ spheroidal_amplitudes_dict.keys()[0] ]
    if isinstance(test_amplitude,(list,tuple,ndarray)):
        if isinstance(times,ndarray):
            error('Amplitudes are given as timeseries, but time are also given and this should not be the case.')
        #
        process_timeseries_amplitudes = True
        alert('Processing spheroidal amplitude timeseries.',verbose=verbose)
    else:
        if not isinstance(times,ndarray):
            error('Amplitudes are given as single values, but time values not input or are input of the wrong type. A numpy array of times must be given. Times and amplitudes must have the same array shape.')
        if len(test_amplitude) != len(times):
            error('length mismatch: amplitudes found to be time series, but of lenth not equal to the times input')
        if isinstance(test_amplitude,(int,float,complex)):
            process_timeseries_amplitudes = False
            alert('Processing spheroidal amplitude values.',verbose=verbose)
        else:
            error('type error: spheroidal amplitude must be numpy array or number')
    
    # End of input validation
    
    # Generate QNM objects, including the related spheroidal harmonics
    # ---
    k_space   = sorted(spheroidal_amplitudes_dict.keys())
    qnmo_list = [ qnmobj(M,a,l,m,n,p=p,use_nr_convention=True,verbose=False)  for l,m,n,p in k_space ]
    
    # Setup data for spherical info containers
    # ---
        
    # Calculate spherical moment timeseries
    # --- 
    
    # define helper function 
    def calc_spherical_moments_helper( spheroidal_moment ):

        # Define list of spherical indices to consider for data storage
        j_space = sorted(set( [ (l,m) for l,m,n,p in k_space ] ))

        #
        spherical_moments_dict = {}
            
        #
        for index,j in enumerate(j_space):
            
            #
            ll,mm = j
            
            #
            spherical_moments_dict[j] = sum( [ spheroidal_amplitudes_dict[k] * qnmo_list[index].ysprod(ll,mm) for index,k in enumerate(k_space) ], axis=0 )
            
        #
        return spherical_moments_dict
        
    
    #
    if process_timeseries_amplitudes:
        
        #-~>-~>-~>-~>-~>-~>-~>-~>-~>-~>-~>-~>-#
        # IF ampltiudes are timeseries        #
        #-~>-~>-~>-~>-~>-~>-~>-~>-~>-~>-~>-~>-#
        
        # Define spheroidal_moments_dict so that the same variable ID is used for both cases of process_timeseries_amplitudes
        spheroidal_moments_dict = spheroidal_amplitudes_dict
        
        # Calculate the spherical moments
        spherical_moments_dict = calc_spherical_moments_helper( spheroidal_moments_dict )
        
    else:
        
        #-~>-~>-~>-~>-~>-~>-~>-~>-~>-~>-~>-~>-#
        # ELSE-IF amplitudes are constant     #
        #-~>-~>-~>-~>-~>-~>-~>-~>-~>-~>-~>-~>-#
        # THEN we assumed the damped sinusoidal time depedence is desired
        
        # Calculate spheroidal moments
        spheroidal_moments_dict = {}
        for index,k in enumerate(k_space):
            exponential_part = exp( 1j * qnmo_list[index].CW * times )
            spheroidal_moments_dict[k] = spheroidal_amplitudes_dict[k] * exponential_part
        
        # Calculate the spherical moments
        spherical_moments_dict = calc_spherical_moments_helper( spheroidal_moments_dict )
        
    #
    return spherical_moments_dict,spheroidal_moments_dict


# Validation for calc_spheroidal_moments
def validate_inputs_for_calc_spheroidal_moments( spherical_moments_dict, a, m, n, p, verbose, s ):
    '''
    Input validation method for calc_spheroidal_moments
    '''
    
    # Import usefuls 
    from numpy import sort,array,double,ndarray
    from numpy.linalg import inv
    
    # Validate inputs
    # ---
    if not isinstance(spherical_moments_dict,dict): 
        error('first input must be dict of spherical harmonic spin weight -2 waveform samples with keys (l,m)')
    Y = None
    for k in spherical_moments_dict:
        if len(k)!=2:
            error('key in first input found to have a length of %i when it should have a length of 2'%len(k))
        ll,mm = k
        if not isinstance(ll,int): 
            error('l index in spherical_moments_dict key found to not be int type')
        if not isinstance(mm,int): 
            error('m index in spherical_moments_dict key found to not be int type')
        if not isinstance(spherical_moments_dict[k],ndarray):
            error('spherical moment data must be ndarrays')
        if Y is None:
            Y = spherical_moments_dict[k]
        else:
            if Y.shape != spherical_moments_dict[k].shape:
                error('not all spehrical moments are the same shape')
    m_test = sum([ mm==m for ll,mm in spherical_moments_dict ]) == len(spherical_moments_dict)
    if not m_test:
        error('all spherical multipole moments must have the same value of m as the desired set of spheroidal moments')
    if not isinstance(n,(int,list,tuple,ndarray)):
        error('n not int or iterable')
    if not isinstance(p,(int,list,tuple,ndarray)):
        error('p not int or iterable')
    if isinstance(n,int):
        if n<0:
            error('n, the overtone index, must be a non-negative integer')
    if isinstance(p,int):
        if not (p in [-1,1]):
            error('p must be either -1 or +1 but it is not')
    if abs(m) < abs(s):
        error('abs(m) must be greater than or equal to abs(s)=2 but it is %i'%abs(m))
    if s!=-2:
        error('this function only works for spin weigth -2 fields')
    if not isinstance(a,(float,double)):
        error('a, the dimensionless BH spin parameter, must be float. For time variable values, please \
        run calc_spheroidal_moments in a loop whith spherical_moments_dict defined by single time samples')


# Validation for calc_spheroidal_moments
def validate_inputs_for_calc_spheroidal_moments_helper( spherical_moments_dict, a, m, n, p, verbose, s ):
    '''
    Input validation method for calc_spheroidal_moments_helper
    '''
    
    # Import usefuls 
    from numpy import sort,array,double,ndarray
    from numpy.linalg import inv
    
    # Validate inputs
    # ---
    if not isinstance(spherical_moments_dict,dict): 
        error('first input must be dict of spherical harmonic spin weight -2 waveform samples with keys (l,m)')
    Y = None
    for k in spherical_moments_dict:
        if len(k)!=2:
            error('key in first input found to have a length of %i when it should have a length of 2'%len(k))
        ll,mm = k
        if not isinstance(ll,int): 
            error('l index in spherical_moments_dict key found to not be int type')
        if not isinstance(mm,int): 
            error('m index in spherical_moments_dict key found to not be int type')
        if not isinstance(spherical_moments_dict[k],ndarray):
            error('spherical moment data must be ndarrays')
        if isinstance(spherical_moments_dict[k],(float,complex)):
            spherical_moments_dict[k] = array([spherical_moments_dict[k]])
        if Y is None:
            Y = spherical_moments_dict[k]
        else:
            if Y.shape != spherical_moments_dict[k].shape:
                error('not all spehrical moments are the same shape')
    m_test = sum([ mm==m for ll,mm in spherical_moments_dict ]) == len(spherical_moments_dict)
    if not m_test:
        error('all spherical multipole moments must have the same value of m as the desired set of spheroidal moments')
    if not isinstance(n,int):
        error('n not int')
    if n<0:
        error('n, the overtone index, must be a non-negative integer')
    if not (p in [-1,1]):
        error('p must be either -1 or +1 but it is not')
    if abs(m) < abs(s):
        error('abs(m) must be greater than or equal to abs(s)=2 but it is %i'%abs(m))
    if s!=-2:
        error('this function only works for spin weigth -2 fields')
    if not isinstance(a,(float,double)):
        error('a, the dimensionless BH spin parameter, must be float. For time variable values, please \
        run calc_spheroidal_moments in a loop whith spherical_moments_dict defined by single time samples')
        
    #
    return spherical_moments_dict


# Calc spheroidal moments from spherical ones
def calc_spheroidal_moments( spherical_moments_dict, a, m, n, p, time, verbose=False, s=-2, spectral=True, method=None,harmonic_norm_convention=None, np=None, derivatives=None ):
    
    '''
    '''
    
    # Import usefuls 
    # ---
    from numpy import ndarray,zeros,sort,array,dot,int64
    from numpy.linalg import inv,pinv
    
    # Validate inputs
    # ---
    validate_inputs_for_calc_spheroidal_moments( spherical_moments_dict, a, m, n, p, verbose, s )
    
    
    # Handle cases:
    # ---
    
    #   1. Both p and n are integers
    both_p_and_n_are_integers = isinstance(p,(int,int64)) and isinstance(n,(int,int64))
    #   2. Either p or n is iterable of integers
    either_p_or_n_is_iterable = isinstance(p,(list,tuple,ndarray)) or isinstance(n,(list,tuple,ndarray))
    
    # 
    if both_p_and_n_are_integers:
        
        # ~- -~ ~- -~ ~- -~ ~- -~ #
        # Case (1)
        # ~- -~ ~- -~ ~- -~ ~- -~ #
        
        #
        spheroidal_moments_dict,L,Z,qnmo_dict = calc_spheroidal_moments_helper( spherical_moments_dict, a, m, n, p, verbose=verbose, s=s, spectral=spectral,np=np )
        
    elif either_p_or_n_is_iterable:
        
        # ~- -~ ~- -~ ~- -~ ~- -~ #
        # Case (2)
        # ~- -~ ~- -~ ~- -~ ~- -~ #

        # 
        if method is None:
            if (len(p)==1) & (len(n)==1):
                alert('Using projection to determine spheroidal moments')
                method = 'projection'
            else:
                alert('Using gradient to determine spheroidal moments')
                method = 'gradient'
        
        #
        if method in ('grad','gradient'):
            
            # Use derivatives of input sperical multipoles as features
            spheroidal_moments_dict, L, Z, qnmo_dict = __calc_spheroidal_moments_via_gradient__(
                spherical_moments_dict, a, m, n, p, verbose=verbose, s=s, spectral=spectral, time=time, np=np, derivatives=derivatives)
            
        else:
            
            # Use projects as features 
            spheroidal_moments_dict,L,Z,qnmo_dict = __calc_spheroidal_moments_via_projection__(spherical_moments_dict, a, m, n, p, verbose=verbose, s=s, spectral=spectral,harmonic_norm_convention=harmonic_norm_convention,np=np)
            
    #
    return spheroidal_moments_dict,L,Z,qnmo_dict
            
        
#
def __calc_spheroidal_moments_via_gradient__(spherical_moments_dict, a, m, n, p, verbose=False, s=-2, spectral=True, spheroidal_moments_dict=None, T_dict=None, V_dict=None, qnmo_dict=None, time=None, np=None, derivatives=None):
    
    # Import usefuls 
    # ---
    from numpy import ndarray,zeros,sort,array,dot,hstack,vstack,arange
    from numpy.linalg import inv,pinv
    from scipy.linalg import lstsq
    
    
    # Determine if iterables are given
    # ---
    
    #
    p_is_iterable = isinstance(p,(list,tuple,ndarray))
    n_is_iterable = isinstance(n,(list,tuple,ndarray))
    
    #
    if not p_is_iterable: p = [p]
    if not n_is_iterable: n = [n]
    
    #
    if time is None:
        error('the gradient method requires a time input')
    
    #
    if (not p_is_iterable) and (not n_is_iterable):
        error('either p or n should be iterable')
    
    #
    p_iterable,n_iterable = p,n 
    
    # #
    # if p_iterable[0]==-1: 
    #     p_iterable = p_iterable[::-1]
    
    
    # Initiate dictionaries
    # ---
    
    # For output spheroidal moments
    spheroidal_moments_dict = {} if spheroidal_moments_dict is None else spheroidal_moments_dict.copy()
    # For the spherical to spheroidal map
    T_dict = {} if T_dict is None else T_dict.copy()
    # For the spheroidal to spherical map
    V_dict = {} if V_dict is None else V_dict.copy()
    # For the QNM objects
    qnmo_dict = {} if qnmo_dict is None else qnmo_dict.copy()
    
    
    # Extract vector of spherical moments from the input dict 
    # ---
    lrange = sort( [ l for l,_ in spherical_moments_dict ] )
    t = time
    Y_raw = array( [ spherical_moments_dict[l,m] for l in lrange ] )
    
    # Construct seed mixing matrices (ie the matrices for the 0th derivatives)
    foo = {}
    for pk in p_iterable:
        for nk in n_iterable:
            foo[nk,pk] = ysprod_matrix(a,m,nk,pk,s=s,lrange=lrange,verbose=verbose,spectral=spectral,qnmo_dict=qnmo_dict,full_output=True)
    
    # Construct dict of derivatives
    DY = {}
    k = 0
    for pk in p_iterable:
        for nk in n_iterable:
            for l in lrange:
                if derivatives is None:
                    # DY[l,m,k] = ffdiff( t, spherical_moments_dict[l,m], n=k,wlim=[0.0,5*abs(qnmo_dict[l,m,nk,pk].cw.real)] )
                    DY[l,m,k] = spline_diff( t, spherical_moments_dict[l,m], n=k )
                else:
                    DY[l,m,k] = derivatives[l,m][k]
            k += 1
            
    # Stack the matrices horizontally
    T1 = None
    for pk in p_iterable:
        for nk in n_iterable:
            T0  = foo[nk,pk]['ys_matrix']
            if T1 is None:
                T1  =  T0
            else:
                T1  = hstack( [ T1, T0] )
    CW1=None
    for pk in p_iterable:
        for nk in n_iterable:
            CW0 = foo[nk,pk]['cw_matrix']
            if CW1 is None:
                CW1 = CW0
            else:
                CW1 = hstack( [CW1,CW0] )
                
    # Add derivative cases by scaling and then vertically stacking
    T,Y = None,None
    k = 0
    CW = CW1
    for pk in p_iterable:
        for nk in n_iterable:
            if T is None:
                T = T1 
            else:
                T = vstack( [T,T1 * ((1j*CW)**k) ] )
            #
            k += 1
            
    #
    L = T 
    Z = inv(L)
            
    #
    N = len(p) * len(n) * len(spherical_moments_dict)
    S = zeros( (N,len(t)), dtype=complex )
    for kref,tref in enumerate(t):
        
        #
        Y = []
        j = 0
        for pk in p_iterable:
            for nk in n_iterable:
                for l in lrange:
                    Y.append( DY[l,m,j][kref] )
                j +=1
        Y = array(Y)    
        
        #    
        S[:,kref] = dot( Z,Y )
        # S[:,kref] = lstsq( L,Y )[0] # Gives basically the same answer as above
        
    # Construct dictionaries
    final_spheroidal_moments_dict = {}
    #
    row_index = -1
    for a,pa in enumerate(p_iterable):
        for b,nb in enumerate(n_iterable):
            for c,lc in enumerate(lrange):
                row_index += 1
                final_spheroidal_moments_dict[ ( lc,m,nb,pa ) ] = S[row_index]
    
    
    # Output
    # ---
    return (final_spheroidal_moments_dict,L,Z,qnmo_dict)

#
def __calc_spheroidal_moments_via_projection__(spherical_moments_dict, a, m, n, p, verbose=False, s=-2,np=None, spectral=True,harmonic_norm_convention=None):
    
    # Import usefuls 
    # ---
    from numpy import ndarray,zeros,sort,array,dot
    from numpy.linalg import inv,pinv
    from scipy.linalg import lstsq
    
    #
    #error('this method does not work; please use the "gradient" method')
    if np is not None: error('this method is not setup to handle the np input')
    if len(p)>1: error('more than one value of p requested but this method only works with one value of p; for multiple values of p use the gradient method')
    
    #
    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
    # PART 1: Construct the linear system's matrix operator
    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
    
    #
    spheroidal_moments_dict = {}
    T_dict, V_dict, qnmo_dict = {}, {}, {}
    
    #
    p_is_iterable = isinstance(p,(list,tuple,ndarray))
    n_is_iterable = isinstance(n,(list,tuple,ndarray))
    
    #
    if not p_is_iterable: p = [p]
    if not n_is_iterable: n = [n]
    
    #
    if (not p_is_iterable) and (not n_is_iterable):
        error('either p or n should be iterable')
    
    #
    p_iterable,n_iterable = p,n 
    
    #
    for pk in p_iterable:
        for nk in n_iterable:
            
            #
            spheroidal_moments_dict,T_dict,V_dict,qnmo_dict = calc_spheroidal_moments_helper( spherical_moments_dict, a, m, nk, pk, verbose=verbose, s=s, spectral=spectral, spheroidal_moments_dict=spheroidal_moments_dict,T_dict=T_dict,V_dict=V_dict, qnmo_dict=qnmo_dict,harmonic_norm_convention=harmonic_norm_convention )
        
    # # 
    # return spheroidal_moments_dict,T_dict,V_dict,qnmo_dict
    
    # -~- -~- -~- -~- -~- -~- -~- -~- -~- -~- -~- -~- -~- -~- -~- -~- -~- -~- -~- #
    
    # The square matrix we wish to constrcut will have a width of widL
    N = len(p) * len(n) * len(spherical_moments_dict)

    # Preallocate the matrix 
    L = zeros( (N,N), dtype=complex )

    # Preallocate the output array
    Y = zeros( (N,), dtype=complex )

    # Y = L X ---> L^-1 Y = X

    #
    lrange = sort( [ l for l,_ in spherical_moments_dict ] )

    #
    row_index = -1
    for a,pa in enumerate(p):
        for b,nb in enumerate(n):
            for c,lc in enumerate(lrange):
                
                # ROWS
                
                #
                row_index += 1
                
                #
                col_index = -1
                for d,pd in enumerate(p):
                    for e,ne in enumerate(n):
                        for f,lf in enumerate(lrange):
                            
                            # COLUMNS
                            
                            #
                            col_index += 1
                            
                            #
                            LT = array([  T_dict[ (ll,m), (lc,m,nb,pa) ] for ll in lrange ],dtype=complex)
                            LV = array([  V_dict[ (lf,m,ne,pd),(ll,m) ] for ll in lrange ],dtype=complex )
                            
                            #
                            # weight = 1.0 / abs(  (qnmo_dict[lc,m,nb,pa].cw) * (qnmo_dict[lf,m,ne,pd].cw.conj())  )
                            weight = 1.0
                            # weight = (1j*qnmo_dict[lc,m,nb,pa].cw) ** ( col_index-1)
                            # weight = (1j*qnmo_dict[lf,m,ne,pd].cw) ** ( 1-pd)
                            
                            # NOTE that this line is corrently wrong
                            # NOTE that dot does not complex conjugate
                            #error('Please try to calculate spheroidal-spheroidal inner product manually rather than use results from calc_spheroidal_moments_helper')
                            L[row_index,col_index] = weight * dot( LT, LV )


    
    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
    # PART 2: Construct the linear system's output vector at all 
    # domain points
    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
    
    # Invert the array representation of the mixing tensor 
    Z = pinv(L)

    # Define index domain and pre-allocate spheroidal output
    # ---
    index_domain = range(len( spherical_moments_dict[ list(spherical_moments_dict.keys())[0] ] ))
    S = zeros( (N,len(index_domain)), dtype=complex )
    
    # Collect spheroidal information over all domain samples
    # ---
    for k in index_domain:
    
        #
        row_index = -1
        for a,pa in enumerate(p):
            for b,nb in enumerate(n):
                for c,lc in enumerate(lrange):
                    row_index += 1
                    Y[row_index] = spheroidal_moments_dict[ lc, m, ( nb, pa ) ][k]
                    
        # Solve for all spheroidal amplitudes 
        S[:,k] = dot( Z, Y )
        
        #
        # S[:,k], RESIDUAL, EFFECTIVE_RANK, SINGULAR_VALUES = lstsq( L,Y )
        # print(RESIDUAL,EFFECTIVE_RANK)
    
    # Construct dictionaries
    final_spheroidal_moments_dict = {}
    #
    row_index = -1
    for a,pa in enumerate(p):
        for b,nb in enumerate(n):
            for c,lc in enumerate(lrange):
                row_index += 1
                final_spheroidal_moments_dict[ ( lc,m,nb,pa ) ] = S[row_index]
                

    # Output
    # ---
    return (final_spheroidal_moments_dict,L,Z,qnmo_dict)


# Calc spheroidal moments from spherical ones
def calc_spheroidal_moments_helper( spherical_moments_dict, a, m, n, p, verbose=False, s=-2, spectral=True, spheroidal_moments_dict=None, T_dict=None, V_dict=None, qnmo_dict=None,harmonic_norm_convention=None,np=None ):
    
    '''
    GENERAL
    ---
    Given a dictionary of spin -2 weighted spheroidal harmonic moments, 
    use the Kerr QNMs also with spin weight s=-2 to determine the 
    effective spheroidal moments as defined by a fixed n and p subset. 
    This method uses the qnmobj class to consistently enforce NR 
    conventions for the Kerr QNMs. Only single values of the background 
    dimensionless spin, a, are accomodated by this method. 
    
    This method solves the simple linear system
    
    Y = V * S
    
    for the spheroidal harmonic moments, S. Y is a vector of spherical 
    harmonic moments at a sinjgle time. A is a matrix that maps 
    spherical harmonic representations to spheroidal ones. V is
    a matrix of spherical-spheroidal inner-products, and its inverse
    enables S to be determined
    
    S = (V^-1) * Y
    
    USAGE
    ---
    spheroidal_moments_dict = calc_spheroidal_moments( spherical_moments_dict, a, m, n, p, verbose=False, spectral=True )
    
    INPUTS
    ---
    spherical_moments_dict,        Dictionary with keys (l,m), and values being waveform complex 
                                   time sample eg Hlm(t), Psi4lm(t) in the NR convention.
                                   See qnmobj.explain_conventions(). Moment values may be arrays or
                                   single points. If single points, data will be converted to arrays
                                   of shape (1,).
    a,                             Dimensionless spin of spacetime background
    m,                             Azimuthal index of input and output moments. This m must be equal 
                                   to all values of m in the spherical_moments_dict
    n,                             Overtone index
    p,                             Parity index in the NR convention for labeling QNMs
    spectral,                      Toggle to use a spectral method for the determination
                                   of spherical-spheroidal inner products. True by default
                                   as it is slightly faster than directo integration as
                                   would be triggered by spectral=False.
                                   
    OUTPUTS
    ---
    spheroidal_moments_dict,       Dictionary with keys (l,m,n,p), and values given by complex waveform data of the 
                                   type input.
                                   
    AUTHOR
    ---
    londonl@mit.edu, pilondon2@gmail.com 2021
    
    '''
    
    # Import usefuls 
    from numpy import sort,array,zeros,dot
    from numpy.linalg import inv,pinv
    
    # Validate inputs
    # ---
    spherical_moments_dict = validate_inputs_for_calc_spheroidal_moments_helper( spherical_moments_dict, a, m, n, p, verbose, s )
    
    
    # Initiate dictionaries
    # ---
    
    # For output spheroidal moments
    spheroidal_moments_dict = {} if spheroidal_moments_dict is None else spheroidal_moments_dict.copy()
    # For the spherical to spheroidal map
    T_dict = {} if T_dict is None else T_dict.copy()
    # For the spheroidal to spherical map
    V_dict = {} if V_dict is None else V_dict.copy()
    # For the QNM objects
    qnmo_dict = {} if qnmo_dict is None else qnmo_dict.copy()
    
    
    # Extract vector of spherical moments from the input dict 
    # ---
    lrange = sort( [ l for l,_ in spherical_moments_dict ] )
    Y = array( [ spherical_moments_dict[l,m] for l in lrange ] )
        
    #
    foo = ysprod_matrix(a,m,n,p,s=s,lrange=lrange,verbose=verbose,spectral=spectral,qnmo_dict=qnmo_dict,norm_convention=harmonic_norm_convention,full_output=True)
    
    #
    T = foo['ys_matrix']
    
    # Invert map
    # ---
    V = foo['adj_ys_matrix'].conj()
        
    # Define index domain and pre-allocate spheroidal output
    # ---
    index_domain = range(len( spherical_moments_dict[ list(spherical_moments_dict.keys())[0] ] ))
    S = zeros( (len(lrange),len(index_domain)), dtype=complex )
    
    # Collect spheroidal information over all domain samples
    # ---
    # error('we wish to change this method so that it only handles a single domain index; related methods must be updated accordingly')
    for k in index_domain:
        
        # Collect ordered spherical moment array at this domain sample
        Y = array( [ spherical_moments_dict[l,m][k] for l in lrange ] )
        
        # Apply spherical to spheroidal to spherical moments to get spheroidal ones
        S[:,k] = dot( V, Y )
    
    # Create a dictionary of spheroidal moments
    # ---
    moments_are_float = len(S[0])==1
    this_spheroidal_moments_dict = { (l,m,(n,p)) : S[k][0] if moments_are_float else S[k] for k,l in enumerate(lrange) }
    
    #
    this_T_dict, this_V_dict = {}, {}
    for j,lj in enumerate(lrange):
        for k,lk in enumerate(lrange):
            
            #
            this_T_dict[ (lj,m), (lk,m,n,p) ] = T[k,j]
            
            #
            this_V_dict[ (lk,m,n,p), (lj,m) ] = V[j,k]
            
            
    # Update the output dictionaries with this instance's information
    # ---
    spheroidal_moments_dict.update( this_spheroidal_moments_dict )
    T_dict.update( this_T_dict )
    V_dict.update( this_V_dict )
    
    # Output
    # ---
    return spheroidal_moments_dict, T_dict, V_dict, qnmo_dict

