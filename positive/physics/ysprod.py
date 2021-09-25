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



# Function to calculate the 1D inner-product using spline interpolation.
def prod(A,B,TH,WEIGHT_FUNCTION=None,k=5):
    '''
    Function to calculate the 1D inner-product using spline interpolation. 
    ---
    NOTE that by default this function assumes that an inner-product over the spherical solid angle is desired and that the azimuthal integral can be trivially evaluated as 2*Pi as encapsulated in the defualt behavior of the WEIGHT_FUNCTION. This convention is extremely useful when working with spherical harmonic inner-products.
    ---
    londonl@mit.edu
    '''
    from numpy import sin,pi
    if WEIGHT_FUNCTION is None:
        WEIGHT_FUNCTION = 2*pi*sin(TH)
    INTGRND = A.conj()*B*WEIGHT_FUNCTION
    RE_INTGRND = INTGRND.real
    IM_INTGRND = INTGRND.imag
    TH0,TH1 = lim(TH)
    return spline(TH,RE_INTGRND,k=k).integral(TH0,TH1) + 1j*spline(TH,IM_INTGRND,k=k).integral(TH0,TH1)



# ------------------------------------------------------------------ #
# Calculate the inner-product between a spherical and spheroidal harmonic
# ------------------------------------------------------------------ #
def ysprod( jf,
            ll,
            mm,
            lmn,
            s = -2,
            N=2**9,         # Number of points in theta to use for trapezoidal integration
            theta = None,   # Pre computed theta domain
            use_nr_convention = True,
            aw=None,
            __force_eval_depreciated__=False,
            __NO_WARN__ = False,
            verbose=False):
            
    #
    if not __force_eval_depreciated__:
        error('This function has been depreciated due to its overly flexible interface which can increase the chance of human error. Please use the qnmobj.ysprod method in the qnmobj class to compute spherical spheroidal inner-products.')

    #
    from positive import slm,sYlm as ylm,warning
    from numpy import pi,linspace,trapz,sin,sqrt
    from positive import spline,lim

    #
    th = theta if not (theta is None) else linspace(0,pi,N)
    ph = 0
    # prod = lambda A,B: 2*pi * trapz( A.conj()*B*sin(th), x=th )

    # Validate the lmn input
    if len(lmn) not in (3,6):
        error('the lmn input must contain only l m and n; note that the p label is handeled here by the sign of jf')

    # Unpack 1st order mode labels
    if len(lmn)==3:
        l,m,n = lmn
        so=False
        m_eff = m
    elif len(lmn)==6:
        l,m,n,_,l2,m2,n2,_ = lmn
        so=True
        m_eff = m+m2
        if verbose: warning('Second Order mode given. An ansatz will be used for what this harmonic looks like: products of the related 1st order spheroidal functions. This ansatz could be WRONG.','ysprod')

    #
    slm_method = __slm_legacy__
    # slm_method = slm
    
    #
    if __force_eval_depreciated__:
        if not __NO_WARN__:
            warning(' The method you have called is prone to inconsistencies due to its overly flexible input structure, and the various possible conventions relevant to the user. Please use the '+blue('qnmobj')+' class with its '+blue('qnmobj.ysprod')+' method.')
    else:
        error(' The method you have called is prone to inconsistencies due to its overly flexible input structure, and the various possible conventions relevant to the user. Please use the '+blue('qnmobj')+' class with its '+blue('qnmobj.ysprod')+' method. To continue to use this function as is, input the __force_eval_depreciated__=True keyword.')
    
    # # Create a QNM OBJECT to test the old workflow
    # Mf = 1
    # qnmo = qnmobj(Mf,jf,l,m,n,p= 1,use_nr_convention=True,verbose=False)

    #
    if m_eff==mm:
        
        #
        y = ylm(s,ll,mm,th,ph)
        
        _s = slm_method(jf,l,m,n,th,ph,s=s,norm=False,__rescale__=False,use_nr_convention=use_nr_convention,aw=aw) if not so else slm_method(jf,l,m,n,th,ph,norm=False,__rescale__=False,use_nr_convention=use_nr_convention,aw=aw)*slm_method(jf,l2,m2,n2,th,ph,norm=False,__rescale__=False,use_nr_convention=use_nr_convention,aw=aw)
        
        #
        # _s = slmy(aw,l,m,th,ph,s=s)
        
        #
        ss = _s / sqrt(prod(_s,_s,th))
        
        # #
        # from matplotlib.pyplot import plot,show,gca,sca
        # from numpy import unwrap,angle
        # ax = qnmo.plot_slm()
        # sca( ax[0] )
        # plot( th, abs(ss), lw=1, ls='--', color='k' )
        # sca( ax[1] )
        # plot( th, unwrap(angle(ss)), lw=1, ls='--', color='k' )
        
        #
        ans = prod( y,ss,th ) # note that this is consistent with the matlab implementation modulo the 2*pi convention
    else:
        # print m,m_eff,mm,list(lmnp)
        ans = 0
        # raise

    return ans


# ------------------------------------------------------------------ #
# Calculate inner product of two spheroidal harmonics at a a given spin
# NOTE that this inner product does not rescale the spheroidal functions so that the spherical normalization is recovered
# ------------------------------------------------------------------ #
def ssprod(jf, z1, z2, N=2**8,verbose=False,theta=None, london=False,s=-2,aw=None, use_nr_convention=True ):

    '''
    To be used outside of slm for general calculation of inner-products. This is NOT the function slm uses to normalize the spheroidal harmonics.
    '''
    
    #
    error('Please use the qnmobj class to perform this calculation')

    #
    from positive import prod
    from numpy import linspace,pi,sqrt

    #
    def helper(x1,x2,th=None):
        if th is None: th = linspace(0,pi,len(x1))
        c1 = sqrt( prod(x1,x1,th) )
        c2 = sqrt( prod(x2,x2,th) )
        s1_ = s1/c1
        s2_ = s2/c2
        return prod(s1_,s2_,th)

    #
    if len(z1)==len(z2)==3:
        #
        l1,m1,n1 = z1
        l2,m2,n2 = z2
        #
        if m1 == m2 :
            th, phi = pi*linspace(0,1,N), 0
            s1 = __slm_legacy__( jf, l1, m1, n1, th, phi, norm=False, __rescale__=False, london=london,aw=aw, use_nr_convention=use_nr_convention,s=s )
            s2 = __slm_legacy__( jf, l2, m2, n2, th, phi, norm=False, __rescale__=False, london=london,aw=aw, use_nr_convention=use_nr_convention,s=s ) if (l2,m2,n2) != (l1,m1,n1) else s1
            #
            ans = helper(s1,s2,th)
        else:
            ans = 0
    else:
        s1,s2 = z1,z2
        if theta is None:
            error('must input theta vals when inputting vectors')
        if len(s1)!=len(s2):
            error('If 2nd and 3rd inputs are spheroidal function then they must be the ame length, and it is assumed that the span theta between 0 and pi')
        ans = helper(s1,s2,theta)

    return ans


# ------------------------------------------------------------------ #
# Calculate inner product of two spheroidal harmonics at a a given spin
# NOTE that this inner product does not rescale the spheroidal functions so that the spherical normalization is recovered
# ------------------------------------------------------------------ #
def internal_ssprod( jf, z1, z2, s=-2, verbose=False, N=2**9, london=False,aw=None ):
    '''
    To be used by slm to normalize output
    '''
    
    #
    error('Please use the qnmobj class to perform this calculation')

    #
    from numpy import linspace,trapz,array,pi,sin

    #
    l1,m1,n1 = z1
    l2,m2,n2 = z2

    #
    if m1 == m2 :
        #
        th, phi = pi*linspace(0,1,N), 0
        # # Handle optional inner product definition
        # if prod is None:
        #     prod = lambda A,B: 2*pi * trapz( A.conj()*B*sin(th), x=th )
        #
        s1 = __slm_legacy__( jf, l1, m1, n1, th, phi,s=s, norm=False, __rescale__=False, london=london,aw=aw )
        s2 = __slm_legacy__( jf, l2, m2, n2, th, phi,s=s, norm=False, __rescale__=False, london=london,aw=aw ) if (l2,m2,n2) != (l1,m1,n1) else s1
        #
        ans = prod(s1,s2,th)
    else:
        ans = 0

    #
    return ans
    



#
def slmcg_ysprod(aw, s, l, m, lmin=None, lmax=None, span=None, case=None):

    '''
    Use Clebsh-Gordan coefficients to calculate spherical-spheroidal inner-products
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
            
    #
    if span is None:
        span = 8

    # Main bits
    # ------------------------------ #
    
    # Use helper function to calculate matrx elements
    Q,vals,vecs,lrange = slmcg_helper( aw, s, l, m, lmin=lmin, lmax=lmax, span=span, case=case )

    #
    dex_map = { ll:lrange.index(ll) for ll in lrange }
    sep_consts = vals
    ysprod_array = vecs

    # Extract spherical-spheroidal inner-products of interest
    ysprod_vec = ysprod_array[ :,dex_map[l] ]
    
    #
    a = { ll:ysprod_vec[k] for k,ll in enumerate(lrange) }
    
    #
    return a


# Helper function for ysprod_matrix that acts with a single spin value
# Function to compute matrix of spheroidal to spherical harmic inner-products
def ysprod_matrix_helper(a,m,n,p,s,lrange,verbose=False,spectral=True,full_output=False, qnmo_dict=None,norm_convention=None,__suppress_warnings__=False,output_only=None):
    '''
    DESCRIPTION
    ---
    Function to compute matrix of spheroidal to spherical harmic inner-products, ( S, Y ).
    
    INPUTS
    ---
    a        Dimensionless BH spin parameter
    m        Azimuthal eigenvalue
    p        Parity number in [-1,1]. NOTE that this means that this function uses NR conventions for teh QNMs (see qnmobj.explain_conventions)
    s        Spin weight of harmonics 
    lrange   List of l values to consider. 
    norm_convention     Convention used for setting norm of harmonics. See qnmobj.__calc_slm__ for further detail.
    
    OUTPUTS
    ---
    ysmat    Matrix of spherical to spheroidal inner-product values
    '''
    
    # Import usefuls 
    from numpy import pi,zeros,ndarray,sort,alltrue,diff,zeros_like,sqrt,int64
    
    # Validate inputs 
    if not isinstance(a,float):
        error('first input must be float')
    if a<0: 
        error('BH dimensionless spin parameter must be positive')
    if not isinstance(m,(int,int64)): 
        error('m must be (int,int64)')
    if not isinstance(s,(int,int64)): 
        error('s must be (int,int64)')
    if not ( lrange is None ):
        if not isinstance(lrange,(list,tuple,ndarray)):
            error('lrange must be iterable')
        for l in lrange:
            if not isinstance(l,(int,int64)):
                error('all values of l must be int')
            if l<abs(s):
                error('all values of l must less than abs(s)=%i'%abs(s))
        if not alltrue( lrange == sort(lrange) ):
            error('input lrange must be sorted ascendingly')
        if sum(diff(lrange)) != len(lrange[:-1]):
            error('lrange must increment by exactly 1')
    if qnmo_dict is None: qnmo_dict = {}
    if output_only:
        if not isinstance(output_only,str):
            error('The output_only key value must be None or string')
    if output_only:
        full_output = True # NOTE that we will only select a SINGLE field of full_output to return 
        
        
    #
    if spectral and not (norm_convention is None):
        if not (norm_convention is 'unit'):
            warning('You hae asked for spectral calculation of inner-products AND a non-unit norm convention. The spectral method does not currently support non-unit norms, and so we will disable it. To disable this warning, use the __suppress_warnings__=True keyword option.',verbose=not __suppress_warnings__)
            # Diable use of the spectral method
            spectral=False
    
    
    # Define workflow constants 
    M = 1.0                          # BH Mass
    lrange = list(lrange)
    lmin,lmax = lim(lrange)
    numl = len(lrange)               # number of ell vals
        
    # Create list of QNM objects
    qnmo = [ qnmobj( M,a,ll,m,n,p,verbose=verbose,use_nr_convention=True,calc_slm = not spectral, harmonic_norm_convention=norm_convention, num_theta=2**9, calc_rlm=False ) for ll in lrange ]
    
    # Define dict of QNM objects for output 
    qnmo_dict.update( { (ll,m,n,p):qnmo[k] for k,ll in enumerate(lrange) } )
    
    # Pre-allocate output
    ysmat = zeros( (numl,numl), dtype=complex )
    A = zeros_like(ysmat)
    
    # NOTE that profiling implies that the Clebsh-Gordan method is indeed slightly faster
    if spectral:
        
        '''
        This method uses the spehroidal eigenvalue problem to define rows of the inner-product matrix. This is a spectral approach that solves for spheroidal
        harmonics as represented in the spherical harmonic basis. 
        '''
        
        # Define span to used in determination of lrange within slmcg_helper
        slmcg_span = max(6,int(len(lrange)))
        
        #
        for k,llk in enumerate( lrange ):

            # Calculate spherical-spheroidal mixing coefficients using matrix method
            aw = qnmo[k].aw

            # Use helper function to calculate matrx elements
            '''NOTE that we do not use the lrange key input for slmcg_helper
            becuase it causes unacceptable errors near lmax. Instead, we use the 
            default behavior, in which lrange_k adjust to be some span about llk'''
            _,vals_k,vecs_k,lrange_k = slmcg_helper(aw,s,llk,m,span=slmcg_span)
            dex_map = { ll:lrange_k.index(ll) for ll in lrange_k }
            raw_ysmat_k = vecs_k[ :,dex_map[llk] ]
            A[ k,k ] = vals_k[ dex_map[llk] ]
            # Determine the min and max l for this k. These are used to determine rows of ysmat for the k'th columns.
            lmin_k = min(lrange_k); lmax_k = max(lrange_k)
            # Create mask for wanted values in raw_ysmat_k
            start_dex_k = lrange_k.index(lmin) if lmin in lrange_k else  0
            end_dex_k   = lrange_k.index(lmax) if lmax in lrange_k else -1
            # Select wanted values 
            wanted_raw_ysmat_k = raw_ysmat_k[ start_dex_k : end_dex_k+1 ]
            # Seed ysmat with wanted values after determining the lrange mask of interest
            start_dex = lrange.index( lrange_k[start_dex_k] )
            end_dex   = lrange.index( lrange_k[end_dex_k] )
            ysmat[start_dex:end_dex+1,k] = wanted_raw_ysmat_k
            
    else:
        
        '''
        This method is a relatively staightforward computation of the integrals. It is slightly less computationally efficient for typical inputs.
        '''

        #
        for j,lj in enumerate(lrange):
            for k,lk in enumerate(lrange):

                #
                ysmat[j,k] = qnmo[k].ysprod(lj,m)
                
        # Set phases such that the diagonal is real. NOTE that this makes the output of direct integration use the same phase convention that is inherent to the spetral route above.
        # for k,lk in enumerate(lrange):
        #     ysmat[:,k] *= (  ysmat[k,k].conj() / abs(ysmat[k,k])  )
            
    #
    if full_output:
        
        '''
        Prepare quantities for full output mode
        '''
        
        # Invert ysmat to get spherical-adjoint-spheroidal inner products for this p and n subset 
        # --- 
        from numpy.linalg import inv 
        '''
        # NOTE that there is a difference of conventions 
        # * the numpy discrete inner-product space does not conjugate in its inner-products
        # * the continuous space does
        # AS A RESULT WE CONJUGATE HERE TO CONVERT TO THE CONTIUOUS SPACE CONVENTION
        '''
        adj_ysmat = inv( ysmat ).conj()
        
        #
        for j,lj in enumerate(lrange):
            adj_ysmat[:,j] *= sqrt(qnmo[j].__slm_norm_constant__).conj()
          
        # Create dictionary representation of inner-product matrix
        # ---
        adj_ysdict,ysdict = {},{}
        Adict = {}
        adj_spheroidal_vector,spheroidal_vector = {},{}
        cwmat = zeros_like(ysmat)
        for j,lj in enumerate(lrange):
            # For spheroidal eigenvalues
            Adict[ lj,m,n,p ] = A[j,j]
            # For the vector representation of the spheroidal harmonic
            spheroidal_vector[ lj,m,n,p ] = ysmat[:,j] 
            # For the vector representation of the adjoint spheroidal harmonic
            adj_spheroidal_vector[ lj,m,n,p ] = adj_ysmat[j,:] 
            #
            for k,lk in enumerate(lrange):
                # For the spherical to spheroidal inner products 
                ysdict[     (lj,m), (lk,m,n,p) ] = ysmat[j,k]
                # For the spherical to adjoint-spheroidal inner products 
                adj_ysdict[ (lj,m), (lk,m,n,p) ] = adj_ysmat[k,j]
                #
                cwmat[j,k] = qnmo[k].cw
        
        #
        foo = {} 
        
        # Matrix of spherical-spheroidal inner products
        foo['ys_matrix']      = ysmat  
        # Dictionary of spherical-spheroidal inner products
        foo['ys_dict']        = ysdict
        
        # Matrix of adjoint spherical-spheroidal inner products
        foo['adj_ys_matrix']   = adj_ysmat  
        # Dictionary of adjoint spherical-spheroidal inner products
        foo['adj_ys_dict']     = adj_ysdict
        
        # Matrix of spheroidal harmonic eigenvalues
        foo['eigval_matrix']  = A 
        # Dictionary of spheroidal harmonic eigenvalues
        foo['eigval_dict']    = Adict
        
        # Matrix of spheroidal harmonic eigenvalues
        foo['spheroidal_vector_dict']  = spheroidal_vector
        # Dictionary of spheroidal harmonic eigenvalues
        foo['adj_spheroidal_vector_dict']    = adj_spheroidal_vector
        
        # Dictionary of QNM class objects
        foo['qnmo_dict']      = qnmo_dict
        
        #
        foo['lrange']         = lrange
        foo['cw_matrix']      = cwmat
        
        #
        if output_only:
            if not output_only in foo.keys():
                error('output_only must be in %s'%cyan(str(list(foo.keys()))))
            else:
                foo = foo[output_only]
            
    #
    if full_output:
        return foo
    else:
        return ysmat

# Function to compute matrix of spheroidal to spherical harmic inner-products
def ysprod_matrix(a_values,m,n,p,s,lrange,verbose=False,spectral=True,full_output=False, qnmo_dict=None,norm_convention=None,__suppress_warnings__=False,output_only=None):
    '''
    DESCRIPTION
    ---
    Function to compute matrix of spheroidal to spherical harmic inner-products, ( S, Y ).
    
    INPUTS
    ---
    a        Dimensionless BH spin parameter
    m        Azimuthal eigenvalue
    p        Parity number in [-1,1]. NOTE that this means that this function uses NR conventions for teh QNMs (see qnmobj.explain_conventions)
    s        Spin weight of harmonics 
    lrange   List of l values to consider. 
    norm_convention     Convention used for setting norm of harmonics. See qnmobj.__calc_slm__ for further detail.
    
    OUTPUTS
    ---
    ysmat    Matrix of spherical to spheroidal inner-product values
    '''
    
    # Import usefuls 
    from numpy import array, ndarray, int64, double
    
    #
    if isinstance(a_values,(ndarray,list,tuple)):
        
        # Process all spin values 
        ans = []
        for a in a_values:
            # Process each single spin value 
            ans.append(
                ysprod_matrix_helper(a,m,n,p,s,lrange,verbose=verbose,spectral=spectral,full_output=full_output, qnmo_dict=qnmo_dict,norm_convention=norm_convention,__suppress_warnings__=__suppress_warnings__,output_only=output_only)
            )
        
    elif isinstance(a_values,(int,int64,float,double)):
        
        # Process the single input spin value 
        a = a_values 
        ans = ysprod_matrix_helper(a,m,n,p,s,lrange,verbose=verbose,spectral=spectral,full_output=full_output, qnmo_dict=qnmo_dict,norm_convention=norm_convention,__suppress_warnings__=__suppress_warnings__,output_only=output_only)
        
    else:
        
        # Error 
        error('Spin parameter must be int or float')
        
    # Return answer 
    return ans


# Calculte matrix of spherical spheroidal harmonic inner-products (sYlm|Slmn)
def __ysprod_matrix_legacy__( dimensionless_spin, lm_space, N_theta=128, s=-2 ):
    '''

    == Calculte matrix of spherical spheroidal harmonic inner-products (sYlm|Slmn) ==

    |a) = sum_j a_j |Yj) # Spherical representation with j=(L,M)
        = sum_k b_k |Sk) # Spheroidal rep with k=(l,m,n)

    -->

    a_j = (Yj|a)
        = sum_k b_k (Yj|Sk)

    * (Yj|Sk) is treated as a matrix and the rest vectors
    * (Yj|Sk) will be enforced to square:

        A. IF lm_space is of length N, THEN sigma = (Yj|Sk) will be NxN
        B. We CHOOSE k = (L,M,0) so that (Yj|Sk) is square
        C. In the point above, we explicitely disregard overtones; they may be added in a future version of this method (if the dimensionality can be sorted ...)

    USAGE
    ---
    sigma = __ysprod_matrix_legacy__( dimensionless_spin, lm_space, N_theta=128, s=-2 )


    londonl@mit.edu 2019

    '''
    
    #
    error('This function is depreciated due to the sloppy input structure of __slm_legacy__')

    # Import usefuls
    from positive import slm,sYlm,prod
    from numpy import pi,linspace,trapz,sin,sqrt,zeros_like,ones,conj
    
    
    # Validate list of l,m values
    def validate_lm_list( lm_space,s=-2 ):

        # Validate list of l,m values
        lmin = abs(s)
        msg = 'lm_space should be iterable of iterables of length 2; eg [(2,2),(3,2)]'
        if not isiterable(lm_space): error(msg)
        for dex,j in enumerate(lm_space):
            if isiterable(j):
                if len(j)!=2:
                    error(msg)
                else:
                    l,m = j
                    if (not isinstance(l,int)) or (not isinstance(m,int)):
                        error('values fo l and m must be integers')
                    else:
                        lm_space[dex] = (l,m)
            else:
                error(msg)
        msg = 'lm_space must be a ordered iterable of unique elements'
        if len(set(lm_space))!=len(lm_space):
            error(msg)

        #
        return None


    # Validate list of l,m values
    validate_lm_list(lm_space,s=s)

    #
    theta = linspace( 0, pi, N_theta )
    phi = 0

    # Define index space for spheroidal basis
    k_space = []
    for l,m in lm_space:
        k_space.append( (l,m,0) )
        
    #
    # warning('can this function be smarter by pulling columns or rows from slmcg?')

    #
    K = len(lm_space)
    ans = ones( (K,K), dtype=complex )

    #
    for j,(l,m) in enumerate(lm_space):

        #
        sYj = sYlm(s,l,m,theta,phi)
        y = sYj/sqrt(prod(sYj,sYj,theta))

        for k in range( len(k_space) ):

            #
            l_,m_,n_ = k_space[k]

            #
            if m==m_:
                
                Sk = __slm_legacy__(dimensionless_spin,l_,m_,n_,theta,phi,norm=False,london=False)
                # Sk = __slm_legacy__(None,l_,m_,n_,theta,phi,norm=False,aw = aw,london=False)
                # Sk = slmcg(aw,s,l_,m_,theta,phi)
                # Q,vals,vecs,lrange = slmcg_helper( aw, s, l_, m_ )
                # dex_map = { ll:lrange.index(ll) for ll in lrange }
                # ysprod_array = vecs
                # # Extract spherical-spheroidal inner-products of interest
                # ysprod_vec = ysprod_array[ :,dex_map[l_] ]
                # ans[j,k] = ysprod_vec[ lrange.index(l) ]
                
                s_ = Sk/sqrt(prod(Sk,Sk,theta))
                ans[j,k] = conj(prod( y,s_, theta ))
            else:
                ans[j,k] = 0
            # print (L,M),(l,m,n), ans[j,k]

    #
    return ans



#
def ysprod_sequence_helper(aw,ly,slm_sequence_dict,ylm_dict):
    '''
    Calculate spherical spheroidal inner-products using slm_sequence.
    ~ londonl@mit.edu 2020 
    '''
    
    #
    from numpy import linspace,pi,cos,exp,sort,array,zeros_like
    
    #
    theta = linspace(0,pi,1024)
    u = cos(theta)
    E = exp(aw*u)
    
    #
    a = slm_sequence_dict
    lvals = sort(a.keys())
    a_arr = array( [ a[ll] for ll in lvals ] )
    
    #
    b_arr = zeros_like(a_arr)
    for k,lj in enumerate(lvals):
        
        #
        Yp = ylm_dict[ly]
        Yj = ylm_dict[lj]
        Qpj = 2*pi*prod( Yp, E*Yj, -u, WEIGHT_FUNCTION=1 )
        
        #
        b_arr[k] = Qpj #if abs(a_arr[k])>1e-10 else 0
    
    #
    ys = sum( a_arr*b_arr )
        
    #
    return ys


#
def ysprod_sequence(aw,l,m,s=-2,sc=None,lmax=None,span=None,case=None):
    '''
    CURRENT: A wrapper for slmcg_ysprod
    OLD: Calculate a sequence of spherical-spheroidal inner-products using slm_sequence.
    ~ londonl@mit.edu 2020 
    '''
    
    #
    from numpy import linspace,pi,cos,exp,sort,array,zeros_like,sqrt
    
    #
    return slmcg_ysprod(aw, s, l, m, lmin=None, lmax=lmax, span=span,case=case)
    
    # # NOTE that london=-4 seems to have the most consistent behavior for all \ell and m
    # if sc is None:
    #     sc = sc_leaver( dtyp(aw), l, m, s, verbose=verbose,adjoint=False, london=-4)[0]
        
    # # Precompute the series precactors
    # if sequence is None:
    #     a = slm_sequence(aw,l,m,s=s,sc=sc)
    # else:
    #     a = sequence
    
    # #
    # theta = linspace(0,pi,1024)
    # u = cos(theta)
    # E = exp(aw*u)
    
    # #
    # lvals = sort(a.keys())
    # a_arr = array( [ a[ll] for ll in lvals ] )
    # ylm_dict = { k:sYlm( s,k,m,theta,0,leaver=True ) for k in lvals }
    
    # #
    # ys_dict = {k:ysprod_sequence_helper(aw,k,a,ylm_dict) for k in lvals}
    
    # #
    # kmax = None
    # for k in sort(ys_dict.keys()):
    #     if k>l: 
    #         if ys_dict[k]>ys_dict[k-1]:
    #             kmax = k-1 
    #             break     
    # # Only keep values within the stable limit of indices
    # if not (kmax is None):
    #     ys_dict = { k:ys_dict[k] if k<=kmax else 0 for k in ys_dict  }
    
    # #
    # kmin = None
    # for k in sort(ys_dict.keys())[::-1]:
    #     if k<l: 
    #         if ys_dict[k]>ys_dict[k+1]:
    #             kmin = k-2
    #             break     
    # # Only keep values within the stable limit of indices
    # if not (kmin is None):
    #     ys_dict = { k:ys_dict[k] if k>=kmin else 0 for k in ys_dict  }
    
    # #
    # lvals = sort(ys_dict.keys())
    # ys_array = array( [ ys_dict[k] for k in lvals ] )
    
    # #
    # norm_cont = sqrt( sum(ys_array * ys_array.conj()) )
    
    # #
    # ys_array /= norm_cont
    
    # #
    # ys = { lvals[k]:ys_array[k] for k in range(len(lvals)) }
        
    # #
    # return ys

