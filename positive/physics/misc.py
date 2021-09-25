#
from __future__ import print_function
from . import *
from positive.api import *
from positive.plotting import *
from positive.learning import *

# High level function for calculating remant mass and spin
def remnant(m1,m2,chi1,chi2,arxiv=None,verbose=False,L_vec=None):
    '''
    High level function for calculating remant mass and spin for nonprecessing BBH systems.

    Available arxiv ids are:
    * 1611.00332 by Jimenez et. al.
    * 1406.7295 by Healy et. al.

    This function automatically imposes m1,m2 conventions.

    spxll'17
    '''

    #
    if not isinstance(chi1,(float,int)):
        arxiv = '1605.01938'
        warning('spin vectors found; we will use a precessing spin formula from 1605.01938 for the final spin and a non-precessing formula from 1611.00332')

    #
    if arxiv in ('1611.00332',161100332,None):
        if verbose: alert('Using method from arxiv:1611.00332 by Jimenez et. al.')
        Mf = Mf161100332(m1,m2,chi1,chi2)
        jf = jf161100332(m1,m2,chi1,chi2)
    elif arxiv in ('1605.01938',160501938,'precessing','p'):
        Mf = Mf161100332(m1,m2,chi1[-1],chi2[-1])
        jf = jf160501938(m1,m2,chi1,chi2,L_vec=L_vec)
    else:
        if verbose:
            alert('Using method from arxiv:1406.7295 by Healy et. al.')
            warning('This method is slow [af]. Please consider using another one.')
        Mf = Mf14067295(m1,m2,chi1,chi2)
        jf = jf14067295(m1,m2,chi1,chi2)

    # Return answer
    ans = (Mf,jf)
    return ans


#####

# Convert phenom frequency domain waveform to time domain
def phenom2td( fstart, N, dt, model_data, plot=False, verbose=False, force_t=False, time_shift=None, fmax=0.5, ringdown_pad=600,window_type='exp',apply_window_n_times=1 ):
    '''
    INPUTS
    ---
    fstart,             Units: M*omega/(2*pi)
    N,                  Number of samples for output (use an NR waveform for reference!). NOTE that this input may be overwrridden by an internal check on waveform length.
    dt,                 Time step of output (use an NR waveform for reference!)
    model_data,         [Mx3] shaped numpy array in GEOMETRIC UNITS: (positive_f,amp,phase)
    plot=False,         Toggle for plotting output
    verbose=False,      Toggle for verbose
    force_t=False       Force the total time duration of the output based on inputs

    OUTPUTS
    ---
    ht,                 Waveform time series (complex)
    t,                  time values
    time_shift          Location of waveform peak
    '''
    # The idea here is to perform the formatting in a parameterized rather than mimicked way.
    '''
    NOTE that the model's phase must be well resolved in order for us to get reasonable results.
    '''

    # Setup plotting backend
    __plot__ = True if plot else False
    if __plot__:
        import matplotlib as mpl
        from mpl_toolkits.mplot3d import axes3d
        mpl.rcParams['lines.linewidth'] = 0.8
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['axes.labelsize'] = 20
        mpl.rcParams['axes.titlesize'] = 20
        from matplotlib.pyplot import plot,xlabel,ylabel,figure,xlim,ylim,axhline
        from matplotlib.pyplot import yscale,xscale,axvline,axhline,subplot
        import matplotlib.gridspec as gridspec
    #
    from scipy.fftpack import fft,fftshift,ifft,fftfreq,ifftshift
    from scipy.stats import mode
    from numpy import array,arange,zeros,ones,unwrap,histogram,zeros_like
    from numpy import argmax,angle,linspace,exp,diff,pi,floor,convolve
    from scipy.interpolate import CubicSpline as spline

    ##%% Construct the model on this domain

    # Copy input data
    model_f   = array( model_data[0] )
    model_amp = array( model_data[1] )
    model_pha = array( model_data[2] )
    
    #
    if min(model_f)<0:
        error('This function is only setup to work with Phenom models defined on f>=0. It would need to be modified to work with more general cases. ')

    # NOTE: Using the regular diff here would result in
    # unpredictable results due to round-off error

    #-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-#
    ''' Determine the index location of the desired time shift.
    The idea here is that the fd phase derivative has units of time
    and is directly proportional to the map between time and frquency '''
    #-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-#
    dmodel_pha = spline_diff(2*pi*model_f,model_pha)
    # dmodel_pha = intrp_diff(2*pi*model_f,model_pha)
    # Define mask over which to consider derivative
    mask = (abs(model_f)<0.4) & (abs(model_f)>0.01)
    # NOTE:
    # * "sum(mask)-1" -- Using the last value places the peak of
    #   the time domain waveform at the end of the vector
    # * "argmax( dmodel_pha[ mask ] )" -- Using this value
    #   places the peak of the time domain waveform just before
    #   the end of the vector
    # #%% Use last value
    # argmax_shift = sum(mask)-1
    # time_shift = dmodel_pha[ mask ][ argmax_shift ]
    #%% Use mode // histogram better than mode funcion for continuus sets
    # This method is likely the most robust
    if time_shift is None:
        hist,edges = histogram( dmodel_pha[mask],50 )
        time_shift = edges[ 1+argmax( hist ) ]
        warning('This function time shifts data by default. If applying to a collection of multipoles for which the relative timeshift is physical, then use the time_shift=0 keyword input option.')
    if time_shift==0:
        ringdown_pad=0
    # #%% Use peak of phase derivative
    # argmax_shift = argmax( dmodel_pha[ mask ] )
    # time_shift = dmodel_pha[ mask ][ argmax_shift ]

    # #
    # figure()
    # plot( model_f[mask], dmodel_pha[ mask ]  )
    # axhline( time_shift, linestyle='--' )
    # axhline( max(dmodel_pha[ mask ]), color='r', alpha=0.5 )
    # # axvline( model_f[kstart], linestyle=':' )
    
    #
    if min(model_f)>=fstart:
        error('The input frequency values for the model are greater than or equal to the desired fstart value. Please regenerate the model values with a starting frequency that is lower than the desired time domain starting frequncy of the waveform. This is needed for phenom2td to construct an appropriate frequency daomin taper. It would be ok if the model FD waveform started at zero or alsmost zero frequency. As implied, it must have only positive frequency content. ')

    #
    ringdown_pad = ringdown_pad     # Time units not index; TD padding for ringdown
    td_window_width = 3.0/fstart    # Used for determining the TD window function
    fmax = fmax                     # Used for tapering the FD ampliutde
    fstart_eff = fstart#/(pi-2)     # Effective starting frequency for taper generation


    #-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-#
    ''' -- DETERMINE WHETHER THE GIVEN N IS LARGE ENOUGH -- '''
    ''' The method below works becuase the stationary phase approximation
    can be applied from the time to frequency domain as well as from the frequency
    domain to the time domain. '''
    #-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-#
    # Estimate the total time needed for the waveform
    # Here the 4 is a safety factor -- techincally the total
    # time needed depends on the window that will be applied in the frequency domain
    # The point is that the total time should be sufficiently long to avoid the waveform
    # overlapping with itself in the time domain.
    T = 4*sum( abs( diff(dmodel_pha[(abs(model_f)<0.4) & (abs(model_f)>fstart_eff)]) ) )
    T += ringdown_pad+td_window_width
    input_T = N*dt
    if verbose:
        print('>> The total time needed for the waveform is %g'%T)
        print('>> The total time provided for the waveform is %g'%input_T)
        if force_t: print('>> The time provided for the waveform will not be adjusted according to the internal estimate becuase teh force_t=True input has been given.')
    if (input_T < T) and (not force_t):
        input_N = N
        N = int( float(N*T)/input_T )
        if verbose:
            print('>> The number of samples is being changed from %i to %i.'%(input_N,N))
    ## INPUTS: N, dt (in some form)
    # Given dt and N (double sided), Create the new frequency domain
    N = int(N)
    _f_ = fftfreq( N, dt )
    t = dt*arange(N)
    df = 1.0/(N*dt)

    # Apply the time shift
    model_pha -= 2*pi*(time_shift+ringdown_pad)*model_f

    if verbose: print('>> shift = %f'%time_shift)
    # figure()
    # plot( model_f[mask],  intrp_diff(2*pi*model_f,model_pha)[mask] )
    # axhline(0,color='k',alpha=0.5)

    '''
    Make the time domain window
    '''
    fd_k_start = find( model_f > fstart )[0]
    t_start = dmodel_pha[ fd_k_start ] - time_shift
    if t_start > 0: t_start -= (N-1)*dt
    if verbose: print('t_start = %f'%t_start)
    # Define the index end of the window; here we take use of the point that
    # dmodel_pha=0 corresponds to the end of the time vector as to corroborate
    # with the application of time_shift
    k_start = find( (t-t[-1]+ringdown_pad)>=(t_start) )[0]-1
    #
    b = k_start
    a = b - int(td_window_width/dt)
    window = maketaper( t, [a,b] )
    window *= maketaper( t, [len(t)-1,len(t)-1-int(0.5*ringdown_pad/dt)] )


    # 1st try hard windowing around fstart and fend

    ##%% Work on positive side
    f_ = _f_[ _f_ > 0 ]

    # Interpolate model over this subdomain
    amp_,pha_ = zeros_like(f_),zeros_like(f_)
    mask = (f_>=min(model_f)) & (f_<=max(model_f))
    amp_[mask] = spline( model_f,model_amp )(f_[mask])
    pha_[mask] = spline( model_f,model_pha )(f_[mask])

    # figure( figsize=2*array([6,2]) )
    # subplot(1,2,1)
    # plot( model_f, model_amp )
    # plot( f_, amp_, '--k' )
    # yscale('log'); xscale('log')
    # subplot(1,2,2)
    # plot( model_f, model_pha )
    # plot( f_, pha_, '--k' )
    # xscale('log')

    ## Work on negative side (which will have zero amplitude). We add the f<0 side for consistent ifft usage
    _f = _f_[ _f_ < 0 ]
    # Make zero
    _amp = zeros( _f.shape )
    _pha = zeros( _f.shape )

    ## Combine positive and negative sides
    _amp_ = zeros( _f_.shape )
    _pha_ = zeros( _f_.shape )
    _amp_[ _f_<0 ] = _amp; _amp_[ _f_>0 ] = amp_
    _pha_[ _f_<0 ] = _pha; _pha_[ _f_>0 ] = pha_

    # Switch FFT convention (or not)
    amp = _amp_
    pha = _pha_
    f = _f_
    # Construct complex waveform
    hf_raw = amp * exp( -1j*pha )

    # -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ #
    # Apply window to FD amplitude to squash unnecessary low frequency power
    # * Apply input window type
    # -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ #
    window = maketaper(f,[ find(f>0)[0], find(f>fstart_eff)[0] ],window_type=window_type)
    # Sharpen the effect of the window by applying it multiple times (one is default)
    hf_raw *= (window**apply_window_n_times)
    # hf_raw *= maketaper(f,[ find(f>fmax)[0], find(f>(fmax-0.1))[0] ],window_type='parzen')
    # -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ -~ #

    #
    fd_window = fft( window )

    # hf = fftshift( convolve(fftshift(fd_window),fftshift(hf_raw),mode='same')/N )
    hf = hf_raw

    #----------------------------------------------#
    # Calculate Time Domain Waveform
    #----------------------------------------------#
    ht = ifft( hf ) * df*N
    # ht *= window

    #----------------------------------------------#
    # Center waveform in time series and set peak
    # time to zero.
    #----------------------------------------------#
    # ind_shift = -argmax(abs(ht))+len(ht)/2
    # ht = ishift( ht, ind_shift )
    if verbose: print('>> The time domain waveform has a peak at index %i of %i'%(argmax(abs(ht)),len(t)))
    t -= t[ argmax(abs(ht)) ]

    if __plot__:

        figure( figsize=2*array([10,2]) )

        gs = gridspec.GridSpec(1,7)
        # figure( figsize=2*array([2.2,2]) )
        # subplot(1,2,1)
        ax1 = subplot( gs[0,0] )
        subplot(1,4,1)
        plot( abs(f), abs(hf) )
        plot( abs(f), abs(hf_raw), '--' )
        plot( abs(f), amp, ':m' )
        plot( abs(f), abs(fd_window),'k',alpha=0.3 )
        axvline( fstart, color='k', alpha=0.5, linestyle=':' )
        yscale('log'); xscale('log')
        xlim( [ fstart/10,fmax*2 ] )
        xlabel('$fM$')
        ylabel(r'$|\tilde{h}(f)|$')
        # subplot(1,2,2)
        # plot( abs(f), unwrap(angle(hf)) )
        # xscale('log')

        # figure( figsize=2*array([6,2]) )
        ax2 = subplot( gs[0,2:-1] )
        axhline( 0, color='k', linestyle='-', alpha=0.5 )
        clr = rgb(3); white = ones( (3,) )
        plot( t, ht.real, color=0.8*white )
        plot( t, ht.imag, color=0.4*white )
        plot( t,abs(ht), color=clr[0] )
        plot( t,-abs(ht), color=clr[0] )
        axvline( t[k_start], color='k', alpha=0.5, linestyle=':' )
        plot( t, window*0.9*max(ylim()),':k',alpha=0.5 )
        xlim(lim(t))
        xlabel('$t/M$')
        ylabel(r'$h(t)$')

    #
    return ht,t,time_shift


###



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




# Compute the Clebsch-Gordan coefficient by wrapping the sympy function.
def clebsh_gordan_wrapper(j1,j2,m1,m2,J,M,precision=16):

    '''
    Compute the Clebsch-Gordan coefficient by wrapping the sympy functionality.
    ---
    USAGE: clebsh_gordan_wrapper(j1,j2,m1,m2,J,M,precision=16)
    '''

    # Import usefuls
    import sympy,numpy
    from sympy.physics.quantum.cg import CG

    # Compute and store for output
    ans = sympy.N( CG(j1, m1, j2, m2, J, M).doit(), precision )

    # return answer
    return complex( numpy.array(ans).astype(numpy.complex128) )



# Function to calculate "chi_p"
def calc_chi_p(m1,X1,m2,X2,L):
    '''
    Calculate chi_p: see eqn 3.4 of https://arxiv.org/pdf/1408.1810.pdf
    '''
    #
    from numpy import dot,array
    from numpy.linalg import norm
    
    #
    if m1<m2:
        m1,m2 = [ float(k) for k in (m2,m1) ]
        X1,X2 = [ array(k) for k in (X2,X1) ]

    #
    l = L/norm(L)
    
    #
    X1_l = l * dot( l, X1 )
    X1_perp = X1 - X1_l
    
    #
    X2_l = l * dot( l, X2 )
    X2_perp = X2 - X2_l
    
    #
    A1 = 2 + (3*m2)/(2*m1)
    A2 = 2 + (3*m1)/(2*m2)
    
    #
    m1_squared = m1*m1
    m2_squared = m2*m2
    S1_perp = norm( X1_perp * m1_squared )
    S2_perp = norm( X2_perp * m2_squared )
    
    #
    B1 = A1 * S1_perp
    B2 = A2 * S2_perp
    
    #
    chip = max( B1,B2 ) / ( A1 * m1_squared )
    
    #
    return chip
    
#
def calc_chi_eff(m1,X1,m2,X2,L):
    '''
    Calculate chi_s: see eqn 2 of https://arxiv.org/pdf/1508.07253.pdf
    '''
    #
    from numpy import dot,array
    from numpy.linalg import norm

    #
    l = L/norm(L)
    
    # #
    # X1_l = l * dot( l, X1 )
    # X2_l = l * dot( l, X2 )
    # X_eff = ( X1_l*m1 + X2_l*m2 ) / (m1+m2)
    # chi_eff = dot( l, X_eff )
    
    #
    chi_eff = (m1*dot( l, X1 ) + m2*dot( l, X2 ))/(m1+m2)
    
    #
    return chi_eff

#
def Schwarzschild_tortoise(r,M):
    '''
    Calculate the Schwazschild radial tortoise coordinate: 
    '''
    #
    from numpy import log
    #

    radial_tortoise = r + 2 * M * log( r / ( 2 * M ) - 1 )

    #
    return radial_tortoise

