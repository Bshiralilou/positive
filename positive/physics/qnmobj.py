
#
from __future__ import print_function
from . import *
from positive.api import *
from positive.plotting import *
from positive.learning import *
from positive.physics import *
# > > > > > > > > >  Import adjacent modules  > > > > > > > > > > #
import positive
modules = list( basename(f)[:-3] for f in glob.glob(dirname(__file__)+"/*.py") if (not ('__init__.py' in f)) and (not (__file__.split('.')[0] in f)) )
for module in modules:
    exec('from .%s import *' % module)
# > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > #



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
''' Class for single QNM Objects '''
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
class qnmobj:
    
    '''
    DESCRIPTION
    ---
    Class for Kerr QNMs. Self-consistent handling of frequencies, and spheroidal harmonics under different conventions.
    
    AUTHOR
    ---
    londonl@mit.edu, pilondon2@gmail.com 2021
    '''
    
    # Initialize the object
    def __init__(this, M, a, l, m, n, p=None, s=-2, verbose=False, calc_slm=True, calc_rlm=True, use_nr_convention=True, refine=False, num_xi=2**14, num_theta=2**9, harmonic_norm_convention=None, amplitude=None, __DEVELOPMENT__=False,theta=None):

        # Import needed things
        from positive.physics import leaver
        from numpy import array,sqrt

        # ----------------------------------------- #
        #              Validate inputs              #
        # ----------------------------------------- #
        this.__validate_inputs__(M,a,l,m,n,p,s,verbose,use_nr_convention,refine,harmonic_norm_convention,amplitude,theta)
        
        # Get dimesionless QNM frequency (under the M=1 convention) and separation constant
        this.cw,this.sc = leaver( this.a,
                                  this.l,
                                  this.m,
                                  this.n,
                                  p=this.p,
                                  Mf=1.0, # NOTE that mass is applied below
                                  s=s,
                                  refine=this.__refine__,
                                  verbose=this.verbose,
                                  use_nr_convention=this.__use_nr_convention__)
                                  
        #
        this.oblateness = this.aw = this.acw = this.a * this.cw 
        
        # # NOTE that this is not used as it is slow
        # this.sc = slmcg_eigenvalue( this.aw, this.s, this.l, this.m )
        
        # Calculate the M=this.M QNM frequency
        this.CW = this.cw / this.M
        
        # Inner and outer radii
        this.rp = this.M + sqrt( this.M**2 - this.a**2 ) # Event horizon
        this.rm = this.M - sqrt( this.M**2 - this.a**2 )
        
        # Calculate the spheroidal harmonic for this QNM and store related information to the current object
        if calc_slm: this.__calc_slm__(__return__=False,theta=theta,num_theta=num_theta,norm_convention=harmonic_norm_convention)
        
        # Calculate the spheroidal harmonic for this QNM and store related information to the current object
        if calc_rlm: this.__calc_rlm__(__return__=False,num_xi=num_xi,__DEVELOPMENT__=__DEVELOPMENT__)

    # Validate inputs
    def __validate_inputs__(this,M,a,l,m,n,p,s,verbose,use_nr_convention,refine,harmonic_norm_convention,amplitude,theta):
        
        #
        from numpy import int64
        
        # Testing
        if M<0:
            error('BH mass must be positive')
        if a<0:
            error('This object uses the convention that a>0. To select the retrograde QNM branch, set p=-1 when using the numerical relativity conventions, or m --> -m for the perturbation theory conventions. See qnmo.explain_conventions for more detail. ')
        if a>M:
            error('Mass=M cannot be less than BH spin/Mass=a')
        if not isinstance(l,(int,int64)):
            error('ell must be integer')
        if not isinstance(m,(int,int64)):
            error('m must be integer')
        if not isinstance(n,(int,int64)):
            error('n must be integer')
        if use_nr_convention and (p is None):
            error('NR convention being used, but p has not been defined to be +1 for prograde modes, or -1 for retrograde ones. Please define p.')
        if use_nr_convention:
            alert(yellow('Using NR convention')+' for organizing solution space and setting the sign of the QNM freuency imaginary part.',verbose=verbose)
            if not (p in [-1,1]):
                error('p must be +1 or -1')
        else:
            alert(yellow('Not using NR convention')+' for organizing solution space and setting the sign of the QNM freuency imaginary part.',verbose=verbose)
            if not ((p is None) or (p is 0)):
                error('p is not used when NR conventions are not used; p must be None (default)')
        if abs(s) != 2:
            error('This class currently only supports |s|=2')
        if abs(m)>l:
            print('l,m = ',l,m)
            error('|m|>l and it should not be due to the structure of Teukolsk\'s angular equation')
        if abs(a)>1:
            error('Kerr parameter must be non-extremal')
        if abs(a)>(1-1e-3):
            warning('You have selected a nearly extremal spin. Please take significant care to ensure that results make sense.')
        if l<abs(s):
            error('l must be >= |s| du to the structure of Teukolsk\'s angular equation')
            
        # Assign basic class properties from inputs
        if p is None: p=0
        this.M,this.a,this.verbose,this.z,this.s = M,a,verbose,(l,m,n,p),s
        this.l,this.m,this.n,this.p = this.z
        this.amplitude = amplitude
        this.__theta__ = theta
        
        #
        this.__slm_norm_constant__ = 1.0
        this.__harmonic_norm_convention__ = harmonic_norm_convention
        this.__use_nr_convention__ = use_nr_convention
        this.__refine__ = refine

    #
    def ysprod(this,lj,mj):
        
        #
        from numpy import pi,sqrt,sin
        from positive import sYlm
        
        #
        if not ('slm' in this.__dict__):
            this.calc_slm(__return__=False)
        
        #
        yj = sYlm(this.s,lj,mj,this.__theta__,this.__phi__)
        yj = yj / sqrt( prod(yj,yj,this.__theta__,WEIGHT_FUNCTION=2*pi*sin(this.__theta__)) )
        
        #
        ys = prod( yj, this.slm, this.__theta__,WEIGHT_FUNCTION=2*pi*sin(this.__theta__)) if mj == this.m else 0
        
        #
        return ys


    #
    def __calc_aslm__(this):
        #
        error('Please use calc_adjoint_slm_subset to calculate a family of spheroidal harmonics and their adjoint duals all at once.')
        
    #
    def eval_time_domain_waveform( this, geometric_complex_amplitude, geometric_times,NDIFF=None ):
        
        '''
        Method to evaluate time domain ringdown in the form of exponential decal for the QNM object current
        '''
        
        #
        from numpy import exp
        
        #
        IW = 1j * this.CW
        
        #
        return geometric_complex_amplitude * exp( IW *  geometric_times ) * ( 1 if None==NDIFF else IW**NDIFF )

    #
    def explain_conventions(this,plot=False):
        
        '''
        This method exists to explain conventions used when referencing QNMs and their related spherodial harmonics for Kerr.
        ~pilondon2@gmail.com/londonl@mit.edu 2021
        '''
        
        #
        alert('General Explaination',header=True)
        
        #
        print( '''Hi, this is an explaination of what the NR convention is for Black Hole(BH)\nQuasiNormal Modes (QNMs). There are approximately two conventions used when\nworking with BH QNMs. They are the NR convention, and the Perturbation\nTheory (PT) convention. 
        ''' )
        
        alert('Numerical Relativity Conventions',header=True)
        
        #
        print( '''
        * QNM are defined by 4 numbers
        * the usual l,m,n, but also a number p which labels whether modes are prograde
            (p=1) or retrograde (p=-1).
        * The QNM frequencies are generally complex valued (ie complex omegas, thus the
            vairable name "cw"). The real part of the frequency, Re(cw), is the time domain 
            waveform's central frequency. The imaginary part is the time domain amplitude's
            expontntial decay rate.
        * In the NR convention, Im(cw)>0, always. This is due to a convention in how the phase
            is defined. In particular, there is no minus sign explicitly present when writing
            down phases.
        * PROGRADE QNMs have frequencies correspond to perturbations which propigate at the
            source *in the direction of* the BH spin.
        * RETROGRADE QNMs have frequencies correspond to perturbations which propigate at the
            source *against the direction of* the BH spin.


                Prograde            Retrograde
        ------------------------------------------
        m>0     Re(cw)>0             Re(cw)<0

        m<0     Re(cw)<0             Re(cw)>0

        ''' )

        alert('Perturabtion Theory Conventions',header=True)

                #
        print( '''
        * QNM are defined by 3 numbers, the usual l,m and n
        * The QNM frequencies are generally complex valued (ie complex omegas, thus the
            vairable name "cw"). The real part of the frequency, Re(cw), is the time domain 
            waveform's central frequency. The imaginary part is the time domain amplitude's
            expontntial decay rate.
        * In the PT convention, Im(cw)<0, always. This is due to a convention in how the phase
            is defined. In particular, there must be a minus sign explicitly present when writing
            down phases.
        * Positive m QNMs have frequencies correspond to perturbations which propigate at the
            source *in the direction of* the BH spin.
        * Negative m QNMs have frequencies correspond to perturbations which propigate at the
            source *against the direction of* the BH spin.
        * There are harmonics defined above and below the x-y plane. Viewing the plane from 
            below corresponds to the transformation of QNM frequencies, cw, where 
                            cw --> -cw.conj() . 
            To accomodate this, the concept of MIRROR MODES is imagined. 
        * When generally writing down radation, mirror modes must be added manually using the 
            conjugate symmetry cited above. Note that this symmetry applies to the spheroial
            harmonics also.

                Prograde            Retrograde
        ------------------------------------------
        m>0     Re(cw)>0      Must manually define "mirror mode"

        m<0     Re(cw)<0      Must manually define "mirror mode"
        ''' )
        
        alert('Final Comments',header=True)
        
        print(''' 
        One must never mix conventions.

        The practical outcomes of using one convention over the other are:

            * Inner products, such as those between spherical and spheroidal harmonics are conjugated between conventions when p=1. When p=-1, they are related by negation and conjugation. 
            * Similarly the spheroidal harmonic functions are similarly related between conventions.
            * Note that the spheroidal harmonic type functions are defined up to a phase which may be unique for each harmonic.
            * There is a factor of (-1)^l when mapping +m to -m spherical-spheroidal inner-products
        ''')
        

    #
    def __calc_rlm__(this,num_xi=2**14,plot=False,__return__=True,__DEVELOPMENT__=False,__REGULARIZE__=False):
        
        #
        from numpy import linspace,pi,mean,median,sqrt,sin,log
        
        # Adjust for conventions
        if this.__use_nr_convention__:
            internal_a  = this.a
            internal_cw = this.cw.conj()
            internal_sc = this.sc.conj()
        else:
            internal_a  = this.a
            internal_cw = this.cw
            internal_sc = this.sc
        
        # Define domain
        # x = (r-rp)/(r-rm)
        zero = 1e-4
        this.xi = linspace(zero,1-zero*100,num_xi) 
        
        # Store radius
        this.r = (-this.rp + this.rm*this.xi)*1.0/(-1 + this.xi) 
        
        # Store tortoise coordinate
        this.rstar = (-this.rp + this.rm*this.xi)*1.0/(-1 + this.xi) + ((this.a**2 + this.rm**2)*log((this.rm - this.rp)*1.0/(-1 + this.xi)) - (this.a**2 + this.rp**2)*log(((this.rm - this.rp)*this.xi)*1.0/(-1 + this.xi)))*1.0/(this.rm - this.rp)
        
        #
        if not __REGULARIZE__:
            rlm_array = rlm_helper( this.a, internal_cw, internal_sc, this.l, this.m, this.xi, this.s,geometric_M=1,london=False)
            # Test whether a spheroidal harmonic array satisfies TK's radial equation
            __rlm_test_quantity__,test_state = test_rlm(rlm_array,internal_sc,this.a,internal_cw,this.l,this.m,this.s,this.xi,verbose=this.verbose,regularized=False)
        else:
            #
            rlm_array = rlm_helper( this.a, internal_cw, internal_sc, this.l, this.m, this.xi, this.s,geometric_M=1,london=False,pre_solution=1)
            # Test whether a REGULARIZED spheroidal harmonic array satisfies TK's radial equation
            __rlm_test_quantity__,test_state = test_rlm(rlm_array,internal_sc,this.a,internal_cw,this.l,this.m,this.s,this.xi,verbose=this.verbose,regularized=True)
            
        # Adjust for conventions
        if this.__use_nr_convention__:
            rlm_array = rlm_array.conj()
            __rlm_test_quantity__  = __rlm_test_quantity__.conj()
        
        #
        if __return__:
            return this.xi, rlm_array, __rlm_test_quantity__
        else:
            this.rlm = rlm_array
            this.__rlm_test_quantity__ = __rlm_test_quantity__

    # Return the spheriodal harmonic at theta and phi for this QNM
    def __calc_slm__(this,theta=None,phi=None,num_theta=2**9,plot=False,__return__=True,norm_convention=None):
        
        #
        from numpy import linspace,pi,mean,median,sqrt,sin
        
        #
        allowed_norm_conventions = ['unit','aw','cw','cwn','cwp','cwnp','cwpn']
        if norm_convention is None:
            norm_convention = 'unit'
        if not ( norm_convention in allowed_norm_conventions ):
            error('unknown option for norm convention; must be in %s'%allowed_norm_conventions)
        
        # Define domain 
        zero = 1e-8
        this.__theta__ = linspace(0+zero,pi-zero,num_theta) if theta is None else theta
        this.__phi__   = 0 if phi==None else phi
        
        # Generate the spheroidal harmonic as an array. 
        # NOTE that slmy is generally more accurate than slm, so we use it here despite its being slightly slower
        slm_array = slmy(this.aw,this.l,this.m,this.__theta__,this.__phi__,s=this.s,sc=this.sc, test=False)
        
        # # Generate the spheroidal harmonic as an array
        # slm_array,_ = slm( this.aw, this.l, this.m, this.__theta__, this.__phi__, this.s, sc=this.sc, verbose=this.verbose, test=False )
                        
        # Normalize NOTE that this includes the factor of 2*pi from the phi integral
        # this line imposes norm convention "unit" for unit norm
        slm_array /= sqrt(  prod(slm_array,slm_array,this.__theta__,WEIGHT_FUNCTION=2*pi*sin(this.__theta__))  )
                        
        # Test whether a spheroidal harmonic array satisfies TK's angular equation
        __slm_test_quantity__,test_state = test_slm(slm_array,this.sc,this.aw,this.l,this.m,this.s,this.__theta__,verbose=this.verbose)
        
        # Initiate default norm convention
        norm_constant = 1.0 # NOTE that this is the default convention applied for norm_convention='unit'
        '''
        NOTE that we use a `1+cw` normalization constant here. This is becuase:
            * If only aw (or some homogeneous function thereof) is used, then when aw=0, the norm constant would be 0, which is nonsense
            * The rigorous view has the norm constant be, approximately (when |aw|<<1), be 1+aw*(a sum of two clecsh gordam coefficients)
        TODO: add clebsch gordan coefficients
        '''
        if norm_convention is 'aw':
            norm_constant = 1 + this.aw
        '''
        NOTE that this brnach of convention options is well behavied in the zero spin limit and thus does not need the addition of 1
        '''
        if norm_convention is 'cw':
            norm_constant = this.cw
        if norm_convention in ('cwn'):
            norm_constant = this.cw ** this.n
        if norm_convention in ('cwp'):
            norm_constant = this.cw ** (1-this.p) 
        if norm_convention in ('cwnp','cwpn'):
            norm_constant = this.cw ** (this.n+this.p)
        #
        slm_array *= sqrt( norm_constant )
        
        #
        if plot: this.plot_slm()
        
        #
        if __return__:
            return slm_array, __slm_test_quantity__
        else:
            this.slm = slm_array
            this.__slm_norm_constant__ = norm_constant
            this.__slm_test_quantity__ = __slm_test_quantity__
            
    #
    def plot_slm(this,ax=None,line_width=1,plot_scale=0.99,colors=None,label=None,show_legend=True,ls='-',show=False):
          
        #
        from matplotlib.pyplot import plot,xlabel,ylabel,figure,figaspect,subplots,yscale,gca,sca,xlim,ylim,grid,title,legend
        from numpy import unwrap,angle,pi
        
        #
        if ax is None:
            fig,ax = subplots( 1,3, figsize=plot_scale*1.5*figaspect(0.618 * 0.45), sharex=True )
            ax = ax.flatten()
            
        #
        if colors is None:
            colors = ['dodgerblue','orange','r']
            
        #
        if label is None:
            if this.p:
                label = r'$(\ell,m,n,p) = %s$'%str(this.z)
            else:
                label = r'$(\ell,m,n) = %s$'%str(this.z[:-1])
        
        #
        sca( ax[0] )
        plot( this.__theta__, abs(this.slm),lw=line_width,color=colors[0], label=label,ls=ls )
        xlim(lim(this.__theta__))
        xlabel(r'$\theta$')
        title(r'$|S_{\ell m n p}|$')
        if show_legend: legend(loc='best')
        
        #
        sca( ax[1] )
        pha = unwrap(angle(this.slm))
        plot( this.__theta__, pha, color=colors[1],lw=line_width, label=label, ls=ls )
        xlabel(r'$\theta$')
        title(r'$\arg(S_{\ell m n p})$')
        if show_legend: legend(loc='best')
        
        #
        sca( ax[2] )
        plot( this.__theta__, abs(this.__slm_test_quantity__), color=colors[2],lw=line_width, label=label, ls=ls )
        yscale('log')
        grid(True)
        xlabel(r'$\theta$')
        title(r'$\mathcal{D}_{\theta}^2 S_{\ell m n p} $')
        if show_legend: legend(loc='best')
        
        #
        if show:
            from matplotlib.pyplot import show 
            show()
        
        #
        return ax
            
            
    #
    def plot_rlm(this,ax=None,line_width=1,plot_scale=0.99,colors=None,label=None,show_legend=True,ls='-',show=False):
          
        #
        from matplotlib.pyplot import plot,xlabel,ylabel,figure,figaspect,subplots,yscale,gca,sca,xlim,ylim,grid,title,legend
        from numpy import unwrap,angle,pi,exp
        
        #
        if ax is None:
            fig,ax = subplots( 1,3, figsize=plot_scale*1.5*figaspect(0.618 * 0.45), sharex=True )
            ax = ax.flatten()
            
        #
        if colors is None:
            colors = ['dodgerblue','orange','r']
            
        #
        if label is None:
            if this.p:
                label = r'$(\ell,m,n,p) = %s$'%str(this.z)
            else:
                label = r'$(\ell,m,n) = %s$'%str(this.z[:-1])
        
        #
        sca( ax[0] )
        plot( this.xi, abs(this.rlm),lw=line_width,color=colors[0], label=label,ls=ls )
        
        # Compute factors to apply spin-corfficient (r**4 for large r or theta=pi/2) and the complex exponential of the tortoise coordinate
        r4 = this.r**4
        exp_rstar = exp( 1j * (this.cw.conj() if this.__use_nr_convention__ else this.cw) * this.rstar )
        # The effect of this factor allows one to estimate Psi4's radial dependence from Rlm
        psi4_factor = r4 * exp_rstar
        
        # Plot the psi4 scaling for reference
        plot( this.xi, abs(this.rlm/psi4_factor),lw=line_width, label=r'$\sim \psi_4^{\ell m}$',ls='--', color='k' )
        
        xlim(lim(this.xi))
        xlabel(r'$\xi$')
        title(r'$|R_{\ell m n p}|$')
        yscale('log')
        if show_legend: legend(loc='best')
        
        #
        sca( ax[1] )
        pha = unwrap(angle(this.rlm))
        plot( this.xi, pha, color=colors[1],lw=line_width, label=label, ls=ls )
        xlabel(r'$\xi$')
        title(r'$\arg(R_{\ell m n p})$')
        if show_legend: legend(loc='best')
        
        #
        sca( ax[2] )
        plot( this.xi, abs(this.__rlm_test_quantity__), color=colors[2],lw=line_width, label=label, ls=ls )
        yscale('log')
        ylim(1e-10,1e10)
        #print(ylim())
        grid(True)
        xlabel(r'$\xi$')
        title(r'$\mathcal{D}_{\xi}^2 R_{\ell m n p} $')
        if show_legend: legend(loc='best')
        
        #
        if show:
            from matplotlib.pyplot import show 
            show()
        
        #
        return ax
            



###

# Function to check whether a spheroidal harmonic array satisfies TK's angular equation
def test_slm(Slm,Alm,aw,l,m,s,theta,tol=1e-5,verbose=True):
    
    '''
    Function to test whether an input spheroidal harmonic satisfies the spheroidal differential equation
    '''
    
    # Import usefuls 
    from numpy import median
    
    # Evaluate spheroidal differential equation
    __slm_test_quantity__ = tkangular( Slm, theta, aw, m, s=s, separation_constant=Alm)
   
    # Perform the test
    test_number = median(abs(__slm_test_quantity__))
    test_state = test_number > tol
    if test_state:
        # Only print test restults if verbose
        alert(red('Check Failed: ')+'This object\'s spheroidal harmonic does not seem to solve Teukolsky\'s angular equation with zero poorly approximated by %s.'%yellow('%1.2e'%test_number),verbose=verbose)
        # Always show warning
        warning('There may be a bug: the calculated spheroidal harmonic does not appear to solve Teukolsky\'s angular equation. The user should decide whether zero is poorly approximated by %s.'%red('%1.2e'%test_number))
    else:
        # Only print test restults if verbose
        alert(blue('Check Passed: ')+'This object\'s spheroidal harmonic solves Teukolsky\'s angular equation with zero approximated by %s.'%magenta('%1.2e'%test_number),verbose=verbose)
        
    #
    return __slm_test_quantity__,test_state


# Function to check whether a radial spheroidal harmonic array satisfies TK's radial equation
def test_rlm(Rlm,Alm,a,cw,l,m,s,x,tol=1e-5,verbose=True,regularized=False):
    
    '''
    Function to test whether an input RADIAL spheroidal harmonic satisfies the RADIAL spheroidal differential equation
    '''
    
    # Import usefuls 
    from numpy import median
    from matplotlib.pyplot import plot,show,xlabel,ylabel,figure,yscale
    
    #
    M = 1 # Mass convention of inputs
    leaver_a,leaver_cw,leaver_M = rlm_change_convention(a,cw,M)
    
    #
    pre_solution,_,_ = rlm_leading_order_scale(x,leaver_M,leaver_a,leaver_cw,m,s)
    
    # Evaluate spheroidal differential equation
    if not regularized:
        __rlm_test_quantity__1 = tkradial( Rlm, x,leaver_M, leaver_a, leaver_cw, m, s,Alm)/pre_solution
        __rlm_test_quantity__2 = tkradialr( Rlm, x,leaver_M, leaver_a, leaver_cw, m, s,Alm)/pre_solution
        __rlm_test_quantity__3 = tkradial_r( Rlm, x,leaver_M, leaver_a, leaver_cw, m, s,Alm, convert_x=True)/pre_solution
        
        #
        __rlm_test_quantity__ = __rlm_test_quantity__1
        
        # #
        # figure()
        # plot( x, abs(__rlm_test_quantity__1), ls='-' )
        # plot( x, abs(__rlm_test_quantity__2), ls='--' )
        # plot( x, abs(__rlm_test_quantity__3), ls='-', lw=4, alpha=0.4 )
        # yscale('log')
        # show()
        
    else:
        __rlm_test_quantity__ = tkradial_regularized( Rlm, x,leaver_M, leaver_a, leaver_cw, m, s,Alm)
   
    # Perform the test
    test_number = median(abs(__rlm_test_quantity__))
    test_state = test_number > tol
    if test_state:
        # Only print test restults if verbose
        alert(red('Check Failed: ')+'This object\'s radial harmonic does not seem to solve Teukolsky\'s radial equation with zero poorly approximated by %s.'%yellow('%1.2e'%test_number),verbose=verbose)
        # Always show warning
        warning('There may be a bug: the calculated radial harmonic does not appear to solve Teukolsky\'s radial equation. The user should decide whether zero is poorly approximated by %s.'%red('%1.2e'%test_number))
    else:
        # Only print test restults if verbose
        alert(blue('Check Passed: ')+'This object\'s radial harmonic solves Teukolsky\'s radial equation with zero approximated by %s.'%magenta('%1.2e'%test_number),verbose=verbose)
        
    #
    return __rlm_test_quantity__,test_state

