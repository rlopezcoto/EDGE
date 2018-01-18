#######################################################
#                                                     #
#         Calculation of electron spectra,            #
#         gamma-ray spectra and electrons             #
#         flux at the Earth for different             #
#              initial parameters                     #
#                                                     #
#######################################################
#                                                     #
#     Ruben Lopez-Coto, MPIK, rlopez@mpi-hd.mpg.de    #
#     Joachim Hahn, MPIK, joachim.hahn@mpi-hd.mpg.de  #
#                                                     #
#######################################################

import os, sys
from math import exp
import math
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.                                                                                                                                                           
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('/Users/rubenlopez/Code/GAMERA-master/lib'))
import gappa as gp
import argparse
import astropy.units as u
from gammapy.astro.population import make_base_catalog_galactic
from scipy.special import erfc
import matplotlib.ticker as mtick
from matplotlib.ticker import OldScalarFormatter, ScalarFormatter

global fp
fu = gp.Utils()
fr = gp.Radiation()
fp = gp.Particles()
fa = gp.Astro()
deg_to_rad = gp.pi / 180.

os.system("mkdir -p Figures")
os.system("mkdir -p Results")

global opts
p = argparse.ArgumentParser(description="Calculate the IC electron spectrum of sources")
p.add_argument("-n", "--name", dest="Name", type=str, default="Source",
               help="Name of the source.")
p.add_argument("-f", "--file", dest="File", type=str, default="Data/GemingaProfile.dat",
               help="File containing the angular profile of the source.")
p.add_argument("-al", "--alpha", dest="ALPHA", type=float, default=2.2,
               help="Spectral index of the injection spectrum")
p.add_argument("-d", "--distance", dest="DIST", type=float, default=0.25,
               help="Distance to the source [kpc]")
p.add_argument("-del", "--delta", dest="DELTA", type=float, default=0.33,
               help="Diffusion index")
p.add_argument("-a", "--age", dest="AGE", type=float, default=3.42e5,
               help="Characteristic age of the source [yr]")
p.add_argument("-emax", "--emax", dest="EMAX", type=float, default=500.,
               help="EMAX of accelerated electrons [TeV]") # You give it in TeV but it is transformed to erg
p.add_argument("-emin", "--emin", dest="EMIN", type=float, default=0.001,
               help="EMIN of accelerated electrons [TeV]") # You give it in TeV but it is transformed to erg
p.add_argument("-m", "--mu", dest="MU", type=float, default=0.5,
               help="Fraction of energy that goes into electrons") 
p.add_argument("-d0", "--d0", dest="D0", type=float, default=4.e27,
               help="Diffusion coefficient [cm^-2]") 
p.add_argument("-s", "--s", dest="SIZE", type=float, default=5.,
               help="Size of the source given by the diffusion coefficient") 
p.add_argument("-kn", "--kn", dest="KN", action='store_true', default=False,
               help="Flag to activate or deactivate the KN option to calculate IC losses") 
p.add_argument("-edens", "--edens", dest="TOT_E_DENS", type=float, default=1.06,
               help="Total energy density. For Thomson losses. [eV/cm^3]") 
p.add_argument("-bfield", "--bfield", dest="BCONT", type=float, default=3.,
               help="Magnetic field [G]") 
p.add_argument("-edot", "--edot", dest="EDOT", type=float, default=3.2e34,
               help="Spin-down power [erg/s]") 
p.add_argument("-brind", "--brind", dest="BRIND", type=float, default=3.,
               help="Breaking index") 
p.add_argument("-tau0", "--tau0", dest="T0", type=float, default=1.2e4,
               help="Initial spin-down timescale [yr]") 
p.add_argument("-p", "--p", dest="P", type=float, default=237.,
               help="Pulsar Period [ms]") 
p.add_argument("-p0", "--p0", dest="P0", type=float, default=40.5,
               help="Initial pulsar period [ms]") 
p.add_argument("-tsupr", "--tsupr", dest="TIMESUPR", type=float, default=0.,
               help="Suppression time for the luminosity [yr]") 

# Running-related inputs
p.add_argument("-birth_period", "--birth_period", dest="BIRTH_PERIOD", action='store_true', default=False,
               help="Flag to calculate initial spin-down characteristic age from birth period") 
p.add_argument("-all_pulsar", "--all_pulsar", dest="ALL_PULSAR", action='store_true', default=False,
               help="Flag to calculate the contribution at the Earth of all pulsars") 
p.add_argument("-only_flux_earth", "--only_flux_earth", dest="ONLY_FLUX_EARTH", action='store_true', default=False,
               help="Only calculate the flux at the Earth and exit") 
p.add_argument("-eps", "--eps", dest="FIG_EPS", action='store_true', default=False,
               help="Save Figures in EPS format") 

# Binning inputs
p.add_argument("-eb", "--ebins", dest="EBINS", type=float, default=100,
               help="Energy bins of the E_R_Array") 
p.add_argument("-rb", "--rbins", dest="RBINS", type=float, default=400,
               help="Radial bins of the E_R_Array") 


# Source-related inputs
p.add_argument("-norm", "--norm", dest="NORM", type=float, default=12.1e-15,
               help="Normalization of the source's flux at a given pivot E") 
p.add_argument("-norm_err", "--norm_err", dest="NORM_ERR", type=float, default=2.5e-15,
               help="Error on the normalization of the source's flux at a given pivot E") 
p.add_argument("-pivot", "--pivot", dest="PIVOT_E", type=float, default=20.,
               help="Pivot energy for the normalization of the flux") 
p.add_argument("-gamma", "--gamma", dest="GAMMA", type=float, default=2.40,
               help="Spectral index of the gamma-ray spectrum") 
p.add_argument("-gamma_err", "--gamma_err", dest="GAMMA_ERR", type=float, default=0.09,
               help="Error on the spectral index of the gamma-ray spectrum") 



# Input parameters
args = p.parse_args()
opts = args

AGE       = opts.AGE                    # 3.e5               # yr      Real Age of the pulsar
TC        = opts.AGE                    # 3.e5               # yr      Characteristic age of the pulsar
DIST      = opts.DIST                   # 0.25               # kpc     Distance
ALPHA     = opts.ALPHA                  # 2.0                #         Spectral index of the injection function
DELTA     = opts.DELTA                  # 0.4                #         Diffusion index              
EMAX      = opts.EMAX * gp.TeV_to_erg   # 500                # erg 
EMIN      = opts.EMIN * gp.TeV_to_erg   # 0.001              # erg 
MU        = opts.MU                     # 0.5                #         Fraction of energy that goes into electrons
D0        = opts.D0                     # 4.e27              #         Diffusion coefficient
SIZE      = opts.SIZE                   # 4.7                # deg
KN        = opts.KN                     # False
TOT_E_DENS= opts.TOT_E_DENS             # 1.06               # eV/cm^3      
BCONT     = opts.BCONT* 1.e-6           # 3.e-6              # muGauss      
EDOT      = opts.EDOT                   # 3.2e34             # erg/s      
BRIND     = opts.BRIND                  # 3
T0        = opts.T0                     # 1.2e4              # yr
P         = opts.P                      # 20.                # ms
P0        = opts.P0                     # 20.                # ms
TIMESUPR  = opts.TIMESUPR               # 0.                 # yr

EBINS     = opts.EBINS                  # 100          
RBINS     = opts.RBINS                  # 400          

BIRTH_PERIOD    = opts.BIRTH_PERIOD     # False
ALL_PULSAR      = opts.ALL_PULSAR       # False
ONLY_FLUX_EARTH = opts.ONLY_FLUX_EARTH  # False
FIG_EPS         = opts.FIG_EPS          # False

NORM      = opts.NORM                   # 12.1e-15           # TeV^-1 cm^-2 s^-1
NORM_ERR  = opts.NORM_ERR               # 2.5e-15            # TeV^-1 cm^-2 s^-1
PIVOT_E   = opts.PIVOT_E                # 20                 # TeV
GAMMA     = opts.GAMMA                  # 2.40           
GAMMA_ERR = opts.GAMMA_ERR              # 0.09          


electron_mass=0.5e-6                       # TeV/c^2
c=3.e10                                    # cm/s
#Edot=3.2e34                                # erg/s
#nu=4.218                                   # Hz      Frequency
#nu_dot=1.952e-13                           # Hz/s    Frequency derivate
#nu_dot_dot_old=1.49e-25                    # Hz/s^2  Frequency second derivate
#nu_0=nu+nu_dot*t+nu_dot_dot_old*pow(t,2) 
#nu_0=nu+nu_dot*t                           # Hz      Initial frequency

l0    =  5.e-20                            # s^-1
E_star=3.e-3 * gp.TeV_to_erg               # erg
I = 1e45                                   # g cm^2  Pulsar moment of inertia

AGEBURST = AGE                             # s
#AGECONT = 2*TC/(BRIND-1.0)-T0              # s
#AGECONT = AGEBURST - TIMEOFFSET            # s
ETA = .1
el_charge=4.80320427e-10                   # StatC

TMIN = 1.                                  # s

DENS = 1e-4
TIR = 20.                                  # K
TOPT = 5e3                                 # K
WIR = 0.3                                  # erg/cm^3
WOPT = 0.3                                 # erg/cm^3
#BCONT = 3.e-6                             # G Magnetic field for continuous emission
BBURST = 3.e-6                             # G Magnetic field for burst emission 
ESN = 2.5e48                               # erg




# Luminosity evolution of a pulsar (simply spin-down)
def CalculateLuminosity(bins,age):
    T = np.logspace(math.log10(TMIN),math.log10(2.*age),bins) # Array with the time 

    if (BIRTH_PERIOD):     
        tau0 = 2*TC/(BRIND-1.)-age
        Ps = P*1e-3
        Pdot = Ps/(2*TC*gp.yr_to_sec)
        print ("Pdot (ms)",Pdot)
        edot = 4*math.pi**2*I*Pdot/(Ps**3)
        lum0= edot/pow(1+age/tau0,-1.*(BRIND+1.)/(BRIND-1.))# erg/s
        lum = MU*lum0*(1.+T/tau0)**(-1.*(BRIND+1.)/(BRIND-1.))         # Array with the luminosity for each of the times
    else:
        edot=EDOT
        tau0 = T0
        lum0= EDOT/pow(1+age/T0,-1.*(BRIND+1.)/(BRIND-1.))# erg/s
        lum = MU*lum0*(1.+T/T0)**(-1.*(BRIND+1.)/(BRIND-1.))         # Array with the luminosity for each of the times
       
    print ("Age ",age)
    print ("Characteristic age ",TC)
    print ("TAU0",tau0)
    print ("Edot",edot)
    print ("LUM0",lum0)
    
    if TIMESUPR != 0.:
        t_index = np.max(np.where(T < TIMESUPR)[0])
        lumBurst = []
        lum_ones=np.ones(np.size(T[0:+t_index]))
        lumoffset=np.concatenate((lum_ones,lum[t_index+0:]))
        lumCont = np.vstack((T, lumoffset)).T
    else:
        lumCont = np.vstack((T, lum)).T    # We stack both arrays, having two columns, the first one for the time and the second for the corresponding luminosity
        lumBurst = []

    return np.log10(lumBurst),np.log10(lumCont),lum0,tau0,age,edot

# Find the real age of the pulsar (t in eq5 from Gansler&Slane 2006)
def FindAge():
    if (BIRTH_PERIOD):
        print ("Birth period - Period [ms] : ",P0, P)
        age = TC*(2/(BRIND-1.))*(1-math.pow(P0/P,(BRIND-1.)))
    else:
        age = 2*TC/(BRIND-1.0)-T0
        #age = TC
    return age

# Diffusion coefficient at energy e (in erg)                                                                                                                                                               
def Diffusion(e):
    return D0 * math.pow(1. + e/E_star, DELTA)

def CalculateEnergyTrajectory(fp):
    e = EMAX
    E = []
    T = []
    LossRates = []
    LossRatesInverse = []
    DiffIntegrand = []
    DiffIntegrandInt = []
    diff_int = 0.
    t = 0.
    while e > EMIN:                       # loop from EMAX to EMIN
        if (KN):
            lr = fp.EnergyLossRate(e)         
            #print "Calculating losses using the KN formula"
        else:
            gamma=e / gp.m_e                                                                                                                                                 
            lr = (TOT_E_DENS * gp.TeV_to_erg * 1.e-12) * (4./3 * gp.sigma_T * gp.c_speed) * pow(gamma,2)   # Loss rate for 1 eV/cm^3 energy density 
            # TOT_E_DENS in eV/cm^3 -> we transform it to erg/cm^3                                                                                                                                                                         
            # Thomson energy losses: 4./3 * sigma_t * c                                                                                                                                                                                   
            # (TOT_E_DENS * gp.TeV_to_erg * 1.e-12) * (4./3 * gp.sigma_T * gp.c_speed) = l0 * m_e[erg] = 4./3 * gp.sigma_T * gp.c_speed/m_e[eV] * m_e[erg] 
            #print "Calculating losses using the Thomson formula"
            #print "(TOT_E_DENS * gp.TeV_to_erg * 1.e-12) * (4./3 * gp.sigma_T * gp.c_speed)", (TOT_E_DENS * gp.TeV_to_erg * 1.e-12) * (4./3 * gp.sigma_T * gp.c_speed)
            #print "lr,e,gamma",lr,e,gamma  
        dt = 1.e-3 * e / lr;              # time increase
        e -= dt * lr;                     # we decrease the energy in steps of DeltaE=dt*lr
        t += dt / gp.yr_to_sec            # and increase the time in steps of dt 
        D = Diffusion(e)   
        diff_int = diff_int + dt * lr * D / lr  # This is Delta E * f(E), we are integrating in E the expression lambda = int(D(E)/E_dot)
        T.append(t)
        E.append(e)
        DiffIntegrandInt.append(diff_int)

    etraj = np.log10(np.array(zip(T,E)))                            # Energy trajectory (array of [(time0,energy0),(time1,energy1),...]). We zip in (T,E)
    etrajinverse = np.log10(np.flipud(np.array(zip(E,T))))          # Energy trajectory inverted (last element, corresponding to the minimum energy, is now the first). We zip in (E,T)
    lamb = np.log10(np.flipud(np.array(zip(E,DiffIntegrandInt))))   # Lambda (minimum energy goes first)

    return etraj,etrajinverse,lamb


def Create_E_R_ArrayOfElectrons(rbins,ebins):
    r = np.logspace(-3.,math.log10(1.e3*DIST*20.),rbins)      # Array of distances to the pulsar [pc]
    e = np.logspace(math.log10(EMIN),math.log10(EMAX),ebins)  # Array of energies [erg]

    Age=FindAge()

    twoDarray = []
    twoDarrayPLOT = []
    thr = 1e-30
    #fig = plt.figure()
    halfwidth = []   # The point at which the density of electrons has gone down to half its maximum value for a given energy
    for ee in e:                                     # For a given energy 
        print ("Energy: %.4f [TeV]" % (ee/gp.TeV_to_erg))
        line = []
        lineplot = []
        linebplot = []
        vlamb,DT,E0,E = FillLambdaVector(ee,1000,Age)   # We fill the mean free path vector for each of the energies
        # vlamb is a vector of the form [(Energy,lambda_integral(Enow=ee)-lambda_integral(Energy))], where ee is varying from EMIN to EMAX and Energy from ee to EMAX 
        vzero = 0.
        fill_halfwidth = True
        for rr in r:
            scont = Spectrum(ee,vlamb,rr,DT,E0,Age)   # Value in 1/(erg*cm^3) of the differential energy spectrum
#            srl = SpectrumRectilinear(ee,rr)                                                                                                                                                                                                                                 
            if len(LUMBURST):
                sburst = SpectrumBurst(ee,rr)
            else:
                sburst = 0.
#            print scont,srl,sburst                                                                                                                                                                                                                                           
            v = scont + sburst# + srl

            #loop to find the halfwidth of the distribution
            if vzero == 0.:
                vzero = v
            if vzero != 0. and fill_halfwidth == True and v < 0.5 * vzero:
                fill_halfwidth = False
                halfwidth.append([ee,rr])
            vplot = ee*ee*v    # In the plot we represent E^2 dN/dE [erg/cm^3]
            if v < thr:
                v = thr
                vplot = thr
            line.append(v)                # 1-D Array of densities for each of the different radii=rr and for a given energy ee [1/(erg*cm^3)] 
            lineplot.append(vplot)        # The same, multiplied by E^2, for plotting [erg/cm^3] 
        twoDarray.append(np.array(line))  # 2-D Array containing all the 1-D Arrays previously mentioned, for each of the energies ee [1/(erg*cm^3)] 
        twoDarrayPLOT.append(np.array(lineplot))

    halfwidth = np.array(halfwidth)
    twoDarray = np.array(twoDarray)
    twoDarrayPLOT = np.array(twoDarrayPLOT)
    #fig.savefig("Figures/lambdas_"+tag+".png")
    return twoDarray,twoDarrayPLOT,halfwidth,e,r

# fill vector of LAMBDA vs lower energy bound of electrons                                                                                                                                                                                                                
def FillLambdaVector(Enow,bins,Age):
    DT = math.pow(10.,np.interp(math.log10(Enow),ETRAJCONTINVERSE[:,0],ETRAJCONTINVERSE[:,1]))  # We interpolate between the first element (that is an array) of ETRAJCONTINVERSE (x=energy) and the second (f(x)=time) to obtain the interpolated time for a given energy   
    if DT < Age:  # If the interpolated time is smaller than the age, we can consider the initial energy E0=EMAX, otherwise we would be on curve 3 of the notes and the maximum energy would not be EMAX but the one calculated in the next step 
        E0 = EMAX
    else:
        E0 = math.pow(10.,np.interp(math.log10(DT-Age),ETRAJCONT[:,0],ETRAJCONT[:,1]))  # We interpolate between the first element (that is an array) of ETRAJCONT (x=time) and the second (f(x)=energy) to obtain the interpolated energy for a given time
    E = np.logspace(math.log10(Enow),math.log10(E0),bins)
    lamb = []
    for e in E[1:]:
        # LAMBCONT is the integral over the energy of dE D(E)/EDOT(E), from EMIN to EMAX
        # We interpolate between the first time element of LAMBCONT (x=energy) and the second (f(x)=integral(dE D(E)/EDOT(E)))
        v = math.pow(10.,np.interp(math.log10(Enow),LAMBCONT[:,0],LAMBCONT[:,1])) - math.pow(10.,np.interp(math.log10(e),LAMBCONT[:,0],LAMBCONT[:,1])) 
        # We are interested on int_E'^Enow{ dE D(E)/EDOT(E) }, therefore we need to break the integral:
        # int_E'^Enow{  } = int_Emax^Enow{  } - int_Emax^E'{  } = lambda(Enow) - lambda(E') 
        # We subtract from the integral for Enow the integral for every energy e and fill a vector with this subtraction
        lamb.append([e,v])

    lamb = np.log10(np.array(lamb))
    return lamb,DT,E0,lamb[:,0]


# main function to calculate the differential number (1/(erg*cm^3)) of electrons at                                                                                                                     
# energy e and radius R from the (point-) source in the *continuous* scenario.                                                                                                                     
def Spectrum(e,vlamb,R,DT,E0,Age):
    tmin = acc_time(E0,BCONT) # minimum acceleration time needed to accelerate the particle to that energy
    if DT <= tmin:
        return 0.
    R = R * gp.pc_to_cm
    spec = []
    if ALPHA == 2.:
        norm = math.log(EMAX/EMIN)
    else:
        norm = 1. / (ALPHA-2.) * (math.pow(EMIN, -ALPHA + 2.) - math.pow(EMAX, -ALPHA + 2.))  # Normalization of the electron spectrum

    vq = np.array(zip(LUMCONT[:,0],np.log10(10.**LUMCONT[:,1] / norm)))                       # Array with Time and luminosity/normalization
    T = np.logspace(math.log10(max(1e-3,Age-DT)),math.log10(Age-tmin),2000)           # Array with Time in logarithmic bins
    T2  = T - (Age - tmin - DT)
    e0 = 10.**np.interp(np.log10(T2),ETRAJCONT[:,0],ETRAJCONT[:,1])
    lamb = 10.**np.interp(np.log10(e0),vlamb[:,0],vlamb[:,1])
    Q = 10.**np.interp(np.log10(T),vq[:,0],vq[:,1])
    #print "Q",Q
    val = Q * e0 ** (-ALPHA) * e0*e0  * np.exp(-R*R/(4.*lamb)) /( e*e * (4.*gp.pi*lamb)**1.5 )
    np.place(val, val!=val, [0.])
    val = fu.Integrate(zip(T*gp.yr_to_sec,val),T[0]*gp.yr_to_sec,T[len(T)-1]*gp.yr_to_sec)    # Differential number of electrons for an energy e and at radius R [1/(erg * cm^3)]
    return val

# acceleration time for particles of energy e. Used to determine starting time
# of injection.
def acc_time(energy,b):
    momentum = ( energy - gp.m_e) / c
    gyrorad = momentum * c / (el_charge * b)
    tacc =  gyrorad / c / ETA
    return tacc / gp.yr_to_sec

def InitialiseGappa(fp,fr,b,age):

    fr.AddThermalTargetPhotons(2.7,0.26*gp.eV_to_erg)   #CMB
    fr.AddThermalTargetPhotons(TIR,WIR*gp.eV_to_erg)   #IR
    fr.AddThermalTargetPhotons(TOPT,WOPT*gp.eV_to_erg) #OPT
    fr.CreateICLossLookup()
    fr.SetBField(b)
    fr.SetAmbientDensity(DENS)
    fr.SetDistance(0.) # This will calculate the luminosity (which is what we want for the LOS integral)                                                                                                                                                                      

    fp.SetBField(b)
    fp.SetICLossLookup(fr.GetICLossLookup())
    fp.SetAmbientDensity(DENS)
    fp.SetAge(age)
    return fp

# Calculate electron column densities for every spectral energy bin along the
# l, b direction (although this model is radial symmetric,
# so only one angle is required...).
def LineOfSightIntegration(l,b,twoDarray,e,r,rbins):
    # twoDarray contains the density of photons in bins of r and e [1/(erg*cm^3)] 
    # l is the vertical angle
    # b?
    # e is the energy [erg]
    # r is an array with distances from the Earth? [pc] Bug?
    rvals = np.logspace(-6.,math.log10(DIST),rbins)
    # make r-steps so that they are very fine at the source [kpc]
    rvals = np.concatenate(((DIST - rvals)[::-1],rvals + DIST)) 
    #rvals = np.linspace(0.,2.*DIST,rbins) 
    vals = []
    los = line_of_sight(l,b,rvals,fa) # Array with xyz positions w.r.t. the Earth for all the elements with angle < l
    integrand = []
    for xyz in los:
        x = xyz[0]
        y = xyz[1]
        z = xyz[2]
        rr = math.sqrt( x * x + y * y + z * z) * 1000. # *1000 to convert it into pc
        r_index = np.where(r > rr)[0][0] # It returns the index of the first element where the condition is fulfilled
        integrandE = []
        for i in xrange(len(e)):
            val = twoDarray.T[r_index][i] # we add for all the energies the twoDarray element with index r_index (Remember, twoDarray [1/(erg*cm^3)])
            integrandE.append(val)        # For every energy, we add a value to the integrandE array, with the density corresponding to the distance r[r_index]
        integrand.append(integrandE) 
        # Array containing, for each xyz value
        # in the line of sight from the Earth, 
        # the integrandE of the densities for 
        # the distance corresponding to r[r_index] for all the energies

    integrand = np.array(integrand).T
    for integr in integrand:
        vals.append(fu.Integrate(zip(rvals*gp.kpc_to_cm,integr),rvals[0]*gp.kpc_to_cm,rvals[len(rvals)-1]*gp.kpc_to_cm))
        # Integrate integr * rvals (rvals is in kpc)
        # from rvals[0]
        # to rvals[len(rvals)-1]
    return vals # Units [1/(erg * cm^2)]


def LineOfSightVolumeIntegration(l,b,twoDarray,e,r,rbins):
    # twoDarray contains the density of photons in bins of r and e [1/(erg*cm^3)] 
    # l is the vertical angle
    # b?
    # e is the energy [erg]
    # r is an array with distances from the pulsar [pc]
    rvals = np.logspace(-6.,math.log10(DIST),rbins)
    # make r-steps so that they are very fine at the source [kpc]
    rvals = np.concatenate(((DIST - rvals)[::-1],rvals + DIST)) 
    #rvals = np.linspace(0.,2.*DIST,rbins) 
    vals = []
    los = line_of_sight(l,b,rvals,fa) # Array with xyz positions w.r.t. the Earth for all the elements with angle < l
    integrand = []
    for xyz in los:
        x = xyz[0]
        y = xyz[1]
        z = xyz[2]
        rr = math.sqrt( x * x + y * y + z * z) * 1000. # *1000 to convert it into pc
        r_index = np.where(r > rr)[0][0] # It returns the index of the first element where the condition is fulfilled
        integrandE = []
        for i in xrange(len(e)):
            val = twoDarray.T[r_index][i] # we add for all the energies the twoDarray element with index r_index (Remember, twoDarray [1/(erg*cm^3)])
            integrandE.append(val)        # For every energy, we add a value to the integrandE array, with the density corresponding to the distance r[r_index]
        integrand.append(integrandE) 
        # Array containing, for each xyz value
        # in the line of sight from the Earth, 
        # the integrandE of the densities for 
        # the distance corresponding to r[r_index] for all the energies

    integrand = np.array(integrand).T
    for integr in integrand:
        vals.append(fu.Integrate(zip(rvals*gp.kpc_to_cm,integr*rvals*rvals*math.pow(gp.kpc_to_cm,2)),rvals[0]*gp.kpc_to_cm,rvals[len(rvals)-1]*gp.kpc_to_cm))
        # Integrate (integr*rvals*rvals) * rvals (rvals is in kpc)
        # from rvals[0]
        # to rvals[len(rvals)-1]
    return vals # Units [1/(erg)]


# creates an array of x,y,z values along a line of sight in the l,b direction                                                                              
def line_of_sight(l,b,rvals,fa):
    xyz_obs = [DIST, 0. ,0. ]
    los = []
    for r in rvals:
        xyz = fa.GetCartesian(r,l,b,xyz_obs)
        # It gives the xyz position of a point w.r.t. the Earth. It is a vector with 3 components [0]=x,[1]=y,[2]=z
        los.append(xyz)

    return np.array(los)

# Calculate the contribution of an homogeneus distributions of
# pulsars in the galaxy
def Homogeneus_distribution_pulsars(age,SN_rate):
    n_sources = age * SN_rate
    table = make_base_catalog_galactic(n_sources=n_sources,
                                       rad_dis='L06',
                                       vel_dis='F06B',
                                       max_age=max_age,
                                       spiralarms=True)
    return table

# ********** HOMOGENEOUS PULSAR CONTRIBUTION ******** 
def Flux_Earth_all_pulsars(E):
    max_age = 1e7 * u.yr
    SN_rate = 2. / (100. * u.yr)
    pulsar_distribution = Homogeneus_distribution_pulsars(max_age,SN_rate);

    x_pc = np.array(pulsar_distribution[6][:]) # in kpc
    y_pc = np.array(pulsar_distribution[7][:]) # in kpc
    age  = np.array(pulsar_distribution[0][:]) * gp.yr_to_sec  # in s

    x_Earth = x_pc - 8.3 # in kpc
    y_Earth = y_pc - 0   # in kpc

    d = np.sqrt(x_Earth * x_Earth + y_Earth * y_Earth) * gp.kpc_to_cm

    # Steady flux
    # Eq 21 Atoyan et al. 1995
    Q0 = 5.e32       # 1/(erg * s)
    f_st_int = []

    for e in E:
        D = Diffusion(e) # cm^2/s
        t_gamma = []
        for t in age:
            if (t < gp.m_e/(l0 * e)):
                t_gamma.append(t)
            else:
                t_gamma.append(gp.m_e/(l0 * e))

        t_gamma = np.array(t_gamma)
        f_st = Q0 * e**-2.4 / (4*gp.pi * D * d) * erfc(d/(2 * np.sqrt(D * t_gamma))) 

        #print "f_st",f_st
        #print "t_gamma",t_gamma
        #print "d",d
        #print "D",D

        #print "Q0 * e**-2.4",Q0 * e**-2.4
        #print "(4*gp.pi * D * d)", (4*gp.pi * D * d)
        #print "D * t_gamma", D * t_gamma
        #print "np.sqrt(D * t_gamma)",np.sqrt(D * t_gamma)
        #print "d/(2 * np.sqrt(D * t_gamma)",d/(2 * np.sqrt(D * t_gamma))
        #print "erfc(d/(2 * np.sqrt(D * t_gamma)))",erfc(d/(2 * np.sqrt(D * t_gamma)))


        # Condition to consider the contribution of pulsars at a distance > 1 kpc.
        # Note: If we do not add this condition, the electron emission extends up to TeV energies
        sum_all_pulsars = sum(f_st[i] for i in range(len(f_st)) if d[i] > 1 * gp.kpc_to_cm) 
        #sum_all_pulsars = sum(f_st)
        f_st_int.append(sum_all_pulsars)
        
        # Check how many pulsars there are in a region of 1 kpc from the Earth
    N_pulsars_1kpc = 0
    print ("Positions of pulsars at distance < 1 kpc")
    print ("Distance  x_Earth  y_Earth")
    for i in range(len(f_st)):
        if d[i] < 1 * gp.kpc_to_cm:
            print (d[i],x_Earth[i],y_Earth[i])
            N_pulsars_1kpc += 1
    print ("The number of pulsars within 1 kpc distance from the Earth is ", N_pulsars_1kpc)

    return f_st_int



# ******************** MAIN FUNCTION ******************

if __name__=='__main__':
    global LUM
    global tag
    #points = opts.File
    #tag = sys.argv[2]
    tag = opts.Name
    #data = np.loadtxt(points)
    AMS_data = np.loadtxt("Data/Data_points/AMS_data.dat",skiprows=2)
    HESS_data = np.loadtxt("Data/Data_points/HESS_data.dat",skiprows=2)
    Fermi_data = np.loadtxt("Data/Data_points/Fermi_data.dat",skiprows=2)
    AMS_positron_fraction = np.loadtxt("Data/Data_points/AMS_positron_fraction.dat",skiprows=2)
    PAMELA_positron_fraction = np.loadtxt("Data/Data_points/PAMELA_positron_fraction.dat",skiprows=2)
    
    # Curves from other papers
    Yuksel_delta04 = np.loadtxt("Data/Predictions_papers/Yuksel_Fig3_dotted_delta04.csv",delimiter=',')
    Aharonian_Fig4 = np.loadtxt("Data/Predictions_papers/Aharonian_1995_Fig4_time_dependent_injection.csv",skiprows=1)

    min_bin_deg = 0.
    max_bin_deg = SIZE+0.2
    nbins       = SIZE*10+3
    #nbins       = 51
    #deg = np.linspace(min_bin_deg,max_bin_deg,nbins)

    deg1=np.linspace(0.01,0.09,9)
    deg2=np.linspace(0.1,SIZE,SIZE*10)
    deg=np.concatenate((deg1,deg2))

    #degs = [1.7, 5.5, 8.6]
    degs = [2.6,SIZE] # IMPROVE ME!: SIZE should be an array with different sizes, just to compare 

    #bin_1dot7=int(1.7/((max_bin_deg-min_bin_deg)/nbins))
    #bin_5dot5=int(5.5/((max_bin_deg-min_bin_deg)/nbins))
    #bin_8dot6=int(8.6/((max_bin_deg-min_bin_deg)/nbins))

    bin_Size    = int(SIZE/((max_bin_deg-min_bin_deg)/nbins)) # Bin for the corresponding size given by diffusion
    bin_Milagro = int(2.6/((max_bin_deg-min_bin_deg)/nbins))  # Bin for the corresponding size given by Milagro's point at FWHM=2.6

    #****************** LUMINOSITY *************
    Age = FindAge()
    LUMBURST,LUMCONT,lum0,tau0,age,edot = CalculateLuminosity(10000,Age)
    fig = plt.figure()
    #print "LUMBURST,LUMCONT",LUMBURST,LUMCONT
    if len(LUMBURST) != 0:
        plt.plot(10.**LUMBURST[:,0],10.**LUMBURST[:,1],label=" ")
    plt.loglog(10.**LUMCONT[:,0],10.**LUMCONT[:,1]/MU,label="Pulsar evolution luminosity")
    #plt.xlim([0.,10.*TC])
    plt.xlim([0.,2*AGE])
    #plt.xlim([1.e5,AGE])
    plt.ylim([edot/10.,lum0*100])
    plt.ylabel(r'L$_e$ [erg/s]')
    plt.xlabel("Age [kyr]")
    plt.plot((1., 2*age),  (edot, edot), label=r'Constant injection luminosity',color='red')
    plt.plot((tau0, tau0), (edot/10., lum0*100), label=r'$\tau_0$',color='black',linestyle = "dashed")
    plt.plot((age, age),   (edot/10., lum0*100), label=r'Now',color='blue',linestyle = "dashed")
    #print tau0,EDOT/10.,LUMCONT[0,1]
    plt.title(r'L$_0$=%.1e erg/s; $\tau_0$=%.1e yr; n = %.1f' %(lum0,tau0,BRIND))
    plt.grid(color="black",alpha=.5)
    plt.legend(prop={'size':10},loc="upper left")
    #plt.legend(title="log10(L0),t0,n =\n"+str(round(math.log10(LUM0),2))+","+str(round(TC,2))+","+str(round(BRIND,2)),loc="upper right")
    if (FIG_EPS):
        fig.savefig("Figures/Luminosity_"+tag+".eps")
    else:
        fig.savefig("Figures/Luminosity_"+tag+".png")

    #************* ELECTRON DENSITY IN SPACE AND ENERGY ***********
    global ETRAJCONT,ETRAJBURST,ETRAJCONTINVERSE,ETRAJBURSTINVERSE,LAMBCONT,LAMBBURST

    fp = InitialiseGappa(fp,fr,BBURST,AGEBURST)
    ETRAJBURST,ETRAJBURSTINVERSE,LAMBBURST = CalculateEnergyTrajectory(fp)

    fp = InitialiseGappa(fp,fr,BCONT,Age)
    ETRAJCONT,ETRAJCONTINVERSE,LAMBCONT = CalculateEnergyTrajectory(fp)

    # This creates an array of electron densities in (E,R) space
    twoDarray,twoDarrayPLOT,halfwidth,E,R = Create_E_R_ArrayOfElectrons(RBINS,EBINS)
    # twoDarray:  2-D Array containing all the electron densities, for each of the radii and the energies [1/(erg*cm^3)] 
    # twoDarrayPLOT: The same * E^2, for plotting  [erg/cm^3)] 
    # halfwidth: Array containing, for each energy, the distance at which the maximum density goes to half 
    # E: Array of the energies [erg]
    # R: Array of the radii [pc]
    
    #print halfwidth

    # plot it!
    fig,ax =  plt.subplots(1, 1,figsize=(7,5))
    logarray = np.log10(twoDarrayPLOT)
    levels = np.linspace(np.amin(logarray),np.amax(logarray),100)
    plt.contourf(np.log10(E/gp.TeV_to_erg),np.log10(R), logarray.T,levels, cmap=plt.get_cmap('viridis'))    # Density of electrons
    #plt.plot(np.log10(halfwidth[:,0]),np.log10(halfwidth[:,1]),color="black",linestyle = "dashed")      # Line limiting half of the density of the electrons for a given energy
    plt.grid(color="black",alpha=.5)
    cbar = plt.colorbar()
    cbar.set_label(r'log$_{10}$(E$^2$ $\frac{\mathrm{dN}}{\mathrm{dE}})$ [erg cm$^{-3}$]')
    plt.ylabel("log$_{10}$ (R) [pc]")
    plt.xlabel("log$_{10}$ (E) [TeV]")
    if (FIG_EPS):
        fig.savefig("Figures/Electrons_E_R_Array_"+tag+".eps")
    else:
        fig.savefig("Figures/Electrons_E_R_Array_"+tag+".png")


    #***************** ELECTRON FLUX EARTH *****************
    GeV_to_erg = 1.e-3 * gp.TeV_to_erg
    ii = np.where(R >= 1000.*DIST)[0][0] 
    # First index where R > 1000*DIST (corresponds to the distance in pc)
    # Since the problem is spherically symmetric, the flux at Earth is equal to the flux at any point of the sphere with radius R=1000*DIST
    #print "-->",R[ii]
    fig = plt.figure()
    fac = 1e4 * c / (4.*gp.pi) # c/4pi  in cm/s, 1e4 transform the cm^-2 to m^-2 in the E^3 J(E) function
    EGeV =  E/gp.TeV_to_erg * 1.e3 # GeV
    plt.loglog(E/gp.TeV_to_erg,EGeV**3.*GeV_to_erg*fac*twoDarray.T[ii],color="black", label ="Pulsar")  # E^3 J(E)
    # GeV_to_erg pass one of the GeV to erg on the numeral and they go away with the one coming from twoDarray [1/(erg*cm^3)]

    #print "EGeV**3.*GeV_to_erg*fac*f_st_int",EGeV**3.*GeV_to_erg*fac*f_st_int
    print ("Flux_Earth",EGeV**3.*GeV_to_erg*fac*twoDarray.T[ii])
    zipped=zip(E/gp.TeV_to_erg,EGeV**3.*GeV_to_erg*fac*twoDarray.T[ii])
    np.savetxt("Results/Flux_Earth"+tag+".txt", zipped)


    # ********** HOMOGENEOUS PULSAR CONTRIBUTION ********
    if (ALL_PULSAR):
        f_st_int=Flux_Earth_all_pulsars(E)
        zipped_all_pulsars=zip(E/gp.TeV_to_erg,EGeV**3.*GeV_to_erg*fac*f_st_int)
        np.savetxt("Results/Flux_Earth_all_pulsars"+tag+".txt", zipped_all_pulsars)    
        plt.loglog(E/gp.TeV_to_erg,EGeV**3.*GeV_to_erg*fac*f_st_int,color="red",label="All pulsars [d > 1 kpc]")  # E^3 J(E)

    # AMS Data all electron flux
    y_AMS = AMS_data[:,3]*pow(AMS_data[:,0],3) # F x E^3
    yerror_AMS = AMS_data[:,4]*pow(AMS_data[:,0],3)
    AMS_points = plt.errorbar(AMS_data[:,0]*1.e-3,y_AMS,yerr=yerror_AMS,fmt='o',color = "black",label="AMS",markeredgecolor='k')

    # HESS Data all electron flux
    y_HESS = HESS_data[:,3]*pow(HESS_data[:,0],3)
    yerror_HESS = HESS_data[:,4]*pow(HESS_data[:,0],3)
    HESS_points = plt.errorbar(HESS_data[:,0]*1.e-3,y_HESS,yerr=yerror_HESS,fmt='^',color = "red",label="HESS",markeredgecolor='k')

    # Fermi Data all electron flux
    y_Fermi = Fermi_data[:,3]*pow(Fermi_data[:,0],3)
    yerror_Fermi = Fermi_data[:,4]*pow(Fermi_data[:,0],3)
    Fermi_points = plt.errorbar(Fermi_data[:,0]*1.e-3,y_Fermi,yerr=yerror_Fermi,fmt='s',color = "blue", label="Fermi",markeredgecolor='k')


    # Values for galactic electrons and positrons 
    # From Moskalenko and Strong (1998), Figure 5, left panel                                                                                                
    primary_el_data = np.loadtxt("Data/Moskalenko_and_Strong/Primary_electrons.txt",skiprows=1)
    secondary_el_data = np.loadtxt("Data/Moskalenko_and_Strong/Secondary_electrons.txt",skiprows=1)
    secondary_pos_data = np.loadtxt("Data/Moskalenko_and_Strong/Secondary_positrons.txt",skiprows=1)

    x_primary_el    = primary_el_data[:,0] * 1e-6 # TeV
    y_primary_el    = primary_el_data[:,1] * 1e-3 * 1e4 # GeV m^-2 s^-1 sr^-1
    x_secondary_el  = secondary_el_data[:,0] * 1e-6 # TeV
    y_secondary_el  = secondary_el_data[:,1] * 1e-3 * 1e4 # GeV m^-2 s^-1 sr^-1
    x_secondary_pos = secondary_pos_data[:,0] * 1e-6 # TeV
    y_secondary_pos = secondary_pos_data[:,1] * 1e-3 * 1e4 # GeV m^-2 s^-1 sr^-1

    primary_el=np.interp(E/gp.TeV_to_erg,x_primary_el,y_primary_el,right=0) * EGeV # GeV^2 m^-2 s^-1 sr^-1                                                                           
    secondary_el=np.interp(E/gp.TeV_to_erg,x_secondary_el,y_secondary_el,right=0) * EGeV # GeV^2 m^-2 s^-1 sr^-1                                                                                  
    secondary_pos=np.interp(E/gp.TeV_to_erg,x_secondary_pos,y_secondary_pos,right=0) * EGeV # GeV^2 m^-2 s^-1 sr^-1                                                                                         
    #primary_el=np.interp(E/gp.TeV_to_erg,x_primary_el,y_primary_el) * EGeV # GeV^2 m^-2 s^-1 sr^-1
    #secondary_el=np.interp(E/gp.TeV_to_erg,x_secondary_el,y_secondary_el) * EGeV # GeV^2 m^-2 s^-1 sr^-1
    #secondary_pos=np.interp(E/gp.TeV_to_erg,x_secondary_pos,y_secondary_pos) * EGeV # GeV^2 m^-2 s^-1 sr^-1
    plt.loglog(E/gp.TeV_to_erg,primary_el, color = "blue", label ="Primary e$^-$")  # E^3 J(E)
    plt.loglog(E/gp.TeV_to_erg,secondary_el, color = "magenta", label ="Secondary e$^-$")  # E^3 J(E)
    plt.loglog(E/gp.TeV_to_erg,secondary_pos, color = "green", label ="Secondary e$^+$")  # E^3 J(E)                                                                                                        
    #print "Primary electrons", primary_el                                                                                                                                                           
    #print "Secondary electrons", secondary_el
    #print "Secondary positrons", secondary_pos
    #zipped=zip(E/gp.TeV_to_erg,primary_el)
    #np.savetxt("Results/Flux_Earth_primary_electrons.txt", zipped)
    #zipped=zip(E/gp.TeV_to_erg,secondary_el)
    #np.savetxt("Results/Flux_Earth_secondary_electrons.txt", zipped)
    #zipped=zip(E/gp.TeV_to_erg,secondary_pos)
    #np.savetxt("Results/Flux_Earth_secondary_positrons.txt", zipped)

    # Yuksel Figure 3, delta=0.4
    #plt.loglog(Yuksel_delta04[:,0]*1.e-3,Yuksel_delta04[:,1],color = '0.75',label ="Yuksel Fig 3 delta 0.4")
    #plt.loglog(Aharonian_Fig4[:,0]*1.e-3,Aharonian_Fig4[:,1],color = "cyan",label ="Aharonian Fig 4")

    plt.ylabel("E$^3$ J(E) [GeV$^2$/(m$^2$s sr)]", fontsize=13)
    plt.xlabel("E [TeV]", fontsize=13)
    plt.grid(color="black",alpha=.5)
    #plt.legend(numpoints=1,handles=[AMS_points,HESS_points,Fermi_points],prop={'size':10},loc="upper right")
    plt.legend(numpoints=1,prop={'size':10},loc="upper right",ncol=3)
    plt.xlim([1e-3,1e1])
    plt.ylim([1e0,1e3])
    if (FIG_EPS):
        fig.savefig("Figures/Flux_Earth_"+tag+".eps")
    else:
        fig.savefig("Figures/Flux_Earth_"+tag+".png")
    # ********** POSITRON FRACTION ***********                                                                                                                                                              
    fig = plt.figure()
    flux_earth_Source=EGeV**3.*GeV_to_erg*fac*twoDarray.T[ii]

    # All pulsars
    #flux_earth_all_pulsars=EGeV**3.*GeV_to_erg*fac*f_st_int
    #fraction=(0.5 * flux_earth_Source+secondary_pos)/(flux_earth_Source + flux_earth_all_pulsars + primary_el + secondary_el + secondary_pos)                                                            
    fraction=(0.5 * flux_earth_Source+secondary_pos)/(flux_earth_Source +  primary_el + secondary_el + secondary_pos)
    plt.loglog(E/gp.TeV_to_erg,fraction,label = "Fraction total")
    zipped=zip(E/gp.TeV_to_erg,fraction)
    print ("Fraction",fraction)
    np.savetxt("Results/Fraction_Total_Positron_Earth_"+tag+".txt", zipped)

    fraction_galactic_positrons=secondary_pos/(flux_earth_Source+primary_el + secondary_el + secondary_pos)
    plt.loglog(E/gp.TeV_to_erg,fraction_galactic_positrons,label = "Galactic e$^+$ fraction")
    #print "fraction galactic positrons",fraction_galactic_positrons
    zipped=zip(E/gp.TeV_to_erg,fraction_galactic_positrons)
    np.savetxt("Results/Fraction_Galactic_Positron_Earth_"+tag+".txt", zipped)

    fraction_Source_positrons=0.5 * flux_earth_Source/(flux_earth_Source+primary_el + secondary_el + secondary_pos)
    plt.loglog(E/gp.TeV_to_erg,fraction_Source_positrons,label = "Source e$^+$ fraction")
    #print "fraction Source positrons",fraction_Source_positrons
    zipped=zip(E/gp.TeV_to_erg,fraction_Source_positrons)
    np.savetxt("Results/Fraction_Source_Positron_Earth_"+tag+".txt", zipped)

    # AMS Data positron fraction
    plt.errorbar(AMS_positron_fraction[:,0]*1.e-3,AMS_positron_fraction[:,3],yerr=AMS_positron_fraction[:,4],fmt='o',color = "black",label="AMS",markeredgecolor='k')
    # PAMELA Data positron fraction
    plt.errorbar(PAMELA_positron_fraction[:,0]*1.e-3,PAMELA_positron_fraction[:,3],yerr=PAMELA_positron_fraction[:,4],fmt='o',color = "red",label="PAMELA",markeredgecolor='k')

    plt.ylabel("e$^+$/(e$^+$+e$^-$)", fontsize=13)
    plt.xlabel("E [TeV]", fontsize=13)
    plt.grid(color="black",alpha=.5)
    plt.xlim([1e-4,1e0])
    plt.ylim([1e-2,1e0])
    plt.legend(numpoints=1,prop={'size':10},loc="upper right")
    if (FIG_EPS):
        fig.savefig("Figures/Fraction_Earth_"+tag+".eps")
    else:
        fig.savefig("Figures/Fraction_Earth_"+tag+".png")

    # Break in case we do not want to calculate the gamma-ray spectrum
    if(ONLY_FLUX_EARTH):
        exit()

    # ************ ELECTRON COLUMN DENSITIES ************
    # This integrates the spectra along the angular distance                                                                                                                                 
    values = []
    for d in deg:
        values.append(LineOfSightIntegration(d,0.,twoDarray,E,R,1e4))
        #print "los,values",d,values
    values = np.array(values)

    IntSpec = []
    for va in values.T: # One per angle definition
        intsp = np.array(fu.IntegratedProfile(zip(deg * gp.pi/180.,2.*gp.pi * va * deg * gp.pi/180. * (1/(4*gp.pi)))))[:,1] # Integration over the solid angle.
        intsp = np.array(intsp)
        #print "intsp",intsp
        #intsp = np.array(fu.IntegratedProfile(zip(deg * gp.pi/180.,2.*gp.pi * va * deg * gp.pi/180.)))[:,1] # Integration over the solid angle. We use deg instead of sin(deg)
        # The (1/(4*gp.pi)) is to take into account that we are integrating over the solid angle
        IntSpec.append(intsp)
    IntSpec = np.array(IntSpec)

    # plot the angular-integrated electron spectra                                                                                                                                                     
    IntSpec = np.array(IntSpec.T)
    fig = plt.figure()
    for s in IntSpec:
        plt.loglog(E/gp.TeV_to_erg,E**2.*s) # E is in erg
    plt.grid(color="black",alpha=.5)
    plt.ylim([1e-3,1e5])
    plt.ylabel("E$^2$ dN/dE [erg/cm$^2$]", fontsize=13)
    plt.xlabel("E [TeV]", fontsize=13)
    if (FIG_EPS):
        fig.savefig("Figures/Electron_Spectra_"+tag+".eps")
    else:
        fig.savefig("Figures/Electron_Spectra_"+tag+".png")

    #************* ELECTRON SPECTRA SOURCE ***********
    values_diff_spectrum = []
    #for d in deg:
    for d in deg:
        print ("los %.2f" % d)
        values_diff_spectrum.append(LineOfSightVolumeIntegration(d,0.,twoDarray,E,R,1e4))
    values_diff_spectrum = np.array(values_diff_spectrum)

    # This integrates the spectra along the angular distance
    IntSpec_volume = []
    for va_volume in values_diff_spectrum.T: # One per angle definition                                                                                                                                                                                                   
        #intsp_volume = np.array(fu.IntegratedProfile(zip(deg * gp.pi/180.,2.*gp.pi * va_volume * deg * gp.pi/180. * (1/(4*gp.pi)))))[:,1] # Integration over the solid angle. 
        intsp_volume = np.array(fu.IntegratedProfile(zip(deg * gp.pi/180.,2.*gp.pi * va_volume * deg * gp.pi/180. )))[:,1] # Integration over the solid angle. 
        # The (1/(4*gp.pi)) is to take into account that we are integrating over the solid angle
        IntSpec_volume.append(intsp_volume)
    IntSpec_volume = np.array(IntSpec_volume)
    IntSpec_volume = np.array(IntSpec_volume.T)

    #IntSpec_volume_all = LineOfSightVolumeIntegration(90.,0.,twoDarray,E,R,1e4)  # 2.*gp.pi comes from the solid angle integral of half a sphere
    #IntSpec_volume_all = 2.* gp.pi * np.array(IntSpec_volume_all)
    #print IntSpec_volume_all

    # plot the angular-integrated electron spectra
    fig = plt.figure()
    #for volume_spectra in IntSpec_volume:
    #    plt.loglog(E/gp.TeV_to_erg,E**2.*volume_spectra) # E is in erg

    #plt.loglog(E/gp.TeV_to_erg,E**2.*IntSpec_volume[bin_1dot7],label='1.7 deg') # E is in erg 
    #plt.loglog(E/gp.TeV_to_erg,E**2.*IntSpec_volume[bin_5dot5],label='5.5 deg') # E is in erg 
    #plt.loglog(E/gp.TeV_to_erg,E**2.*IntSpec_volume[bin_8dot6],label='8.6 deg') # E is in erg 
    for d in degs :
        ind=np.where(deg >= d)[0][0]
        print ("ind %i,d %.2f" % (ind,d))
        plt.loglog(E/gp.TeV_to_erg,E**2.*IntSpec_volume[ind-1],label='%s deg' % d)
    #plt.loglog(E/gp.TeV_to_erg,E**2.*IntSpec_volume[bin_Milagro],label='2.6 deg [Milagro]') # E is in erg 
    #plt.loglog(E/gp.TeV_to_erg,E**2.*IntSpec_volume[bin_Size],label='%s deg' % SIZE) # E is in erg 
    #plt.loglog(E/gp.TeV_to_erg,E**2.*IntSpec_volume_all,label='All') # E is in erg 
    #plt.loglog(x_Mehr,y_Mehr,label="Mehr flux",color='black')
    plt.grid(color="black",alpha=.5)
    plt.ylim([1e40,1e47])
    plt.ylabel("E$^2$ dN/dE [erg]", fontsize=13)
    plt.xlabel("E [TeV]", fontsize=13)
    plt.legend(prop={'size':9},loc="upper right")
    if (FIG_EPS):
        fig.savefig("Figures/Electron_Spectra_Volume_"+tag+".eps")
    else:
        fig.savefig("Figures/Electron_Spectra_Volume_"+tag+".png")
    zipped=zip(E/gp.TeV_to_erg,E**2.*IntSpec_volume[ind-1])
    np.savetxt("Results/Electron_Spectra_Volume_%s_%sdeg.txt" %(tag,d), zipped)


    # *************  GAMMA SPECTRUM ***************
    # calculate the corresponding gamma-ray spectra of the angular integrated
    # column densities                                                                                                                                                                       
    sp = []
    inds = []
    for d in degs :
        inds.append(np.where(deg >= d)[0][0])
    #print inds
    #print IntSpec
    for i,d in zip(inds,degs):
        print ("i,d",i,d)
        print ("IntSpec[i-1]",IntSpec[i-1])
        s = IntSpec[i-1]
        fr.SetElectrons(zip(E,s))
        fr.CalculateDifferentialPhotonSpectrum(E) 
        sp.append(np.array(fr.GetTotalSED()))
        
    sp = np.array(sp)    
    fig = plt.figure()
    for s,d in zip(sp,degs):
        print ("sp",s)
        plt.loglog(s[:,0],s[:,1],label=r'%s deg' % (d)) # s[:,0] contains the Energy [TeV] and s[:,1] directly the SED [erg cm^-2 s^-1]
        zipped=zip(s[:,0],s[:,1])
        np.savetxt("Results/Gamma_Spectra_%s_%sdeg.txt" %(tag,d), zipped)

    # Include in the plot Source's spectral energy distribution
    x_Source = np.arange(1., 100., 0.1)
    y_Source = NORM*pow(x_Source/PIVOT_E,-GAMMA) * pow(x_Source,2) * gp.TeV_to_erg                           # Norm is given in TeV^-1 cm^-2 s^-1, but when multiplied by E^2 it is converted to TeV
    plt.loglog(x_Source,y_Source,label=tag,color='black')
    # Butterfly
    y_max_Source_down = (NORM+NORM_ERR)*pow(x_Source/PIVOT_E,-(GAMMA+GAMMA_ERR)) * pow(x_Source,2) * gp.TeV_to_erg
    y_min_Source_down = (NORM-NORM_ERR)*pow(x_Source/PIVOT_E,-(GAMMA-GAMMA_ERR)) * pow(x_Source,2) * gp.TeV_to_erg
    y_max_Source_up   = (NORM+NORM_ERR)*pow(x_Source/PIVOT_E,-(GAMMA-GAMMA_ERR)) * pow(x_Source,2) * gp.TeV_to_erg
    y_min_Source_up   = (NORM-NORM_ERR)*pow(x_Source/PIVOT_E,-(GAMMA+GAMMA_ERR)) * pow(x_Source,2) * gp.TeV_to_erg
    plt.fill_between(x_Source,y_min_Source_down,y_max_Source_down,where=x_Source<20,color='grey', alpha='0.5')
    plt.fill_between(x_Source,y_min_Source_up,y_max_Source_up,where=x_Source>20,color='grey', alpha='0.5')
    # Milagro point
    x_Milagro = 20. 
    y_Milagro = 6.9e-15 * pow(x_Milagro,2) * gp.TeV_to_erg  
    y_err_Milagro = 1.6e-15 * pow(x_Milagro,2) * gp.TeV_to_erg
    #plt.errorbar(x_Milagro,y_Milagro,yerr=y_err_Milagro,fmt='o',color = "red",label="Milagro")

    plt.ylabel("E$^2$ dN/dE [erg s$^{-1}$cm$^{-2}$]", fontsize=13)
    plt.xlabel("E [TeV]", fontsize=13)
    plt.ylim([1e-16,1e-8])
    plt.grid(color="black",alpha=.5)
    plt.legend(numpoints=1,prop={'size':9},loc="upper right")
    if (FIG_EPS):
        fig.savefig("Figures/Gamma_Spectra_"+tag+".eps")
    else:
        fig.savefig("Figures/Gamma_Spectra_"+tag+".png")
    #fig.savefig("Figures/Gamma_Spectra_"+tag+".eps")


    # *************  GAMMA SPECTRUM VOLUME ***************
    # calculate the corresponding gamma-ray spectra of the angular integrated
    # column densities                                                                                                                                                                       
    sp_volume = []
    inds = []
    for d in degs :
        inds.append(np.where(deg >= d)[0][0])
    #print inds
    #print IntSpec
    for i,d in zip(inds,degs):
        print ("i,d",i,d)
        print ("IntSpec_volume[i-1]",IntSpec_volume[i-1])
        s = IntSpec_volume[i-1]
        fr.SetElectrons(zip(E,s))
        #fr.SetElectrons(zip(Mehr_data[:,0]*1.e-9,Mehr_data[:,2]*1.e9/ gp.TeV_to_erg))
        fr.SetDistance(DIST*1.e3)
        fr.CalculateDifferentialPhotonSpectrum(E) 
        sp_volume.append(np.array(fr.GetTotalSED()))
        

    sp_volume = np.array(sp_volume)    
    fig = plt.figure()
    for s,d in zip(sp_volume,degs):
        print ("sp_volume",s)
        plt.loglog(s[:,0],s[:,1],label=r'%s deg' % (d)) # s[:,0] contains the Energy [TeV] and s[:,1] directly the SED [erg cm^-2 s^-1]
    plt.loglog(x_Source,y_Source,label=tag,color='black')
    plt.ylabel("E$^2$ dN/dE [erg s$^{-1}$cm$^{-2}$]", fontsize=13)
    plt.xlabel("E [TeV]", fontsize=13)
    plt.ylim([1e-16,1e-8])
    plt.grid(color="black",alpha=.5)
    plt.legend(prop={'size':9},loc="upper right")
    if (FIG_EPS):
        fig.savefig("Figures/Gamma_Spectra_Volume_"+tag+".eps")
    else:
        fig.savefig("Figures/Gamma_Spectra_Volume_"+tag+".png")

    # *********** GAMMA-RAY ANGULAR PROFILES ***********                                                                                                                                             
    # this gives an array of line-of-sight integrated electron spectra vs. angular
    # distance from the source                                                                                                                                                    
    # corr = np.diff(deg) * deg_to_rad * deg[1:] * deg_to_rad * gp.pi # sr^2   Delta_theta * theta * pi (area of the circular section)
    deg_sqr = deg**2.
    corr = np.diff(deg_sqr)  * gp.pi # deg^2   Delta_theta^2 * pi (area of the ring)
    # integrand for the solid angle.

    # calculate the corresponding gamma-ray spectra of the *not* angular integrated
    # column densities (aka surface brightness)                                                                                                                                                         
    fig = plt.figure()
    sb = []
    sb_20TeV = []
    diff_sb_20TeV = []
    intspec_profile_1TeV = []
    intspec_profile_20TeV = []
    diffspec_profile_20TeV = []
    diffspec_profile = []

    EE_1TeV = 1. * gp.TeV_to_erg   # We compute the profile above 1 TeV
    EE_20TeV = 20. * gp.TeV_to_erg # We compute the profile above 20 TeV
    EE_20TeV_list=[20 * gp.TeV_to_erg]
    ee = np.logspace(math.log10(EE_1TeV),math.log10(EMAX),40)

    IntSpec_diff = np.diff(IntSpec,axis=0) 
    # IntSpec is a 2D array with the values of the electron spectrum [100] x values for each d. 
    # This is dN/dE [d_i+1] - dN/dE [d_i]    
    for s in IntSpec_diff:
        #print "IntSpec,axis=0",s
        fr.SetElectrons(zip(E,s))
        fr.SetDistance(0.)
        fr.CalculateDifferentialPhotonSpectrum(ee)
        intspec_profile_1TeV = fu.Integrate(fr.GetTotalSpectrum(),EE_1TeV,EMAX)   # The E range should be given in erg  
        intspec_profile_20TeV = fu.Integrate(fr.GetTotalSpectrum(),EE_20TeV,EMAX) # The E range should be given in erg  
        sb.append(intspec_profile_1TeV) # Integrated surface brigthness of photons 
        sb_20TeV.append(intspec_profile_20TeV) # Integrated surface brigthness of photons 
        # Differential spectrum
        fr.CalculateDifferentialPhotonSpectrum(EE_20TeV_list)
        diffspec_profile_20TeV = fr.GetTotalSpectrum()[0][1]
        diff_sb_20TeV.append(diffspec_profile_20TeV)

    sb = np.array(sb)
    sb_20TeV = np.array(sb_20TeV)
    diff_sb_20TeV=np.array(diff_sb_20TeV)


    profile = []
    #for p,c in zip(s[:,1],corr):
    for p,c in zip(sb,corr[1:]):
        print ("s,corr 1 TeV",p,c)
        profile.append(p/c) # Wrong, should be multiplied?
        print ("p/c 1 TeV",p/c)

    profile_20TeV = []
    for p,c in zip(sb_20TeV,corr[1:]):
        print ("s,corr 20 TeV",p,c)
        profile_20TeV.append(p/c) # Wrong, should be multiplied?
        print ("p/c 20 TeV",p/c)


    # Calculate the contribution for the first bin in theta
    fr.SetElectrons(zip(E,IntSpec[0]))
    fr.SetDistance(0.)
    fr.CalculateDifferentialPhotonSpectrum(EE_20TeV_list)
    Total = fr.GetTotalSpectrum()[0][1] # The total differential flux per bin in theta
    print ("Flux from theta[0] to theta[1]=",Total)

    profile_20TeV_diff = []
    profile_20TeV_only_flux = []
    for p,c,d in zip(diff_sb_20TeV,corr[1:],deg[2:]):
        print ("s,corr 20 TeV diff",p,c)
        profile_20TeV_diff.append(p/c) # Wrong, should be multiplied?
        profile_20TeV_only_flux.append(p)
        print ("p/c 20 TeV diff",p/c)
        Total=Total+p
        print ("Total differential %f [erg^-1 s^1 cm^-2] for deg %f" % (Total,d))

    #print "int = ",fu.Integrate(zip(gp.pi*deg_sqr[1:],profile),deg_sqr[1],deg_sqr[len(deg_sqr)-1])
    profile = np.array(profile)
    profile_20TeV = np.array(profile_20TeV)
    #plt.plot(deg_reduced[1:],profile,label = "E >"+str(EE_1TeV/gp.TeV_to_erg)+" TeV")          # Profile above 1 TeV 
    plt.plot(deg[2:],profile_20TeV,label = "E >"+str(EE_20TeV/gp.TeV_to_erg)+" TeV")    # Profile above 20 TeV
    ax = fig.add_subplot(111)
    plt.ylabel("Surface luminosity [1/(s cm$^{2}$ deg$^{2}$)]", fontsize=13)
    #plt.xlabel("angular distance [deg$^2$]", fontsize=13)
    plt.xlabel("Angular distance [deg]", fontsize=13)
    #plt.ylim([-120.,500.])
    #ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    #ax.yaxis.labelpad = -10
    #plt.ticklabel_format(style='sci', axis='y')
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.grid(color="black",alpha=.5)
    plt.legend(prop={'size':8},loc="upper right")
    if (FIG_EPS):
        fig.savefig("Figures/Gamma_Profiles_"+tag+".eps")
    else:
        fig.savefig("Figures/Gamma_Profiles_"+tag+".png")

    # Differential profile
    profile_20TeV_diff = np.array(profile_20TeV_diff)
    fig = plt.figure()
    plt.plot(deg[2:],profile_20TeV_diff,label = "E = 20 TeV")
    ax = fig.add_subplot(111)
    plt.ylabel("Surface luminosity [1/(erg s cm$^{2}$ deg$^{2}$)]", fontsize=13)
    plt.xlabel("Angular distance [deg]", fontsize=13)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #plt.ticklabel_format(style='sci', axis='y')
    #plt.ylim([-120.,500.])
    #ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    #ax.yaxis.labelpad = -10
    plt.grid(color="black",alpha=.5)
    plt.legend(prop={'size':8},loc="upper right")
    if (FIG_EPS):
        fig.savefig("Figures/Gamma_Profile_Differential_"+tag+".eps")
    else:
        fig.savefig("Figures/Gamma_Profile_Differential_"+tag+".png")
    zipped=zip(deg[2:],profile_20TeV_diff)
    np.savetxt("Results/Gamma_Profile_Differential_flux_solid_angle"+tag+".txt", zipped)

    # Differential profile (only flux)
    profile_20TeV_only_flux = np.array(profile_20TeV_only_flux)
    fig = plt.figure()
    plt.plot(deg[2:],profile_20TeV_only_flux,'o', label = "E = 20 TeV")
    ax = fig.add_subplot(111)
    plt.ylabel("Surface luminosity [1/(erg s cm$^{2}$)]", fontsize=13)
    plt.xlabel("Angular distance [deg]", fontsize=13)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #plt.ticklabel_format(style='sci', axis='y')
    #ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    #ax.yaxis.labelpad = -10
    plt.grid(color="black",alpha=.5)
    plt.legend(prop={'size':8},loc="upper right")
    if (FIG_EPS):
        fig.savefig("Figures/Gamma_Profile_Differential_flux_"+tag+".eps")
    else:
        fig.savefig("Figures/Gamma_Profile_Differential_flux_"+tag+".png")
    zipped=zip(deg[2:],profile_20TeV_diff)
    np.savetxt("Results/Gamma_Profile_Differential_flux"+tag+".txt", zipped)
    
