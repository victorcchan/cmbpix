cimport cython
from cython.parallel import prange
from cython.cimports.libc.stdlib cimport rand, srand, RAND_MAX
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def Psi_and_A_cy(int L, double[:] ClTTunlensed, double[:] ClTTfiducial, \
                 double[:] ClTTtotal, double[:] Clphiphi, int l1min, \
                 int l1max, int l2min, int l2max, int dl1, int dl2):
    
    cdef int l1x
    cdef int l1y
    cdef int l2x
    cdef int l2y
    cdef int nl1
    cdef int l1max_round
    nl1 = l1max/dl1
    l1max_round = nl1*dl1
    cdef int nl2
    cdef int l2max_round
    nl2 = l2max/dl2
    l2max_round = nl2*dl2
    #print(lmax_round)
    
    cdef int l1dotl1minusL
    cdef int l2dotl2minusl1
    cdef int Lminusl2dotl1minusl2
    cdef int l2dotl2minusL
    
    cdef int modl1
    cdef int modLminusl1
    cdef int modl2
    cdef int modLminusl2
    cdef int modl1minusl2
    
    cdef double integrand = 0.
    cdef double integral = 0.
    cdef double AL = 0.
    
    #for l1x from -lmax <= l1x < lmax by dl:
    for l1x in prange(-l1max_round, l1max_round+1, dl1, nogil = True):
        for l1y from -l1max_round <= l1y < l1max_round+1 by dl1:
            modl1 = <int> ((l1x*l1x+l1y*l1y)**(0.5))
            modLminusl1 = <int> (((L-l1x)*(L-l1x) + l1y*l1y)**(0.5))
            if (modl1 < l1min or modl1 > l1max):
                integral += 0.
            elif (modLminusl1 < l1min or modLminusl1 > l1max):
                integral += 0.
            else:
                l1dotl1minusL = l1x*(l1x-L) + l1y*(l1y)
                for l2x from -l2max_round <= l2x < l2max_round+1 by dl2:
                    for l2y from -l2max_round <= l2y < l2max_round+1 by dl2:
                        modl2 = <int> ((l2x*l2x+l2y*l2y)**(0.5))
                        modLminusl2 = <int> (((L-l2x)*(L-l2x) + l2y*l2y)**(0.5))
                        if (modl2 < l2min or modl2 > l2max):
                            integral += 0.
                        elif (modLminusl2 < l2min or modLminusl2 > l2max):
                            integral += 0.
                        else:
                            l2dotl2minusl1 = l2x*(l2x-l1x) + l2y*(l2y-l1y)
                            Lminusl2dotl1minusl2 = (L-l2x)*(l1x-l2x) + (-l2y)*(l1y-l2y)
                            l2dotl2minusL = l2x*(l2x-L) + l2y*(l2y)
                            modl1minusl2 = <int> (((l1x-l2x)*(l1x-l2x) + (l1y-l2y)*(l1y-l2y))**(0.5))

                            integrand = 2. \
                                * 1./(ClTTtotal[modl1] * ClTTtotal[modLminusl1]) \
                                * (ClTTunlensed[modl2]*ClTTfiducial[modl2]/ClTTtotal[modl2]) \
                                * (ClTTunlensed[modLminusl2]*ClTTfiducial[modLminusl2]/ClTTtotal[modLminusl2]) \
                                * l1dotl1minusL * l2dotl2minusl1 * Lminusl2dotl1minusl2 * l2dotl2minusL
                            
                            AL += integrand
                            integral += integrand*Clphiphi[modl1minusl2]
    
    integral *= dl1*dl1*dl2*dl2
    AL *= dl1*dl1*dl2*dl2
    AL = 1./AL
    # Results need to be scaled by (2\pi)^4
    
    return integral, AL

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def Psi_and_A_cy_mc(int L, double[:] ClTTunlensed, double[:] ClTTfiducial, \
                    double[:] ClTTtotal, double[:] Clphiphi, int l1min, \
                    int l1max, int l2min, int l2max, int n_samples, int dl=1):
    
    cdef int l1x
    cdef int l1y
    cdef int l2x
    cdef int l2y
    
    cdef int l1dotl1minusL
    cdef int l2dotl2minusl1
    cdef int Lminusl2dotl1minusl2
    cdef int l2dotl2minusL
    
    cdef int modl1
    cdef int modLminusl1
    cdef int modl2
    cdef int modLminusl2
    cdef int modl1minusl2
    
    cdef int j
    
    cdef double integrand = 0.
    cdef double integral = 0.
    cdef double AL = 0.
    
    cdef np.ndarray[np.int_t,
                    ndim=2,
                    negative_indices=False,
                    mode='c'] l1_vals = np.random.randint(-l1max/dl, l1max/dl, (n_samples, 2))*dl
    cdef np.ndarray[np.int_t,
                    ndim=2,
                    negative_indices=False,
                    mode='c'] l2_vals = np.random.randint(-l2max/dl, l2max/dl, (n_samples, 2))*dl
    
    for j in prange(n_samples, nogil = True):
        l1x = l1_vals[j,0]
        l1y = l1_vals[j,1]
        
        l2x = l2_vals[j,0]
        l2y = l2_vals[j,1]
        modl1 = <int> ((l1x*l1x+l1y*l1y)**(0.5))
        modLminusl1 = <int> (((L-l1x)*(L-l1x) + l1y*l1y)**(0.5))
        if (modl1 < l1min or modl1 > l1max):
            continue
        elif (modLminusl1 < l1min or modLminusl1 > l1max):
            continue
        else:
            modl2 = <int> ((l2x*l2x+l2y*l2y)**(0.5))
            modLminusl2 = <int> (((L-l2x)*(L-l2x) + l2y*l2y)**(0.5))
            if (modl2 < l2min or modl2 > l2max):
                continue
            elif (modLminusl2 < l2min or modLminusl2 > l2max):
                continue
            else:
                l1dotl1minusL = l1x*(l1x-L) + l1y*(l1y)
                l2dotl2minusl1 = l2x*(l2x-l1x) + l2y*(l2y-l1y)
                Lminusl2dotl1minusl2 = (L-l2x)*(l1x-l2x) + (-l2y)*(l1y-l2y)
                l2dotl2minusL = l2x*(l2x-L) + l2y*(l2y)
                modl1minusl2 = <int> (((l1x-l2x)*(l1x-l2x) + (l1y-l2y)*(l1y-l2y))**(0.5))
                integrand = 2. \
                    * 1./(ClTTtotal[modl1] * ClTTtotal[modLminusl1]) \
                    * (ClTTunlensed[modl2]*ClTTfiducial[modl2]/ClTTtotal[modl2]) \
                    * (ClTTunlensed[modLminusl2]*ClTTfiducial[modLminusl2]/ClTTtotal[modLminusl2]) \
                    * l1dotl1minusL * l2dotl2minusl1 * Lminusl2dotl1minusl2 * l2dotl2minusL
                AL += integrand
                integral += integrand*Clphiphi[modl1minusl2]
    integral *=1.*(2.*l1max)*(2.*l1max)*(2.*l2max)*(2.*l2max)/n_samples
    AL *=1.*(2.*l1max)*(2.*l1max)*(2.*l2max)*(2.*l2max)/n_samples
    AL = 1./AL
    # Results need to be scaled by (2\pi)^4
    
    return integral, AL