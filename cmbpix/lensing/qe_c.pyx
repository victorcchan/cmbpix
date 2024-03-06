cimport cython
from cython.parallel import prange
from cython.cimports.libc.stdlib cimport rand, srand, RAND_MAX
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def N1_bias_integral_cy_Kesden(int LL, double[:] ClTTunlensed, double[:] ClTTtotal, double[:] Clphiphi, int lmin, int lmax, int dl):
    
    cdef int l1x
    cdef int l1y
    cdef int l2x
    cdef int l2y
    cdef int L = LL
    
    cdef int Ldotl2
    cdef int LdotLminusl2
    cdef int Ldotl1
    cdef int LdotLminusl1
    
#     cdef int l1dotl1minusl2
#     cdef int Lminusl1dotl1minusl2
#     cdef int l1dotl1minusLplusl2
#     cdef int l2dotl1minusLplusl2

    cdef int l1dotl2minusl1
    cdef int l2dotl2minusl1
    cdef int Lminusl1dotl1minusl2
    cdef int Lminusl2dotl1minusl2
    cdef int Lminusl1minusl2dotl1
    cdef int Lminusl1minusl2dotLminusl2
    cdef int l1minusLplusl2dotLminusl1
    cdef int l1minusLplusl2dotl2
    
    cdef int modl1
    cdef int modLminusl1
    cdef int modl2
    cdef int modLminusl2
    cdef int modl1minusl2
    cdef int modl1minusLplusl2
    
    cdef double integral = 0
    
    #for l1x from -lmax <= l1x < lmax by dl:
    for l1x in prange(-lmax, lmax, dl, nogil = True):
        Ldotl1 = l1x*L
        LdotLminusl1 = L*(L-l1x)
        for l1y from -lmax <= l1y < lmax by dl:
            modl1 = <int> ((l1x*l1x+l1y*l1y)**(0.5))
            modLminusl1 = <int> (((L-l1x)*(L-l1x) + l1y*l1y)**(0.5))
            for l2x from -lmax <= l2x < lmax by dl:
                Ldotl2 = l2x*L
                LdotLminusl2 = L*(L-l2x)
                for l2y from -lmax <= l2y < lmax by dl:
                    modl2 = <int> ((l2x*l2x+l2y*l2y)**(0.5))
                    modLminusl2 = <int> (((L-l2x)*(L-l2x) + l2y*l2y)**(0.5))
                    modl1minusLplusl2 = <int> (((l1x-L+l2x)*(l1x-L+l2x) + (l1y+l2y)*(l1y+l2y))**(0.5))
                    modl1minusl2 = <int> (((l1x-l2x)*(l1x-l2x) + (l1y-l2y)*(l1y-l2y))**(0.5))
                    #l1dotl1minusl2 = l1x*(l1x-l2x)+l1y*(l1y-l2y)
                    #Lminusl1dotl1minusl2 = (L-l1x)*(l1x-l2x) + (-l1y)*(l1y-l2y)
                    #l1dotl1minusLplusl2 = l1x*(l1x-L+l2x) + l1y*(l1y+l2y)
                    #l2dotl1minusLplusl2 = l2x*(l1x-L+l2x) + l2y*(l1y+l2y)
                    
                    l1dotl2minusl1 = l1x*(l1x-l2x)+l1y*(l1y-l2y)
                    l2dotl2minusl1 = l2x*(l2x-l1x)+l2y*(l2y-l1y)
                    Lminusl1dotl1minusl2 = (L-l1x)*(l1x-l2x) + (-l1y)*(l1y-l2y)
                    Lminusl2dotl1minusl2 = (L-l2x)*(l1x-l2x) + (-l2y)*(l1y-l2y)
                    Lminusl1minusl2dotl1 = (L-l1x-l2x)*l1x + (-l1y-l2y)*l1y
                    Lminusl1minusl2dotLminusl2 = (L-l1x-l2x)*(L-l2x) + (-l1y-l2y)*(-l2y)
                    l1minusLplusl2dotLminusl1 = (l1x-L+l2x)*(L-l1x) + (l1y+l2y)*(-l1y)
                    l1minusLplusl2dotl2 = (l1x-L+l2x)*l2x + (l1y+l2y)*l2y
                    
                    if (modl1 < lmin or modl1 > lmax):
                        integral += 0.
                    elif (modl2 < lmin or modl2 > lmax):
                        integral += 0.
                    elif (modLminusl1 < lmin or modLminusl1 > lmax):
                        integral += 0.
                    elif (modLminusl2 < lmin or modLminusl2 > lmax):
                        integral += 0.
                    else:
                        integral += 1. \
                            * ( Ldotl1*ClTTunlensed[modl1] + LdotLminusl1*ClTTunlensed[modLminusl1] ) \
                            * 1./(2. * ClTTtotal[modl1] * ClTTtotal[modLminusl1]) \
                            * ( Ldotl2*ClTTunlensed[modl2] + LdotLminusl2*ClTTunlensed[modLminusl2] ) \
                            * 1./(2. * ClTTtotal[modl2] * ClTTtotal[modLminusl2]) \
                            * ( Clphiphi[modl1minusl2] \
                            * ( -l1dotl2minusl1*ClTTunlensed[modl1] + l2dotl2minusl1*ClTTunlensed[modl2] ) \
                            * ( -Lminusl1dotl1minusl2*ClTTunlensed[modLminusl1] + Lminusl2dotl1minusl2*ClTTunlensed[modLminusl2] ) \
                            + Clphiphi[modl1minusLplusl2] \
                            * ( -Lminusl1minusl2dotl1*ClTTunlensed[modl1] + Lminusl1minusl2dotLminusl2*ClTTunlensed[modLminusl2] ) \
                            * ( -l1minusLplusl2dotLminusl1*ClTTunlensed[modLminusl1] + l1minusLplusl2dotl2*ClTTunlensed[modl2] ) )
    
    integral *= dl*dl*dl*dl
    
    return integral

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def N1_bias_integral_cy_Kesden_mc(int LL, double[:] ClTTunlensed, double[:] ClTTtotal, double[:] Clphiphi, int lmin, int lmax, int dl, int n_samples):
    
    cdef int l1x
    cdef int l1y
    cdef int l2x
    cdef int l2y
    cdef int L = LL
    
    cdef int Ldotl2
    cdef int LdotLminusl2
    cdef int Ldotl1
    cdef int LdotLminusl1
    
#     cdef int l1dotl1minusl2
#     cdef int Lminusl1dotl1minusl2
#     cdef int l1dotl1minusLplusl2
#     cdef int l2dotl1minusLplusl2

    cdef int l1dotl2minusl1
    cdef int l2dotl2minusl1
    cdef int Lminusl1dotl1minusl2
    cdef int Lminusl2dotl1minusl2
    cdef int Lminusl1minusl2dotl1
    cdef int Lminusl1minusl2dotLminusl2
    cdef int l1minusLplusl2dotLminusl1
    cdef int l1minusLplusl2dotl2
    
    cdef int modl1
    cdef int modLminusl1
    cdef int modl2
    cdef int modLminusl2
    cdef int modl1minusl2
    cdef int modl1minusLplusl2

    cdef int j
    
    cdef double integral = 0
    
    cdef np.ndarray[np.int_t,
                    ndim=2,
                    negative_indices=False,
                    mode='c'] l1_vals = np.random.randint(-lmax/dl, lmax/dl, (n_samples, 2))*dl
    cdef np.ndarray[np.int_t,
                    ndim=2,
                    negative_indices=False,
                    mode='c'] l2_vals = np.random.randint(-lmax/dl, lmax/dl, (n_samples, 2))*dl
    
    #for l1x from -lmax <= l1x < lmax by dl:
    for j in prange(n_samples, nogil = True):
        l1x = l1_vals[j,0]
        l1y = l1_vals[j,1]
        
        l2x = l2_vals[j,0]
        l2y = l2_vals[j,1]

        modl1 = <int> ((l1x*l1x+l1y*l1y)**(0.5))
        modLminusl1 = <int> (((L-l1x)*(L-l1x) + l1y*l1y)**(0.5))
        
        if (modl1 < lmin or modl1 > lmax):
            continue
        elif (modLminusl1 < lmin or modLminusl1 > lmax):
            continue
        else:
            modl2 = <int> ((l2x*l2x+l2y*l2y)**(0.5))
            modLminusl2 = <int> (((L-l2x)*(L-l2x) + l2y*l2y)**(0.5))
            if (modl2 < lmin or modl2 > lmax):
                continue
            elif (modLminusl2 < lmin or modLminusl2 > lmax):
                continue
            else:
                Ldotl1 = l1x*L
                LdotLminusl1 = L*(L-l1x)
                Ldotl2 = l2x*L
                LdotLminusl2 = L*(L-l2x)

                modl1minusLplusl2 = <int> (((l1x-L+l2x)*(l1x-L+l2x) + (l1y+l2y)*(l1y+l2y))**(0.5))
                modl1minusl2 = <int> (((l1x-l2x)*(l1x-l2x) + (l1y-l2y)*(l1y-l2y))**(0.5))
                #l1dotl1minusl2 = l1x*(l1x-l2x)+l1y*(l1y-l2y)
                #Lminusl1dotl1minusl2 = (L-l1x)*(l1x-l2x) + (-l1y)*(l1y-l2y)
                #l1dotl1minusLplusl2 = l1x*(l1x-L+l2x) + l1y*(l1y+l2y)
                #l2dotl1minusLplusl2 = l2x*(l1x-L+l2x) + l2y*(l1y+l2y)
                
                l1dotl2minusl1 = l1x*(l1x-l2x)+l1y*(l1y-l2y)
                l2dotl2minusl1 = l2x*(l2x-l1x)+l2y*(l2y-l1y)
                Lminusl1dotl1minusl2 = (L-l1x)*(l1x-l2x) + (-l1y)*(l1y-l2y)
                Lminusl2dotl1minusl2 = (L-l2x)*(l1x-l2x) + (-l2y)*(l1y-l2y)
                Lminusl1minusl2dotl1 = (L-l1x-l2x)*l1x + (-l1y-l2y)*l1y
                Lminusl1minusl2dotLminusl2 = (L-l1x-l2x)*(L-l2x) + (-l1y-l2y)*(-l2y)
                l1minusLplusl2dotLminusl1 = (l1x-L+l2x)*(L-l1x) + (l1y+l2y)*(-l1y)
                l1minusLplusl2dotl2 = (l1x-L+l2x)*l2x + (l1y+l2y)*l2y

                integral += 1. \
                    * ( Ldotl1*ClTTunlensed[modl1] + LdotLminusl1*ClTTunlensed[modLminusl1] ) \
                    * 1./(2. * ClTTtotal[modl1] * ClTTtotal[modLminusl1]) \
                    * ( Ldotl2*ClTTunlensed[modl2] + LdotLminusl2*ClTTunlensed[modLminusl2] ) \
                    * 1./(2. * ClTTtotal[modl2] * ClTTtotal[modLminusl2]) \
                    * ( Clphiphi[modl1minusl2] \
                    * ( -l1dotl2minusl1*ClTTunlensed[modl1] + l2dotl2minusl1*ClTTunlensed[modl2] ) \
                    * ( -Lminusl1dotl1minusl2*ClTTunlensed[modLminusl1] + Lminusl2dotl1minusl2*ClTTunlensed[modLminusl2] ) \
                    + Clphiphi[modl1minusLplusl2] \
                    * ( -Lminusl1minusl2dotl1*ClTTunlensed[modl1] + Lminusl1minusl2dotLminusl2*ClTTunlensed[modLminusl2] ) \
                    * ( -l1minusLplusl2dotLminusl1*ClTTunlensed[modLminusl1] + l1minusLplusl2dotl2*ClTTunlensed[modl2] ) )
    
    integral *= 1.*(2.*lmax)*(2.*lmax)*(2.*lmax)*(2.*lmax)/n_samples
    
    return integral