import numpy as np

cdef double cabs(double complex z):
    return (z.real*z.real + z.imag*z.imag)

def getJuliaEscapeTime(double complex c, double complex z):
    cdef int i
    for i in range(128):
        if cabs(z) > 4:
            return i
        z = z**2 + c
    return i

def getMandelbrotEscapeTime(complex c):
    cdef complex z = 0
    cdef int i
    for i in range(128):
        if cabs(z) > 4:
            return i
        z = z**2 + c
    return i

def getBurningShipEscapeTime(complex c):
    cdef complex z = 0
    cdef int i
    for i in range(128):
        if cabs(z) > 4:
            return i
        z = (abs(z.real) + abs(z.imag)*1j)**2 + c
    return i

def getMultibrotEscapeTime(complex c, int n):
    cdef complex z = 0
    cdef int i
    for i in range(128):
        if cabs(z) > 4:
            return i
        z = z**n + c
    return i
