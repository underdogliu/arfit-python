#!/usr/bin/env python3
# Stepwise least squares estimation of multivariate AR model
#
# This is a python intepretation of ARfit toolkit:
#   https://github.com/tapios/arfit
# 
# This script so far only works for 1-dimensional time series
#   so for multi-variate data it is recommeneded to check on
#   MATLAB script. I will update those accordingly in near future.

import numpy as np
import numpy.linalg

def arfit_frames(data, pmin=1, pmax=40, selector='sbc'):
    '''acquire in put data matrix and call arfit for each frame
    @data: input array in shape [num_frames, frame_len]
    @pmin: minimal AR order to select, default 1
    @pmax: maximum AR order to select, default 40
    @selector: order selection criterion
    @no_const: decision on no intercept vector to be fitted

    @return: an integer of selected order
    '''
    num_frames = data.shape[0]
    ar_orders = np.zeros((num_frames,))
    for i in range(num_frames):
        ar_orders[i] = arfit(data[i,:], pmin=pmin, pmax=pmax, 
                             selector=selector)
    return ar_orders


def arfit(data, pmin=1, pmax=40, selector='sbc'):
    '''perform auto-regeressive order selection/estimation
    @data: input array in shape [frame_len, ]
    @pmin: minimal AR order to select, default 1
    @pmax: maximum AR order to select, default 40
    @selector: order selection criterion
    @no_const: decision on no intercept vector to be fitted

    @return: an integer of selected order
    '''
    n = data.shape[0]
    m = 1 # number of variables (dimension of state vectors)
    ntr = 1 # number of realizations (trials)

    mcor = 1 # num. intercept vectors

    assert pmin <= pmax
    assert n > pmax

    ne = ntr * (n-pmax)
    npmax = m * pmax + mcor

    # compute QR factorization for model of order pmax
    R, scale = arqr(data, pmax, mcor)

    # compute approximate order selection criteria for models 
    # of order pmin:pmax
    sbc, fpe = arord(R, m, mcor, ne, pmin, pmax)

    # get index iopt of order that minimizes the order selection 
    # criterion specified by the variable selector
    if selector == 'sbc':
        iopt = np.argmin(sbc)
    else:
        iopt = np.argmin(fpe)

    # select order of model
    popt = pmin + iopt # estimated optimum order 

    return popt


def arqr(data, p, mcor=0):
    '''perform AR-QR factorization for least squares estimation 
        of AR model.
    @data: input array in shape [num_frames, ]
    @p: AR order
    @mcor: number of intercept vectors, default 0

    @return: An upper triangular matrix R; A vector of scaling
        factors $scale, used to regularize the QR factorization.
    '''
    n = data.shape[0]
    m = 1 # number of variables (dimension of state vectors)
    ntr = 1 # number of realizations (trials)

    ne = ntr * (n - p) # number of block equations of size m
    np_vec = m * p + mcor # number of parameter vectors of size m

    # Initialize the data matrix K 
    # (of which a QR factorization will be computed)
    K = np.zeros((ne, np_vec+m))
    if mcor == 1:
        K[:, 0] = 1
    
    # assemble 'predictors' u in K
    for j in range(0, p):
        K[0:n-p, mcor+m*j] = data[p-(j+1):n-(j+1)]
    # Add `observations' v (left hand side of regression model) to K
    K[0:n-p, np_vec] = data[p:n]

    # Compute regularized QR factorization of K: The regularization
    # parameter delta is chosen according to Higham's (1996) Theorem
    # 10.7 on the stability of a Cholesky factorization. Replace the
    # regularization parameter delta below by a parameter that depends
    # on the observational error if the observational error dominates
    # the rounding error (cf. Neumaier, A. and T. Schneider, 2001:
    # "Estimation of parameters and eigenmodes of multivariate
    # autoregressive models", ACM Trans. Math. Softw., 27, 27--57.).
    q = np_vec + m              # number of columns of K
    delta = (q**2 + q + 1) * np.finfo(float).eps  # Higham's choice for a Cholesky factorization
    scale = np.sqrt(delta) * np.sqrt(np.sum(K**2, axis=0))
    R = np.triu(numpy.linalg.qr(np.concatenate([K, np.diag(scale)], axis=0))[1])
    return R, scale


def arord(R, m, mcor, ne, pmin, pmax):
    '''Evaluates criteria for selecting the order of an AR model.
    It returns approximate values of the order selection criteria 
        SBC and FPE for models of order pmin:pmax.
    @R: An upper triangular matrix
    @m: dimension of state vectors
    @mcor: a flag indicates whether or not an intercept vector
        is being fitted
    @ne: number of block equations of size m
    @pmin: minimum order
    @pmax: maximum order
    @return: SBC and FPE estimation results, in vectors
    '''
    imax = pmax - pmin + 1 # maximum index of output vectors

    sbc = np.zeros((imax,)).T
    fpe = np.zeros((imax,)).T
    logdp = np.zeros((imax,)).T
    np_vec = np.zeros((imax,), dtype=np.int).T
    np_vec[imax-1] = int(m * pmax + mcor)

    # Get lower right triangle R22 of R: 
    #     | R11  R12 |
    # R=  |          |
    #     | 0    R22 |
    R22 = R[np_vec[imax-1]-1:np_vec[imax-1]+m-1, np_vec[imax-1]-1:np_vec[imax-1]+m-1]
    # get inverse of residual cross-product matrix for model
    # of order pmax
    invR22 = numpy.linalg.inv(R22)
    Mp = invR22 * invR22.T

    # For order selection, get determinant of residual cross-product matrix
    #   logdp = log det(residual cross-product matrix)
    logdp[imax-1] = 2 * np.log(np.abs(R22))

    # Compute approximate order selection criteria for models of 
    #   order pmin:pmax
    i = imax - 1
    for p in reversed(range(pmin, pmax)):
        np_vec[i-1] = m * p + mcor
        if p < pmax:
            # Downdate determinant of residual cross-product matrix
            # Rp: Part of R to be added to Cholesky factor of covariance matrix
            Rp = R[np_vec[i-1]:np_vec[i-1]+m, np_vec[imax-1]:np_vec[imax-1]+m]

            # Get Mp, the downdated inverse of the residual cross-product
            # matrix, using the Woodbury formula
            L = numpy.linalg.cholesky(np.eye(m) + Rp * Mp * Rp.T)
            # note: here since we only deal with 1-dimensional data,
            # in order to solve LN = Rp*Mp (where in matlab we call 'L \ Rp*Mp'),
            # we simply perform division to get value of N
            N = Rp * Mp / L
            Mp -= N**2

            # get downgraded logarithm of determinant
            logdp[i-1] = logdp[i] + 2 * np.log(np.abs(L))
    
        # SBC
        sbc[i-1] = logdp[i-1] / m - np.log(ne) * (ne-np_vec[i-1]) / ne

        # FPE
        fpe[i-1] = logdp[i-1] / m - np.log(ne * (ne-np_vec[i-1]) / (ne+np_vec[i-1]))
        
        i -= 1
    
    return sbc, fpe
