#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 13:14:53 2018

@author: achim
"""

from __future__ import print_function, division
from statsmodels.compat.python import range

import numpy as np
import numpy.linalg as npl
from numpy.linalg import slogdet
from scipy.linalg import solve

from statsmodels.tools.numdiff import (approx_hess, approx_fprime)
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.tsa.vector_ar.var_model import VARProcess, \
                                                        VARResults

import statsmodels.tsa.vector_ar.util as util
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.compat.numpy import np_matrix_rank

from datetime import datetime
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
import scipy.optimize as sopt 

mat = np.array

class SVAR_CV(tsbase.TimeSeriesModel):
    def __init__(self, endog, SB, start =None, end = None, freq = None,\
                 format = None, date_vector = None, max_iter = 50, crit = 0.001,\
                 restriction_matrix = None, missing=None):

        '''super(SVAR_CV, self).__init__(endog, None, dates, freq, missing=missing) 
        #(self.endog, self.names,
        # self.dates) = data_util.interpret_data(endog, names, dates)

        self.y = self.endog #keep alias for now
        self.neqs = self.endog.shape[1]
        self.SB = SB

        #initialize A, B as I if not given
        #Initialize SVAR masks
        A = np.identity(self.neqs)
        self.A_mask = A_mask = np.zeros(A.shape, dtype=bool)

        B = np.identity(self.neqs)
        self.B_mask = B_mask = np.zeros(B.shape, dtype=bool)

        # convert A and B to numeric
        #TODO: change this when masked support is better or with formula
        #integration
        Anum = np.zeros(A.shape, dtype=float)
        Anum[~A_mask] = A[~A_mask]
        Anum[A_mask] = np.nan
        self.A = Anum

        Bnum = np.zeros(B.shape, dtype=float)
        Bnum[~B_mask] = B[~B_mask]
        Bnum[B_mask] = np.nan
        self.B = Bnum

        #LikelihoodModel.__init__(self, endog)

        #super(SVAR, self).__init__(endog)
        '''
        self._get_var_object(endog)
        if(SB.isnumeric()):
            self.SB_character = None
        else:
            self.SB_character = SB
            self.SB = self._get_structural_break(SB=SB, start=start, end=end, \
                freq=freq, format=format, date_vector = date_vector)
        
        TB = self.SB - self.p

        resid1 = self.u[1:TB-1,]
        resid2 = self.u[TB:self.Tob,]
        sigma_hat1 = np.cross(resid1, resid1) / (TB-1)
        sigma_hat2 = np.cross(resid2, resid2) / (self.Tob-TB+1)

        if(restriction_matrix != None):
            result_unrestricted = self._identify_volatility(self.endog ,self.SB, u=self.u, k=self.k, y=None, restriction_matrix=restriction_matrix, sigma_hat1=sigma_hat1,\
        sigma_hat2=sigma_hat2, p=self.p, TB=TB, crit=crit, y_out=None, type=type)
        #TODO: Skript zum Testen der Klasse schreiben!!!


    def fit(self, A_guess=None, B_guess=None, maxlags=None, method='ols',
            ic=None, trend='c', verbose=False, s_method='mle',
            solver="bfgs", override=False, maxiter=500, maxfun=500):
        pass
    def _get_var_object(self, endog):
        """Get VAR object from Class VARResultsWrapper"""
        if(type(endog) != VARResultsWrapper):
            raise ValueError("Input have to be of type VARResult")
        self.u = endog.resid
        self.Tob = endog.resid.shape[0]
        self.k = endog.resid.shape[1]
        self.residY = endog.resid
        self.p = endog.k_ar
        self.y = endog.endog.T
        self.yOut = endog.endog
        self.type = endog.trend        
        self.coef_x = endog.coefs

    def _get_structural_break(self, SB, date_vector=None, start=None, end=None, \
        freq=None, format=None):
        if(format==None):
            raise ValueError("Format is missing with no default")
        
        if(date_vector==None and start==None and end==None):
            raise ValueError("Please provide either a valid number of observation or proper date \
                specifications")
        
        if(date_vector==None):
            date_vector = np.arange(start=datetime.strptime(start, format).date(), stop=datetime.strptime(end, format).date(), step=freq,dtype="datetime64")
        
        if((self.Tob + self.p) != len(date_vector)):
            raise ValueError("Date Vector and data have different length")
        
        return list(np.where([datetime.strptime(SB) in i for i in date_vector])[0])
    
    def _identify_volatility(self, x, SB, u, k, y, restriction_matrix, sigma_hat1,\
        sigma_hat2, p, TB, crit, y_out, type):
        if(restriction_matrix != None):
            B = np.transpose(np.multiply(np.linalg.cholesky(1/self.Tob), np.cross(u,u)))
            na_elements = np.isnan(restriction_matrix)
            B = B[na_elements]
            restrictions = len(restriction_matrix[not np.isnan(restriction_matrix)])
        else:
            B = np.transpose(np.linalg.cholesky(1/self.Tob)*np.cross(u,u))
            B = [B]
        
        lambda_ = np.ones(k)
        S = [B, lambda_]

        MLE = sopt.minimize(fun=self._likelihood, p=S, args=(k,TB, sigma_hat1, sigma_hat2, self.Tob, restriction_matrix, restrictions),method='BFGS', jac=True, opt={"maxiter": 150}) 
        
        if restriction_matrix is not None:
            na_elements = np.isnan(restriction_matrix)
            B_hat = restriction_matrix
            B_hat[na_elements] = MLE.x[0:sum(na_elements)]
            lambda_hat = np.diag(MLE.x[sum(na_elements) + 1 : len(MLE.x)])
        else:
            B_hat = np.array(MLE.x[0:(k*k)]).reshape((-1,k))
            lambda_hat = np.diag(MLE.x[(k*k+1):(k*k+k)])
            restrictions = 0
        
        #TODO: Minimum von MLE herausfinden
        ll = MLE.min()

        yl = y

        # B_hat[na_elements] = #MLE$estimate

        return True
    
    def _likelihood(self, S, Tob, sigma_hat1, k, sigma_hat2, restriction_matrix, restrictions):
        if restriction_matrix is not None:
            #if restriction_matrix is not matrix:
            #    raise ValueError("Please provide a valide input matrix")
            na_elements = np.isnan(restriction_matrix)
            to_fill_matrix = restriction_matrix
            to_fill_matrix[na_elements] = S[:np.sum(na_elements)]
            W = to_fill_matrix
        else:
            W = np.asmatrix(S[:(k*k)].reshape((k,-1)))
            restrictions = 0
        
        Psi = np.diag(S[(k*k-restrictions):(k*k+k-restrictions)])

        MMM = np.cross(W,W.T)
        MMM2 = W @ np.cross(Psi,W.T) 
        MW = np.linalg.det(MMM)
        MW2 = np.linalg.det(MMM2)

        if any(Psi < 0 or MW < 0.01 or MW2 < 0.01): #Wieso hier any und or?
            return 1e25
        
        L = -(((TB-1) / 2) * (np.log(MW) + np.sum(np.diag((sigma_hat1 @ solve(MMM)))))) - 
            (((Tob - TB + 1) / 2) * (np.log(MW2) + np.sum(np.diag((sigma_hat2 @ solve(MMM2))))))

        return -L

    def _y_lag_cr(self, y, lag_length):
        y_lag = np.full((y.shape()[0], y.shape()[1] * lag_length), np.nan)
        for i in range(lag_length):
            y_lag[(1+i):y.shape()[0], (((i-1) * y.shape()[1]) + 1):(i*y.shape()[1])] = y[:(y.shape()[0] - i), :y.shape()[0]]
        
        y_lag_mask = np.ones(len(y_lag), dtype=bool)
        y_lag_mask[:lag_length] = False
        y_lag = np.asmatrix(y_lag[y_lag_mask,])
        out = {'lags':y_lag} #In R wird eine list zurückgegeben...

    def _get_init_params(self, A_guess, B_guess):
        pass
    def _estimate_svar(self, start_params, lags, maxiter, maxfun,
                       trend='c', solver="nm", override=False):
        pass
    def loglike(self, params):
        pass
    def score(self, AB_mask):
        pass
    def hessian(self, AB_mask):
        pass
    def _solve_AB(self, start_params, maxiter, maxfun, override=False,
            solver='bfgs'):
        pass
    def _compute_J(self, A_solve, B_solve):
        pass
    def check_order(self, J):
        if np.size(J, axis=0) < np.size(J, axis=1):
            raise ValueError("Order condition not met: "
                             "solution may not be unique")

    def check_rank(self, J):
        rank = np_matrix_rank(J)
        if rank < np.size(J, axis=1):
            raise ValueError("Rank condition not met: "
                             "solution may not be unique.")

class SVAR_CV_Process(VARProcess):
    def __init__(self, coefs, intercept, sigma_u, A_solve, B_solve,
    names=None):
        self.k_ar = len(coefs)
        self.neqs = coefs.shape[1]
        self.coefs = coefs
        self.intercept = intercept
        self.sigma_u = sigma_u
        self.A_solve = A_solve
        self.B_solve = B_solve
        self.names = names

    def orth_ma_rep(self, maxn=10, P=None):
        raise NotImplementedError

    def svar_ma_rep(self, maxn=10, P=None):
        pass

class SVAR_CV_Results(SVAR_CV_Process, VARResults):

    _model_type = 'SVAR_CV'

    def __init__(self, endog, endog_lagged, params, sigma_u, lag_order,
    A=None, B=None, A_mask=None, B_mask=None, model=None, trend='c',
    names=None, dates=None):
        self.model = model
        self.y = self.endog = endog
        self.ys_lagged = self.endog_lagged = endog_lagged
        self.dates = dates

        self.n_totobs, self.neqs = self.endog.shape
        self.nobs = self.n_totobs - lag_order
        k_trend = util.get_trendorder(trend)
        if k_trend > 0:
            trendorder = k_trend - 1
        else:
            trendorder = None
        self.k_trend = k_trend
        self.k_exog = k_trend
        self.trendorder = trendorder

        self.exog_names = util.make_lag_names(names, lag_order, k_trend)
        self.params = params
        self.sigma_u = sigma_u
        
        # Each matrix needs to be transposed
        reshaped = self.params[self.k_trend:]
        reshaped = reshaped.reshaped((lag_order, self.neqs, self.neqs))

        intercept = self.params[0]
        coefs = reshaped.swapaxes(1,2).copy()

        self.A = A
        self.B = B
        self.A_mask = A_mask
        self.B_mask = B_mask

        super(SVAR_CV_Results, self).__init__(coefs, intercept, sigma_u, A, B, names=names)

    def fevd(self):
        pass

    def irf(self):
        pass

    def logLik(self):
        pass

    def Phi(self):
        pass

    def print(self):
        pass

    def summary(self):
        pass
