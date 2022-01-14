#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:40:51 2021

@author: StRoSato
"""

import autograd.numpy as np
import LieGroupsSO3 as so3

def hat(x):
    return _aux_hom_matrix(so3.hat(x[0:3]), x[3:6], 0)

def check(x):
    return np.hstack((so3.check(x[0:3,0:3]), x[0:3,3]))

def inv(x):
    return _aux_hom_matrix(x[0:3,0:3].transpose(), -x[0:3,0:3].transpose()@x[0:3,3], 1)

def cay(x):
    X = hat(x)
    XX = X@X
    M2 = -(XX).trace()/2
    return np.eye(4) + 2*X + 2/(1 + M2)*XX@(np.eye(4) + X)

def inv_cay(x):
    Y0 = so3.inv_cay(x[0:3,0:3])
    Y1 = (np.eye(3) - so3.hat(Y0))@x[0:3,3]/2
    return np.hstack((Y0, Y1))

def exp(x, **kwargs):
    if 'theta' in kwargs:
        theta = kwargs['theta']
    else:
        theta = np.linalg.norm(x[0:3])
    return _aux_hom_matrix(so3.exp(x[0:3], theta=theta), so3.dexp_R(x[0:3], theta=theta)@x[3:6], 1)

def log(x):
    temp = so3.log(x[0:3,0:3])
    return np.hstack((temp,so3.inv_dexp_R(temp)@x[0:3,3]))

def left_trans(x):
    return _aux_alg_matrix(so3.left_trans(x[0:3]),np.zeros(3,3),so3.group_element(x[0:3]))

def right_trans(x):
    return _aux_alg_matrix(so3.right_trans(x[0:3]), -so3.hat(x[3:6]), np.eye(3))

def inv_left_trans(x):
    return _aux_alg_matrix(so3.inv_left_trans(x[0:3]),np.zeros(3,3),so3.inv(so3.group_element(x[0:3])))

def inv_right_trans(x):
    Z = so3.inv_right_trans(x[0:3])
    return _aux_alg_matrix(Z, so3.hat(x[3:6])@Z, np.eye(3))

def dleft_trans(x,y):
    return _aux_alg_matrix(so3.dleft_trans(x[0:3],y[0:3]), -so3.group_element(x[0:3])@so3.hat(y[3:6]), np.zeros((3,3)))

def dright_trans(x,y):
    return _aux_alg_matrix(so3.dright_trans(x[0:3],y[0:3]), -so3.hat(y[0:3])@so3.hat(x[3:6]), so3.hat(y[0:3]))

def Ad(x):
    return _aux_tangent_matrix(x[0:3,0:3], so3.hat(x[0:3,3])@x[0:3,0:3])

def ad_alg(x):
    return _aux_tangent_matrix(so3.hat(x[0:3]), so3.hat(x[3:6]))

def dexp_R(x, **kwargs):
    if 'theta' in kwargs:
        theta = kwargs['theta']
    else:
        theta = np.linalg.norm(x[0:3])
    Y = so3.hat(x[0:3])
    Z = so3.hat(x[3:6])
    YY = Y@Y
    if theta > 1e-4:
        W = (Z/2 + (theta - np.sin(theta))/(theta**3)*(Y@Z + Z@Y) +
            (np.cos(theta) + (theta**2)/2 - 1)/(theta**4)*(YY@Z + Z@YY) +
            (1 - np.cos(theta) - theta*np.sin(theta)/2)/(theta**4)*Y@Z@Y +
            (theta - 3*np.sin(theta)/2 + theta*np.cos(theta)/2)/(theta**5)*(YY@Z@Y + Y@Z@YY) +
            ((theta**2)/2 + theta*np.sin(theta)/2 + 2*np.cos(theta) - 2)/(theta**6)*YY@Z@YY)
    else:
        W = (Z/2 + (1/6 - 1/120*(theta**2))*(Y@Z + Z@Y) +
            (1/24 - 1/720*(theta**2))*(YY@Z + Z@YY) +
            (1/24 - 1/360*(theta**2))*Y@Z@Y +
            (1/120 - 1/2520*(theta**2))*(YY@Z@Y + Y@Z@YY) +
            (1/720 - 1/20160*(theta**2))*YY@Z@YY)
    return _aux_tangent_matrix(so3.dexp_R(x[0:3], theta=theta), W)
    
def dexp_L(x, **kwargs):
    return dexp_R(-np.array(x), **kwargs)

def inv_dexp_R(x, **kwargs):
    if 'theta' in kwargs:
        theta = kwargs['theta']
    else:
        theta = np.linalg.norm(x[0:3])
    Y = so3.hat(x[0:3])
    Z = so3.hat(x[3:6])
    YY = Y@Y
    if theta > 1e-4:
        W = (- Z/2 + (theta*np.sin(theta)/2 + np.cos(theta) - 1)/((theta**2)*(np.cos(theta) - 1))*(Y@Z + Z@Y) +
            ((theta**2)/4 + theta*np.sin(theta)/4 + np.cos(theta) - 1)/((theta**4)*(np.cos(theta) - 1))*(YY@Z@Y + Y@Z@YY))
    else:
        W = (- Z/2 + (1/12 + 1/720*(theta**2))*(Y@Z + Z@Y) +
            (-1/720 - 1/15120*(theta**2))*(YY@Z@Y + Y@Z@YY))
    return _aux_tangent_matrix(so3.inv_dexp_R(x[0:3], theta=theta), W)
    
def inv_dexp_L(x, **kwargs):
    return inv_dexp_R(-np.array(x), **kwargs)

def dcay_R(x):
    X = so3.hat(x[0:3])
    V = so3.hat(x[3:6])
    M2 = -(X@X).trace()/2
    temp = np.eye(3) + X
    return _aux_alg_matrix(temp, V@temp, np.eye(3)*(1 + M2) + X@temp)*2/(1 + M2)
    
def dcay_L(x):
    return dcay_R(-np.array(x))

def inv_dcay_R(x):
    X = so3.hat(x[0:3])
    V = so3.hat(x[3:6])
    M2 = -(X@X).trace()/2
    temp = np.eye(3) - X
    return _aux_alg_matrix(np.eye(3)*(1 + M2) - temp@X, -temp@V, temp)/2
    
def inv_dcay_L(x):
    return inv_dcay_R(-np.array(x))

def ddexp_R(x, y, **kwargs):
    if 'theta' in kwargs:
        theta = kwargs['theta']
    else:
        theta = np.linalg.norm(x[0:3])
    o = np.array(x[0:3])
    v = np.array(x[3:6])
    s = np.array(y[0:3])
    w = np.array(y[3:6])
    O = so3.hat(o)
    V = so3.hat(v)
    S = so3.hat(s)
    OO = O@O
    OV = O@V
    VO = V@O
    OOV = OO@V
    VOO = V@OO
    OVO = O@V@O
    OOVO = OOV@O
    OVOO = O@VOO
    
    termA1 = (OV + VO)
    termA2 = (OOV + VOO)
    termA3 = OVO
    termA4 = (OOVO + OVOO)
    termA5 = O@OVOO
    termB1 = - so3.hat(V@s) - V@S
    termB2 = - so3.hat(OV@s) - O@so3.hat(V@s) - V@so3.hat(O@s) - VO@S
    termB3 = - so3.hat(VO@s) - OV@S
    termB4 = - so3.hat(OVO@s) - O@so3.hat(VO@s) - OOV@S - so3.hat(VOO@s) - OV@so3.hat(O@s) - OVO@S
    termB5 = - so3.hat(OVOO@s) - O@so3.hat(VOO@s) - OOV@so3.hat(O@s) - OOVO@S

    if theta > 1e-4:
        T1 = ((theta - np.sin(theta))/(theta**3)*termB1 +
            (np.cos(theta) + (theta**2)/2 - 1)/(theta**4)*termB2 +
            (1 - np.cos(theta) - theta*np.sin(theta)/2)/(theta**4)*termB3 +
            (theta - 3*np.sin(theta)/2 + theta*np.cos(theta)/2)/(theta**5)*termB4 +
            ((theta**2)/2 + theta*np.sin(theta)/2 + 2*np.cos(theta) - 2)/(theta**6)*termB5)
        T2 = ((3*np.sin(theta) - theta*(np.cos(theta) + 2))/(theta**5)*termA1 +
            (4*(1 - np.cos(theta)) - theta*(np.sin(theta) + theta))/(theta**6)*termA2 +
            (5*theta*np.sin(theta) - 8 - ((theta**2) - 8)*np.cos(theta))/(2*(theta**6))*termA3 +
            ((15 - (theta**2))*np.sin(theta) - (7*np.cos(theta) + 8)*theta)/(2*(theta**7))*termA4 +
            ((np.cos(theta) - 4)*(theta**2) - 9*theta*np.sin(theta) + 24*(1 - np.cos(theta)))/(2*(theta**8))*termA5)
    else:
        T1 = ((1/6 - 1/120*(theta**2))*termB1 +
            (1/24 - 1/720*(theta**2))*termB2 +
            (1/24 - 1/360*(theta**2))*termB3 +
            (1/120 - 1/2520*(theta**2))*termB4 +
            (1/720 - 1/20160*(theta**2))*termB5)
        T2 = ((-1/60*theta + 1/1260*(theta**3))*termA1 +
            (-1/360*theta + 1/10080*(theta**3))*termA2 +
            (-1/180*theta + 1/3360*(theta**3))*termA3 +
            (-1/1260*theta + 1/30240*(theta**3))*termA4 +
            (-1/10080*theta + 1/302400*(theta**3))*termA5)
    T = T1 + T2@np.kron(s[:,np.newaxis],o)
    return _aux_tangent_matrix(so3.ddexp_R(o, s, theta=theta), T + so3.ddexp_R(o, w, theta=theta))

def ddexp_L(x, y, **kwargs):
    return -ddexp_R(-np.array(x),-np.array(y), **kwargs)

def ddcay_R(x,y):
    o = x[0:3]
    v = x[3:6]
    s = y[0:3]
    w = y[3:6]
    O = so3.hat(o)
    V = so3.hat(v)
    W = so3.hat(w)
    S = so3.hat(s)
    M2 = -(O@O).trace()/2
    TW = np.hstack((np.zeros(3),w))
    TO = np.hstack((o,np.zeros(3)))
    TT = dcay_R(x)@y - 2*TW
    Z1 = np.kron(TT[:,np.newaxis],TO)
    Z2 = _aux_alg_matrix(S, V@S + W + so3.hat(O@w) + O@W, so3.hat((np.eye(3) + O)@s))
    return -2*(Z1 + Z2)/(1+M2)
    
def ddcay_L(x,y):
    return -ddcay_R(-np.array(x),-np.array(y))

def _aux_hom_matrix(A,B,c):
    return np.vstack((np.hstack((          A,  B[:,np.newaxis])),
                      np.hstack((np.zeros(3),               c))))

def _aux_tangent_matrix(A,B):
    return _aux_alg_matrix(A,B,A)

def _aux_alg_matrix(A,B,C):
    return np.vstack((np.hstack((A, np.zeros((3, 3)))),
                      np.hstack((B,               C))))