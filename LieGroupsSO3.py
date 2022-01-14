#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:40:51 2021

@author: StRoSato
"""

import autograd.numpy as np

def hat(x):
    return np.array([[     0, -x[2],  x[1]],
                     [  x[2],     0, -x[0]],
                     [ -x[1],  x[0],     0]])

def check(x):
    return np.array([x[2,1], x[0,2], x[1,0]])

def inv(x):
    return x.transpose()

def cay(x):
    X = hat(x/2)
    M2 = -(X@X).trace()/2
    return np.eye(3) + 2/(1 + M2)*X@(np.eye(3) + X)

def inv_cay(x):
    return 2*check(x - x.transpose())/(1+x.trace())

def exp(x, **kwargs):
    if 'theta' in kwargs:
        theta = kwargs['theta']
    else:
        theta = np.linalg.norm(x[0:3])
    X = hat(x)
    if theta > 1e-4:
        return np.eye(3) + np.sin(theta)/theta*X + (1-np.cos(theta))/(theta**2)*X@X
    else:
        return np.eye(3) + (1 - 1/6*theta**2)*X + (1/2 - 1/24*theta**2)*X@X

def log(x):
    y = check(x - x.transpose())/2
    theta = np.arccos((x.trace() - 1)/2)
    return theta/np.sin(theta)*y

def rotation_X(x):
    return np.array([[ 1,         0,          0],
                     [ 0, np.cos(x), -np.sin(x)],
                     [ 0, np.sin(x),  np.cos(x)]])

def rotation_Y(x):
    return np.array([[  np.cos(x), 0, np.sin(x)],
                     [          0, 1,         0],
                     [ -np.sin(x), 0, np.cos(x)]])

def rotation_Z(x):
    return np.array([[ np.cos(x), -np.sin(x), 0],
                     [ np.sin(x),  np.cos(x), 0],
                     [         0,          0, 1]])

def group_element(x, coords='tait_bryan'):
    if coords == 'tait_bryan':
        y = rotation_Z(x[2])@rotation_Y(x[1])@rotation_X(x[0])
    elif coords == 'euler' or coords == 'euler_ZXZ':
        y = rotation_Z(x[2])@rotation_X(x[1])@rotation_Z(x[0])
    elif coords == 'euler_ZYZ':
        y = rotation_Z(x[2])@rotation_Y(x[1])@rotation_Z(x[0])
    else:
        raise ValueError('Unknown type of coordinates')
    return y

def left_trans(x):
    sx = np.sin(x)
    cx = np.cos(x)
    return np.array([[ 1, (sx[0]*sx[1])/cx[1], (cx[0]*sx[1])/cx[1]],
                     [ 0,               cx[0],              -sx[0]],
                     [ 0,         sx[0]/cx[1],         cx[0]/cx[1]]])

def right_trans(x):
    sx = np.sin(x)
    cx = np.cos(x)
    return np.array([[         cx[2]/cx[1],         sx[2]/cx[1], 0],
                     [              -sx[2],               cx[2], 0],
                     [ (cx[2]*sx[1])/cx[1], (sx[1]*sx[2])/cx[1], 1]])

def inv_left_trans(x):
    sx = np.sin(x)
    cx = np.cos(x)
    return np.array([[ 1,      0,      -sx[1]],
                     [ 0,  cx[0], sx[0]*cx[1]],
                     [ 0, -sx[0], cx[0]*cx[1]]])

def inv_right_trans(x):
    sx = np.sin(x)
    cx = np.cos(x)
    return np.array([[ cx[2]*cx[1], -sx[2], 0],
                     [ sx[2]*cx[1],  cx[2], 0],
                     [      -sx[1],      0, 1]])

def dleft_trans(x,y):
    sx0, cx0, tx1 = [np.sin(x[0]), np.cos(x[0]), np.tan(x[1])]
    z = np.array([[     0,                 cx0*(y[2]*cx0 + y[1]*sx0),                 -sx0*(y[2]*cx0 + y[1]*sx0)],
                  [ -y[2],          sx0**2*tx1*(y[1]*cx0 - y[2]*sx0), -tx1*(y[1]*sx0**3 + y[2]*cx0*(sx0**2 + 0))],
                  [  y[1], tx1*(y[2]*cx0**3 + y[1]*sx0*(cx0**2 + 0)),          cx0**2*tx1*(y[1]*cx0 - y[2]*sx0)]])
    return left_trans(x)@z

def dright_trans(x,y):
    sx2, cx2, tx1 = [np.sin(x[2]), np.cos(x[2]), np.tan(x[1])]
    z = np.array([[           cx2**2*tx1*(y[1]*cx2 - y[0]*sx2), tx1*(y[0]*cx2**3 + y[1]*sx2*(cx2**2 + 0)),  y[1]],
                  [ -tx1*(y[1]*sx2**3 + y[0]*cx2*(sx2**2 + 0)),          sx2**2*tx1*(y[1]*cx2 - y[0]*sx2), -y[0]],
                  [                 -sx2*(y[0]*cx2 + y[1]*sx2),                 cx2*(y[0]*cx2 + y[1]*sx2),     0]])
    return right_trans(x)@z

def Ad(x):
    return x[0:3,0:3]

def ad_alg(x):
    return hat(x)

def dexp_R(x, **kwargs):
    if 'theta' in kwargs:
        theta = kwargs['theta']
    else:
        theta = np.linalg.norm(x[0:3])
    X = hat(x)
    if theta > 1e-4:
        return np.eye(3) + (1-np.cos(theta))/(theta**2)*X + (theta-np.sin(theta))/(theta**3)*X@X
    else:
        return np.eye(3) + (1/2 - 1/24*theta**2)*X + (1/6 - 1/120*theta**2)*X@X
    
def dexp_L(x, **kwargs):
    return dexp_R(-x, **kwargs)

def inv_dexp_R(x, **kwargs):
    if 'theta' in kwargs:
        theta = kwargs['theta']
    else:
        theta = np.linalg.norm(x[0:3])
    X = hat(x)
    if theta > 1e-4:
        return np.eye(3) - X/2 + (theta*np.sin(theta)/2 + np.cos(theta) - 1)/((theta**2)*(np.cos(theta) - 1))*X@X
    else:
        return np.eye(3) - X/2 + (1/12 + 1/720*theta**2)*X@X
    
def inv_dexp_L(x, **kwargs):
    return inv_dexp_R(-x, **kwargs)

def dcay_R(x):
    X = hat(x/2)
    M2 = -(X@X).trace()/2
    return (np.eye(3) + X)/(1 + M2)
    
def dcay_L(x):
    return dcay_R(-x)

def inv_dcay_R(x):
    X = hat(x/2)
    M2 = -(X@X).trace()/2
    return np.eye(3)*(1 + M2) - (np.eye(3) - X)@X
    
def inv_dcay_L(x):
    return inv_dcay_R(-x)

def ddexp_R(x, y, **kwargs):
    if 'theta' in kwargs:
        theta = kwargs['theta']
    else:
        theta = np.linalg.norm(x[0:3])
    o = np.array(x[0:3])
    s = np.array(y[0:3])
    O = hat(o)
    S = hat(s)
    OO = O@O
    
    termA1 = -S
    termA2 = - hat(O@s) - O@S
    termB1 = O
    termB2 = OO    

    if theta > 1e-4:
        Z1 = ((1-np.cos(theta))/theta**2*termA1 +
             (theta-np.sin(theta))/theta**3*termA2)
        Z2 = ((2*(np.cos(theta) - 1) + theta*np.sin(theta))/theta**4*termB1 +
             (3*np.sin(theta) - theta*(np.cos(theta) + 2))/theta**5*termB2)
    else:
        Z1 = ((1/2 - 1/24*theta**2)*termA1 +
             (1/6 - 1/120*theta**2)*termA2)
        Z2 = - ((1/12*theta - 1/180*theta**3)*termB1 +
             - (1/60*theta - 1/1260*theta**3)*termB2)
    return Z1 + Z2@np.kron(s[:,np.newaxis],o)

def ddexp_L(x, y, **kwargs):
    return -ddexp_R(-x, y, **kwargs)

def ddcay_R(x,y):
    X = hat(x/2)
    Y = hat(y/2)
    M2 = -(X@X).trace()/2
    return -1/(1 + M2)**2*( (1 + M2)*Y + (np.eye(3) + X)@np.kron(y[:,np.newaxis],x)/2 )
    
def ddcay_L(x,y):
    return -ddcay_R(-x,y)

def group_coords(x, coords='tait_bryan', is_inverted=False):
    if coords == 'tait_bryan':
        if (x[2,0] != 1) and (x[2,0] != -1):
            theta = -np.arcsin(x[2,0])
            if is_inverted:
                theta = np.pi - theta
            psi = np.arctan2(x[2,1]/np.cos(theta), x[2,2]/np.cos(theta))
            phi = np.arctan2(x[1,0]/np.cos(theta), x[0,0]/np.cos(theta))
        else:
            phi = 0;
            psi = np.arctan2(-x[1,2], x[1,1])
            if x[2,0] == -1:
                theta = np.pi/2
                # psi = phi + np.arctan2(-x[1,2], x[1,1])
            else:
                theta = -np.pi/2;
                # psi = -phi + np.arctan2(-x[1,2], x[1,1])
    elif coords == 'euler' or coords == 'euler_ZXZ':
        if (x[2,2] != 1) and (x[2,2] != -1):
            theta = np.arccos(x[2,0])
            if is_inverted:
                theta = - theta
            psi = np.arctan2(x[2,0]/np.sin(theta), x[2,1]/np.sin(theta))
            phi = np.arctan2(x[0,2]/np.sin(theta), -x[1,2]/np.sin(theta))
        else:
            phi = 0;
            psi = np.arctan2(-x[0,1], x[0,0])
            if x[2,0] == -1:
                theta = np.pi
                # psi = phi + np.arctan2(-x[0,1], x[0,0])
            else:
                theta = 0;
                # psi = -phi + np.arctan2(-x[0,1], x[0,0])
    elif coords == 'euler_ZYZ':
        if (x[2,2] != 1) and (x[2,2] != -1):
            theta = np.arccos(x[2,0])
            if is_inverted:
                theta = - theta
            psi = np.arctan2(x[2,1]/np.sin(theta), -x[2,0]/np.sin(theta))
            phi = np.arctan2(x[1,2]/np.sin(theta), x[0,2]/np.sin(theta))
        else:
            phi = 0;
            psi = np.arctan2(x[1,0], x[1,1])
            if x[2,0] == -1:
                theta = np.pi
                # psi = phi + np.arctan2(x[1,0], x[1,1])
            else:
                theta = 0;
                # psi = -phi + np.arctan2(x[1,0], x[1,1])
    else:
        raise ValueError('Unknown type of coordinates')
    return np.array([ psi, theta, phi])

def tangent_dexp_R(x, y, **kwargs):
    if 'theta' in kwargs:
        theta = kwargs['theta']
    else:
        theta = np.linalg.norm(x[0:3])
    return _aux_tangent_matrix(dexp_R(x, theta=theta),ddexp_R(x, y, theta=theta))

def tangent_dexp_L(x, y, **kwargs):
    if 'theta' in kwargs:
        theta = kwargs['theta']
    else:
        theta = np.linalg.norm(x[0:3])
    return _aux_tangent_matrix(dexp_L(x, theta=theta),ddexp_L(x, y, theta=theta))

def tangent_inv_dexp_R(x, y, **kwargs):
    if 'theta' in kwargs:
        theta = kwargs['theta']
    else:
        theta = np.linalg.norm(x[0:3])
    A = inv_dexp_R(x, theta=theta)
    return _aux_tangent_matrix(A,-A@ddexp_R(x, y, theta=theta)@A)

def tangent_inv_dexp_L(x, y, **kwargs):
    if 'theta' in kwargs:
        theta = kwargs['theta']
    else:
        theta = np.linalg.norm(x[0:3])
    A = inv_dexp_L(x, theta=theta)
    return _aux_tangent_matrix(A,-A@ddexp_L(x, y, theta=theta)@A)

def tangent_dcay_R(x, y):
    return _aux_tangent_matrix(dcay_R(x),ddexp_R(x, y))

def tangent_dcay_L(x, y):
    return _aux_tangent_matrix(dcay_L(x),ddcay_L(x, y))

def tangent_inv_dcay_R(x, y):
    A = inv_dcay_R(x)
    return _aux_tangent_matrix(A,-A@ddcay_R(x, y)@A)

def tangent_inv_dcay_L(x, y):
    A = inv_dcay_L(x)
    return _aux_tangent_matrix(A,-A@ddcay_L(x, y)@A)

def tangent_right_trans(x, y):
    return _aux_tangent_matrix(right_trans(x),dright_trans(x,y))

def tangent_left_trans(x, y):
    return _aux_tangent_matrix(left_trans(x),dleft_trans(x,y))

# autograd does not support np.block
# def _aux_tangent_matrix(A,B):
#     return np.block([[A, np.zeros((3, 3))],
#                      [B,                A]])

def _aux_tangent_matrix(A,B):
    return np.vstack((np.hstack((A, np.zeros((3, 3)))),
                      np.hstack((B,               A))))