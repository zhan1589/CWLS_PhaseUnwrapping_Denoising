#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Jiacheng Zhang, Email: zhan1589@purdue.edu

import numpy as np
import scipy.sparse as scysparse
import scipy.sparse.linalg as splinalg
import scipy.linalg as linalg
import sys
from scipy.spatial import distance

from NumericalLinearOperators import LinearOperatorGeneration

class PhaseUnwrappingWeightedLeastSquares4D():
  
  def __init__(self,Xn,Yn,Zn,Un,Vn,Wn,Venc_u,Venc_v,Venc_w,fluid_maskn,unwrapping_maskn):
    """
    # Perform phase unwrapping of 4D flow data using weighted least-squares algorithm
    # The weights are determined from the phase singularites and the divergence calculated from the wrapped/corrected phase difference. 
    # Although this module is named 4d, it actually take 3D inputs.
    # This module serves as a tool box for the 4D unwrapping function.
    Inputs:
      Xn,Yn,Zn: 3d mesh grids.
      Un,Vn,Wn: 3d fields of phase values (should be bounded within [-pi,pi]), shaped (Ny,Nx,Nz)
      Venc_u,Venc_v,Venc_w: the encoding velocity for each of the three velocity components, can be different from each other.
      fluid_maskn: 3d fluid mask, in the shape of (Ny,Nx,Nz)
      unwrapping_maskn: 3d mask for the region to perfrom unwrapping. This set of mask should be at least as large as the fluid_maskn.
        The addition region of unwrapping_maskn compared with fluid_maksn is employed as the reference points (zero flow, no wrapping).
    """
    
    self.Xn = Xn
    self.Yn = Yn
    self.Zn = Zn
    self.dx = self.Xn[0,1,0] - self.Xn[0,0,0]
    self.dy = self.Yn[1,0,0] - self.Yn[0,0,0]
    self.dz = self.Zn[0,0,1] - self.Zn[0,0,0]
    self.Phase = {}
    self.Phase['U'] = Un
    self.Phase['V'] = Vn
    self.Phase['W'] = Wn
    self.Vencs = {}
    self.Vencs['U'] = Venc_u
    self.Vencs['V'] = Venc_v
    self.Vencs['W'] = Venc_w
    self.fluid_maskn = fluid_maskn
    self.unwrapping_maskn = unwrapping_maskn
    self.Ny, self.Nx, self.Nz = np.shape(self.fluid_maskn)

    # Determine the mask for the phase differences along the 3 spatial axis for both maks sets.
    self.fluid_mask_Gx = np.logical_and(self.fluid_maskn[:,1:,:],self.fluid_maskn[:,:-1,:])
    self.fluid_mask_Gy = np.logical_and(self.fluid_maskn[1:,:,:],self.fluid_maskn[:-1,:,:])
    self.fluid_mask_Gz = np.logical_and(self.fluid_maskn[:,:,1:],self.fluid_maskn[:,:,:-1])
    self.unwrapping_mask_Gx = np.logical_and(self.unwrapping_maskn[:,1:,:],self.unwrapping_maskn[:,:-1,:])
    self.unwrapping_mask_Gy = np.logical_and(self.unwrapping_maskn[1:,:,:],self.unwrapping_maskn[:-1,:,:])
    self.unwrapping_mask_Gz = np.logical_and(self.unwrapping_maskn[:,:,1:],self.unwrapping_maskn[:,:,:-1])

    # Assigning zero values to the phase data outside of the fluid_maskn
    for vel_key in ['U','V','W']:
      self.Phase[vel_key][self.fluid_maskn==False] = 0.0
    
    # Define the unwrapped phases
    self.Phase_unwrapped = {}
    self.Phase_unwrapped['U'] = np.zeros(self.Phase['U'].shape)
    self.Phase_unwrapped['V'] = np.zeros(self.Phase['V'].shape)
    self.Phase_unwrapped['W'] = np.zeros(self.Phase['W'].shape)

    # Calculate the phase spatial phase differences
    self.Phase_diff_x = {}
    self.Phase_diff_y = {}
    self.Phase_diff_z = {}
    for vel_key in ['U','V','W']:
      self.Phase_diff_x[vel_key] = self.Phase[vel_key][:,1:,:] - self.Phase[vel_key][:,:-1,:]
      self.Phase_diff_y[vel_key] = self.Phase[vel_key][1:,:,:] - self.Phase[vel_key][:-1,:,:]
      self.Phase_diff_z[vel_key] = self.Phase[vel_key][:,:,1:] - self.Phase[vel_key][:,:,:-1]

    # Prepare a mask of the additional reference values to be None. 
    # This would be updated if we have the addition reference values given to the system.
    self.mask_ref_values = None
    self.add_ref_values = None
    self.sigma_ref_values = None

    
  def additional_reference_points(self,ref_uvw,mask_ref_uvw,sigma_ref_uvw):
    # This fuction adds a set of reference points to the field. 
    # This will control the unwrapped values at the reference locations with a certain confidience.
    # Inputs: 
    # ref_uvw: 4D arrays with shape (Ny,Nx,Nz,3) for the additional reference values for the U, V, W phases. 
    # mask_ref_uvw: the masks (4D arrays) of the additional reference values, True means the values would be used.
    # sigma_ref_uvw: 4D arrays of the uncertainty (standard deviation) for each given reference value, should be positive.
    
    self.add_ref_values = {}
    self.add_ref_values['U'] = ref_uvw[:,:,:,0]
    self.add_ref_values['V'] = ref_uvw[:,:,:,1]
    self.add_ref_values['W'] = ref_uvw[:,:,:,2]
    self.mask_ref_values = {}
    self.mask_ref_values['U'] = mask_ref_uvw[:,:,:,0]
    self.mask_ref_values['V'] = mask_ref_uvw[:,:,:,1]
    self.mask_ref_values['W'] = mask_ref_uvw[:,:,:,2]
    self.sigma_ref_values = {}
    self.sigma_ref_values['U'] = sigma_ref_uvw[:,:,:,0]
    self.sigma_ref_values['V'] = sigma_ref_uvw[:,:,:,1]
    self.sigma_ref_values['W'] = sigma_ref_uvw[:,:,:,2]
    # Set the lower bound of the sigma_ref_values.
    min_ref_sigma = 1e-3
    for vel_key in ['U','V','W']:
      self.sigma_ref_values[vel_key][self.sigma_ref_values[vel_key] < min_ref_sigma] = min_ref_sigma
    

  def RMS_phase_difference(self):
    # Calculate the RMS of the phase differences within the unwrapping_maskn
    # This is employed when the peak diastole & systole timings are not available.
    # The frame with lowest RMS will be selected as the starting frame to perform unwrapping.
    # And the frame with highest RMS will be selected as the ending frame.
    sum_diff = 0.0
    jx,ix,kx = np.where(self.unwrapping_mask_Gx==True)
    jy,iy,ky = np.where(self.unwrapping_mask_Gy==True)
    jz,iz,kz = np.where(self.unwrapping_mask_Gz==True)
    for vel_key in ['U','V','W']:
      sum_diff += np.sum(self.Phase_diff_x[vel_key][jx,ix,kx]**2)
      sum_diff += np.sum(self.Phase_diff_y[vel_key][jy,iy,ky]**2)
      sum_diff += np.sum(self.Phase_diff_z[vel_key][jz,iz,kz]**2)
    self.RMS_phase_diff = (sum_diff / (3.0*(len(jx)+len(jy)+len(jz))))**0.5


  def wrap_phase_difference(self):
    # Wrap the spatial phase differences.
    self.Phase_diff_x_Wrap = {}
    self.Phase_diff_y_Wrap = {}
    self.Phase_diff_z_Wrap = {}
    for vel_key in ['U','V','W']:
      self.Phase_diff_x_Wrap[vel_key] = self.Phase_diff_x[vel_key] - np.round(self.Phase_diff_x[vel_key] / (2.0*np.pi)) * 2.0*np.pi
      self.Phase_diff_y_Wrap[vel_key] = self.Phase_diff_y[vel_key] - np.round(self.Phase_diff_y[vel_key] / (2.0*np.pi)) * 2.0*np.pi
      self.Phase_diff_z_Wrap[vel_key] = self.Phase_diff_z[vel_key] - np.round(self.Phase_diff_z[vel_key] / (2.0*np.pi)) * 2.0*np.pi
    # Assign zero to the phase differences outside the mask
    for vel_key in ['U','V','W']:
      self.Phase_diff_x_Wrap[vel_key][self.unwrapping_mask_Gx==False] = 0.0
      self.Phase_diff_y_Wrap[vel_key][self.unwrapping_mask_Gy==False] = 0.0
      self.Phase_diff_z_Wrap[vel_key][self.unwrapping_mask_Gz==False] = 0.0


  def update_wrapped_phase_difference(self, vel_key, Phase_diff_x_in, Phase_diff_y_in, Phase_diff_z_in):
    # Update the wrapped phase differences (self.Phase_diff_x_Wrap, etc) by external values
    self.Phase_diff_x_Wrap[vel_key] = Phase_diff_x_in
    self.Phase_diff_y_Wrap[vel_key] = Phase_diff_y_in
    self.Phase_diff_z_Wrap[vel_key] = Phase_diff_z_in
    self.Phase_diff_x_Wrap[vel_key][self.unwrapping_mask_Gx==False] = 0.0
    self.Phase_diff_y_Wrap[vel_key][self.unwrapping_mask_Gy==False] = 0.0
    self.Phase_diff_z_Wrap[vel_key][self.unwrapping_mask_Gz==False] = 0.0 


  def phase_diff_error_pole(self):
    # Evaluate the pole field for the phase differences (phase singularity)
    # Estimate the error magnitude due to phase singularity on phase differences. 

    # Determine the mask for the phase sigularites on each plane 
    self.mask_phase_sg_xy = (self.unwrapping_maskn[1:,1:,:] * self.unwrapping_maskn[1:,:-1,:] * self.unwrapping_maskn[:-1,1:,:] * self.unwrapping_maskn[:-1,:-1,:]).astype('bool')
    self.mask_phase_sg_yz = (self.unwrapping_maskn[1:,:,1:] * self.unwrapping_maskn[1:,:,:-1] * self.unwrapping_maskn[:-1,:,1:] * self.unwrapping_maskn[:-1,:,:-1]).astype('bool')
    self.mask_phase_sg_xz = (self.unwrapping_maskn[:,1:,1:] * self.unwrapping_maskn[:,1:,:-1] * self.unwrapping_maskn[:,:-1,1:] * self.unwrapping_maskn[:,:-1,:-1]).astype('bool')
    # Determine the phase singularites by intergrating along the 2x2 loops.
    self.Phase_sg_xy = {}
    self.Phase_sg_yz = {}
    self.Phase_sg_xz = {}
    for vel_key in ['U','V','W']:
      self.Phase_sg_xy[vel_key] = self.Phase_diff_x_Wrap[vel_key][:-1,:,:] - self.Phase_diff_x_Wrap[vel_key][1:,:,:] 
      self.Phase_sg_xy[vel_key] += self.Phase_diff_y_Wrap[vel_key][:,1:,:] - self.Phase_diff_y_Wrap[vel_key][:,:-1,:]
      self.Phase_sg_yz[vel_key] = self.Phase_diff_y_Wrap[vel_key][:,:,:-1] - self.Phase_diff_y_Wrap[vel_key][:,:,1:]
      self.Phase_sg_yz[vel_key] += self.Phase_diff_z_Wrap[vel_key][1:,:,:] - self.Phase_diff_z_Wrap[vel_key][:-1,:,:]
      self.Phase_sg_xz[vel_key] = self.Phase_diff_x_Wrap[vel_key][:,:,:-1] - self.Phase_diff_x_Wrap[vel_key][:,:,1:]
      self.Phase_sg_xz[vel_key] += self.Phase_diff_z_Wrap[vel_key][:,1:,:] - self.Phase_diff_z_Wrap[vel_key][:,:-1,:]
    # From the phase singularities, estimate the error component 
    self.Phase_diff_x_error_sg = {}
    self.Phase_diff_y_error_sg = {}
    self.Phase_diff_z_error_sg = {}
    for vel_key in ['U','V','W']:
      phase_diff_x_count = np.zeros(self.Phase_diff_x_Wrap[vel_key].shape)
      phase_diff_y_count = np.zeros(self.Phase_diff_y_Wrap[vel_key].shape)
      phase_diff_z_count = np.zeros(self.Phase_diff_z_Wrap[vel_key].shape)
      self.Phase_diff_x_error_sg[vel_key] = np.zeros(self.Phase_diff_x_Wrap[vel_key].shape)
      self.Phase_diff_y_error_sg[vel_key] = np.zeros(self.Phase_diff_y_Wrap[vel_key].shape)
      self.Phase_diff_z_error_sg[vel_key] = np.zeros(self.Phase_diff_z_Wrap[vel_key].shape)

      j,i,k = np.where(self.mask_phase_sg_xy==True)
      self.Phase_diff_x_error_sg[vel_key][j,i,k] += self.Phase_sg_xy[vel_key][j,i,k]**2
      phase_diff_x_count[j,i,k] += 1.0
      self.Phase_diff_x_error_sg[vel_key][j+1,i,k] += self.Phase_sg_xy[vel_key][j,i,k]**2
      phase_diff_x_count[j+1,i,k] += 1.0
      self.Phase_diff_y_error_sg[vel_key][j,i,k] += self.Phase_sg_xy[vel_key][j,i,k]**2
      phase_diff_y_count[j,i,k] += 1.0
      self.Phase_diff_y_error_sg[vel_key][j,i+1,k] += self.Phase_sg_xy[vel_key][j,i,k]**2
      phase_diff_y_count[j,i+1,k] += 1.0

      j,i,k = np.where(self.mask_phase_sg_yz==True)
      self.Phase_diff_y_error_sg[vel_key][j,i,k] += self.Phase_sg_yz[vel_key][j,i,k]**2
      phase_diff_y_count[j,i,k] += 1.0
      self.Phase_diff_y_error_sg[vel_key][j,i,k+1] += self.Phase_sg_yz[vel_key][j,i,k]**2
      phase_diff_y_count[j,i,k+1] += 1.0
      self.Phase_diff_z_error_sg[vel_key][j,i,k] += self.Phase_sg_yz[vel_key][j,i,k]**2
      phase_diff_z_count[j,i,k] += 1.0
      self.Phase_diff_z_error_sg[vel_key][j+1,i,k] += self.Phase_sg_yz[vel_key][j,i,k]**2
      phase_diff_z_count[j+1,i,k] += 1.0

      j,i,k = np.where(self.mask_phase_sg_xz==True)
      self.Phase_diff_x_error_sg[vel_key][j,i,k] += self.Phase_sg_xz[vel_key][j,i,k]**2
      phase_diff_x_count[j,i,k] += 1.0
      self.Phase_diff_x_error_sg[vel_key][j,i,k+1] += self.Phase_sg_xz[vel_key][j,i,k]**2
      phase_diff_x_count[j,i,k+1] += 1.0
      self.Phase_diff_z_error_sg[vel_key][j,i,k] += self.Phase_sg_xz[vel_key][j,i,k]**2
      phase_diff_z_count[j,i,k] += 1.0
      self.Phase_diff_z_error_sg[vel_key][j,i+1,k] += self.Phase_sg_xz[vel_key][j,i,k]**2
      phase_diff_z_count[j,i+1,k] += 1.0
      
      j,i,k = np.where(phase_diff_x_count > 0.0)
      self.Phase_diff_x_error_sg[vel_key][j,i,k] = (self.Phase_diff_x_error_sg[vel_key][j,i,k] / phase_diff_x_count[j,i,k])**0.5
      j,i,k = np.where(phase_diff_y_count > 0.0)
      self.Phase_diff_y_error_sg[vel_key][j,i,k] = (self.Phase_diff_y_error_sg[vel_key][j,i,k] / phase_diff_y_count[j,i,k])**0.5
      j,i,k = np.where(phase_diff_z_count > 0.0)
      self.Phase_diff_z_error_sg[vel_key][j,i,k] = (self.Phase_diff_z_error_sg[vel_key][j,i,k] / phase_diff_z_count[j,i,k])**0.5

    # Bound the phase_diff_error_sg by 1e-6
    for vel_key in ['U','V','W']:
      self.Phase_diff_x_error_sg[vel_key] += 1e-3
      self.Phase_diff_y_error_sg[vel_key] += 1e-3
      self.Phase_diff_z_error_sg[vel_key] += 1e-3


  def update_phase_diff_error_sg(self, vel_key, Phase_diff_x_sg, Phase_diff_y_sg, Phase_diff_z_sg):
    # Update the phase difference error magnitude due to singularity (for a single frame)
    self.Phase_diff_x_error_sg[vel_key] = Phase_diff_x_sg
    self.Phase_diff_y_error_sg[vel_key] = Phase_diff_y_sg
    self.Phase_diff_z_error_sg[vel_key] = Phase_diff_z_sg


  def phase_diff_error_divergence(self):
    # Estimate the phase difference errors from the divergence field evaluated from the wrapped phase differences.
    # This is only applied to points without significant phase sg errors 
    # Only apply to frame ct

    # Evaluate the divergence.
    # Exclude the points out of the flow domain (or with significant error_sg)
    phase_u_diff_x = np.copy(self.Phase_diff_x_Wrap['U'])
    # phase_u_diff_x[self.Phase_diff_x_error_sg['U']>=2.0*np.pi] = np.nan
    phase_u_diff_x[self.fluid_mask_Gx==False] = np.nan
    phase_v_diff_y = np.copy(self.Phase_diff_y_Wrap['V'])
    # phase_v_diff_y[self.Phase_diff_y_error_sg['V']>=2.0*np.pi] = np.nan
    phase_v_diff_y[self.fluid_mask_Gy==False] = np.nan
    phase_w_diff_z = np.copy(self.Phase_diff_z_Wrap['W'])
    # phase_w_diff_z[self.Phase_diff_z_error_sg['W']>=2.0*np.pi] = np.nan
    phase_w_diff_z[self.fluid_mask_Gz==False] = np.nan
    
    # Convert the phase differences to the velcoity differeces.
    u_diff_x = phase_u_diff_x * self.Vencs['U']/np.pi
    v_diff_y = phase_v_diff_y * self.Vencs['V']/np.pi
    w_diff_z = phase_w_diff_z * self.Vencs['W']/np.pi

    # Evaluate the divergence field from the velocity differences with SOC-type scheme.
    div_field = np.full(self.fluid_maskn.shape, np.nan)
    div_field[1:-1,1:-1,1:-1] = (u_diff_x[1:-1,1:,1:-1] + u_diff_x[1:-1,:-1,1:-1])/(2.0*self.dx)
    div_field[1:-1,1:-1,1:-1] += (v_diff_y[1:,1:-1,1:-1] + v_diff_y[:-1,1:-1,1:-1])/(2.0*self.dy)
    div_field[1:-1,1:-1,1:-1] += (w_diff_z[1:-1,1:-1,1:] + w_diff_z[1:-1,1:-1,:-1])/(2.0*self.dz)
    
    # Construct the linear operator which transforms phase values to div. 
    # This is tricky, needs lots of book keeping
    # For divergence field
    mask_div_field = np.isnan(div_field)==False
    div_index_field = -np.ones(self.fluid_maskn.shape).astype('int')
    j_div,i_div,k_div = np.where(mask_div_field==True)
    Npts_div = len(j_div)
    div_index_field[j_div,i_div,k_div] = range(Npts_div)
    div_field_vect = div_field[j_div,i_div,k_div]
    # Figure out the mask and index for phase_u
    mask_phase_u = np.zeros(self.fluid_maskn.shape).astype('bool')
    mask_phase_u[:,1:,:] = mask_div_field[:,:-1,:]
    mask_phase_u[:,:-1,:] = np.logical_or(mask_phase_u[:,:-1,:], mask_div_field[:,1:,:])
    phase_u_index_field = -np.ones(self.fluid_maskn.shape).astype('int')
    ju,iu,ku = np.where(mask_phase_u==True)
    Npts_u = len(ju)
    phase_u_index_field[ju,iu,ku] = range(Npts_u)
    # Figure out the mask and index for phase_v
    mask_phase_v = np.zeros(self.fluid_maskn.shape).astype('bool')
    mask_phase_v[1:,:,:] = mask_div_field[:-1,:,:]
    mask_phase_v[:-1,:,:] = np.logical_or(mask_phase_v[:-1,:,:], mask_div_field[1:,:,:])
    phase_v_index_field = -np.ones(self.fluid_maskn.shape).astype('int')
    jv,iv,kv = np.where(mask_phase_v==True)
    Npts_v = len(jv)
    phase_v_index_field[jv,iv,kv] = range(Npts_v)
    # Figure out the mask and index for phase_w
    mask_phase_w = np.zeros(self.fluid_maskn.shape).astype('bool')
    mask_phase_w[:,:,1:] = mask_div_field[:,:,:-1]
    mask_phase_w[:,:,:-1] = np.logical_or(mask_phase_w[:,:,:-1], mask_div_field[:,:,1:])
    phase_w_index_field = -np.ones(self.fluid_maskn.shape).astype('int')
    jw,iw,kw = np.where(mask_phase_w==True)
    Npts_w = len(jw)
    phase_w_index_field[jw,iw,kw] = range(Npts_w)
    # Generate the linear operator
    iC = div_index_field[j_div,i_div,k_div]
    iE = phase_u_index_field[j_div,i_div+1,k_div]
    iW = phase_u_index_field[j_div,i_div-1,k_div]
    iN = phase_v_index_field[j_div+1,i_div,k_div]
    iS = phase_v_index_field[j_div-1,i_div,k_div]
    iT = phase_w_index_field[j_div,i_div,k_div+1]
    iB = phase_w_index_field[j_div,i_div,k_div-1]
    Opt_div_u_part = scysparse.csr_matrix((Npts_div,Npts_u),dtype=np.float) # shape of the operators may change later.
    Opt_div_v_part = scysparse.csr_matrix((Npts_div,Npts_v),dtype=np.float)
    Opt_div_w_part = scysparse.csr_matrix((Npts_div,Npts_w),dtype=np.float)
    Opt_div_u_part[iC,iE] += 0.5/self.dx
    Opt_div_u_part[iC,iW] += 0.5/self.dx
    Opt_div_v_part[iC,iN] += 0.5/self.dy
    Opt_div_v_part[iC,iS] += 0.5/self.dy
    Opt_div_w_part[iC,iT] += 0.5/self.dz
    Opt_div_w_part[iC,iB] += 0.5/self.dz
    # Combine the operator for augemented
    Opt_div_aug = scysparse.bmat([[Opt_div_u_part,Opt_div_v_part,Opt_div_w_part]],format='csr')
    # Solve for the phase errors by least-squares
    vel_error_vect,istop,_,_,_,_,_,_,_,_ = splinalg.lsqr(Opt_div_aug, div_field_vect)
    # Assign the value to the phase error fields.
    self.Phase_error_div = {}
    self.Phase_error_div['U'] = np.full(self.fluid_maskn.shape,np.nan)
    self.Phase_error_div['V'] = np.full(self.fluid_maskn.shape,np.nan)
    self.Phase_error_div['W'] = np.full(self.fluid_maskn.shape,np.nan)
    self.Phase_error_div['U'][ju,iu,ku] = vel_error_vect[:Npts_u] * np.pi/self.Vencs['U']
    self.Phase_error_div['V'][jv,iv,kv] = vel_error_vect[Npts_u:Npts_u+Npts_v] * np.pi/self.Vencs['V']
    self.Phase_error_div['W'][jw,iw,kw] = vel_error_vect[Npts_u+Npts_v:] * np.pi/self.Vencs['W']
    
    # From the determined phase errors, try to estimate the local variance (uncertainty) as the WSTD of neighboring error values using a Gaussian kernel distance based
    j,i,k = np.where(self.fluid_maskn==True)
    x_list = self.Xn[j,i,k]
    y_list = self.Yn[j,i,k]
    z_list = self.Zn[j,i,k]
    self.Phase_unc_div = {}
    for vel_key in ['U','V','W']:
      self.Phase_unc_div[vel_key] = np.zeros(self.fluid_maskn.shape)
      j_err, i_err, k_err = np.where(np.isnan(self.Phase_error_div[vel_key])==False)
      x_err = self.Xn[j_err,i_err,k_err]
      y_err = self.Yn[j_err,i_err,k_err]
      z_err = self.Zn[j_err,i_err,k_err]
      phase_error_vect = self.Phase_error_div[vel_key][j_err,i_err,k_err]
      pairwise_dist = distance.cdist(np.array([x_list,y_list,z_list]).T, np.array([x_err,y_err,z_err]).T)
      kernel_coefs = np.exp(-0.5*pairwise_dist**2/(self.dx**2))
      self.Phase_unc_div[vel_key][j,i,k] = np.dot(kernel_coefs, phase_error_vect**2) / np.dot(kernel_coefs, np.ones(len(j_err)))
      self.Phase_unc_div[vel_key] = self.Phase_unc_div[vel_key]**0.5

    # From uncertainty of the phase, get the ucertainty of the phase diff.
    self.Phase_diff_x_unc_div = {}
    self.Phase_diff_y_unc_div = {}
    self.Phase_diff_z_unc_div = {}
    for vel_key in ['U','V','W']:
      self.Phase_diff_x_unc_div[vel_key] = (self.Phase_unc_div[vel_key][:,1:,:]**2 + self.Phase_unc_div[vel_key][:,:-1,:]**2)**0.5
      self.Phase_diff_y_unc_div[vel_key] = (self.Phase_unc_div[vel_key][1:,:,:]**2 + self.Phase_unc_div[vel_key][:-1,:,:]**2)**0.5
      self.Phase_diff_z_unc_div[vel_key] = (self.Phase_unc_div[vel_key][:,:,1:]**2 + self.Phase_unc_div[vel_key][:,:,:-1]**2)**0.5


  def phase_diff_unc_mag(self,mag_Un,mag_Vn,mag_Wn):
    # Estimate the phase difference error component based on the magnitude of the measurement.
    # First, determine a global ratio (C0) betwenn phase error and the 1.0/magnitude.
    # This untilizes the phase errors estimated from divergence to determine the C0

    self.Phase_mag = {}
    self.Phase_mag['U'] = mag_Un
    self.Phase_mag['V'] = mag_Vn
    self.Phase_mag['W'] = mag_Wn
    C0_u = self.Phase_error_div['U']*self.Phase_mag['U']
    C0_v = self.Phase_error_div['V']*self.Phase_mag['V']
    C0_w = self.Phase_error_div['W']*self.Phase_mag['W']
    C0_u_vect = C0_u[np.where(np.isnan(C0_u)==False)]
    C0_v_vect = C0_v[np.where(np.isnan(C0_v)==False)]
    C0_w_vect = C0_w[np.where(np.isnan(C0_w)==False)]
    C0_global = np.mean(np.concatenate((C0_u_vect,C0_v_vect,C0_w_vect)).flatten()**2)**0.5
    # From the phase magnitude and CO_global, evaluate the phase uncertainty
    self.Phase_unc_mag = {}
    for vel_key in ['U','V','W']:
      self.Phase_unc_mag[vel_key] = C0_global * self.Phase_mag[vel_key]**(-1)
    # From the phase uncertainty, determine the uncertainty of the phase differences.
    self.Phase_diff_x_unc_mag = {}
    self.Phase_diff_y_unc_mag = {}
    self.Phase_diff_z_unc_mag = {}
    for vel_key in ['U','V','W']:
      self.Phase_diff_x_unc_mag[vel_key] = (self.Phase_unc_mag[vel_key][:,1:,:]**2 + self.Phase_unc_mag[vel_key][:,:-1,:]**2)**0.5
      self.Phase_diff_y_unc_mag[vel_key] = (self.Phase_unc_mag[vel_key][1:,:,:]**2 + self.Phase_unc_mag[vel_key][:-1,:,:]**2)**0.5
      self.Phase_diff_z_unc_mag[vel_key] = (self.Phase_unc_mag[vel_key][:,:,1:]**2 + self.Phase_unc_mag[vel_key][:,:,:-1]**2)**0.5


  def phase_diff_error_combined(self):
    # This function combines the phase difference errors to formulate an overall error magnitude for each phase difference value. 
    # This combined error magniutde can be used to determine the weights for the WLS reconstruction.
    # Combine the error and uncertainty of the random noise component.
    self.Phase_diff_x_unc_noise = {}
    self.Phase_diff_y_unc_noise = {}
    self.Phase_diff_z_unc_noise = {}
    for vel_key in ['U','V','W']:
      self.Phase_diff_x_unc_noise[vel_key] = np.nanmean([self.Phase_diff_x_unc_div[vel_key]**2, self.Phase_diff_x_unc_mag[vel_key]**2],axis=0)**0.5
      self.Phase_diff_y_unc_noise[vel_key] = np.nanmean([self.Phase_diff_y_unc_div[vel_key]**2, self.Phase_diff_y_unc_mag[vel_key]**2],axis=0)**0.5
      self.Phase_diff_z_unc_noise[vel_key] = np.nanmean([self.Phase_diff_z_unc_div[vel_key]**2, self.Phase_diff_z_unc_mag[vel_key]**2],axis=0)**0.5
    self.Phase_diff_x_unc_combined = {}
    self.Phase_diff_y_unc_combined = {}
    self.Phase_diff_z_unc_combined = {}
    for vel_key in ['U','V','W']:
      self.Phase_diff_x_unc_combined[vel_key] = (self.Phase_diff_x_unc_noise[vel_key]**2 + self.Phase_diff_x_error_sg[vel_key]**2)**0.5
      self.Phase_diff_y_unc_combined[vel_key] = (self.Phase_diff_y_unc_noise[vel_key]**2 + self.Phase_diff_y_error_sg[vel_key]**2)**0.5
      self.Phase_diff_z_unc_combined[vel_key] = (self.Phase_diff_z_unc_noise[vel_key]**2 + self.Phase_diff_z_error_sg[vel_key]**2)**0.5
      # Bound the combined uncertainty to avoid zero uncertainty.
      low_unc = 1e-3
      self.Phase_diff_x_unc_combined[vel_key][self.Phase_diff_x_unc_combined[vel_key]<low_unc] = low_unc
      self.Phase_diff_y_unc_combined[vel_key][self.Phase_diff_y_unc_combined[vel_key]<low_unc] = low_unc
      self.Phase_diff_z_unc_combined[vel_key][self.Phase_diff_z_unc_combined[vel_key]<low_unc] = low_unc


  def weights_from_errors(self):
    # Construct the weights from the estimated uncertainties of phase differences

    j,i,k = np.where(self.unwrapping_maskn==True)
    Npts = len(j)
    jx,ix,kx = np.where(self.unwrapping_mask_Gx==True)
    jy,iy,ky = np.where(self.unwrapping_mask_Gy==True)
    jz,iz,kz = np.where(self.unwrapping_mask_Gz==True)
    # Initialized the weight vector as the reciprocal of the square of uncertainty of the phase differences.
    self.weights_diff_x = {}
    self.weights_diff_y = {}
    self.weights_diff_z = {}
    for vel_key in ['U','V','W']:
      self.weights_diff_x[vel_key] = 1.0 / self.Phase_diff_x_unc_combined[vel_key][jx,ix,kx]**2
      self.weights_diff_y[vel_key] = 1.0 / self.Phase_diff_y_unc_combined[vel_key][jy,iy,ky]**2
      self.weights_diff_z[vel_key] = 1.0 / self.Phase_diff_z_unc_combined[vel_key][jz,iz,kz]**2

    # Notice that the weights are not compatible with the scale of the gradient operators
    # as the effect of grid size were not considered (phase difference is not same as phase gradient).
    # For a compatible weights, the square of the grid size need to be multiplied.
    self.weights_grad_x = {}
    self.weights_grad_y = {}
    self.weights_grad_z = {}
    for vel_key in ['U','V','W']:
      self.weights_grad_x[vel_key] = self.weights_diff_x[vel_key] * self.dx**2
      self.weights_grad_y[vel_key] = self.weights_diff_y[vel_key] * self.dy**2
      self.weights_grad_z[vel_key] = self.weights_diff_z[vel_key] * self.dz**2

    # Find the mean of the weights and adjust the lower & upper bounds of the weights:
    # For the phase diffference weights
    mean_weights = 0.0
    for vel_key in ['U','V','W']:
      mean_weights += np.sum(self.weights_diff_x[vel_key])
      mean_weights += np.sum(self.weights_diff_y[vel_key])
      mean_weights += np.sum(self.weights_diff_z[vel_key])
    mean_weights = mean_weights / (3.0*(len(jx)+len(jy)+len(jz)))
    self.mean_weights_diff = mean_weights
    # Set bounds
    low_weights = 1e-3*mean_weights
    high_weights = 1e3*mean_weights
    for vel_key in ['U','V','W']:
      self.weights_diff_x[vel_key][self.weights_diff_x[vel_key]<low_weights] = low_weights
      self.weights_diff_x[vel_key][self.weights_diff_x[vel_key]>high_weights] = high_weights
      self.weights_diff_y[vel_key][self.weights_diff_y[vel_key]<low_weights] = low_weights
      self.weights_diff_y[vel_key][self.weights_diff_y[vel_key]>high_weights] = high_weights
      self.weights_diff_z[vel_key][self.weights_diff_z[vel_key]<low_weights] = low_weights
      self.weights_diff_z[vel_key][self.weights_diff_z[vel_key]>high_weights] = high_weights

    # For the phase gradient weights:
    mean_weights = 0.0
    for vel_key in ['U','V','W']:
      mean_weights += np.sum(self.weights_grad_x[vel_key])
      mean_weights += np.sum(self.weights_grad_y[vel_key])
      mean_weights += np.sum(self.weights_grad_z[vel_key])
    mean_weights = mean_weights / (3.0*(len(jx)+len(jy)+len(jz)))
    self.mean_weights_grad = mean_weights
    # Set bounds
    low_weights = 1e-3*mean_weights
    high_weights = 1e3*mean_weights
    for vel_key in ['U','V','W']:
      self.weights_grad_x[vel_key][self.weights_grad_x[vel_key]<low_weights] = low_weights
      self.weights_grad_x[vel_key][self.weights_grad_x[vel_key]>high_weights] = high_weights
      self.weights_grad_y[vel_key][self.weights_grad_y[vel_key]<low_weights] = low_weights
      self.weights_grad_y[vel_key][self.weights_grad_y[vel_key]>high_weights] = high_weights
      self.weights_grad_z[vel_key][self.weights_grad_z[vel_key]<low_weights] = low_weights
      self.weights_grad_z[vel_key][self.weights_grad_z[vel_key]>high_weights] = high_weights


  def construct_linear_system_WLS(self, divfree_regularization=1e3, uniform_weights=False):
    # This function is employed in the main function.
    # Construct the weighted least-squares with divergence free constraints linear system.
    # The divergence constraint is added as a highly weighted term to ensure the resulted field would be nearly divergence-free.
    # The input divfree_regularization controls the strength of divergence-free constraint relative to the gradients integration.
    # uniform_weights: if True, use OLS insteaed of WLS for the gradient integration.

    # Genearte the gradient operator from the unwrapping_maskn 
    linear_operator_generator = LinearOperatorGeneration(self.Xn,self.Yn,self.Zn,self.unwrapping_maskn)
    OperatorGx, OperatorGy, OperatorGz, fluid_mask_Gx, fluid_mask_Gy, fluid_mask_Gz = linear_operator_generator.generate_gradient_operator_mask_staggered()
    # Generate the divergence (gradient operators) for the flow field (notice there are two sets of masks)
    linear_operator_generator_1 = LinearOperatorGeneration(self.Xn,self.Yn,self.Zn,self.fluid_maskn)
    OperatorDDX, OperatorDDY, OperatorDDZ = linear_operator_generator_1.generate_operator_gradient_two_masks(self.unwrapping_maskn)
    # The OperatorDDX, OperatorDDY, OperatorDDZ will be used for the divergence free constraint.
    # We need to modify it to account for the effect of different Vencs, to add the transformation from phase to velocity.
    OperatorDDX = OperatorDDX * self.Vencs['U']/np.pi
    OperatorDDY = OperatorDDY * self.Vencs['V']/np.pi
    OperatorDDZ = OperatorDDZ * self.Vencs['W']/np.pi

    
    # indexing for different masks
    j,i,k = np.where(self.fluid_maskn==True)
    Npts = len(j)
    jx,ix,kx = np.where(self.unwrapping_mask_Gx==True)
    jy,iy,ky = np.where(self.unwrapping_mask_Gy==True)
    jz,iz,kz = np.where(self.unwrapping_mask_Gz==True)

    j_uw,i_uw,k_uw = np.where(self.unwrapping_maskn==True)
    Npts_uw = len(j_uw)

    # Construct the augmented linear system Ax = b

    # Some place holder for bmat operation.
    zeros_Gx = scysparse.csr_matrix(OperatorGx.shape,dtype=np.float)
    zeros_Gy = scysparse.csr_matrix(OperatorGy.shape,dtype=np.float)
    zeros_Gz = scysparse.csr_matrix(OperatorGz.shape,dtype=np.float)

    # Define an arbintray reference point with zero phase value to ensure the system is non-singular
    ref_operator = scysparse.csr_matrix((1,Npts_uw),dtype=np.float)
    ref_operator[0,0] = 1.0/self.dx # The operator is multiplied by 1.0/self.dx such that the linear system is more balanced
    zeros_ref = scysparse.csr_matrix((1,Npts_uw),dtype=np.float) # Place holder for bmat operation.
    weights_vector_ref = np.ones(3) * 1e3 * self.dx**2  # The weight vector for the 3 zero reference values.

    # Specify the weights vector for penalizing the divergence
    # The strenght is realtive to the weights of the gradient terms
    weights_vector_div = np.ones(Npts) * divfree_regularization**2 * self.mean_weights_grad * (np.pi/np.mean([self.Vencs['U'],self.Vencs['V'],self.Vencs['W']]))

    # Construct the reference operator for the added reference points and their weights.
    if self.mask_ref_values is None:
      add_ref_operator_u = scysparse.csr_matrix((0,Npts_uw), dtype=np.float)
      add_ref_operator_v = scysparse.csr_matrix((0,Npts_uw), dtype=np.float)
      add_ref_operator_w = scysparse.csr_matrix((0,Npts_uw), dtype=np.float)
      weights_addref_combined = np.array([])
      rhs_addref_combined = np.array([]) 
    else:
      index_uw_field = -np.ones(self.fluid_maskn.shape).astype('int')
      index_uw_field[j_uw,i_uw,k_uw] = range(Npts_uw)
      # For U phase:
      j_addref, i_addref, k_addref = np.where(self.mask_ref_values['U']==True)
      Npts_addref = len(j_addref)
      add_ref_operator_u = scysparse.csr_matrix((Npts_addref,Npts_uw), dtype=np.float)
      index_list_addref = index_uw_field[j_addref, i_addref, k_addref]
      add_ref_operator_u[range(Npts_addref), index_list_addref] = 1.0/self.dx # multiplied by 1.0/self.dx for more balanced system
      rhs_add_ref_u = self.add_ref_values['U'][j_addref, i_addref, k_addref] * (1.0/self.dx)
      weights_addref_u = (1.0 / self.sigma_ref_values['U'][j_addref, i_addref, k_addref]**2) * self.dx**2
      # For V phase:
      j_addref, i_addref, k_addref = np.where(self.mask_ref_values['V']==True)
      Npts_addref = len(j_addref)
      add_ref_operator_v = scysparse.csr_matrix((Npts_addref,Npts_uw), dtype=np.float)
      index_list_addref = index_uw_field[j_addref, i_addref, k_addref]
      add_ref_operator_v[range(Npts_addref), index_list_addref] = 1.0/self.dx # multiplied by 1.0/self.dx for more balanced system
      rhs_add_ref_v = self.add_ref_values['V'][j_addref, i_addref, k_addref] * (1.0/self.dx)
      weights_addref_v = (1.0 / self.sigma_ref_values['V'][j_addref, i_addref, k_addref]**2) * self.dx**2
      # For W phase:
      j_addref, i_addref, k_addref = np.where(self.mask_ref_values['W']==True)
      Npts_addref = len(j_addref)
      add_ref_operator_w = scysparse.csr_matrix((Npts_addref,Npts_uw), dtype=np.float)
      index_list_addref = index_uw_field[j_addref, i_addref, k_addref]
      add_ref_operator_w[range(Npts_addref), index_list_addref] = 1.0/self.dx # multiplied by 1.0/self.dx for more balanced system
      rhs_add_ref_w = self.add_ref_values['W'][j_addref, i_addref, k_addref] * (1.0/self.dx)
      weights_addref_w = (1.0 / self.sigma_ref_values['W'][j_addref, i_addref, k_addref]**2) * self.dx**2
      # combine the weights vector and rhs vector.
      weights_addref_combined = np.concatenate((weights_addref_u, weights_addref_v, weights_addref_w))
      rhs_addref_combined = np.concatenate((rhs_add_ref_u, rhs_add_ref_v, rhs_add_ref_w))

    
    # Construct the combined linear operator for the WLS system.
    self.Operator_A = scysparse.bmat([[OperatorGx, zeros_Gx, zeros_Gx],
                                      [OperatorGy, zeros_Gy, zeros_Gy],
                                      [OperatorGz, zeros_Gz, zeros_Gz],
                                      [zeros_Gx, OperatorGx, zeros_Gx],
                                      [zeros_Gy, OperatorGy, zeros_Gy],
                                      [zeros_Gz, OperatorGz, zeros_Gz],
                                      [zeros_Gx, zeros_Gx, OperatorGx],
                                      [zeros_Gy, zeros_Gy, OperatorGy],
                                      [zeros_Gz, zeros_Gz, OperatorGz],
                                      [ref_operator, zeros_ref, zeros_ref],
                                      [zeros_ref, ref_operator, zeros_ref],
                                      [zeros_ref, zeros_ref, ref_operator],
                                      [OperatorDDX,OperatorDDY,OperatorDDZ],
                                      [add_ref_operator_u, None, None],
                                      [None, add_ref_operator_v, None],
                                      [None, None, add_ref_operator_w]], dtype=np.float)
    
    # For the corresponding weight vector to form the weight matrix
    weight_vector = np.concatenate((self.weights_grad_x['U'], self.weights_grad_y['U'], self.weights_grad_z['U'],
                                    self.weights_grad_x['V'], self.weights_grad_y['V'], self.weights_grad_z['V'],
                                    self.weights_grad_x['W'], self.weights_grad_y['W'], self.weights_grad_z['W'],
                                    weights_vector_ref, weights_vector_div, weights_addref_combined))

    if uniform_weights == True:
      Npts_grads = len(np.concatenate((self.weights_grad_x['U'], self.weights_grad_y['U'], self.weights_grad_z['U'],
                                       self.weights_grad_x['V'], self.weights_grad_y['V'], self.weights_grad_z['V'],
                                       self.weights_grad_x['W'], self.weights_grad_y['W'], self.weights_grad_z['W'])))
      weight_vector[:Npts_grads] = self.mean_weights_grad
    
    # For the rhs of the linear system
    # The phase diffferences need to be rescaled to the gradients (dividing by the grid size).
    self.rhs_b = np.concatenate((self.Phase_diff_x_Wrap['U'][jx,ix,kx]/self.dx, self.Phase_diff_y_Wrap['U'][jy,iy,ky]/self.dy, self.Phase_diff_z_Wrap['U'][jz,iz,kz]/self.dz,
                                 self.Phase_diff_x_Wrap['V'][jx,ix,kx]/self.dx, self.Phase_diff_y_Wrap['V'][jy,iy,ky]/self.dy, self.Phase_diff_z_Wrap['V'][jz,iz,kz]/self.dz,
                                 self.Phase_diff_x_Wrap['W'][jx,ix,kx]/self.dx, self.Phase_diff_y_Wrap['W'][jy,iy,ky]/self.dy, self.Phase_diff_z_Wrap['W'][jz,iz,kz]/self.dz,
                                 np.zeros(3), np.zeros(Npts), rhs_addref_combined))

    # Form the diagnonal weight matrix
    self.Weight_matrix = scysparse.diags(weight_vector,format='csr',dtype=np.float)

    # For an augmented system which is compatible for a lsqr solver (the weight matrix need to be decomposed and multiplied).
    self.Operator_augmented = self.Weight_matrix.power(0.5) * self.Operator_A
    # rhs of the augmented system.
    self.rhs_augmented = self.Weight_matrix.power(0.5).dot(self.rhs_b)

    
  def solve_linear_system_WLS_lsqr(self, Phase_u_0=None, Phase_v_0=None, Phase_w_0=None, initialize_lsqr=True):

    # The sparse linear system is solved using lsqr
    # Some initial guess can be given 
    # Only work for one snapshot

    j,i,k = np.where(self.fluid_maskn==True)
    Npts = len(j)
    j_uw,i_uw,k_uw = np.where(self.unwrapping_maskn==True)
    Npts_uw = len(j_uw)
    
    if Phase_u_0 is None:
      x0 = None
    else:
      x0 = np.zeros(3*Npts_uw)
      x0[:Npts_uw] = Phase_u_0[j_uw,i_uw,k_uw]
      x0[Npts_uw : 2*Npts_uw] = Phase_v_0[j_uw,i_uw,k_uw]
      x0[2*Npts_uw : 3*Npts_uw] = Phase_w_0[j_uw,i_uw,k_uw]
    
    if initialize_lsqr==True:
      self.phase_vect,istop,itn,_,_,_,_,_,_,_ = splinalg.lsqr(self.Operator_augmented,self.rhs_augmented,x0=x0)
    else:
      self.phase_vect,istop,itn,_,_,_,_,_,_,_ = splinalg.lsqr(self.Operator_augmented,self.rhs_augmented)
    # print('istop = '+str(istop))
    if istop == 3:
      print('istop = '+str(istop))
    # Assign to the phase field.
    self.Phase_unwrapped['U'][j_uw,i_uw,k_uw] = self.phase_vect[:Npts_uw]
    self.Phase_unwrapped['V'][j_uw,i_uw,k_uw] = self.phase_vect[Npts_uw : 2*Npts_uw]
    self.Phase_unwrapped['W'][j_uw,i_uw,k_uw] = self.phase_vect[2*Npts_uw : 3*Npts_uw]
    self.istop = istop
    self.itn = itn


  def solve_linear_system_WLS_augmented_direct(self):

    # The sparse linear system is solved using lsqr
    # Some initial guess can be given 
    # Only work for one snapshot

    j,i,k = np.where(self.fluid_maskn==True)
    Npts = len(j)
    j_uw,i_uw,k_uw = np.where(self.unwrapping_maskn==True)
    Npts_uw = len(j_uw)
    
    self.phase_vect = splinalg.spsolve(self.Operator_augmented,self.rhs_augmented)
    
    # Assign to the phase field.
    self.Phase_unwrapped['U'][j_uw,i_uw,k_uw] = self.phase_vect[:Npts_uw]
    self.Phase_unwrapped['V'][j_uw,i_uw,k_uw] = self.phase_vect[Npts_uw : 2*Npts_uw]
    self.Phase_unwrapped['W'][j_uw,i_uw,k_uw] = self.phase_vect[2*Npts_uw : 3*Npts_uw]


  def correct_phase_offset(self,vel_key,ref_points,ref_values=None):
    # As the phase have offsets. Try to correct them
    # ct is the frame to work on
    # vel_key is 'U' or 'V' or 'W' which says which phase field is corrected    
    # ref_points: the index locations (i,j,k) of the reference points where the phase values were known
    # ref_values: the values of the index location. If none assuem unchanged
    # The correction is done by minimizing the difference between unwrapped phase and original phase of the ref_points.

    j_ref, i_ref, k_ref = ref_points
    if ref_values is None:
      phase_refs = self.Phase[vel_key][j_ref,i_ref,k_ref]
    else:
      phase_refs = ref_values
    phase_unwrapped_refs = self.Phase_unwrapped[vel_key][j_ref,i_ref,k_ref]
    phase_offset = np.median(phase_unwrapped_refs - phase_refs)
    self.Phase_unwrapped[vel_key] -= phase_offset
    self.Phase_unwrapped[vel_key][np.where(self.unwrapping_maskn==False)] = 0.0











