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
from UnwrappingWLS4DVencs import PhaseUnwrappingWeightedLeastSquares4D

def phase_unwrapping_WLS_initialized(Xn,Yn,Zn,Un,Vn,Wn,U0,V0,W0,fluid_maskn,fluid_mask0,unwrapping_maskn,Mag_u,Mag_v,Mag_w,Venc_u,Venc_v,Venc_w,phase_offset_correction='ref',
                                     val_addref_uvw=None, mask_addref_uvw=None, sigma_addref_uvw=None, divfree_regularization=1e3, uniform_weights=False,
                                     return_details=False, initialize_lsqr=True):
  # Perform the phase unwrapping on a single snapshot of phase with initial guess U0,V0,W0
  """
  Inputs:
    Xn,Yn,Zn: 3D cartesian grids of the data
    Un,Vn,Wn: 3D array of phase values for U, V, W. bounded by [-pi,pi]
    U0,V0,W0: 3D array of initial guess for unwrapped U,V,W. infered from temporal differences.
    fluid_maskn: 3D array of mask for flow regions.
    fluid_maskn: 3D array of mask for flow regions for U0.
    unwrapping_maskn: 3D array of mask for the region to unwrapping, should be larger than fluid_maskn
    Mag_u,Mag_v,Mag_w: 3D array of the magnitude of the phase data for the frame. 
    Venc_u, Venc_v, Venc_w: the venc for each of the three velocity components
    phase_offset_correction: the option for correcting the phase offset as follows:
                              "ref" use a layer outside the lumen as the reference points.
                              "median" treats the majority of the voxels are not aliased.
    val_addref_uvw: 4D array with the shape of (Ny,Nx,Nz,3) contains the additional reference value for phase u,v,w
    mask_addref_uvw: 4D array (binary mask) with the shape of (Ny,Nx,Nz,3) specify the locations of the addtional reference values.
    sigma_addref_uvw: 4D array with the shape of (Ny,Nx,Nz,3) specify the uncertainty of the additional reference values (for weights generation).
    divfree_regularization: the strenght of divfree regulariziation compared to the gradient terms.
    uniform_weights: if True, use OLS instead of WLS for the part of the gradient integration part.  
    return details: If true, return additionally the istop and itn for the lsqr calculation.  
    initialize_lsqr: if True, the lsqr calculation will be initialized, otherwise, not initialized (initialized as zeros)
  Outputs: 
    Phase_unwrapped: unwrapped phase field.
  """
  Phase_unwrapped = {}

  unwrapping_tool = PhaseUnwrappingWeightedLeastSquares4D(Xn,Yn,Zn,Un,Vn,Wn,Venc_u,Venc_v,Venc_w,fluid_maskn,unwrapping_maskn)
  # Add additional reference values for the phase fields if specified:
  if val_addref_uvw is not None:
    unwrapping_tool.additional_reference_points(val_addref_uvw,mask_addref_uvw,sigma_addref_uvw)
  # Calculate the wrapped phase difference.
  unwrapping_tool.wrap_phase_difference()
  # Extract the phase differences
  Phase_diff_x = {}
  Phase_diff_y = {}
  Phase_diff_z = {}
  for vel_key in ['U','V','W']:
    Phase_diff_x[vel_key] = unwrapping_tool.Phase_diff_x_Wrap[vel_key]
    Phase_diff_y[vel_key] = unwrapping_tool.Phase_diff_y_Wrap[vel_key]
    Phase_diff_z[vel_key] = unwrapping_tool.Phase_diff_z_Wrap[vel_key]

  # Use the phase differences of the initial guess as another set.
  Phase_0 = {}
  Phase_0['U'] = np.copy(U0)
  Phase_0['V'] = np.copy(V0)
  Phase_0['W'] = np.copy(W0)
  Phase_wrapped = {}
  Phase_wrapped['U'] = np.copy(Un)
  Phase_wrapped['V'] = np.copy(Vn)
  Phase_wrapped['W'] = np.copy(Wn)

  fluid_mask_comb = np.logical_and(fluid_mask0,fluid_maskn)
  for vel_key in ['U','V','W']:
    Phase_0[vel_key][fluid_mask_comb==False] = np.nan
  Phase_diff_x_0 = {}
  Phase_diff_y_0 = {}
  Phase_diff_z_0 = {}
  for vel_key in ['U','V','W']:
    Phase_diff_x_0[vel_key] = Phase_0[vel_key][:,1:,:] - Phase_0[vel_key][:,:-1,:]
    Phase_diff_y_0[vel_key] = Phase_0[vel_key][1:,:,:] - Phase_0[vel_key][:-1,:,:]
    Phase_diff_z_0[vel_key] = Phase_0[vel_key][:,:,1:] - Phase_0[vel_key][:,:,:-1]

  # Combine the two sets of phase differences (take the average)
  Phase_diff_x_comb = {}
  Phase_diff_y_comb = {}
  Phase_diff_z_comb = {}
  for vel_key in ['U','V','W']:
    Phase_diff_x_comb[vel_key] = np.nanmean([Phase_diff_x[vel_key],Phase_diff_x_0[vel_key]],axis=0)
    Phase_diff_y_comb[vel_key] = np.nanmean([Phase_diff_y[vel_key],Phase_diff_y_0[vel_key]],axis=0)
    Phase_diff_z_comb[vel_key] = np.nanmean([Phase_diff_z[vel_key],Phase_diff_z_0[vel_key]],axis=0)
  # Determine the STD of the phase differnces from the two sets.
  Phase_diff_x_std_sg = {}
  Phase_diff_y_std_sg = {}
  Phase_diff_z_std_sg = {}
  for vel_key in ['U','V','W']:
    Phase_diff_x_std_sg[vel_key] = np.abs(Phase_diff_x[vel_key] - Phase_diff_x_comb[vel_key])
    Phase_diff_y_std_sg[vel_key] = np.abs(Phase_diff_y[vel_key] - Phase_diff_y_comb[vel_key])
    Phase_diff_z_std_sg[vel_key] = np.abs(Phase_diff_z[vel_key] - Phase_diff_z_comb[vel_key])

  # Use the combined the phase difference 
  for vel_key in ['U','V','W']:
    unwrapping_tool.update_wrapped_phase_difference(vel_key, Phase_diff_x_comb[vel_key], Phase_diff_y_comb[vel_key], Phase_diff_z_comb[vel_key])
  # Compute the sg error.
  unwrapping_tool.phase_diff_error_pole()
  # Extract the phase_diff_x_error_sg and update by adding the std part.
  Phase_diff_x_error_sg = {}
  Phase_diff_y_error_sg = {}
  Phase_diff_z_error_sg = {}
  for vel_key in ['U','V','W']:
    Phase_diff_x_error_sg[vel_key] = (unwrapping_tool.Phase_diff_x_error_sg[vel_key]**2 + Phase_diff_x_std_sg[vel_key]**2)**0.5
    Phase_diff_y_error_sg[vel_key] = (unwrapping_tool.Phase_diff_y_error_sg[vel_key]**2 + Phase_diff_y_std_sg[vel_key]**2)**0.5
    Phase_diff_z_error_sg[vel_key] = (unwrapping_tool.Phase_diff_z_error_sg[vel_key]**2 + Phase_diff_z_std_sg[vel_key]**2)**0.5
  for vel_key in ['U','V','W']:
    unwrapping_tool.update_phase_diff_error_sg(vel_key, Phase_diff_x_error_sg[vel_key], Phase_diff_y_error_sg[vel_key], Phase_diff_z_error_sg[vel_key])
  
  # Proceed with the spatial unwrapping to finish the unwrapping job 
  unwrapping_tool.phase_diff_error_divergence()
  unwrapping_tool.phase_diff_unc_mag(Mag_u,Mag_v,Mag_w)
  unwrapping_tool.phase_diff_error_combined()
  unwrapping_tool.weights_from_errors()
  print('Constructing linear system')
  unwrapping_tool.construct_linear_system_WLS(divfree_regularization=divfree_regularization, uniform_weights=uniform_weights)
  print('Solving linear system')
  # unwrapping_tool.solve_linear_system_WLS_augmented_direct()
  unwrapping_tool.solve_linear_system_WLS_lsqr(Phase_u_0=U0, Phase_v_0=V0, Phase_w_0=W0, initialize_lsqr=initialize_lsqr)
  for vel_key in ['U','V','W']:
    Phase_unwrapped[vel_key] = unwrapping_tool.Phase_unwrapped[vel_key]
  # Correct for phase offset
  if phase_offset_correction == 'ref':
    j_ref,i_ref,k_ref = np.where(np.logical_and(fluid_maskn==False, unwrapping_maskn==True)==True)
    ref_points = (j_ref,i_ref,k_ref)
    ref_values = np.zeros(len(j_ref))
    for vel_key in ['U','V','W']:
      unwrapping_tool.correct_phase_offset(vel_key,ref_points,ref_values)
      Phase_unwrapped[vel_key] = unwrapping_tool.Phase_unwrapped[vel_key]
  elif phase_offset_correction == 'median':
    j,i,k = np.where(fluid_maskn==True)
    for vel_key in ['U','V','W']:
      phase_offset_list = Phase_unwrapped[vel_key] - Phase_wrapped[vel_key]
      Phase_unwrapped[vel_key] -= np.median(phase_offset_list[j,i,k])
  
  if return_details == False:
    return Phase_unwrapped['U'], Phase_unwrapped['V'], Phase_unwrapped['W']
  else:
    return Phase_unwrapped['U'], Phase_unwrapped['V'], Phase_unwrapped['W'], unwrapping_tool.istop, unwrapping_tool.itn


def phase_unwrapping_WLS_4D(Xn,Yn,Zn,Un,Vn,Wn,fluid_maskn,unwrapping_maskn,Mag_u,Mag_v,Mag_w,Venc_u,Venc_v,Venc_w,ct_start=None,ct_end=None,phase_offset_correction='ref',
                            val_addref_uvw=None, mask_addref_uvw=None, sigma_addref_uvw=None, divfree_regularization=1e3, uniform_weights=False,
                            return_details=False, Un_0=None, Vn_0=None, Wn_0=None, initialize_lsqr=True):
  # Perform the phase uwrapping using WLS on 4D flow data.
  """
  Inputs:
    Xn,Yn,Zn: 3D cartesian grids of the data
    Un,Vn,Wn: 4D array of phase values for U, V, W. bounded by [-pi,pi]
    fluid_maskn: 4D array of mask for flow regions. can be dynamic.
    unwrapping_maskn: 4D array of mask for the region to unwrapping, can be dyanmic, should be larger than fluid_maskn
    Mag_u,Mag_v,Mag_w: 4D array of the magnitude of the phase data. 
    Venc_u, Venc_v, Venc_w: the venc for each of the three velocity components
    ct_start: the starting frame. if None, determine by RMS of phase_diffs.
    ct_end: the ending frame. if None, determine by RMS pf phase_diffs
    phase_offset_correction: the option for correcting the phase offset as follows:
                              "ref" use a layer outside the lumen as the reference points.
                              "median" treats the majority of the voxels are not aliased.
    val_addref_uvw: 5D array with the shape of (Nt,Ny,Nx,Nz,3) contains the additional reference value for phase u,v,w
    mask_addref_uvw: 5D array (binary mask) with the shape of (Nt,Ny,Nx,Nz,3) specify the locations of the addtional reference values.
    sigma_addref_uvw: 5D array with the shape of (Nt,Ny,Nx,Nz,3) specify the uncertainty of the additional reference values (for weights generation).
    divfree_regularization: the strenght of divfree regulariziation compared to the gradient terms.
    uniform_weights: if True, use OLS instead of WLS for the part of the gradient integration part.       
    return_details: if True, the istop and itn for each lsqr calculation will be returned.
    Un_0, Vn_0, Wn_0: the initial guess of the phase fields, which will be used for initializing the lsqr calculation.  
    initialize_lsqr: if True, the lsqr calculation will be initialized, otherwise, not initialized (initialized as zeros) 
  Outputs: 
    Phase_unwrapped: unwrapped phase field.
  """

  # Set zeros to the flow data out of fluid_maskn
  Un[np.where(fluid_maskn==False)] = 0.0
  Vn[np.where(fluid_maskn==False)] = 0.0
  Wn[np.where(fluid_maskn==False)] = 0.0

  Nt,Ny,Nx,Nz = np.shape(Un)
  Phase_wrapped = {}
  Phase_wrapped['U'] = np.copy(Un)
  Phase_wrapped['V'] = np.copy(Vn)
  Phase_wrapped['W'] = np.copy(Wn)
  Phase_unwrapped = {}
  for vel_key in ['U','V','W']:
    Phase_unwrapped[vel_key] = np.zeros(Un.shape)
  istop = np.zeros(Nt)
  itn = np.zeros(Nt)

  # Normalize the Xn, Yn, Zn
  dx = Xn[1,1,1] - Xn[0,0,0]
  Xn = Xn/dx
  Yn = Yn/dx
  Zn = Zn/dx
  
  # Determine the RMS of the phase difference from each snapshot
  print('Determine starting frame')
  RMS_phase_diffs = np.zeros(Nt)
  for ct in range(Nt):
    unwrapping_tool = PhaseUnwrappingWeightedLeastSquares4D(Xn,Yn,Zn,Un[ct],Vn[ct],Wn[ct],Venc_u,Venc_v,Venc_w,fluid_maskn[ct],unwrapping_maskn[ct],)
    unwrapping_tool.RMS_phase_difference()
    RMS_phase_diffs[ct] = unwrapping_tool.RMS_phase_diff

  if ct_start is None:
    ct_start = np.argmin(RMS_phase_diffs)
  if ct_end is None:
    ct_end = np.argmax(RMS_phase_diffs)

  print('ct_start = '+str(ct_start))
  print('ct_end = '+str(ct_end))
  
  # Peform the phase unwrapping in ct_start (done pure spatially)
  print('Unwrapping ct_start = '+str(ct_start))
  unwrapping_tool = PhaseUnwrappingWeightedLeastSquares4D(Xn,Yn,Zn,Un[ct_start],Vn[ct_start],Wn[ct_start],Venc_u,Venc_v,Venc_w,fluid_maskn[ct_start],unwrapping_maskn[ct_start])
  # add additional reference values of phases if available
  if val_addref_uvw is not None:
    unwrapping_tool.additional_reference_points(val_addref_uvw[ct_start], mask_addref_uvw[ct_start], sigma_addref_uvw[ct_start])
  # Continue the operations
  unwrapping_tool.wrap_phase_difference()
  unwrapping_tool.phase_diff_error_pole()
  unwrapping_tool.phase_diff_error_divergence()
  unwrapping_tool.phase_diff_unc_mag(Mag_u[ct_start],Mag_v[ct_start],Mag_w[ct_start])
  unwrapping_tool.phase_diff_error_combined()
  unwrapping_tool.weights_from_errors()
  print('Constructing linear system')
  unwrapping_tool.construct_linear_system_WLS(divfree_regularization=divfree_regularization, uniform_weights=uniform_weights)
  print('Solving linear system')
  # unwrapping_tool.solve_linear_system_WLS_augmented_direct()
  unwrapping_tool.solve_linear_system_WLS_lsqr()
  for vel_key in ['U','V','W']:
    Phase_unwrapped[vel_key][ct_start] = unwrapping_tool.Phase_unwrapped[vel_key]
  # Correct for phase offset
  if phase_offset_correction == 'ref':
    j_ref,i_ref,k_ref = np.where(np.logical_and(fluid_maskn[ct_start]==False, unwrapping_maskn[ct_start]==True)==True)
    ref_points = (j_ref,i_ref,k_ref)
    ref_values = np.zeros(len(j_ref))
    for vel_key in ['U','V','W']:
      unwrapping_tool.correct_phase_offset(vel_key,ref_points,ref_values)
      Phase_unwrapped[vel_key][ct_start] = unwrapping_tool.Phase_unwrapped[vel_key]
  elif phase_offset_correction == 'median':
    j,i,k = np.where(fluid_maskn[ct_start]==True)
    for vel_key in ['U','V','W']:
      phase_offset_list = Phase_unwrapped[vel_key][ct_start] - Phase_wrapped[vel_key][ct_start]
      Phase_unwrapped[vel_key][ct_start] -= np.median(phase_offset_list[j,i,k])

  # Rearange the input time series, let ct_start frame to be at the start and ends with the ct_end
  if ct_start < ct_end:
    sequence_1 = list(range(ct_start,ct_end+1))
    sequence_2 = list(range(ct_start+1)[::-1])
    sequence_2 += list(range(ct_end,Nt)[::-1])

  else:
    sequence_1 = list(range(ct_end,ct_start+1)[::-1])
    sequence_2 = list(range(ct_start,Nt))
    sequence_2 += list(range(ct_end+1))


  # Starting from the ct_start, perform the unwrapping (temporal prediction then spatial integration) following sequence 1.
  for ct_sequence in range(len(sequence_1)-1):
    ct_0 = sequence_1[ct_sequence]
    ct = sequence_1[ct_sequence+1]
    print('Unwrapping ct = '+str(ct))
    # Make a initial prediction on the following snapshot.
    # Calculate the temporal phase diff
    Phase_diff_t = {}
    Phase_diff_t['U'] = Un[ct] - Phase_unwrapped['U'][ct_0]
    Phase_diff_t['V'] = Vn[ct] - Phase_unwrapped['V'][ct_0]
    Phase_diff_t['W'] = Wn[ct] - Phase_unwrapped['W'][ct_0]
    # Wrap the temp diffs
    for vel_key in ['U','V','W']:
      Phase_diff_t[vel_key] = Phase_diff_t[vel_key] - np.round(Phase_diff_t[vel_key] / (2.0*np.pi)) * 2.0*np.pi
    # Make prediction on the next frame.
    Phase_initial = {}
    for vel_key in ['U','V','W']:
      Phase_initial[vel_key] = Phase_unwrapped[vel_key][ct_0] + Phase_diff_t[vel_key]
    # Perform the unwrapping on the ct
    if val_addref_uvw is not None:
      val_addref_uvw_ct = val_addref_uvw[ct]
      mask_addref_uvw_ct = mask_addref_uvw[ct]
      sigma_addref_uvw_ct = sigma_addref_uvw[ct]
    else:
      val_addref_uvw_ct = None
      mask_addref_uvw_ct = None
      sigma_addref_uvw_ct = None

    if initialize_lsqr==False: # The lsqr will not be initialized
      Phase_unwrapped['U'][ct], Phase_unwrapped['V'][ct], Phase_unwrapped['W'][ct], istop[ct], itn[ct] = \
        phase_unwrapping_WLS_initialized(Xn,Yn,Zn,Un[ct],Vn[ct],Wn[ct],Phase_initial['U'],Phase_initial['V'],Phase_initial['W'],
                                         fluid_maskn[ct],fluid_maskn[ct_0],unwrapping_maskn[ct],Mag_u[ct],Mag_v[ct],Mag_w[ct],Venc_u,Venc_v,Venc_w,
                                         phase_offset_correction=phase_offset_correction,
                                         val_addref_uvw=val_addref_uvw_ct, mask_addref_uvw=mask_addref_uvw_ct, sigma_addref_uvw=sigma_addref_uvw_ct,
                                         divfree_regularization=divfree_regularization, uniform_weights=uniform_weights,
                                         return_details=True, initialize_lsqr=initialize_lsqr)

    elif Un_0 is None: # The lsqr will be initialized based on the time unwrapping results
      Phase_unwrapped['U'][ct], Phase_unwrapped['V'][ct], Phase_unwrapped['W'][ct], istop[ct], itn[ct] = \
        phase_unwrapping_WLS_initialized(Xn,Yn,Zn,Un[ct],Vn[ct],Wn[ct],Phase_initial['U'],Phase_initial['V'],Phase_initial['W'],
                                         fluid_maskn[ct],fluid_maskn[ct_0],unwrapping_maskn[ct],Mag_u[ct],Mag_v[ct],Mag_w[ct],Venc_u,Venc_v,Venc_w,
                                         phase_offset_correction=phase_offset_correction,
                                         val_addref_uvw=val_addref_uvw_ct, mask_addref_uvw=mask_addref_uvw_ct, sigma_addref_uvw=sigma_addref_uvw_ct,
                                         divfree_regularization=divfree_regularization, uniform_weights=uniform_weights,
                                         return_details=True, initialize_lsqr=initialize_lsqr)

    else: # The lsqr will be initialized based on given fields.
      Phase_unwrapped['U'][ct], Phase_unwrapped['V'][ct], Phase_unwrapped['W'][ct], istop[ct], itn[ct] = \
        phase_unwrapping_WLS_initialized(Xn,Yn,Zn,Un[ct],Vn[ct],Wn[ct],Un_0[ct],Vn_0[ct],Wn_0[ct],
                                         fluid_maskn[ct],fluid_maskn[ct_0],unwrapping_maskn[ct],Mag_u[ct],Mag_v[ct],Mag_w[ct],Venc_u,Venc_v,Venc_w,
                                         phase_offset_correction=phase_offset_correction,
                                         val_addref_uvw=val_addref_uvw_ct, mask_addref_uvw=mask_addref_uvw_ct, sigma_addref_uvw=sigma_addref_uvw_ct,
                                         divfree_regularization=divfree_regularization, uniform_weights=uniform_weights,
                                         return_details=True, initialize_lsqr=initialize_lsqr)

  # Store the end frame for future averaging
  Phase_unwrapped_end = {}
  for vel_key in ['U','V','W']:
    Phase_unwrapped_end[vel_key] = 0.5*Phase_unwrapped[vel_key][ct_end]
    
  # Starting from the ct_start and do backwards.
  for ct_sequence in range(len(sequence_2)-1):
    ct_0 = sequence_2[ct_sequence]
    ct = sequence_2[ct_sequence+1]
    print('Unwrapping ct = '+str(ct))
    # Make a initial prediction on the following snapshot.
    # Calculate the temporal phase diff
    Phase_diff_t = {}
    Phase_diff_t['U'] = Un[ct] - Phase_unwrapped['U'][ct_0]
    Phase_diff_t['V'] = Vn[ct] - Phase_unwrapped['V'][ct_0]
    Phase_diff_t['W'] = Wn[ct] - Phase_unwrapped['W'][ct_0]
    # Wrap the temp diffs
    for vel_key in ['U','V','W']:
      Phase_diff_t[vel_key] = Phase_diff_t[vel_key] - np.round(Phase_diff_t[vel_key] / (2.0*np.pi)) * 2.0*np.pi
    # Make prediction on the next frame.
    Phase_initial = {}
    for vel_key in ['U','V','W']:
      Phase_initial[vel_key] = Phase_unwrapped[vel_key][ct_0] + Phase_diff_t[vel_key]
    # Perform the unwrapping on the ct
    if val_addref_uvw is not None:
      val_addref_uvw_ct = val_addref_uvw[ct]
      mask_addref_uvw_ct = mask_addref_uvw[ct]
      sigma_addref_uvw_ct = sigma_addref_uvw[ct]
    else:
      val_addref_uvw_ct = None
      mask_addref_uvw_ct = None
      sigma_addref_uvw_ct = None

    if initialize_lsqr==False: # The lsqr will not be initialized
      Phase_unwrapped['U'][ct], Phase_unwrapped['V'][ct], Phase_unwrapped['W'][ct], istop[ct], itn[ct] = \
        phase_unwrapping_WLS_initialized(Xn,Yn,Zn,Un[ct],Vn[ct],Wn[ct],Phase_initial['U'],Phase_initial['V'],Phase_initial['W'],
                                         fluid_maskn[ct],fluid_maskn[ct_0],unwrapping_maskn[ct],Mag_u[ct],Mag_v[ct],Mag_w[ct],Venc_u,Venc_v,Venc_w,
                                         phase_offset_correction=phase_offset_correction,
                                         val_addref_uvw=val_addref_uvw_ct, mask_addref_uvw=mask_addref_uvw_ct, sigma_addref_uvw=sigma_addref_uvw_ct,
                                         divfree_regularization=divfree_regularization, uniform_weights=uniform_weights,
                                         return_details=True, initialize_lsqr=initialize_lsqr)

    elif Un_0 is None: # The lsqr will be initialized based on the time unwrapping results
      Phase_unwrapped['U'][ct], Phase_unwrapped['V'][ct], Phase_unwrapped['W'][ct], istop[ct], itn[ct] = \
        phase_unwrapping_WLS_initialized(Xn,Yn,Zn,Un[ct],Vn[ct],Wn[ct],Phase_initial['U'],Phase_initial['V'],Phase_initial['W'],
                                         fluid_maskn[ct],fluid_maskn[ct_0],unwrapping_maskn[ct],Mag_u[ct],Mag_v[ct],Mag_w[ct],Venc_u,Venc_v,Venc_w,
                                         phase_offset_correction=phase_offset_correction,
                                         val_addref_uvw=val_addref_uvw_ct, mask_addref_uvw=mask_addref_uvw_ct, sigma_addref_uvw=sigma_addref_uvw_ct,
                                         divfree_regularization=divfree_regularization, uniform_weights=uniform_weights,
                                         return_details=True, initialize_lsqr=initialize_lsqr)

    else: # The lsqr will be initialized based on given fields.
      Phase_unwrapped['U'][ct], Phase_unwrapped['V'][ct], Phase_unwrapped['W'][ct], istop[ct], itn[ct] = \
        phase_unwrapping_WLS_initialized(Xn,Yn,Zn,Un[ct],Vn[ct],Wn[ct],Un_0[ct],Vn_0[ct],Wn_0[ct],
                                         fluid_maskn[ct],fluid_maskn[ct_0],unwrapping_maskn[ct],Mag_u[ct],Mag_v[ct],Mag_w[ct],Venc_u,Venc_v,Venc_w,
                                         phase_offset_correction=phase_offset_correction,
                                         val_addref_uvw=val_addref_uvw_ct, mask_addref_uvw=mask_addref_uvw_ct, sigma_addref_uvw=sigma_addref_uvw_ct,
                                         divfree_regularization=divfree_regularization, uniform_weights=uniform_weights,
                                         return_details=True, initialize_lsqr=initialize_lsqr)
    
  # Average with previous iteration
  for vel_key in ['U','V','W']:
    Phase_unwrapped_end[vel_key] += 0.5*Phase_unwrapped[vel_key][ct_end]
  for vel_key in ['U','V','W']:
    Phase_unwrapped[vel_key][ct_end] = Phase_unwrapped_end[vel_key]

  # Return the unwrapped phase maps.
  if return_details:
    return Phase_unwrapped['U'], Phase_unwrapped['V'], Phase_unwrapped['W'], istop, itn
  else:
    return Phase_unwrapped['U'], Phase_unwrapped['V'], Phase_unwrapped['W']
  

  



