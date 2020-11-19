#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 15:42:45 2020

@author: mwall
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
from pyNBS import network_propagation as prop
import numpy.matlib as matlib
from scipy.optimize import nnls

#************ UNCHANGED from Core *********************
# Function to quantile normalize a pandas DataFrame
# Code taken from: https://github.com/ShawnLYU/Quantile_Normalize/blob/master/quantile_norm.py
# Using implementation described on Wikipedia: https://en.wikipedia.org/wiki/Quantile_normalization
# data: Pandas DataFrame (propagated genotypes) where rows are samples (samples), and columns are features (genes)
# Returns df_out: Quantile normalized Pandas DataFrame with same orientation as data df
def qnorm(data):
    df = data.T
    df_out = df.copy(deep=True)
    dic = {}
    # Sort each gene's propagation value for each patient
    for col in df:
        dic.update({col:sorted(df[col])})
    sorted_df = pd.DataFrame(dic)
    # Rank averages for each gene across samples
    ranked_avgs = sorted_df.mean(axis = 1).tolist()
    # Construct quantile normalized Pandas DataFrame by assigning ranked averages to ranks of each gene for each sample
    for col in df_out:
        t = stats.rankdata(df[col]).astype(int)
        df_out[col] = [ranked_avgs[i-1] for i in t]
    qnorm_data = df_out.T
    return qnorm_data

def subsample_sm_mat(sm_mat, propNet=None, pats_subsample_p=0.8, gene_subsample_p=0.8, min_muts=10):          
    #Import libraries
    from numpy import random

    # Number of indiv/features for sampling
    (Nind, Dfeat) = sm_mat.shape
    Nsample = round(Nind*pats_subsample_p)
    Dsample = round(Dfeat*gene_subsample_p)
    # Sub sample patients
    pats_subsample = random.choice(sm_mat.index, int(Nsample),replace=False)
    # Sub sample genes
    gene_subsample = random.choice(sm_mat.columns, int(Dsample),replace=False)
    # Sub sampled data mat
    gind_sample = sm_mat.loc[pats_subsample,gene_subsample]
    # Filter by mutation count
    gind_sample = gind_sample[gind_sample.sum(axis=1) > min_muts]
    # Filter columns by network nodes only if network is given
    if propNet is not None:
        # Check if network node names intersect with somatic mutation matrix column names
        # If there is no intersection, throw an error, gene names are not matched
        if len(set(list(propNet.nodes)).intersection(set(sm_mat.columns)))==0:
            raise ValueError('No mutations found in network nodes. Gene names may be mismatched.')
        gind_sample_filt = gind_sample.T.loc[list(propNet.nodes),:].fillna(0).T

    else:
        gind_sample_filt = gind_sample
    return gind_sample_filt


#************ UNCHANGED from Core *********************
# Adapted from Matan Hofree's Matlab code in NBS
# data = features-by-samples propagated (or not) mutation profiles
# KNN_glap = Graph laplacian of regularization network
# Note: Make sure that the rows of Y are aligned correctly with the rows and columns of KNN_glap before passing into function
# k = Number of clusters for factorization
# l = Network regularization term constant
# eps = Small number precision
# Loop break conditions:
#   err_tol = Maximum value of reconstruction error function allowed for break
#   err_delta_tol = Maximum change in reconstruction error allowed for break
#   maxiter = Maximum number of iterations to execute before break
# verbose = print statements on update progress
def mixed_netNMF(data, KNN_glap, k=3, l=200, maxiter=250, 
    eps=1e-15, err_tol=1e-4, err_delta_tol=1e-8, verbose=False):
    # Initialize H and W Matrices from data array if not given
    r, c = data.shape[0], data.shape[1]
    # Initialize H
    H_init = np.random.rand(k,c)
    H = np.maximum(H_init, eps)
    # Initialize W
    W_init = np.linalg.lstsq(H.T, data.T)[0].T
    W_init = np.dot(W_init, np.diag(1/sum(W_init)))
    W = np.maximum(W_init, eps)
    
    if verbose:
        print 'W and H matrices initialized'
    
    # Get graph matrices from laplacian array
    D = np.diag(np.diag(KNN_glap)).astype(float)
    A = (D-KNN_glap).astype(float)
    if verbose:
        print 'D and A matrices calculated'
    # Set mixed netNMF reconstruction error convergence factor
    XfitPrevious = np.inf
    
    # Updating W and H
    for i in range(maxiter):
        XfitThis = np.dot(W, H)
        WHres = np.linalg.norm(data-XfitThis) # Reconstruction error

        # Change in reconstruction error
        if i == 0:
            fitRes = np.linalg.norm(XfitPrevious)
        else:
            fitRes = np.linalg.norm(XfitPrevious-XfitThis)
        XfitPrevious = XfitThis

        # Reporting netNMF update status
        if (verbose) & (i%10==0):
            print 'Iteration >>', i, 'Mat-res:', WHres, 'Lambda:', l, 'Wfrob:', np.linalg.norm(W)
        if (err_delta_tol > fitRes) | (err_tol > WHres) | (i+1 == maxiter):
            if verbose:
                print 'NMF completed!'
                print 'Total iterations:', i+1
                print 'Final Reconstruction Error:', WHres
                print 'Final Reconstruction Error Delta:', fitRes
            numIter = i+1
            finalResidual = WHres
            break

        # Note about this part of the netNMF function:
        # There used to be a small block of code that would dynamically change l
        # to improve the convergence of the algorithm. We did not see any mathematical
        # or statistical support to have this block of code here. It seemed to just
        # add confusion in the final form of the algorithm. Therefore it has been removed.
        # The default l parameter is fine here, but the regularization constant can
        # be changed by the user if so desired.

        # Terms to be scaled by regularization constant: l
        KWmat_D = np.dot(D,W) 
        KWmat_W = np.dot(A,W)
            
        # Update W with network constraint
        W = W*((np.dot(data, H.T) + l*KWmat_W + eps) / (np.dot(W,np.dot(H,H.T)) + l*KWmat_D + eps))
        W = np.maximum(W, eps)
        # Normalize W across each gene (row-wise)
        W = W/matlib.repmat(np.maximum(sum(W),eps),len(W),1);        
        
        # Update H
        H = np.array([nnls(W, data[:,j])[0] for j in range(c)]).T 
        # ^ Hofree uses a custom fast non-negative least squares solver here, we will use scipy's implementation here
        H=np.maximum(H,eps)

    return W, H, numIter, finalResidual






def NBS_single_ext(sm_mat, regNet_glap, propNet=None, propNet_kernel=None, 
    k=3, verbose=False, **kwargs):
    
    import networkx as nx

    # Check for correct input data
    if type(sm_mat)!=pd.DataFrame:
        raise TypeError('Somatic mutation data must be given as Pandas DataFrame')
    if propNet is not None:
        if type(propNet)!=nx.Graph:
            raise TypeError('Networkx graph object required for propNet')
    if regNet_glap is not None:
        if type(regNet_glap)!=pd.DataFrame:
            raise TypeError('netNMF regularization network laplacian (regNet_glap) must be given as Pandas DataFrame')

    # Load or set subsampling parameters
    pats_subsample_p, gene_subsample_p, min_muts = 0.8, 0.8, 10
    if 'pats_subsample_p' in kwargs:
        pats_subsample_p = float(kwargs['pats_subsample_p'])
    if 'gene_subsample_p' in kwargs:
        gene_subsample_p = float(kwargs['gene_subsample_p'])
    if 'min_muts' in kwargs:
        min_muts = int(kwargs['min_muts'])  

    # Subsample Data
    sm_mat_subsample = subsample_sm_mat(sm_mat, propNet=propNet, 
        pats_subsample_p=pats_subsample_p, gene_subsample_p=gene_subsample_p, min_muts=min_muts)
    if verbose:
        print('Somatic mutation data sub-sampling complete')

    # Throw exception if subsampling returned empty dataframe
    if sm_mat_subsample.shape[0]==0:
        raise ValueError('Subsampled somatic mutation matrix contains no patients.')

    # Propagate data if network object is provided
    if propNet is not None:
        # Determine if propagation is can be based on pre-computed propagation kernel
        if propNet_kernel is None:
            # If kernel is not given and some propagation parameters are given in kwargs, set propagation parameters
            # Otherwise set default values
            alpha, symmetric_norm, save_prop = 0.7, False, False
            if 'prop_alpha' in kwargs:
                alpha = float(kwargs['prop_alpha'])
            if 'prop_symmetric_norm' in kwargs:
                symmetric_norm = ((kwargs['prop_symmetric_norm']=='True') | (kwargs['prop_symmetric_norm']==True))
            if 'save_prop' in kwargs:
                save_prop = ((kwargs['save_prop']=='True') | (kwargs['save_prop']==True))
            # Save propagation step data if desired (indicated in kwargs)
            if save_prop:
                prop_sm_data = prop.network_propagation(propNet, sm_mat_subsample, alpha=alpha, symmetric_norm=symmetric_norm, **kwargs)
            else:
                prop_sm_data = prop.network_propagation(propNet, sm_mat_subsample, alpha=alpha, symmetric_norm=symmetric_norm)
        else:
            # Save propagation step data if desired (indicated in kwargs)
            save_prop = False
            if 'save_prop' in kwargs:
                save_prop = ((kwargs['save_prop']=='True') | (kwargs['save_prop']==True))
            if save_prop:
                prop_sm_data = prop.network_kernel_propagation(propNet, propNet_kernel, sm_mat_subsample, **kwargs)
            else:
                prop_sm_data = prop.network_kernel_propagation(propNet, propNet_kernel, sm_mat_subsample)
        if verbose:
            print 'Somatic mutation data propagated'
    else:
        prop_sm_data = sm_mat_subsample
        if verbose:
            print('Somatic mutation data not propagated')

    # Quantile Normalize Data
    qnorm_data = True
    if 'qnorm_data' in kwargs:
        qnorm_data = ((kwargs['qnorm_data']=='True') | (kwargs['qnorm_data']==True))
    if qnorm_data:
        prop_data_qnorm = qnorm(prop_sm_data)
        if verbose:
            print('Somatic mutation data quantile normalized')
    else:
        prop_data_qnorm = prop_sm_data
        if verbose:
            print('Somatic mutation data not quantile normalized')

    # Prepare data for mixed netNMF function (align propagated profile columns with regularization network laplacian rows)
    if propNet is not None:
        propNet_nodes = list(propNet.nodes)
        data_arr = np.array(prop_data_qnorm.T.loc[propNet_nodes,:])
        regNet_glap_arr = np.array(regNet_glap.loc[propNet_nodes,propNet_nodes])
    else:
        propNet_nodes = list(regNet_glap.index)
        data_arr = np.array(prop_data_qnorm.T.loc[propNet_nodes,:].fillna(0))
        regNet_glap_arr = np.array(regNet_glap)

    # Set netNMF parameters from kwargs if given, otherwise use defaults
    netNMF_lambda, netNMF_maxiter, netNMF_verbose = 200, 250, False
    netNMF_eps, netNMF_err_tol, netNMF_err_delta_tol = 1e-15, 1e-4, 1e-8
    if 'netNMF_lambda' in kwargs:
        netNMF_lambda = float(kwargs['netNMF_lambda'])
    if 'netNMF_maxiter' in kwargs:
        netNMF_maxiter = int(kwargs['netNMF_maxiter'])
    if 'netNMF_eps' in kwargs:
        netNMF_eps = float(kwargs['netNMF_eps'])
    if 'netNMF_err_tol' in kwargs:
        netNMF_err_tol = float(kwargs['netNMF_err_tol'])
    if 'netNMF_err_delta_tol' in kwargs:
        netNMF_err_delta_tol = float(kwargs['netNMF_err_delta_tol']) 

    # Mixed netNMF Result
    W, H, numIter, finalResid = mixed_netNMF(data_arr, regNet_glap_arr, k=k, 
        l=netNMF_lambda, maxiter=netNMF_maxiter, eps=netNMF_eps, 
        err_tol=netNMF_err_tol, err_delta_tol=netNMF_err_delta_tol, verbose=False)

    # Return netNMF result (dimension-reduced propagated gene profiles)
    W_df = pd.DataFrame(W,index=prop_data_qnorm.T.index)
    W_df.columns = list(range(1,W_df.shape[1]+1))

    # Return netNMF result (dimension-reduced propagated patient profiles)
    H_df = pd.DataFrame(H.T, index=prop_data_qnorm.index)
    H_df.columns = list(range(1,H_df.shape[1]+1))

    # Save netNMF result
    # Saving the propagation result
    if 'outdir' in kwargs:
        if 'job_name' in kwargs:
            if 'iteration_label' in kwargs:
                save_path = kwargs['outdir']+str(kwargs['job_name'])+'_H_'+str(kwargs['iteration_label'])+'.csv'
                save_path_W = kwargs['outdir']+str(kwargs['job_name'])+'_W_'+str(kwargs['iteration_label'])+'.csv'
            else:
                save_path = kwargs['outdir']+str(kwargs['job_name'])+'_H.csv'
                save_path_W = kwargs['outdir']+str(kwargs['job_name'])+'_W.csv'
        else:
            if 'iteration_label' in kwargs:
                save_path = kwargs['outdir']+'H_'+str(kwargs['iteration_label'])+'.csv'
                save_path_W = kwargs['outdir']+'W_'+str(kwargs['iteration_label'])+'.csv'
            else:
                save_path = kwargs['outdir']+'H.csv'
                save_path = kwargs['outdir']+'W.csv'
        H_df.to_csv(save_path)
        W_df.to_csv(save_path_W)

        if verbose:
            print('H and W matrices saved:', save_path, save_path_W)
    else:
        pass
    if verbose:
        print('pyNBS iteration complete')
    return H_df, W_df

def consensus_networks(Hlist,Wlist,cluster_assignment_df=None,min_ratio = 2,min_percentile = 20,n_consensus_misses = 2):
    if cluster_assignment_df is None:
        from pyNBS import consensus_clustering as cc
        table, linkage, cluster_assignment_df = cc.consensus_hclust_hard(Hlist, k=Hlist[0].shape[1])
    #Set minimum hits
    min_hits = len(Hlist)-n_consensus_misses

    # Cluster assignments to disctionary
    cluster_assignments = {i:list(cluster_assignment_df.index[cluster_assignment_df==i])
     for i in cluster_assignment_df.unique()}

    # Instantiate results dictionary
    gene_subnetworks = {i:[] for i in list(cluster_assignments.keys())}


    #set iteration
    for index in range(len(Hlist)):

        #Identify tmp H matrix
        tmp_H = Hlist[index]
        pat_ix = [p for p in cluster_assignment_df.index if p in tmp_H.index]
        tmp_H = tmp_H.loc[pat_ix,:]

        #Identify tmp W matrix
        tmp_W = Wlist[index]

        #Iterate over each patient cluster
        for c in list(cluster_assignments.keys()):
            clust_match = tmp_H.loc[cluster_assignments[c],:].sum(axis=0).argmax()
            tmp_W_c = tmp_W.loc[:,[i for i in tmp_W.columns if i != clust_match]]

            #Set minimum ratio and value to select subnetwork genes
            min_value = np.percentile(tmp_W,min_percentile)

            #Subset genes by min_ratio
            tmp_diff = tmp_W.loc[:,clust_match]/tmp_W_c.mean(axis=1)
            tmp_diff.sort_values(ascending=False,inplace=True)

            #Subset genes by min_value
            subnetwork = tmp_diff.index[tmp_diff>min_ratio]
            subnetwork = subnetwork[tmp_W.loc[subnetwork,clust_match]>min_value]

            #Add subnetwork to dictionary
            gene_subnetworks[c].append(subnetwork)


    # Instantiate consensus networks dictionary
    consensus_subnetworks = {}

    # Identify consensus subnetworks
    for key in list(gene_subnetworks.keys()):
        tmp_count = Counter(np.hstack(gene_subnetworks[key]))
        tmp_mc = tmp_count.most_common()
        tmp_consensus = [tmp_mc[i][0] for i in range(len(tmp_mc)) if tmp_mc[i][1] >= min_hits]
        consensus_subnetworks[key] = tmp_consensus
        
    return consensus_subnetworks

def cluster_kmplot(cluster_assign,surv,lr_test = True,verbose = False,tmax = -1):
    import seaborn as sns
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test as multiv_lr_test
    import matplotlib.pyplot as plt

    # Initialize KM plotter
    kmf = KaplanMeierFitter()

    # Number of clusters
    clusters = sorted(list(cluster_assign.value_counts().index))
    k = len(clusters)

    #Set title
    title = "Survival plot k = "+str(k)

    # Initialize KM Plot Settings
    fig = plt.figure(figsize=(10, 7)) 
    ax = plt.subplot(1,1,1)
    colors = sns.color_palette('hls', k)
    cluster_cmap = {clusters[i]:colors[i] for i in range(k)}

    for clust in clusters:
        clust_pats = list(cluster_assign[cluster_assign==clust].index)
        if len(set(clust_pats)&set(surv.index)) < 2:
            continue
        clust_surv_data = surv.loc[clust_pats,:].dropna()
        kmf.fit(clust_surv_data.duration, clust_surv_data.observed, label='Group '+str(clust)+' (n=' +  str(len(clust_surv_data)) + ')')
        kmf.plot(ax=ax, color=cluster_cmap[clust], ci_show=False)

    if tmax!=-1:
        plt.xlim((0,tmax))
    plt.xlabel('Time (Days)', fontsize=16)
    plt.ylabel('Survival Probability', fontsize=16)
    _=plt.xticks(FontSize = 16)
    _=plt.yticks(FontSize = 16)
    # Multivariate logrank test
    if lr_test:
        cluster_survivals = pd.concat([surv, cluster_assign], axis=1).dropna().astype(int)
        p = multiv_lr_test(np.array(cluster_survivals.duration), 
                           np.array(cluster_survivals[cluster_assign.name]), t_0=tmax,
                           event_observed=np.array(cluster_survivals.observed)).p_value
        if verbose:
            print('Multi-Class Log-Rank P:', p)
        plt.title(title+'\np='+repr(round(p, 4)), fontsize=20, y=1.02)
    else:
        plt.title(title, fontsize=20, y=1.02)

    return