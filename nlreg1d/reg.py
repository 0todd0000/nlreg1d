

import collections
import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd
from scipy.integrate import trapz, cumtrapz
import fdasrsf.utility_functions as uf
import skfda





def _align_fPCA(f, time, num_comp=3, smoothdata=False, MaxItr=50, parallel=False, verbose=True):
    """
    
    This is a customization of the "align_fPCA" function in fdasrsf/time_warping.py
    https://github.com/jdtuck/fdasrsf_python/blob/master/fdasrsf/time_warping.py
    
    See also:
    https://github.com/jdtuck/fdasrsf_python/blob/master/notebooks/example_warping.ipynb
    
    Original function documentation follows below.
    --------
    
    aligns a collection of functions while extracting principal components.
    The functions are aligned to the principal components

    :param f: numpy ndarray of shape (M,N) of N functions with M samples
    :param time: vector of size M describing the sample points
    :param num_comp: number of fPCA components
    :param showplot: Shows plots of results using matplotlib (default = T)
    :param smooth_data: Smooth the data using a box filter (default = F)
    :param sparam: Number of times to run box filter (default = 25)
    :type sparam: double
    :type smooth_data: bool
    :type f: np.ndarray
    :type time: np.ndarray

    :rtype: tuple of numpy array
    :return fn: aligned functions - numpy ndarray of shape (M,N) of N
                functions with M samples
    :return qn: aligned srvfs - similar structure to fn
    :return q0: original srvf - similar structure to fn
    :return mqn: srvf mean or median - vector of length M
    :return gam: warping functions - similar structure to fn
    :return q_pca: srsf principal directions
    :return f_pca: functional principal directions
    :return latent: latent values
    :return coef: coefficients
    :return U: eigenvectors
    :return orig_var: Original Variance of Functions
    :return amp_var: Amplitude Variance
    :return phase_var: Phase Variance

    """
    
    lam = 0.0
    # MaxItr = 50
    coef = np.arange(-2., 3.)
    Nstd = coef.shape[0]
    M = f.shape[0]
    N = f.shape[1]
    # if M > 500:
    #     parallel = True
    # elif N > 100:
    #     parallel = True
    # else:
    #     parallel = False

    eps = np.finfo(np.double).eps
    f0 = f

    # Compute SRSF function from data
    f, g, g2 = uf.gradient_spline(time, f, smoothdata)
    q = g / np.sqrt(abs(g) + eps)

    if verbose:
        print ("Initializing...")
    mnq = q.mean(axis=1)
    a = mnq.repeat(N)
    d1 = a.reshape(M, N)
    d = (q - d1) ** 2
    dqq = np.sqrt(d.sum(axis=0))
    min_ind = dqq.argmin()

    if verbose:
        print("Aligning %d functions in SRVF space to %d fPCA components..."
          % (N, num_comp))
    itr = 0
    mq = np.zeros((M, MaxItr + 1))
    mq[:, itr] = q[:, min_ind]
    fi = np.zeros((M, N, MaxItr + 1))
    fi[:, :, 0] = f
    qi = np.zeros((M, N, MaxItr + 1))
    qi[:, :, 0] = q
    gam = np.zeros((M, N, MaxItr + 1))
    cost = np.zeros(MaxItr + 1)

    while itr < MaxItr:
        if verbose:
            print("updating step: r=%d" % (itr + 1))
        if verbose and (itr == MaxItr):
            print("maximal number of iterations is reached")

        # PCA Step
        a = mq[:, itr].repeat(N)
        d1 = a.reshape(M, N)
        qhat_cent = qi[:, :, itr] - d1
        K = np.cov(qi[:, :, itr])
        U, s, V = svd(K)

        alpha_i = np.zeros((num_comp, N))
        for ii in range(0, num_comp):
            for jj in range(0, N):
                alpha_i[ii, jj] = trapz(qhat_cent[:, jj] * U[:, ii], time)

        U1 = U[:, 0:num_comp]
        tmp = U1.dot(alpha_i)
        qhat = d1 + tmp

        # Matching Step
        if parallel:
            out = Parallel(n_jobs=-1)(
                delayed(uf.optimum_reparam)(qhat[:, n], time, qi[:, n, itr],
                                            "DP", lam) for n in range(N))
            gam_t = np.array(out)
            gam[:, :, itr] = gam_t.transpose()
        else:
            gam[:, :, itr] = uf.optimum_reparam(qhat, time, qi[:, :, itr], "DP",  lam)

        for k in range(0, N):
            time0 = (time[-1] - time[0]) * gam[:, k, itr] + time[0]
            fi[:, k, itr + 1] = np.interp(time0, time, fi[:, k, itr])
            qi[:, k, itr + 1] = uf.f_to_srsf(fi[:, k, itr + 1], time)

        qtemp = qi[:, :, itr + 1]
        mq[:, itr + 1] = qtemp.mean(axis=1)

        cost_temp = np.zeros(N)

        for ii in range(0, N):
            cost_temp[ii] = norm(qtemp[:, ii] - qhat[:, ii]) ** 2

        cost[itr + 1] = cost_temp.mean()

        if abs(cost[itr + 1] - cost[itr]) < 1e-06:
            break

        itr += 1

    if itr >= MaxItr:
        itrf = MaxItr
    else:
        itrf = itr+1
    cost = cost[1:(itrf+1)]

    # Aligned data & stats
    fn = fi[:, :, itrf]
    qn = qi[:, :, itrf]
    q0 = qi[:, :, 0]
    mean_f0 = f0.mean(axis=1)
    std_f0 = f0.std(axis=1)
    mqn = mq[:, itrf]
    gamf = gam[:, :, 0]
    for k in range(1, itr):
        gam_k = gam[:, :, k]
        for l in range(0, N):
            time0 = (time[-1] - time[0]) * gam_k[:, l] + time[0]
            gamf[:, l] = np.interp(time0, time, gamf[:, l])

    # Center Mean
    gamI = uf.SqrtMeanInverse(gamf)
    gamI_dev = np.gradient(gamI, 1 / float(M - 1))
    time0 = (time[-1] - time[0]) * gamI + time[0]
    mqn = np.interp(time0, time, mqn) * np.sqrt(gamI_dev)
    for k in range(0, N):
        qn[:, k] = np.interp(time0, time, qn[:, k]) * np.sqrt(gamI_dev)
        fn[:, k] = np.interp(time0, time, fn[:, k])
        gamf[:, k] = np.interp(time0, time, gamf[:, k])

    mean_fn = fn.mean(axis=1)
    std_fn = fn.std(axis=1)

    # Get Final PCA
    mididx = np.round(time.shape[0] / 2)
    
    mididx = int(mididx)
    
    m_new = np.sign(fn[mididx, :]) * np.sqrt(np.abs(fn[mididx, :]))
    mqn2 = np.append(mqn, m_new.mean())
    qn2 = np.vstack((qn, m_new))
    K = np.cov(qn2)

    U, s, V = svd(K)
    stdS = np.sqrt(s)

    # compute the PCA in the q domain
    q_pca = np.ndarray(shape=(M + 1, Nstd, num_comp), dtype=float)
    for k in range(0, num_comp):
        for l in range(0, Nstd):
            q_pca[:, l, k] = mqn2 + coef[l] * stdS[k] * U[:, k]

    # compute the correspondence in the f domain
    f_pca = np.ndarray(shape=(M, Nstd, num_comp), dtype=float)
    for k in range(0, num_comp):
        for l in range(0, Nstd):
            q_pca_tmp = q_pca[0:M, l, k] * np.abs(q_pca[0:M, l, k])
            q_pca_tmp2 = np.sign(q_pca[M, l, k]) * (q_pca[M, l, k] ** 2)
            f_pca[:, l, k] = uf.cumtrapzmid(time, q_pca_tmp, q_pca_tmp2, int(np.floor(time.shape[0]/2)))

    N2 = qn.shape[1]
    c = np.zeros((N2, num_comp))
    for k in range(0, num_comp):
        for l in range(0, N2):
            c[l, k] = sum((np.append(qn[:, l], m_new[l]) - mqn2) * U[:, k])

    mean_f0 = f0.mean(axis=1)
    std_f0 = f0.std(axis=1)
    mean_fn = fn.mean(axis=1)
    std_fn = fn.std(axis=1)
    tmp = np.zeros(M)
    tmp[1:] = cumtrapz(mqn * np.abs(mqn), time)
    fmean = np.mean(f0[1, :]) + tmp

    fgam = np.zeros((M, N))
    for k in range(0, N):
        time0 = (time[-1] - time[0]) * gamf[:, k] + time[0]
        fgam[:, k] = np.interp(time0, time, fmean)

    var_fgam = fgam.var(axis=1)
    orig_var = trapz(std_f0 ** 2, time)
    amp_var = trapz(std_fn ** 2, time)
    phase_var = trapz(var_fgam, time)

    K = np.cov(fn)

    U, s, V = svd(K)

    align_fPCAresults = collections.namedtuple('align_fPCA', ['fn', 'qn',
                                               'q0', 'mqn', 'gam', 'q_pca',
                                               'f_pca', 'latent', 'coef',
                                               'U', 'orig_var', 'amp_var',
                                               'phase_var', 'cost'])

    out = align_fPCAresults(fn, qn, q0, mqn, gamf, q_pca, f_pca, s, c,
                            U, orig_var, amp_var, phase_var, cost)
    return out
    
    
    
def fpca(y, ncomp=5, smooth=False, niter=5, parallel=False, verbose=True):
    '''
    Wrapper for align_fPCA
    '''
    Q         = y.shape[1]
    q         = np.linspace(0, 1, Q)
    results   = _align_fPCA(y.T, q, num_comp=ncomp, smoothdata=smooth, MaxItr=niter, parallel=parallel, verbose=verbose)
    w         = Q * (results.gam.T - q)
    return results.fn.T, w





def elastic(y, q=None, penalty=0):
    Q       = y.shape[1]
    if q is None:
        q   = np.linspace(0, 1, Q)
    fd      = skfda.FDataGrid( data_matrix=y, grid_points=q)
    er      = skfda.preprocessing.registration.ElasticRegistration(penalty=penalty)
    fdr     = er.fit_transform(fd)
    yr      = fdr.data_matrix[:,:,0]
    wr      = er.warping_.data_matrix[:,:,0]
    wr      = Q * (wr - q)
    return yr,wr