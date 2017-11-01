"""
Relative Multi-dimensional Scaling (Relative MDS)

Adapted from mds.py, refer to mds.py for complete doc
"""

# License: BSD

import numpy as np

import warnings

from ..base import BaseEstimator
from ..metrics import euclidean_distances
from ..utils import check_random_state, check_array, check_symmetric
from ..externals.joblib import Parallel
from ..externals.joblib import delayed
from ..isotonic import IsotonicRegression


def _set_fixed_points(X, fixed_points, fixed_ids):
    for idx, fixed_id in enumerate(fixed_ids):
        X[fixed_id] = fixed_points[idx]
    return X

def _smacof_single(dissimilarities, metric=True, n_components=2, init=None,
                   max_iter=300, verbose=0, eps=1e-3, random_state=None, fixed_points=None, fdissim=None, fixed_ids=None):
    """Computes multidimensional scaling using SMACOF algorithm

    If fdissim is None, compute using fixed points replacement in SMACOF
        dissimilarities will contain all dissimilarities including the fixed points (n = m + f)
    If fdissim is not None, compute using modified stress value and original SMACOF over moving points
        dissimilarities will only contain all dissimilarities between the moving points and fixed points (m = n - f)

    References
    ----------
    "Conditions for Optimal Efficiency of Relative MDS" Bernataviciene, Jolita;
    Informatica (2007)
    """
    dissimilarities = check_symmetric(dissimilarities, raise_exception=True)
    n_samples = dissimilarities.shape[0]
    f_samples = fixed_points.shape[0]

    # assert parameters are passed in correctly for given algorithm
    if (fdissim is not None and n_samples != fdissim.shape[0]) or \
            (fixed_ids is None and fdissim is None):
        raise Exception

    random_state = check_random_state(random_state)

    sim_flat = ((1 - np.tri(n_samples)) * dissimilarities).ravel()
    sim_flat_w = sim_flat[sim_flat != 0]
    if init is None:
        # Randomly choose initial configuration
        X = random_state.rand(n_samples * n_components)
        X = X.reshape((n_samples, n_components))
    else:
        # overrides the parameter p
        n_components = init.shape[1]
        if n_samples != init.shape[0]:
            raise ValueError("init matrix should be of shape (%d, %d)" %
                             (n_samples, n_components))
        X = init

    # set fixed points in X
    if fixed_ids is not None:
        X = _set_fixed_points(X, fixed_points, fixed_ids)

    # continue with SMACOF
    old_stress = None
    ir = IsotonicRegression()
    for it in range(max_iter):
        # Compute distance and monotonic regression
        dis = euclidean_distances(X)
        fdis = euclidean_distances(X, fixed_points)

        if metric:
            disparities = dissimilarities
        else:
            dis_flat = dis.ravel()
            # dissimilarities with 0 are considered as missing values
            dis_flat_w = dis_flat[sim_flat != 0]

            # Compute the disparities using a monotonic regression
            disparities_flat = ir.fit_transform(sim_flat_w, dis_flat_w)
            disparities = dis_flat.copy()
            disparities[sim_flat != 0] = disparities_flat
            disparities = disparities.reshape((n_samples, n_samples))
            disparities *= np.sqrt((n_samples * (n_samples - 1) / 2) /
                                   (disparities ** 2).sum())

        # Compute stress
        stress = ((dis.ravel() - disparities.ravel()) ** 2).sum() / 2
        if fdissim is not None:
            stress_interpoint = ((dis.ravel() - disparities.ravel()) ** 2).sum() / 2
            stress_fixed = ((fdis.ravel() - fdissim.ravel()) ** 2).sum() / 2
            stress = stress_interpoint + stress_fixed

        # Update X using the Guttman transform
        dis[dis == 0] = 1e-5
        ratio = disparities / dis
        B = - ratio
        B[np.arange(len(B)), np.arange(len(B))] += ratio.sum(axis=1)
        X = 1. / n_samples * np.dot(B, X)

        # reset fixed points in X
        if fixed_ids is not None:
            X = _set_fixed_points(X, fixed_points, fixed_ids)


        dis = np.sqrt((X ** 2).sum(axis=1)).sum()
        if verbose >= 2:
            print('it: %d, stress %s' % (it, stress))
        if old_stress is not None:
            if(old_stress - stress / dis) < eps:
                if verbose:
                    print('breaking at iteration %d with stress %s' % (it,
                                                                       stress))
                break
        old_stress = stress / dis

    return X, stress, it + 1

def smacof(dissimilarities, metric=True, n_components=2, init=None, n_init=8,
           n_jobs=1, max_iter=300, verbose=0, eps=1e-3, random_state=None,
           return_n_iter=False, fixed_points=None, fdissim=None, fixed_ids=None):
    """Computes multidimensional scaling using the SMACOF algorithm.

    The SMACOF (Scaling by MAjorizing a COmplicated Function) algorithm is a
    multidimensional scaling algorithm which minimizes an objective function
    (the *stress*) using a majorization technique. Stress majorization, also
    known as the Guttman Transform, guarantees a monotone convergence of
    stress, and is more powerful than traditional techniques such as gradient
    descent.

    The SMACOF algorithm for metric MDS can summarized by the following steps:

    1. Set an initial start configuration, randomly or not.
    2. Compute the stress
    3. Compute the Guttman Transform
    4. Iterate 2 and 3 until convergence.

    The nonmetric algorithm adds a monotonic regression step before computing
    the stress.
    """

    dissimilarities = check_array(dissimilarities)
    random_state = check_random_state(random_state)

    if hasattr(init, '__array__'):
        init = np.asarray(init).copy()
        if not n_init == 1:
            warnings.warn(
                'Explicit initial positions passed: '
                'performing only one init of the MDS instead of %d'
                % n_init)
            n_init = 1

    best_pos, best_stress = None, None

    if n_jobs == 1:
        for it in range(n_init):
            pos, stress, n_iter_ = _smacof_single(
                dissimilarities, metric=metric,
                n_components=n_components, init=init,
                max_iter=max_iter, verbose=verbose,
                eps=eps, random_state=random_state,
                fixed_points=fixed_points, fdissim=fdissim, fixed_ids=fixed_ids)
            if best_stress is None or stress < best_stress:
                best_stress = stress
                best_pos = pos.copy()
                best_iter = n_iter_
    else:
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=max(verbose - 1, 0))(
            delayed(_smacof_single)(
                dissimilarities, metric=metric, n_components=n_components,
                init=init, max_iter=max_iter, verbose=verbose, eps=eps,
                random_state=seed, fixed_points=fixed_points, fdissim=fdissim, fixed_ids=fixed_ids)
            for seed in seeds)
        positions, stress, n_iters = zip(*results)
        best = np.argmin(stress)
        best_stress = stress[best]
        best_pos = positions[best]
        best_iter = n_iters[best]

    if return_n_iter:
        return best_pos, best_stress, best_iter
    else:
        return best_pos, best_stress


class RMDS(BaseEstimator):
    """
    Relative Multidimensional scaling

    Parameters
    ----------
    fixed_points : ndarray, shape (n_basis, n_components)
        Coordinates of the projection of basis data

    fdissim : ndarray, shape (n_sample, n_basis)
        Precomputed actual distance of samples from basis

    fixed_ids : ndarray, shape (n_basis, 1)
        Index values of the fixed points to re-insert the fixed points in the correct place
    """
    def __init__(self, n_components=2, metric=True, n_init=4,
                 max_iter=300, verbose=0, eps=1e-3, n_jobs=1,
                 random_state=None, dissimilarity="euclidean", fixed_points=None, fdissim=None, fixed_ids=None):
        self.n_components = n_components
        self.dissimilarity = dissimilarity
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.fixed_points = fixed_points
        self.fdissim = fdissim
        self.fixed_ids = fixed_ids

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def fit(self, X, y=None, init=None):
        """
        Computes the position of the points in the embedding space
        """
        self.fit_transform(X, init=init)
        return self

    def fit_transform(self, X, y=None, init=None):
        """
        Fit the data from X, and returns the embedded coordinates
        """
        X = check_array(X)
        if X.shape[0] == X.shape[1] and self.dissimilarity != "precomputed":
            warnings.warn("The MDS API has changed. ``fit`` now constructs an"
                          " dissimilarity matrix from data. To use a custom "
                          "dissimilarity matrix, set "
                          "``dissimilarity='precomputed'``.")

        if self.dissimilarity == "precomputed":
            self.dissimilarity_matrix_ = X
        elif self.dissimilarity == "euclidean":
            self.dissimilarity_matrix_ = euclidean_distances(X)
        else:
            raise ValueError("Proximity must be 'precomputed' or 'euclidean'."
                             " Got %s instead" % str(self.dissimilarity))

        self.embedding_, self.stress_, self.n_iter_ = smacof(
            self.dissimilarity_matrix_, metric=self.metric,
            n_components=self.n_components, init=init, n_init=self.n_init,
            n_jobs=self.n_jobs, max_iter=self.max_iter, verbose=self.verbose,
            eps=self.eps, random_state=self.random_state, return_n_iter=True,
            fixed_points=self.fixed_points, fdissim=self.fdissim, fixed_ids=self.fixed_ids)

        return self.embedding_
