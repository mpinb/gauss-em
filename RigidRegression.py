"""RigidRegression.py

Defines a classifier class and helper functions for a regression that
is compatible with scikit-learn. The regression implements "rigid-body"
regression, which provides the optimal fit between two points sets
using rotation and translation (and optionally uniform scale) only.

Copyright (C) 2023 Max Planck Institute for Neurobiology of Behavior - caesar

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import scipy.linalg as lin

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def rigid_regression_loss(true, pred):
    return ((true - pred)**2).sum(1)


class RigidRegression(BaseEstimator, RegressorMixin):

    def __init__(self, demo_param='demo_param'):
        """A scikit-learn compatible "rigid-body" regression.

        Note:
            See init for scikit-learn regression classes.
        """
        self.demo_param = demo_param

    # NOTE: X and y are simply the points, i.e., the transformation includes a translation even though
    #   X is not augmented to contain a constant column of ones (as opposed to scipy linear regression).
    def fit(self, X, y):
        """scikit-learn compatiable fit method for "rigid-body" regression.

        Note:
            See fit for scikit-learn classifiers fit method.
            X and y are simply the points, i.e., the transformation includes a translation even though
            X is not augmented to contain a constant column of ones (as opposed to scipy linear regression).
        """
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True, multi_output=True)

        self.coef_, self.translation_, a, self.scale_ = RigidRegression.rigid_transform(X,y)

        if self.coef_ is not None:
            # computed angle is only valid for 2d rotations
            if X.shape[1] == 2: self.angle_ = a
            self.is_fitted_ = True
        else:
            self.is_fitted_ = False

        # `fit` should always return `self`
        return self

    def predict(self, X):
        """scikit-learn compatiable predict method for "rigid-body" regression.

        Note:
            See predict for scikit-learn classifiers fit method.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        return np.dot(X, self.coef_[:-1,:-1].T) + self.translation_

    @staticmethod
    def rigid_transform(A, B, scale=False):
        """Regression for the orthonormal prucrustes problem, i.e., rigid body.

        Also known as the Kabsch–Umeyama algorithm.
        Only rotation and translation (and optionally scale) are fit.
        Modified from: http://nghiaho.com/uploads/code/rigid_transform_3D.py_
        See also: https://en.wikipedia.org/wiki/Kabsch_algorithm

        Args:
            A (ndarray shape (npts, ndims)): Source points.
            B (ndarray shape (npts, ndims)): Destination points.
            scale (bool): Whether to fit uniform scale or not.

        Returns:
            R (ndarray shape (ndims+1, ndims+1)): Augmented affine matrix.
            t (ndarray shape (ndims)): Translation vector
            a (float): rotation angle in radians, NOTE: only valid for 2D rotation
            s (float): uniform scale, if scale==True, else 1.
        """
        npts, ndims = A.shape
        assert( npts == B.shape[0] and ndims == B.shape[1] ) # must be same number and dimension of points
        cA = A.mean(axis=0, dtype=np.double)
        cB = B.mean(axis=0, dtype=np.double)
        AA = A - cA
        BB = B - cB
        U, s, Vt = lin.svd(np.dot(AA.T,BB),overwrite_a=True,full_matrices=False)
        V = Vt.T

        R = np.dot(V,U.T)
        if lin.det(R) < 0:
            # this prevents fitting reflection which is not a rotation.
            V[:,-1] = -V[:,-1]
            s[-1] = -s[-1]
            R = np.dot(V,U.T)
        if scale:
            # this is the Umeyama addition to the Kabsch algorithm.
            # see also:
            #   https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/
            #   SO - 13432805/finding-translation-and-scale-on-two-sets-of-points-to-get-least-square-error-in
            s = s.sum() / npts / np.var(AA) / ndims
        else:
            s = 1.
        #https://math.stackexchange.com/questions/301319/derive-a-rotation-from-a-2d-rotation-matrix
        a = np.arctan2(R[1,0], R[0,0])

        # scale the rotation matrix, then fit the translation
        R *= s
        t = -np.dot(R, cA.T) + cB.T

        # return the augmented affine (rotation and translation) matrix
        Ra = np.zeros((ndims+1,ndims+1), dtype=np.double)
        Ra[:ndims,:ndims] = R
        Ra[:ndims,ndims] = t
        Ra[ndims,ndims] = 1

        return Ra, t, a, np.array([s])
