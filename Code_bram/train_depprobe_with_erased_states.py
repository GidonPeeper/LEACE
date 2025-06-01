#!/usr/bin/python3
from collections import defaultdict
import argparse, logging, os, sys, time
import hashlib, pickle
from datasets import Dataset
from dataclasses import dataclass
from torch import Tensor
from typing import Literal

import math

import gc
import torch
import torch.nn as nn
import transformers

import logging, os, sys, time

import numpy as np
import torch
from collections import defaultdict

import logging, os, re

from collections import OrderedDict

from functools import wraps
from typing import Callable

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


"""
Eleuther AI caching functions
"""
def cached_property(func: Callable) -> property:
    """Decorator that converts a method into a lazily-evaluated cached property"""
    # Create a secret attribute name for the cached property
    attr_name = "_cached_" + func.__name__

    @property
    @wraps(func)
    def _cached_property(self):
        # If the secret attribute doesn't exist, compute the property and set it
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))

        # Otherwise, return the cached property
        return getattr(self, attr_name)

    return _cached_property


def invalidates_cache(dependent_prop_name: str) -> Callable:
    """Invalidates a cached property when the decorated function is called"""
    attr_name = "_cached_" + dependent_prop_name

    # The actual decorator
    def _invalidates_cache(func: Callable) -> Callable:
        # The wrapper function
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if the secret attribute exists; if so delete it so that
            # the cached property is recomputed
            if hasattr(self, attr_name):
                delattr(self, attr_name)

            return func(self, *args, **kwargs)

        return wrapper

    return _invalidates_cache

"""
Oracle functions
"""

@dataclass(frozen=True)
class OracleEraser:
    """Surgically erases a concept from a representation, given concept labels."""

    coef: Tensor
    mean_z: Tensor

    @classmethod
    def fit(cls, x: Tensor, z: Tensor, **kwargs) -> "OracleEraser":
        """Convenience method to fit an OracleEraser on data and return it."""
        return OracleFitter.fit(x, z, **kwargs).eraser

    def __call__(self, x: Tensor, z: Tensor) -> Tensor:
        """Replace `x` with the OLS residual given `z`."""
        # Ensure Z is at least 2D
        z = z.reshape(len(z), -1).type_as(x)
        expected_x = (z - self.mean_z) @ self.coef.T

        return x.sub(expected_x).type_as(x)


class OracleFitter:
    """Compute stats needed for surgically erasing a concept Z from a random vector X.

    Unlike `LeaceFitter`, the resulting erasure function requires oracle concept labels
    at inference time. In exchange, it achieves more surgical edits.
    """

    mean_x: Tensor
    """Running mean of X."""

    mean_z: Tensor
    """Running mean of Z."""

    sigma_xz_: Tensor
    """Unnormalized cross-covariance matrix X^T Z."""

    sigma_zz_: Tensor
    """Unnormalized covariance matrix Z^T Z."""

    n: Tensor
    """Number of X samples seen so far."""

    @classmethod
    def fit(cls, x: Tensor, z: Tensor, **kwargs) -> "OracleFitter":
        """Convenience method to fit a OracleFitter on data and return it."""
        n, d = x.shape
        _, k = z.reshape(n, -1).shape

        fitter = OracleFitter(d, k, device=x.device, dtype=x.dtype, **kwargs)
        return fitter.update(x, z)

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        shrinkage: bool = False,
        svd_tol: float = 0.01,
    ):
        """Initialize a `OracleFitter`.

        Args:
            x_dim: Dimensionality of the representation.
            z_dim: Dimensionality of the concept.
            method: Type of projection matrix to use.
            device: Device to put the statistics on.
            dtype: Data type to use for the statistics.
            shrinkage: Whether to use shrinkage to estimate the covariance matrix of X.
            svd_tol: Threshold for singular values of the covariance matrix of Z.
        """
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim

        self.shrinkage = shrinkage

        assert svd_tol > 0.0, "`svd_tol` must be positive for numerical stability."
        self.svd_tol = svd_tol

        self.mean_x = torch.zeros(x_dim, device=device, dtype=dtype)
        self.mean_z = torch.zeros(z_dim, device=device, dtype=dtype)

        self.n = torch.tensor(0, device=device)
        self.sigma_xz_ = torch.zeros(x_dim, z_dim, device=device, dtype=dtype)
        self.sigma_zz_ = torch.zeros(z_dim, z_dim, device=device, dtype=dtype)

    @torch.no_grad()
    @invalidates_cache("eraser")
    def update(self, x: Tensor, z: Tensor) -> "OracleFitter":
        """Update the running statistics with a new batch of data."""
        d, c = self.sigma_xz_.shape
        x = x.reshape(-1, d).type_as(self.mean_x)
        n, d2 = x.shape

        assert d == d2, f"Unexpected number of features {d2}"
        self.n += n

        # Welford's online algorithm
        delta_x = x - self.mean_x
        self.mean_x += delta_x.sum(dim=0) / self.n

        z = z.reshape(n, -1).type_as(x)
        assert z.shape[-1] == c, f"Unexpected number of classes {z.shape[-1]}"

        delta_z = z - self.mean_z
        self.mean_z += delta_z.sum(dim=0) / self.n
        delta_z2 = z - self.mean_z

        self.sigma_xz_.addmm_(delta_x.mH, delta_z2)
        self.sigma_zz_.addmm_(delta_z.mH, delta_z2)

        return self

    @cached_property
    def eraser(self) -> OracleEraser:
        """Erasure function lazily computed given the current statistics."""
        return OracleEraser(
            self.sigma_xz @ torch.linalg.pinv(self.sigma_zz, atol=self.svd_tol),
            self.mean_z,
        )

    @property
    def sigma_zz(self) -> Tensor:
        """The covariance matrix of Z."""
        assert self.n > 1, "Call update() before accessing sigma_xx"
        assert (
            self.sigma_zz_ is not None
        ), "Covariance statistics are not being tracked for X"

        # Accumulated numerical error may cause this to be slightly non-symmetric
        S_hat = (self.sigma_zz_ + self.sigma_zz_.mH) / 2

        # Apply Random Matrix Theory-based shrinkage
        if self.shrinkage:
            return optimal_linear_shrinkage(S_hat / self.n, self.n)

        # Just apply Bessel's correction
        else:
            return S_hat / (self.n - 1)

    @property
    def sigma_xz(self) -> Tensor:
        """The cross-covariance matrix."""
        assert self.n > 1, "Call update() with labels before accessing sigma_xz"
        return self.sigma_xz_ / (self.n - 1)

"""
Functions needed for Leace
"""

def optimal_linear_shrinkage(
    S_n: Tensor, n: int | Tensor, *, inplace: bool = False
) -> Tensor:
    """Optimal linear shrinkage for a sample covariance matrix or batch thereof.

    Given a sample covariance matrix `S_n` of shape (*, p, p) and a sample size `n`,
    this function computes the optimal shrinkage coefficients `alpha` and `beta`, then
    returns the covariance estimate `alpha * S_n + beta * Sigma0`, where `Sigma0` is
    an isotropic covariance matrix with the same trace as `S_n`.

    The formula is distribution-free and asymptotically optimal in the Frobenius norm
    among all linear shrinkage estimators as the dimensionality `p` and sample size `n`
    jointly tend to infinity, with the ratio `p / n` converging to a finite positive
    constant `c`. The derivation is based on Random Matrix Theory and assumes that the
    underlying distribution has finite moments up to 4 + eps, for some eps > 0.

    See "On the Strong Convergence of the Optimal Linear Shrinkage Estimator for Large
    Dimensional Covariance Matrix" <https://arxiv.org/abs/1308.2608> for details.

    Args:
        S_n: Sample covariance matrices of shape (*, p, p).
        n: Sample size.
    """
    p = S_n.shape[-1]
    assert S_n.shape[-2:] == (p, p)

    trace_S = trace(S_n)

    # Since sigma0 is I * tr(S_n) / p, its squared Frobenius norm is tr(S_n) ** 2 / p.
    sigma0_norm_sq = trace_S**2 / p
    S_norm_sq = S_n.norm(dim=(-2, -1), keepdim=True) ** 2

    prod_trace = sigma0_norm_sq
    top = trace_S * trace_S.conj() * sigma0_norm_sq / n
    bottom = S_norm_sq * sigma0_norm_sq - prod_trace * prod_trace.conj()

    # Epsilon prevents dividing by zero for the zero matrix. In that case we end up
    # setting alpha = 0, beta = 1, but it doesn't matter since we're shrinking toward
    # tr(0)*I = 0, so it's a no-op.
    eps = torch.finfo(S_n.dtype).eps
    alpha = 1 - (top + eps) / (bottom + eps)
    beta = (1 - alpha) * (prod_trace + eps) / (sigma0_norm_sq + eps)

    ret = S_n.mul_(alpha) if inplace else alpha * S_n
    diag = beta * trace_S / p
    torch.linalg.diagonal(ret).add_(diag.squeeze(-1))
    return ret

def trace(matrices: Tensor) -> Tensor:
    """Version of `torch.trace` that works for batches of matrices."""
    diag = torch.linalg.diagonal(matrices)
    return diag.sum(dim=-1, keepdim=True).unsqueeze(-1)


def cached_property(func: Callable) -> property:
    """Decorator that converts a method into a lazily-evaluated cached property"""
    # Create a secret attribute name for the cached property
    attr_name = "_cached_" + func.__name__

    @property
    @wraps(func)
    def _cached_property(self):
        # If the secret attribute doesn't exist, compute the property and set it
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))

        # Otherwise, return the cached property
        return getattr(self, attr_name)

    return _cached_property


def invalidates_cache(dependent_prop_name: str) -> Callable:
    """Invalidates a cached property when the decorated function is called"""
    attr_name = "_cached_" + dependent_prop_name

    # The actual decorator
    def _invalidates_cache(func: Callable) -> Callable:
        # The wrapper function
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if the secret attribute exists; if so delete it so that
            # the cached property is recomputed
            if hasattr(self, attr_name):
                delattr(self, attr_name)

            return func(self, *args, **kwargs)

        return wrapper

    return _invalidates_cache

"""
Leace functions
"""
ErasureMethod = Literal["leace", "orth"]


@dataclass(frozen=True)
class LeaceEraser:
    """LEACE eraser that surgically erases a concept from a representation.

    Since the LEACE projection matrix is guaranteed to be a rank k - 1 perturbation of
    the identity, we store it implicitly in the d x k matrices `proj_left` and
    `proj_right`. The full matrix is given by `torch.eye(d) - proj_left @ proj_right`.
    """

    proj_left: Tensor
    proj_right: Tensor
    bias: Tensor | None

    @classmethod
    def fit(cls, x: Tensor, z: Tensor, **kwargs) -> "LeaceEraser":
        """Convenience method to fit a LeaceEraser on data and return it."""
        return LeaceFitter.fit(x, z, **kwargs).eraser

    @property
    def P(self) -> Tensor:
        """The projection matrix."""
        eye = torch.eye(
            self.proj_left.shape[0],
            device=self.proj_left.device,
            dtype=self.proj_left.dtype,
        )
        return eye - self.proj_left @ self.proj_right

    def __call__(self, x: Tensor) -> Tensor:
        """Apply the projection to the input tensor."""
        delta = x - self.bias if self.bias is not None else x

        # Ensure we do the matmul in the most efficient order.
        x_ = x - (delta @ self.proj_right.mH) @ self.proj_left.mH
        return x_.type_as(x)


class LeaceFitter:
    """Fits an affine transform that surgically erases a concept from a representation.

    This class implements Least-squares Concept Erasure (LEACE) from
    https://arxiv.org/abs/2306.03819. You can also use a slightly simpler orthogonal
    projection-based method by setting `method="orth"`.

    This class stores all the covariance statistics needed to compute the LEACE eraser.
    This allows the statistics to be updated incrementally with `update()`.
    """

    mean_x: Tensor
    """Running mean of X."""

    mean_z: Tensor
    """Running mean of Z."""

    sigma_xz_: Tensor
    """Unnormalized cross-covariance matrix X^T Z."""

    sigma_xx_: Tensor | None
    """Unnormalized covariance matrix X^T X."""

    n: Tensor
    """Number of X samples seen so far."""

    @classmethod
    def fit(cls, x: Tensor, z: Tensor, **kwargs) -> "LeaceFitter":
        """Convenience method to fit a LeaceFitter on data and return it."""
        n, d = x.shape
        _, k = z.reshape(n, -1).shape

        fitter = LeaceFitter(d, k, device=x.device, dtype=x.dtype, **kwargs)
        return fitter.update(x, z)

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        method: ErasureMethod = "leace",
        *,
        affine: bool = True,
        constrain_cov_trace: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        shrinkage: bool = True,
        svd_tol: float = 0.01,
    ):
        """Initialize a `LeaceFitter`.

        Args:
            x_dim: Dimensionality of the representation.
            z_dim: Dimensionality of the concept.
            method: Type of projection matrix to use.
            affine: Whether to use a bias term to ensure the unconditional mean of the
                features remains the same after erasure.
            constrain_cov_trace: Whether to constrain the trace of the covariance of X
                after erasure to be no greater than before erasure. This is especially
                useful when injecting the scrubbed features back into a model. Without
                this constraint, the norm of the model's hidden states may diverge in
                some cases.
            device: Device to put the statistics on.
            dtype: Data type to use for the statistics.
            shrinkage: Whether to use shrinkage to estimate the covariance matrix of X.
            svd_tol: Singular values under this threshold are truncated, both during
                the phase where we do SVD on the cross-covariance matrix, and at the
                phase where we compute the pseudoinverse of the projected covariance
                matrix. Higher values are more numerically stable and result in less
                damage to the representation, but may leave trace correlations intact.
        """
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim

        self.affine = affine
        self.constrain_cov_trace = constrain_cov_trace
        self.method = method
        self.shrinkage = shrinkage

        assert svd_tol > 0.0, "`svd_tol` must be positive for numerical stability."
        self.svd_tol = svd_tol

        self.mean_x = torch.zeros(x_dim, device=device, dtype=dtype)
        self.mean_z = torch.zeros(z_dim, device=device, dtype=dtype)

        self.n = torch.tensor(0, device=device)
        self.sigma_xz_ = torch.zeros(x_dim, z_dim, device=device, dtype=dtype)

        if self.method == "leace":
            self.sigma_xx_ = torch.zeros(x_dim, x_dim, device=device, dtype=dtype)
        elif self.method == "orth":
            self.sigma_xx_ = None
        else:
            raise ValueError(f"Unknown projection type {self.method}")

    @torch.no_grad()
    @invalidates_cache("eraser")
    def update(self, x: Tensor, z: Tensor) -> "LeaceFitter":
        """Update the running statistics with a new batch of data."""
        d, c = self.sigma_xz_.shape
        x = x.reshape(-1, d).type_as(self.mean_x)
        n, d2 = x.shape

        assert d == d2, f"Unexpected number of features {d2}"
        self.n += n

        # Welford's online algorithm
        delta_x = x - self.mean_x
        self.mean_x += delta_x.sum(dim=0) / self.n
        delta_x2 = x - self.mean_x

        # Update the covariance matrix of X if needed (for LEACE)
        if self.method == "leace":
            assert self.sigma_xx_ is not None
            self.sigma_xx_.addmm_(delta_x.mH, delta_x2)

        z = z.reshape(n, -1).type_as(x)
        assert z.shape[-1] == c, f"Unexpected number of classes {z.shape[-1]}"

        delta_z = z - self.mean_z
        self.mean_z += delta_z.sum(dim=0) / self.n
        delta_z2 = z - self.mean_z

        # Update the cross-covariance matrix
        self.sigma_xz_.addmm_(delta_x.mH, delta_z2)

        return self

    @cached_property
    def eraser(self) -> LeaceEraser:
        """Erasure function lazily computed given the current statistics."""
        eye = torch.eye(self.x_dim, device=self.mean_x.device, dtype=self.mean_x.dtype)

        # Compute the whitening and unwhitening matrices
        if self.method == "leace":
            sigma = self.sigma_xx
            L, V = torch.linalg.eigh(sigma)

            # Threshold used by torch.linalg.pinv
            mask = L > (L[-1] * sigma.shape[-1] * torch.finfo(L.dtype).eps)

            # Assuming PSD; account for numerical error
            L.clamp_min_(0.0)

            W = V * torch.where(mask, L.rsqrt(), 0.0) @ V.mH
            W_inv = V * torch.where(mask, L.sqrt(), 0.0) @ V.mH
        else:
            W, W_inv = eye, eye

        u, s, _ = torch.linalg.svd(W @ self.sigma_xz, full_matrices=False)

        # Throw away singular values that are too small
        u *= s > self.svd_tol

        proj_left = W_inv @ u
        proj_right = u.mH @ W

        if self.constrain_cov_trace and self.method == "leace":
            P = eye - proj_left @ proj_right

            # Prevent the covariance trace from increasing
            sigma = self.sigma_xx
            old_trace = torch.trace(sigma)
            new_trace = torch.trace(P @ sigma @ P.mH)

            # If applying the projection matrix increases the variance, this might
            # cause instability, especially when erasure is applied multiple times.
            # We regularize toward the orthogonal projection matrix to avoid this.
            if new_trace.real > old_trace.real:
                Q = eye - u @ u.mH

                # Set up the variables for the quadratic equation
                x = new_trace
                y = 2 * torch.trace(P @ sigma @ Q.mH)
                z = torch.trace(Q @ sigma @ Q.mH)
                w = old_trace

                # Solve for the mixture of P and Q that makes the trace equal to the
                # trace of the original covariance matrix
                discr = torch.sqrt(
                    4 * w * x - 4 * w * y + 4 * w * z - 4 * x * z + y**2
                )
                alpha1 = (-y / 2 + z - discr / 2) / (x - y + z)
                alpha2 = (-y / 2 + z + discr / 2) / (x - y + z)

                # Choose the positive root
                alpha = torch.where(alpha1.real > 0, alpha1, alpha2).clamp(0, 1)
                P = alpha * P + (1 - alpha) * Q

                # TODO: Avoid using SVD here
                u, s, vh = torch.linalg.svd(eye - P)
                proj_left = u * s.sqrt()
                proj_right = vh * s.sqrt()

        return LeaceEraser(
            proj_left, proj_right, bias=self.mean_x if self.affine else None
        )

    @property
    def sigma_xx(self) -> Tensor:
        """The covariance matrix of X."""
        assert self.n > 1, "Call update() before accessing sigma_xx"
        assert (
            self.sigma_xx_ is not None
        ), "Covariance statistics are not being tracked for X"

        # Accumulated numerical error may cause this to be slightly non-symmetric
        S_hat = (self.sigma_xx_ + self.sigma_xx_.mH) / 2

        # Apply Random Matrix Theory-based shrinkage
        if self.shrinkage:
            return optimal_linear_shrinkage(S_hat / self.n, self.n, inplace=True)

        # Just apply Bessel's correction
        else:
            return S_hat / (self.n - 1)

    @property
    def sigma_xz(self) -> Tensor:
        """The cross-covariance matrix."""
        assert self.n > 1, "Call update() with labels before accessing sigma_xz"
        return self.sigma_xz_ / (self.n - 1)
    

"""
Universal Dependencies functions and classes for parsing and filtering UD data.
"""
#
# Primary Universal Dependencies Data Classes
#


class UniversalDependencies:
    def __init__(self, treebanks=[]):
        self._treebanks = treebanks
        self._index_map = self._build_index_map() # corpus_index -> (treebank_index, sentence_index)

    def __repr__(self):
        return f'<UniversalDependencies: {len(self._treebanks)} treebanks, {len(self)} sentences>'

    def __len__(self):
        # returns total number of sentences across all treebanks
        return len(self._index_map)

    def __getitem__(self, key):
        if type(key) is slice:
            return [self._treebanks[tbidx][sidx] for tbidx, sidx in self._index_map[key]]
        elif type(key) is list:
            return [self._treebanks[self._index_map[key_idx][0]][self._index_map[key_idx][1]] for key_idx in key]
        else:
            tbidx, sidx = self._index_map[key]
            return self._treebanks[tbidx][sidx]

    def __setitem__(self, key, val):
        if type(key) is slice:
            for vidx, (tbidx, sidx) in enumerate(self._index_map[key]):
                self._treebanks[tbidx][sidx] = val[vidx]
        elif type(key) is list:
            for kidx, v in zip(key, val):
                tbidx, sidx = self._index_map[kidx]
                self._treebanks[tbidx][sidx] = v
        else:
            tbidx, sidx = self._index_map[key]
            self._treebanks[tbidx][sidx] = val

    def _build_index_map(self):
        index_map = []

        for tbidx, tb in enumerate(self._treebanks):
            tb_sentence_indices = list(range(len(tb))) # [0 ... num_sentences-1]
            tb_index_map = list(zip([tbidx for _ in range(len(tb))], tb_sentence_indices)) # [(tbidx, 0) ... (tbidx, num_sentences-1)]

            index_map += tb_index_map

        return index_map

    def _get_sentences_by_criterion(self, criterion):
        cur_key, cur_sentences = None, []
        for sidx in range(len(self)):
            # check if current key has changed based on criterion function
            if (criterion(sidx) != cur_key) or (sidx == len(self) - 1):
                if cur_key is not None:
                    yield cur_key, (cur_sentences + [self[sidx]] if (sidx == len(self) - 1) else cur_sentences)
                # start gathering sentences of new key
                cur_key = criterion(sidx)
                cur_sentences = []
            cur_sentences.append(self[sidx])

    @staticmethod
    def from_directory(path, ud_filter=None, verbose=False):
        treebanks = []
        cursor = 0

        # gather treebank directories
        for tb_dir in sorted(os.listdir(path)):
            tb_path = os.path.join(path, tb_dir)

            print(tb_path)

            # skip non-directories
            if not os.path.isdir(tb_path):
                continue

            # parse TB dirname
            tb_name_match = re.match(r'UD_(.+)-(.+)', tb_dir)
            if not tb_name_match:
                continue
            language = tb_name_match[1].replace('_', ' ')
            tb_name = tb_name_match[2]

            # initialize TB metadata
            tb_meta = {
                'Language': language,
                'Treebank': tb_name
            }

            # iterate over files in TB directory
            for tbf in sorted(os.listdir(tb_path)):
                tbf_path = os.path.join(tb_path, tbf)
                print(tbf_path)

                # if README
                if tbf.startswith('README'):
                    # parse README metadata
                    with open(tbf_path, 'r', encoding='utf8') as fp:
                        readme = fp.read()
                    metadata = re.search(r'[-=]+ Machine[-\s]readable metadata(.+)', readme, flags=re.DOTALL)
                    if metadata is None: continue

                    for meta_line in metadata[1].split('\n'):
                        meta_line = meta_line.strip()
                        # skip comments
                        if meta_line.startswith('==='): continue
                        # extract metadata from 'key: value'
                        if len(meta_line.split(': ')) != 2: continue
                        meta_key, meta_value = meta_line.split(': ')
                        tb_meta[meta_key] = meta_value

                # skip non-conllu files
                if os.path.splitext(tbf)[1] != '.conllu': continue

                # load treebank
                treebank = UniversalDependenciesTreebank.from_conllu(tbf_path, name=tbf, meta=tb_meta, start_idx=cursor, ud_filter=ud_filter)
                treebanks.append(treebank)
                cursor += len(treebank)

                # print statistics (if verbose)
                if verbose:
                    info = f"Loaded {treebank}."
                    if logging.getLogger().hasHandlers():
                        logging.info(info)
                    else:
                        print(info)

        return UniversalDependencies(treebanks=treebanks)

    def get_treebanks(self):
        return self._treebanks

    def get_domains(self):
        return sorted({d for tb in self.get_treebanks() for d in tb.get_domains()})

    def get_relations(self, include_subtypes=False):
        relations = set()
        for sidx in range(len(self)):
            sentence = self[sidx]
            if sentence is None: continue
            relations |= set(sentence.get_dependencies(include_subtypes=include_subtypes)[1])

        return sorted(relations)

    def get_language_of_index(self, key):
        return self._treebanks[self._index_map[key][0]].get_language()

    def get_treebank_name_of_index(self, key):
        return self._treebanks[self._index_map[key][0]].get_treebank_name()

    def get_treebank_file_of_index(self, key):
        return self._treebanks[self._index_map[key][0]].get_name()

    def get_domains_of_index(self, key):
        return self._treebanks[self._index_map[key][0]].get_domains()

    def get_sentences_by_language(self):
        for language, sentences in self._get_sentences_by_criterion(self.get_language_of_index):
            yield language, sentences

    def get_sentences_by_treebank(self):
        cursor = 0
        for treebank, sentences in self._get_sentences_by_criterion(self.get_treebank_name_of_index):
            yield f'{self.get_language_of_index(cursor)}-{treebank}', sentences
            cursor += len(sentences)

    def get_sentences_by_file(self):
        for tb_file, sentences in self._get_sentences_by_criterion(self.get_treebank_file_of_index):
            yield tb_file, sentences


class UniversalDependenciesTreebank:
    def __init__(self, sentences=[], name=None, meta={}):
        self._sentences = sentences
        self._name = name
        self._meta = meta

    def __repr__(self):
        return f'<UniversalDependenciesTreebank{f" ({self._name})" if self._name else ""}: {len(self._sentences)} sentences>'

    def __len__(self):
        return len(self._sentences)

    def __getitem__(self, key):
        return self._sentences[key]

    def __setitem__(self, key, val):
        self._sentences[key] = val

    @staticmethod
    def from_conllu(path, name=None, meta=None, start_idx=0, ud_filter=None):
        sentences = []
        with open(path, 'r', encoding='utf8') as fp:
            cur_lines = []
            for line_idx, line in enumerate(fp):
                line = line.strip()
                # on blank line, construct full sentence from preceding lines
                if line == '':
                    try:
                        # parse sentence from current set of lines
                        sentence = UniversalDependenciesSentence.from_conllu(start_idx + len(sentences), cur_lines)
                        # if filter is set, set any sentences not matching the filter to None
                        if (ud_filter is not None) and (not ud_filter(sentence, meta)): sentence = None
                        # append sentence to results
                        sentences.append(sentence)
                    except Exception as err:
                        warn_msg = f"[Warning] UniversalDependenciesTreebank: Unable to parse '{path}' line {line_idx} ({err}). Skipping."
                        if logging.getLogger().hasHandlers():
                            logging.warning(warn_msg)
                        else:
                            print(warn_msg)
                    cur_lines = []
                    continue
                cur_lines.append(line)
        return UniversalDependenciesTreebank(sentences=sentences, name=name, meta=meta)

    def to_tokens(self):
        sentences = []
        for sentence in self:
            sentences.append(sentence.to_tokens())
        return sentences

    def to_words(self):
        sentences = []
        for sentence in self:
            sentences.append(sentence.to_words())
        return sentences

    def to_conllu(self, comments=True, resolve=False):
        return ''.join([s.to_conllu(comments, resolve) for s in self._sentences])

    def get_sentences(self):
        return self._sentences

    def get_name(self):
        return self._name

    def get_treebank_name(self):
        return self._meta.get('Treebank', 'Unknown')

    def get_language(self):
        return self._meta.get('Language', 'Unknown')

    def get_domains(self):
        return sorted(self._meta.get('Genre', '').split(' '))

    def get_statistics(self):
        statistics = {
            'sentences': len(self._sentences),
            'tokens': 0,
            'words': 0,
            'metadata': set()
        }

        for sidx, sentence in enumerate(self):
            statistics['tokens'] += len(sentence.to_tokens(as_str=False))
            statistics['words'] += len(sentence.to_words(as_str=False))
            statistics['metadata'] |= set(sentence.get_metadata().keys())

        statistics['metadata'] = list(sorted(statistics['metadata']))

        return statistics


class UniversalDependenciesSentence:
    def __init__(self, idx, tokens, comments=[]):
        self.idx = idx
        self._tokens = tokens
        self._comments = comments

    def __repr__(self):
        return f"<UniversalDependenciesSentence: ID {self.idx}, {len(self._tokens)} tokens, {len(self._comments)} comments>"

    @staticmethod
    def from_conllu(idx, lines):
        tokens, comments = [], []
        line_idx = 0
        while line_idx < len(lines):
            # check for comment
            if lines[line_idx].startswith('#'):
                comments.append(lines[line_idx])
                line_idx += 1
                continue

            # process tokens
            tkn_words = []
            tkn_line_split = lines[line_idx].split('\t')
            tkn_idx_str = tkn_line_split[0]
            # check for multiword token in 'a-b' format
            num_words = 1
            if '-' in tkn_idx_str:
                tkn_idx_split = tkn_idx_str.split('-')
                # convert token id to tuple signifying range (e.g. (3,4))
                tkn_span = (int(tkn_idx_split[0]), int(tkn_idx_split[1]))
                # collect the number of words in the current span
                while (line_idx + num_words + 1) < len(lines):
                    num_words += 1
                    # get current index as float due to spans such as '1-2; 1; 2; 2.1; ... 3' (e.g. Arabic data)
                    span_str = lines[line_idx+num_words].split('\t')[0]
                    if '-' in span_str: break
                    span_tkn_idx = float(span_str)
                    if int(span_tkn_idx) > tkn_span[1]: break
            # check for multiword token in decimal format '1; 1.1; 1.2; ... 2' or '0.1; 0.2; ... 1' (e.g. Czech data)
            elif re.match(r'^\d+\.\d+', tkn_idx_str)\
                or ((line_idx < (len(lines) - 1)) and re.match(r'^\d+\.\d+\t', lines[line_idx+1])):
                # count words that are part of multiword token
                while (line_idx + num_words) < len(lines):
                    if not re.match(r'^\d+\.\d+\t', lines[line_idx+num_words]):
                        break
                    num_words += 1
                # token span for decimal indices is (a.1, a.n)
                tkn_span_start = float(tkn_idx_str) if re.match(r'^\d+\.\d+', tkn_idx_str) else int(tkn_idx_str) + .1
                tkn_span_end = tkn_span_start + (.1 * (num_words - 1))
                tkn_span = (tkn_span_start, tkn_span_end)
            # if single word token
            else:
                # convert token id to tuple with range 1 (e.g. (3,3))
                tkn_span = (int(tkn_idx_str), int(tkn_idx_str))

            # construct words contained in token
            for word_line in lines[line_idx:line_idx + num_words]:
                tkn_words.append(UniversalDependenciesWord.from_conllu(word_line))
            # construct and append token
            tokens.append(UniversalDependenciesToken(idx=tkn_span, words=tkn_words))
            # increment line index by number of words in token
            line_idx += num_words

        return UniversalDependenciesSentence(idx=idx, tokens=tokens, comments=comments)

    def to_text(self):
        return ''.join([t.to_text() for t in self._tokens])

    def to_tokens(self, as_str=True):
        return [(t.get_form() if as_str else t) for t in self._tokens]

    def to_words(self, as_str=True):
        return [(w.get_form() if as_str else w) for token in self._tokens for w in token.to_words()]

    def to_conllu(self, comments=True, resolve=False):
        conllu = '\n'.join(self._comments) + '\n' if comments and self._comments else ''

        conllu += '\n'.join([t.to_conllu(resolve=resolve) for t in self._tokens])
        conllu += '\n\n'

        return conllu

    def get_dependencies(self, offset=-1, include_subtypes=True):
        heads = [
            (w.head + offset)
            for token in self._tokens for w in token.to_words()
        ]
        labels = [
            w.deprel if include_subtypes else w.deprel.split(':')[0]
            for token in self._tokens for w in token.to_words()
        ]
        return heads, labels

    def get_pos(self):
        pos = [
            w.upostag for token in self._tokens for w in token.to_words()
        ]
        return pos

    def get_comments(self, stripped=True):
        return [c[1:].strip() for c in self._comments]

    def get_metadata(self):
        """Returns metadata from the comments of a sentence.

        Comment should follow the UD metadata guidelines '# FIELD = VALUE' or '# FIELD VALUE.
        Lines not following this convention are exported in the 'unknown' field.

        Returns a dict of metadata field and value pairs {'FIELD': 'VALUE'}.
        """
        metadata = {}
        md_patterns = [r'^# ?(.+?) ?= ?(.+)', r'^# ?([^\s]+?)\s([^\s]+)$']
        for comment in self._comments:
            for md_pattern in md_patterns:
                md_match = re.match(md_pattern, comment)
                if md_match:
                    metadata[md_match[1]] = md_match[2]
                    break
            else:
                metadata['unknown'] = metadata.get('unknown', []) + [comment[1:].strip()]
        return metadata


class UniversalDependenciesToken:
    def __init__(self, idx, words):
        self.idx = idx # expects int or float tuple
        self._words = words # first element is token form, all following belong to potential multiword tokens

    def to_text(self):
        return self._words[0].to_text()

    def to_words(self):
        # if single word token
        if len(self._words) == 1:
            return self._words if self._words[0].head is not None else []
        # if multiword token
        else:
            # return words which have a dependency head
            return [w for w in self._words if w.head is not None]

    def to_conllu(self, resolve=False):
        # resolve multiword tokens into its constituents
        if resolve:
            # if form token has no head (e.g. 'i-j' token), get constituent words
            if (self._words[0].head is None) and (len(self._words) > 1):
                return '\n'.join([w.to_conllu() for w in self._words[1:] if w.head is not None])
            # if form token has head or it is not a multiword token, return itself
            elif self._words[0].head is not None:
                return self._words[0].to_conllu()
            # if token consists of only one word which has no head, omit (e.g. Coptic '0.1')
            else:
                return ''
        # otherwise return full set of words
        else:
            return '\n'.join([w.to_conllu() for w in self._words])

    def get_form(self):
        return self._words[0].get_form()


class UniversalDependenciesWord:
    """
    ID: Word index, integer starting at 1 for each new sentence; may be a range for tokens with multiple words.
    FORM: Word form or punctuation symbol.
    LEMMA: Lemma or stem of word form.
    UPOSTAG: Universal part-of-speech tag drawn from our revised version of the Google universal POS tags.
    XPOSTAG: Language-specific part-of-speech tag; underscore if not available.
    FEATS: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
    HEAD: Head of the current token, which is either a value of ID or zero (0).
    DEPREL: Universal Stanford dependency relation to the HEAD (root iff HEAD = 0) or a defined language-specific subtype of one.
    DEPS: List of secondary dependencies (head-deprel pairs).
    MISC: Any other annotation.

    [1] https://universaldependencies.org/docs/format.html
    """
    def __init__(self, idx, form, lemma, upostag, xpostag, feats, head, deprel, deps=None, misc=None):
        self.idx = idx # expects int, float or str
        self.form = form
        self.lemma = lemma
        self.upostag = upostag
        self.xpostag = xpostag
        self.feats = feats # expects dict
        self.head = head # expects int
        self.deprel = deprel # expects str
        self.deps = deps
        self.misc = misc # expects dict

    def __repr__(self):
        return f'<UniversalDependenciesWord: ID {self.idx}, "{self.form}">'

    @staticmethod
    def from_conllu(line):
        # split line and initially convert '_' values to None
        idx_str, form, lemma, upostag, xpostag, feats, head, deprel, deps, misc = [(v if v != '_' else None) for v in line.split('\t')]
        # parse idx string (int 1, decimal 1.1 or string '1-2')
        idx = idx_str
        if re.match(r'^\d+\.\d+$', idx_str): idx = float(idx_str)
        elif re.match(r'^\d+$', idx_str): idx = int(idx_str)
        # parse form and lemma (special case '_')
        form = form if form is not None else '_'
        lemma = lemma if form != '_' else '_'
        # parse dependency head idx (int)
        head = int(head) if head is not None else head
        # parse FEATS dictionaries
        try:
            feats = {f.split('=')[0]:f.split('=')[1] for f in feats.split('|')}
        except:
            feats = {}
        # parse MISC dictionary
        try:
            misc = {m.split('=')[0]:m.split('=')[1] for m in misc.split('|')}
        except:
            misc = {}
        # construct word
        word = UniversalDependenciesWord(
            idx,
            form, lemma, # form and lemma are str
            upostag, xpostag, # upostag and xpostag are str
            feats,
            head, deprel, deps, # dependency information as str
            misc
        )
        return word

    def to_text(self):
        text = self.get_form() + ' ' # form + space by default
        # if 'SpaceAfter=No' remove trailing space
        if ('SpaceAfter' in self.misc) and (self.misc['SpaceAfter'] == 'No'):
            text = text[:-1]

        return text

    def to_conllu(self):
        conllu = ''

        # convert dictionaries
        feats_str = '|'.join([f'{k}={v}' for k, v in sorted(self.feats.items())]) if self.feats else None
        misc_str = '|'.join([f'{k}={v}' for k, v in sorted(self.misc.items())]) if self.misc else None

        conllu_values = [
            self.idx,
            self.form, self.lemma,
            self.upostag, self.xpostag,
            feats_str,
            self.head, self.deprel, self.deps,
            misc_str
        ]
        # convert None to '_'
        conllu_values = [str(v) if v is not None else '_' for v in conllu_values]

        conllu = '\t'.join(conllu_values)
        return conllu

    def get_form(self, empty_as_unk=True):
        form = self.form if self.form else ''
        form = form.replace('\xad', '-') # sanitize soft hyphens
        form = form.replace('\x92', '') # sanitize single quotation mark
        form = form.replace('\x97', '') # sanitize acute accent below
        form = form.replace('\U000fe4fa', '') # sanitize Unicode character U+FE4FA
        form = form.replace('\ue402', '') # sanitize Unicode character U+E402
        form = form.replace('\ufeff', '') # sanitize Unicode character U+FEFF (zero width no-break space)
        form = form.replace('�', '') # sanitize Unicode replacement character
        # replace form with [UNK] if empty
        form = form if (len(form) > 0) or (not empty_as_unk) else '[UNK]'
        return form


#
# Universal Dependency Filtering Classes and Functions
#

class UniversalDependenciesFilter:
    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    def __call__(self, sentence, meta):
        return True


class UniversalDependenciesFilterCombination(UniversalDependenciesFilter):
    """Combines multiple filters."""
    def __init__(self, filters, mode='any'):
        self._filters = filters
        self._mode = mode

    def __repr__(self):
        return f"<{self.__class__.__name__}: match {self._mode} of {len(self._filters)} filters>"

    def __call__(self, sentence, meta):
        if (self._mode == 'all') and not all([filt(sentence, meta) for filt in self._filters]):
            return False
        if (self._mode == 'any') and not any([filt(sentence, meta) for filt in self._filters]):
            return False
        return True


class UniversalDependenciesIndexFilter(UniversalDependenciesFilter):
    """Filters out any sentences which are not in the set of specified indices (corpus-level)."""
    def __init__(self, indices):
        self._indices = indices
        self._cursor = -1

    def __repr__(self):
        return f"<{self.__class__.__name__}: {len(self._indices)} indices>"

    def __call__(self, sentence=None, meta=None):
        self._cursor += 1
        if self._cursor not in self._indices:
            return False
        return True


class UniversalDependenciesMetadataFilter(UniversalDependenciesFilter):
    """Filters out sentences based on a value in the treebank metadata."""
    def __init__(self, field, values, mode='include'):
        self._field = field
        self._values = values
        self._mode = mode

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self._mode} {len(self._values)} '{self._field}' value(s)>"

    def __call__(self, sentence, meta):
        if (self._mode == 'include') and (meta[self._field] not in self._values):
            return False
        if (self._mode == 'exclude') and (meta[self._field] in self._values):
            return False
        return True


class UniversalDependenciesDomainFilter(UniversalDependenciesFilter):
    """Filters out sentences based on the treebank domains or a provided domain distribution."""
    def __init__(self, domains, source=None, mode='include'):
        self._domains = set(domains)
        self._source = source # format {'domains': ['label0', ...], 'domain_dist': np.array}
        self._mode = mode
        self._cursor = -1

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self._mode} {len(self._domains)} domains{'' if self._source is None else ' based on provided distribution'}>"

    def __call__(self, sentence, meta):
        self._cursor += 1
        # if no source is provided, use treebank metadata
        if self._source is None:
            domains = set(meta.get('Genre', '').split(' '))
        # if source is provided use maximum probability assigned domain
        else:
            domains = {self._source['domains'][self._source['domain_dist'][self._cursor].argmax()]}

        if (self._mode == 'include') and (len(domains & self._domains) < 1):
            return False
        if (self._mode == 'exclude') and (len(domains & self._domains) > 0):
            return False
        if (self._mode == 'exact') and (domains != self._domains):
            return False
        return True

#
# Universal Dependency Grouping Classes and Functions
#


class UniversalDependenciesGrouper:
    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    def __call__(self, sentences):
        return OrderedDict()


class UniversalDependenciesCommentGrouper(UniversalDependenciesGrouper):
    """Groups sentences by a comment pattern into numbered groups.

    Examples:
        * comment('newdoc'): Groups each sentence by the last preceeding 'newdoc' comment, otherwise files under ID = 'unknown'
    """
    def __init__(self, comment_regex):
        self._comment_regex = comment_regex

    def __call__(self, sentences):
        """Returns an OrderedDict of sentence lists grouped by the values of the provided metadata key."""
        groups = OrderedDict()
        # initialize current group as 'unknown' in case no key is encountered in the first lines
        num_groups = 0
        cur_group = 'unknown'
        for sentence in sentences:
            for comment in sentence.get_comments():
                # e.g. encountered 'newdoc'
                if re.match(self._comment_regex, comment):
                    # set current group to 
                    cur_group = f'group-{num_groups}'
                    num_groups += 1
                    break
            # if group is new, initialize empty list
            if cur_group not in groups:
                groups[cur_group] = []
            # add sentences to the active group
            groups[cur_group].append(sentence)
        return groups


class UniversalDependenciesMetadataGrouper(UniversalDependenciesGrouper):
    """Groups sentences by a metadata key.

    Examples:
        * metadata('newdoc id'): Groups each sentence by the last preceeding ID specified in 'newdoc id = ID', otherwise files under ID = 'unknown'
        * metadata('source'): Groups each sentence by the same source, e.g. if each sentence is commented with the same 'source = SOURCE'
    """
    def __init__(self, key, value_regex=''):
        self._key = key
        self._value_regex = value_regex if len(value_regex) > 0 else None

    def __call__(self, sentences):
        """Returns an OrderedDict of sentence lists grouped by the values of the provided metadata key."""
        groups = OrderedDict()
        # initialize current group as 'unknown' in case no key is encountered in the first lines
        cur_group = 'unknown'
        for sentence in sentences:
            metadata = sentence.get_metadata()
            # e.g. encountered 'newdoc id'
            if self._key in metadata:
                # set current group to appropriate metadata value (e.g. document ID)
                cur_group = metadata[self._key]
                # check whether group must be extracted from the metadata value
                if self._value_regex:
                    value_match = re.match(self._value_regex, cur_group)
                    if value_match:
                        cur_group = ''
                        for match_key, match_val in sorted(value_match.groupdict().items()):
                            if not match_key.startswith('val'):
                                continue
                            cur_group += match_val
                    else:
                        cur_group = 'unknown'
            # if group is new, initialize empty list
            if cur_group not in groups:
                groups[cur_group] = []
            # add sentences to the active group
            groups[cur_group].append(sentence)
        return groups


def parse_grouper(grouper_str):
    grouper_map = {
        'comment': UniversalDependenciesCommentGrouper,
        'metadata': UniversalDependenciesMetadataGrouper
    }

    # match the grouper syntax "grouper_name('arg1', 'arg2')"
    grouper_match = re.match(r'(.+?)\((.+)\)', grouper_str)
    if not grouper_match:
        return None

    # look for grouper in list of constructors
    grouper_key = grouper_match[1]
    if grouper_key not in grouper_map:
        return None

    # parse grouper arguments
    grouper_args = tuple()
    for arg_str in re.split(r",\s*(?=')", grouper_match[2]):
        grouper_args += (arg_str[1:-1], ) # remove surrounding quotes

    # instantiate grouper
    grouper = grouper_map[grouper_key](*grouper_args)

    return grouper

#
# Global Universal Dependencies Variables
#


# all 307 UD 2.8 dependency relations with subtypes
UD_RELATIONS = [
    'acl', 'acl:adv', 'acl:attr', 'acl:cleft', 'acl:fixed', 'acl:inf', 'acl:relat', 'acl:relcl',
    'advcl', 'advcl:abs', 'advcl:cau', 'advcl:cleft', 'advcl:cmpr', 'advcl:cond', 'advcl:coverb', 'advcl:eval', 'advcl:lcl', 'advcl:lto', 'advcl:mcl', 'advcl:pred', 'advcl:relcl', 'advcl:sp', 'advcl:svc', 'advcl:tcl',
    'advmod', 'advmod:arg', 'advmod:cau', 'advmod:comp', 'advmod:deg', 'advmod:det', 'advmod:df', 'advmod:emph', 'advmod:eval', 'advmod:fixed', 'advmod:foc', 'advmod:freq', 'advmod:lfrom', 'advmod:lmod', 'advmod:lmp', 'advmod:locy', 'advmod:lto', 'advmod:mmod', 'advmod:mode', 'advmod:neg', 'advmod:obl', 'advmod:que', 'advmod:tfrom', 'advmod:tlocy', 'advmod:tmod', 'advmod:to', 'advmod:tto',
    'amod', 'amod:att', 'amod:attlvc', 'amod:flat',
    'appos', 'appos:trans',
    'aux', 'aux:aff', 'aux:aspect', 'aux:caus', 'aux:clitic', 'aux:cnd', 'aux:ex', 'aux:imp', 'aux:nec', 'aux:neg', 'aux:opt', 'aux:part', 'aux:pass', 'aux:pot', 'aux:q', 'aux:tense',
    'case', 'case:acc', 'case:adv', 'case:aff', 'case:det', 'case:gen', 'case:loc', 'case:pred', 'case:voc',
    'cc', 'cc:nc', 'cc:preconj',
    'ccomp', 'ccomp:cleft', 'ccomp:obj', 'ccomp:obl', 'ccomp:pmod', 'ccomp:pred',
    'clf',
    'compound', 'compound:a', 'compound:affix', 'compound:dir', 'compound:ext', 'compound:lvc', 'compound:nn', 'compound:preverb', 'compound:prt', 'compound:quant', 'compound:redup', 'compound:smixut', 'compound:svc', 'compound:vo', 'compound:vv',
    'conj', 'conj:expl', 'conj:extend', 'conj:svc',
    'cop', 'cop:expl', 'cop:locat', 'cop:own',
    'csubj', 'csubj:cleft', 'csubj:cop', 'csubj:pass',
    'dep', 'dep:aff', 'dep:agr', 'dep:alt', 'dep:ana', 'dep:aux', 'dep:comp', 'dep:conj', 'dep:cop', 'dep:emo', 'dep:infl', 'dep:mark', 'dep:mod', 'dep:pos', 'dep:redup', 'dep:ss',
    'det', 'det:adj', 'det:noun', 'det:numgov', 'det:nummod', 'det:poss', 'det:predet', 'det:pron', 'det:rel',
    'discourse', 'discourse:emo', 'discourse:filler', 'discourse:intj', 'discourse:sp',
    'dislocated', 'dislocated:cleft', 'dislocated:csubj', 'dislocated:nsubj', 'dislocated:obj', 'dislocated:subj',
    'expl', 'expl:comp', 'expl:impers', 'expl:pass', 'expl:poss', 'expl:pv', 'expl:subj',
    'fixed',
    'flat', 'flat:abs', 'flat:dist', 'flat:foreign', 'flat:name', 'flat:num', 'flat:range', 'flat:repeat', 'flat:sibl', 'flat:title', 'flat:vv',
    'goeswith',
    'iobj', 'iobj:agent', 'iobj:appl', 'iobj:patient',
    'list',
    'mark', 'mark:adv', 'mark:advmod', 'mark:aff', 'mark:prt', 'mark:q', 'mark:rel',
    'nmod', 'nmod:agent', 'nmod:appos', 'nmod:arg', 'nmod:att', 'nmod:attlvc', 'nmod:attr', 'nmod:bahuv', 'nmod:cau', 'nmod:comp', 'nmod:flat', 'nmod:gen', 'nmod:gobj', 'nmod:gsubj', 'nmod:lfrom', 'nmod:lmod', 'nmod:npmod', 'nmod:obj', 'nmod:obl', 'nmod:part', 'nmod:poss', 'nmod:pred', 'nmod:prp', 'nmod:redup', 'nmod:relat', 'nmod:subj', 'nmod:tmod',
    'nsubj', 'nsubj:advmod', 'nsubj:aff', 'nsubj:bfoc', 'nsubj:caus', 'nsubj:cleft', 'nsubj:cop', 'nsubj:ifoc', 'nsubj:lfoc', 'nsubj:lvc', 'nsubj:nc', 'nsubj:obj', 'nsubj:pass', 'nsubj:periph',
    'nummod', 'nummod:det', 'nummod:entity', 'nummod:flat', 'nummod:gov',
    'obj', 'obj:advmod', 'obj:advneg', 'obj:agent', 'obj:appl', 'obj:caus', 'obj:lvc', 'obj:obl', 'obj:periph',
    'obl', 'obl:advmod', 'obl:agent', 'obl:appl', 'obl:arg', 'obl:cau', 'obl:cmp', 'obl:cmpr', 'obl:comp', 'obl:dat', 'obl:freq', 'obl:inst', 'obl:lfrom', 'obl:lmod', 'obl:lmp', 'obl:lto', 'obl:lvc', 'obl:mcl', 'obl:mod', 'obl:npmod', 'obl:orphan', 'obl:own', 'obl:patient', 'obl:pmod', 'obl:poss', 'obl:prep', 'obl:sentcon', 'obl:smod', 'obl:tmod',
    'orphan', 'orphan:missing',
    'parataxis', 'parataxis:appos', 'parataxis:conj', 'parataxis:coord', 'parataxis:deletion', 'parataxis:discourse', 'parataxis:dislocated', 'parataxis:hashtag', 'parataxis:insert', 'parataxis:mod', 'parataxis:newsent', 'parataxis:nsubj', 'parataxis:obj', 'parataxis:parenth', 'parataxis:rel', 'parataxis:rep', 'parataxis:restart', 'parataxis:rt', 'parataxis:sentence', 'parataxis:trans', 'parataxis:url',
    'punct',
    'reparandum',
    'root',
    'vocative', 'vocative:cl', 'vocative:mention',
    'xcomp', 'xcomp:cleft', 'xcomp:ds', 'xcomp:obj', 'xcomp:pred', 'xcomp:sp', 'xcomp:subj'
]

# UD dependency relations without subtypes (N=37)
UD_RELATION_TYPES = sorted({r.split(':')[0] for r in UD_RELATIONS})

"""
Probing functions
"""

class StructuralProbe(nn.Module):
    def __init__(self, emb_model, dep_dim):
        super().__init__()
        # internal models
        self._emb = emb_model
        self._arc = UndirectedGraphPredictor(self._emb.emb_dim, dep_dim)

    def __repr__(self):
        return \
            f'{self.__class__.__name__}:\n' \
            f'  {self._emb}\n' \
            f'  <{self._arc.__class__.__name__}: {self._arc._emb_dim} -> {self._arc._out_dim}>'

    def get_trainable_parameters(self):
        return list(self._arc.parameters())

    def train(self, mode=True):
        super().train(mode)
        self._emb.eval()
        return self

    def forward(self, sentences, decode=True):
        # embed sentences (batch_size, seq_length) -> (batch_size, max_length, emb_dim)
        with torch.no_grad():
            emb_sentences, att_sentences = self._emb(sentences)

        # calculate distances in dependency space
        # dep_embeddings: (batch_size, dep_dim)
        # distances: (batch_size, max_len, max_len)
        dep_embeddings, distances = self._arc(emb_sentences.detach())

        # construct minimal result set
        results = {
            'dependency_embeddings': dep_embeddings,
            'distances': distances
        }

        # decode undirected graph
        if decode:
            # construct MST starting at node 0 (no explicit directionality)
            edges = self._arc.to_edges(distances.detach(), att_sentences.detach())

            # add undirected graph to results
            results['graphs'] = edges

        return results


class DirectedProbe(nn.Module):
    def __init__(self, emb_model, dep_dim):
        super().__init__()
        # internal models
        self._emb = emb_model
        self._arc = DirectedGraphPredictor(self._emb.emb_dim, dep_dim)

    def __repr__(self):
        return \
            f'{self.__class__.__name__}:\n' \
            f'  {self._emb}\n' \
            f'  <{self._arc.__class__.__name__}: {self._arc._emb_dim} -> {self._arc._out_dim}>'

    def get_trainable_parameters(self):
        return list(self._arc.parameters())

    def train(self, mode=True):
        super().train(mode)
        self._emb.eval()
        return self

    def forward(self, sentences, decode=True):
        # embed sentences (batch_size, seq_length) -> (batch_size, max_length, emb_dim)
        with torch.no_grad():
            emb_sentences, att_sentences = self._emb(sentences)

        # calculate distances in dependency space
        # depths: (batch_size, max_len)
        # distances: (batch_size, max_len, max_len)
        depths, distances, _ = self._arc(emb_sentences)

        # construct minimal result set
        results = {
            'depths': depths,
            'distances': distances
        }

        # decode into directed graph using CLE
        if decode:
            # convert to graph with idx -> head (batch_size, max_len)
            graphs = self._arc.to_graph(depths, distances, att_sentences)

            # add directed graph to results
            results['graphs'] = graphs

        return results


#
# Graph Predictors
#


class UndirectedGraphPredictor(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super().__init__()
        self._emb_dim = embedding_dim
        self._out_dim = output_dim
        # trainable parameters
        self._transform = nn.Linear(self._emb_dim, self._out_dim, bias=False)

    def forward(self, emb_sentences):
        dep_embeddings = self._transform(emb_sentences)
        batch_size, max_len, out_dim = dep_embeddings.size()

        # calculate differences
        dup_transformed = dep_embeddings.unsqueeze(2)
        dup_transformed = dup_transformed.expand(-1, -1, max_len, -1)
        dup_transposed = dup_transformed.transpose(1, 2)
        differences = dup_transformed - dup_transposed # (batch_size, max_len, max_len, dep_dim)
        squared_diffs = differences.pow(2)
        distances = torch.sum(squared_diffs, -1)

        return dep_embeddings, distances

    def to_edges(self, distances, mask):
        graphs = torch.ones_like(mask, dtype=torch.long) * -2 # (batch_size, max_len)

        # iterate over sentences
        for sidx in range(graphs.shape[0]):
            # get current sentence length
            sen_len = int(torch.sum(mask[sidx]))

            # always set first node's head to -1 (root)
            graphs[sidx, 0] = -1

            # gather initial nodes
            tree_nodes = [0]
            free_nodes = [n for n in range(sen_len) if n != 0]

            # while there are free nodes, keep adding to graph
            while free_nodes:
                # look for minimum distance between tree and free nodes
                cur_tree_dists = distances[sidx, tree_nodes, :] # (num_tree_nodes, max_len)
                cur_dists = cur_tree_dists[:, free_nodes] # (num_tree_nodes, num_free_nodes)
                min_dist_idx = torch.argmin(cur_dists) # returns argmin of flattened distances # returns tree node, free node
                min_tree = tree_nodes[min_dist_idx // len(free_nodes)] # tree node of minimum distance pair
                min_free = free_nodes[min_dist_idx % len(free_nodes)] # free node of minimum distance pair

                # set head node of free node to tree node (point towards root)
                graphs[sidx, min_free] = min_tree

                # housekeeping
                tree_nodes.append(min_free)
                free_nodes.remove(min_free)

        return graphs


class DirectedGraphPredictor(nn.Module):
    """
    Distance + Depth-based Graph Predictor

    Chu-Liu-Edmonds Algorithm implementation adapted from AllenNLP. [2]

    [2] https://github.com/allenai/allennlp/blob/v2.6.0/allennlp/nn/chu_liu_edmonds.py
    """
    def __init__(self, embedding_dim, output_dim):
        super().__init__()
        self._emb_dim = embedding_dim
        self._out_dim = output_dim
        # trainable parameters
        self._depth_transform = nn.Linear(self._emb_dim, self._out_dim, bias=False)
        self._distance_transform = nn.Linear(self._emb_dim, self._out_dim, bias=False)

    def forward(self, emb_sentences):
        batch_size, max_len, emb_dim = emb_sentences.shape

        # depth prediction
        emb_depths = self._depth_transform(emb_sentences)
        # calculate norms
        norms = torch.bmm(
            emb_depths.view(batch_size * max_len, 1, self._out_dim),
            emb_depths.view(batch_size * max_len, self._out_dim, 1))
        norms = norms.view(batch_size, max_len)

        emb_distances = self._distance_transform(emb_sentences)
        # calculate squared differences
        dup_transformed = emb_distances.unsqueeze(2)
        dup_transformed = dup_transformed.expand(-1, -1, max_len, -1)
        dup_transposed = dup_transformed.transpose(1, 2)
        differences = dup_transformed - dup_transposed
        squared_diffs = differences.pow(2)
        distances = torch.sum(squared_diffs, -1)

        return norms, distances, differences

    def to_graph(self, depths, distances, mask):
        graphs = torch.ones_like(depths, dtype=torch.int) * -2 # (batch_size, max_len)

        # iterate over sentences
        for sidx in range(graphs.shape[0]):
            # get current sentence length
            sen_len = int(torch.sum(mask[sidx]))

            # initialize energy matrix
            energy = np.ones((sen_len+1, sen_len+1)) * float('-inf')

            # root node is shallowest
            root_idx = torch.argmin(depths[sidx, :sen_len])
            # set root node to maximum energy
            energy[0, root_idx+1] = 0

            # construct energy matrix
            for hidx in range(sen_len):
                for cidx in range(sen_len):
                    # skip self
                    if hidx == cidx: continue
                    # skip if potential child is shallower than head
                    if depths[sidx, cidx] < depths[sidx, hidx]: continue
                    # if potential head is shallower than child, add score
                    energy[hidx+1, cidx+1] = -distances[sidx, hidx, cidx]

            graph = self.decode_mst(energy, sen_len+1)
            graph = graph[1:] - 1  # remove dummy root node (idx=0) and offset by -1
            graphs[sidx, :sen_len] = torch.tensor(graph, dtype=torch.int)

        return graphs

    def decode_mst(self, energy, length):
        input_shape = energy.shape
        max_length = input_shape[-1]

        # Our energy matrix might have been batched -
        # here we clip it to contain only non padded tokens.
        energy = energy[:length, :length]
        label_id_matrix = None
        # get original score matrix
        original_score_matrix = energy
        # initialize score matrix to original score matrix
        score_matrix = np.array(original_score_matrix, copy=True)

        old_input = np.zeros([length, length], dtype=np.int32)
        old_output = np.zeros([length, length], dtype=np.int32)
        current_nodes = [True for _ in range(length)]
        representatives = []

        for node1 in range(length):
            original_score_matrix[node1, node1] = 0.0
            score_matrix[node1, node1] = 0.0
            representatives.append({node1})

            for node2 in range(node1 + 1, length):
                old_input[node1, node2] = node1
                old_output[node1, node2] = node2

                old_input[node2, node1] = node2
                old_output[node2, node1] = node1

        final_edges = {}

        # The main algorithm operates inplace.
        self.chu_liu_edmonds(
            length, score_matrix, current_nodes, final_edges, old_input, old_output, representatives
        )

        heads = np.zeros([max_length], np.int32)

        for child, parent in final_edges.items():
            heads[child] = parent

        return heads

    def chu_liu_edmonds(self, length, score_matrix,	current_nodes, final_edges,	old_input, old_output, representatives):
        # Set the initial graph to be the greedy best one.
        parents = [-1]
        for node1 in range(1, length):
            parents.append(0)
            if current_nodes[node1]:
                max_score = score_matrix[0, node1]
                for node2 in range(1, length):
                    if node2 == node1 or not current_nodes[node2]:
                        continue

                    new_score = score_matrix[node2, node1]
                    if new_score > max_score:
                        max_score = new_score
                        parents[node1] = node2

        # Check if this solution has a cycle.
        has_cycle, cycle = self._find_cycle(parents, length, current_nodes)
        # If there are no cycles, find all edges and return.
        if not has_cycle:
            final_edges[0] = -1
            for node in range(1, length):
                if not current_nodes[node]:
                    continue

                parent = old_input[parents[node], node]
                child = old_output[parents[node], node]
                final_edges[child] = parent
            return

        # Otherwise, we have a cycle so we need to remove an edge.
        # From here until the recursive call is the contraction stage of the algorithm.
        cycle_weight = 0.0
        # Find the weight of the cycle.
        index = 0
        for node in cycle:
            index += 1
            cycle_weight += score_matrix[parents[node], node]

        # For each node in the graph, find the maximum weight incoming
        # and outgoing edge into the cycle.
        cycle_representative = cycle[0]
        for node in range(length):
            if not current_nodes[node] or node in cycle:
                continue

            in_edge_weight = float("-inf")
            in_edge = -1
            out_edge_weight = float("-inf")
            out_edge = -1

            for node_in_cycle in cycle:
                if score_matrix[node_in_cycle, node] > in_edge_weight:
                    in_edge_weight = score_matrix[node_in_cycle, node]
                    in_edge = node_in_cycle

                # Add the new edge score to the cycle weight
                # and subtract the edge we're considering removing.
                score = (
                        cycle_weight
                        + score_matrix[node, node_in_cycle]
                        - score_matrix[parents[node_in_cycle], node_in_cycle]
                )

                if score > out_edge_weight:
                    out_edge_weight = score
                    out_edge = node_in_cycle

            score_matrix[cycle_representative, node] = in_edge_weight
            old_input[cycle_representative, node] = old_input[in_edge, node]
            old_output[cycle_representative, node] = old_output[in_edge, node]

            score_matrix[node, cycle_representative] = out_edge_weight
            old_output[node, cycle_representative] = old_output[node, out_edge]
            old_input[node, cycle_representative] = old_input[node, out_edge]

        # For the next recursive iteration, we want to consider the cycle as a
        # single node. Here we collapse the cycle into the first node in the
        # cycle (first node is arbitrary), set all the other nodes not be
        # considered in the next iteration. We also keep track of which
        # representatives we are considering this iteration because we need
        # them below to check if we're done.
        considered_representatives = []
        for i, node_in_cycle in enumerate(cycle):
            considered_representatives.append(set())
            if i > 0:
                # We need to consider at least one
                # node in the cycle, arbitrarily choose
                # the first.
                current_nodes[node_in_cycle] = False

            for node in representatives[node_in_cycle]:
                considered_representatives[i].add(node)
                if i > 0:
                    representatives[cycle_representative].add(node)

        self.chu_liu_edmonds(
            length, score_matrix, current_nodes, final_edges, old_input, old_output, representatives
        )

        # Expansion stage.
        # check each node in cycle, if one of its representatives
        # is a key in the final_edges, it is the one we need.
        found = False
        key_node = -1
        for i, node in enumerate(cycle):
            for cycle_rep in considered_representatives[i]:
                if cycle_rep in final_edges:
                    key_node = node
                    found = True
                    break
            if found:
                break

        previous = parents[key_node]
        while previous != key_node:
            child = old_output[parents[previous], previous]
            parent = old_input[parents[previous], previous]
            final_edges[child] = parent
            previous = parents[previous]

    def _find_cycle(self, parents, length, current_nodes):
        added = [False for _ in range(length)]
        added[0] = True
        cycle = set()
        has_cycle = False
        for i in range(1, length):
            if has_cycle:
                break
            # don't redo nodes we've already
            # visited or aren't considering.
            if added[i] or not current_nodes[i]:
                continue
            # Initialize a new possible cycle.
            this_cycle = set()
            this_cycle.add(i)
            added[i] = True
            has_cycle = True
            next_node = i
            while parents[next_node] not in this_cycle:
                next_node = parents[next_node]
                # If we see a node we've already processed,
                # we can stop, because the node we are
                # processing would have been in that cycle.
                if added[next_node]:
                    has_cycle = False
                    break
                added[next_node] = True
                this_cycle.add(next_node)

            if has_cycle:
                original = next_node
                cycle.add(original)
                next_node = parents[original]
                while next_node != original:
                    cycle.add(next_node)
                    next_node = parents[next_node]
                break

        return has_cycle, list(cycle)



"""
Loss functions
"""

class RootedDependencyLoss(nn.Module):
    def __init__(self):
        super(RootedDependencyLoss, self).__init__()
        # set up spatial loss
        self._distance_loss = DependencyDistanceLoss()
        # set up label loss (ignore -1 padding labels)
        self._label_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
        # stats
        self.stats = {
            'loss_dist': None,
            'loss_label': None
        }

    def forward(self, parse, targets, use_trgt_graphs=False):
        pred_distances, pred_label_logits = parse['distances'], parse['label_logits']

        # calculate distance loss
        dist_loss = self._distance_loss(pred_distances, targets['distances'])
        self.stats['loss_dist'] = float(dist_loss.detach())

        # flatten logits and labels across all sequences
        pred_label_logits = torch.flatten(pred_label_logits, start_dim=0, end_dim=1) # (batch_size * max_len, num_labels)
        flat_trgt_labels = torch.flatten(targets['rels']) # (batch_size * max_len, )
        # calculate cross-entropy loss over label predictions
        label_loss = self._label_loss(pred_label_logits, flat_trgt_labels)
        self.stats['loss_label'] = float(label_loss.detach())

        loss = dist_loss + label_loss

        return loss


class StructuralProbingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # set up spatial loss
        self._distance_loss = DependencyDistanceLoss()
        # stats
        self.stats = {
            'loss_dist': None
        }

    def forward(self, parse, targets, use_trgt_graphs=False):
        pred_distances = parse['distances']

        # calculate distance loss
        loss = self._distance_loss(pred_distances, targets['distances'])
        self.stats['loss_dist'] = float(loss.detach())

        return loss


class DirectedProbingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # set up spatial loss
        self._depth_loss = DependencyDepthLoss()
        self._distance_loss = DependencyDistanceLoss()
        # stats
        self.stats = {
            'loss_depth': None,
            'loss_dist': None
        }

    def forward(self, parse, targets, use_trgt_graphs=False):
        pred_depths, pred_distances = parse['depths'], parse['distances']

        # calculate distance loss
        depth_loss = self._depth_loss(pred_depths, targets['depths'])
        dist_loss = self._distance_loss(pred_distances, targets['distances'])
        self.stats['loss_depth'] = float(depth_loss.detach())
        self.stats['loss_dist'] = float(dist_loss.detach())

        loss = depth_loss + dist_loss

        return loss

#
# L1-Distance Loss in Dependency Space
#
# 	Based on Hewitt & Manning, (2019) [1]
#
# 	[1] https://github.com/john-hewitt/structural-probes/blob/master/structural-probes/loss.py
#


class DependencyDepthLoss(nn.Module):
    def __init__(self):
        super(DependencyDepthLoss, self).__init__()

    def forward(self, pred_depths, trgt_depths):
        depth_mask = (trgt_depths != -1) # (batch_size, max_len)
        num_sentences = trgt_depths.shape[0] # scalar
        len_sentences = torch.sum(depth_mask, dim=-1) # (batch_size, )

        # calculate depth loss
        # sum absolute differences between predicted depth and target (both are positive)
        sum_token_depths = torch.sum(
            torch.abs((trgt_depths - pred_depths) * depth_mask),
            dim=-1
        ) # (batch_size, )
        # average over individual sentence lengths
        nrm_sentence_depths = sum_token_depths / len_sentences # (batch_size, )
        # mean over batch
        depth_loss = torch.sum(nrm_sentence_depths) / num_sentences # scalar

        return depth_loss


class DependencyDistanceLoss(nn.Module):
    def __init__(self):
        super(DependencyDistanceLoss, self).__init__()

    def forward(self, pred_distances, trgt_distances):
        dists_mask = (trgt_distances != -1).float()  # (batch_size, max_len, max_len)
        num_sentences = float(trgt_distances.shape[0])  # scalar
        len_sentences = torch.sum((trgt_distances != -1)[:, :, 0], dim=-1).float()  # (batch_size, )

        # calculate distance loss
        # mask distances to disregard padding
        trgt_distances_masked = trgt_distances * dists_mask
        pred_distances_masked = pred_distances * dists_mask
        # sum absolute differences between predicted distances (positive because of square) and target (positive)
        sum_token_loss = torch.sum(
            torch.abs(trgt_distances_masked - pred_distances_masked),
            dim=[1, 2]
        )  # (batch_size, )
        # average over all token pairs per sentence
        nrm_sentence_loss = sum_token_loss / (len_sentences ** 2)  # (batch_size, )
        # mean over batch
        dist_loss = torch.sum(nrm_sentence_loss) / num_sentences  # scalar

        return dist_loss

"""
Setup functions
"""

def setup_output_directory(out_path):
    print(f"in setup.py out_path = {out_path}")
    if os.path.exists(out_path):
        response = None
        while response not in ['y', 'n']:
            response = input(f"Path '{out_path}' already exists. Overwrite? [y/n] ")
        if response == 'n':
            exit(1)
    # if output dir does not exist, create it
    else:
        print(f"Path '{out_path}' does not exist. Creating...")
        os.mkdir(out_path)
    return True


def setup_logging(log_path):
    log_format = '%(message)s'
    log_level = logging.INFO
    logging.basicConfig(filename=log_path, filemode='w', format=log_format, level=log_level)

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))


def setup_data(ud_path, split_path, is_treebank, skip_root_label=False):
    # load data split definition (if supplied)
    ud_filter = None
    splits = None
    if split_path:
        with open(split_path, 'rb') as fp:
            splits = pickle.load(fp)
        # create filter to load only relevant indices (train, dev)
        relevant_idcs = set(splits['train']) | set(splits['dev'])
        ud_filter = UniversalDependenciesIndexFilter(relevant_idcs)
        logging.info(f"Loaded data splits {', '.join([f'{s}: {len(idcs)}' for s, idcs in splits.items()])} with filter {ud_filter}.")

    # load single Universal Dependencies treebank
    if is_treebank:
        treebanks = []
        splits = {}
        cursor = 0
        # iterate over files in TB directory
        for tbf in sorted(os.listdir(ud_path)):
            print(f"tbf = {tbf}")
            # skip non-conllu files
            if os.path.splitext(tbf)[1] != '.conllu': continue

            # load treebank
            tbf_path = os.path.join(ud_path, tbf)
            treebank = UniversalDependenciesTreebank.from_conllu(tbf_path, name=tbf, start_idx=cursor)
            treebanks.append(treebank)

            # identify split
            split = os.path.splitext(tbf)[0].split('-')[-1]
            splits[split] = list(range(cursor, cursor + len(treebank)))
            cursor += len(treebank)
            logging.info(f"Loaded {treebank} ({split}) from '{tbf}'.")
        # construct UD instance from relevant treebanks
        ud = UniversalDependencies(treebanks=treebanks)
        logging.info(f"Constructed UD corpus from treebank '{ud_path}'.")

    # load full Universal Dependencies
    else:
        print(f"ud_path = {ud_path}")
        ud = UniversalDependencies.from_directory(ud_path, ud_filter=ud_filter, verbose=True)
    # load UD dependency relations and map to indices
    rel_map = {r: i for i, r in enumerate(UD_RELATION_TYPES)}
    # set 'root' to -1 such that it's skipped (only for separate root prediction)
    if skip_root_label:
        rel_map['root'] = -1
    logging.info(f"Loaded {ud} with {len(rel_map)} dependency relations.")

    # use all of UD for each split if none are provided
    # if splits is None:
    #     splits = {split: list(range(len(ud))) for split in ['train', 'dev', 'test']}
    if splits is None:
        total_size = len(ud)
        train_size = int(0.7 * total_size)
        dev_size = int(0.1 * total_size)
        test_size = total_size - train_size - dev_size  # Ensure all indices are accounted for

        # Assign indices for each split
        splits = {
            'train': list(range(0, train_size)),
            'dev': list(range(train_size, train_size + dev_size)),
            'test': list(range(train_size + dev_size, total_size))
        }

    return ud, splits, rel_map


def setup_model(lm_name, dep_dim, parser_type='depprobe', state_dict=None, emb_layers=None, emb_cache=None, dataset_path = None, erasure_method = None, linguistic_feature = None, batch_size = 64, splits=None, hidden_state_path = None):
    # load pre-computed embedding cache if specified
    if emb_cache is not None:
        if emb_cache == 'local':
            emb_cache = {}
        else:
            # load pre-computed embeddings {hash: torch.Tensor (sen_len, emb_dim)}
            with open(emb_cache, 'rb') as fp:
                emb_cache = pickle.load(fp)

    print("loading embedding model")
    # load transformer embedding model
    emb_model = EmbeddingModel(lm_name, layers=emb_layers, cache=emb_cache, dataset_path=dataset_path, erasure_method = erasure_method, linguistic_feature = linguistic_feature, batch_size=batch_size, splits = splits, hidden_state_path = hidden_state_path)

    # build structural probe
    if parser_type == 'structural':
        assert len(emb_layers) == 1, f"[Error] StructuralProbe requires one embedding layer, received {len(emb_layers)}."
        parser = StructuralProbe(
            emb_model=emb_model,
            dep_dim=dep_dim
        )
    # build directed probe
    elif parser_type == 'directed':
        assert len(emb_layers) == 1, f"[Error] DirectedProbe requires one embedding layer, received {len(emb_layers)}."
        parser = DirectedProbe(
            emb_model=emb_model,
            dep_dim=dep_dim
        )
    # build dependency probe
    elif parser_type == 'depprobe':
        assert len(emb_layers) == 2, f"[Error] DepProbe requires two embedding layers, received {len(emb_layers)}."
        parser = DepProbe(
            emb_model=emb_model,
            dep_dim=dep_dim,
            dep_rels=UD_RELATION_TYPES
        )
    # build dependency probe over mixture of layers
    elif parser_type == 'depprobe-mix':
        parser = DepProbeMix(
            emb_model=emb_model,
            dep_dim=dep_dim,
            dep_rels=UD_RELATION_TYPES
        )
    else:
        logging.error(f"[Error] Unknown model type '{parser_type}.")

    logging.info(f"Constructed '{parser_type}' model:")
    logging.info(parser)

    # load existing state if provided
    if state_dict is not None:
        keys = parser.load_state_dict(state_dict, strict=False)
        assert len([k for k in keys.missing_keys if not k.startswith('_emb.')]) == 0, \
            f"[Error] State dict is missing keys ({keys.missing_keys})."
        logging.info(f"Loaded weights from predefined state dict.")

    # check CUDA availability
    if torch.cuda.is_available():
        parser.to(torch.device('cuda'))
        logging.info(f"Moved parser to CUDA device ('{torch.device('cuda')}').")

    return parser


def setup_criterion(parser_type='depprobe'):
    # use structural distance loss
    if parser_type == 'structural':
        criterion = StructuralProbingLoss()
        logging.info(
            f"Using {criterion.__class__.__name__} with "
            f"{criterion._distance_loss.__class__.__name__}.")
    # use directed (depth + distance) loss
    elif parser_type == 'directed':
        criterion = DirectedProbingLoss()
        logging.info(
            f"Using {criterion.__class__.__name__} with "
            f"{criterion._depth_loss.__class__.__name__} and {criterion._distance_loss.__class__.__name__}.")
    # use depprobe loss
    #else parser_type.startswith('depprobe'):
    else:
        assert parser_type.startswith('depprobe'), "parser_type should start with depprobe or else statement is faulty"
        criterion = RootedDependencyLoss()
        logging.info(
            f"Using {criterion.__class__.__name__} with "
            f"{criterion._distance_loss.__class__.__name__} and "
            f"{criterion._label_loss.__class__.__name__}.")

    return criterion


def get_accuracies(parse, targets, match_all=True):
    accuracies = {}

    # compute graph accuracy (i.e. UAS)
    if 'graphs' in parse:
        # gather predictions and targets (these fields are always available)
        pred_graphs, trgt_graphs = parse['graphs'].detach(), targets['heads'].detach()

        # calculate mask to ignore padding (same for graphs and labels)
        mask = (trgt_graphs != -2).float()

        head_matches = (pred_graphs == trgt_graphs).float()
        num_head_matches = torch.sum(head_matches * mask, dim=-1)
        accuracies['graph'] = float(torch.sum(num_head_matches) / torch.sum(mask))

    # compute label accuracy
    if 'labels' in parse:
        pred_labels, trgt_labels = parse['labels'].detach(), targets['rels'].detach()
        # only count tokens with correct heads and correct labels
        if match_all:
            num_label_matches = torch.sum((pred_labels == trgt_labels) * mask * head_matches, dim=-1)
        # assume correct heads and count correct labels
        else:
            num_label_matches = torch.sum((pred_labels == trgt_labels) * mask, dim=-1)
        accuracies['label'] = float(torch.sum(num_label_matches) / torch.sum(mask))

    # compute sentence reconstruction accuracy
    if 'sentences' in parse:
        num_correct, num_total = 0, 0
        for sidx in range(len(parse['sentences'])):
            for widx in range(len(parse['sentences'][sidx])):
                if parse['sentences'][sidx][widx] == targets['sentences'][sidx][widx]:
                    num_correct += 1
                num_total += 1
        accuracies['word'] = num_correct / num_total

    return accuracies


def run(parser, criterion, optimizer, dataset, mode='train', decode=True):
    stats = defaultdict(list)

    # set model to training mode
    if mode == 'train':
        parser.train()
    # set model to eval mode
    elif mode == 'eval':
        parser.eval()

    # iterate over batches
    for bidx, batch_data in enumerate(dataset):
        sentences, targets, num_remaining = batch_data

        try:
            # when training, perform both forward and backward pass
            if mode == 'train':
                stats['tokens'].append(sum([len(s) for s in sentences]))
                stats['time'].append(time.time())

                # zero out previous gradients
                optimizer.zero_grad()

                # forward pass (use teacher forcing for label prediction)
                parse = parser(sentences, decode=decode, batch_num = bidx, mode = mode)

                # propagate loss
                loss = criterion(parse, targets, use_trgt_graphs=True)
                loss.backward()
                optimizer.step()

                stats['time'][-1] = time.time() - stats['time'][-1]

                # calculate accuracy (assume gold graph for labels)
                accuracies = get_accuracies(parse, targets, match_all=False)

            # when evaluating, perform forward pass without gradients
            elif mode == 'eval':
                with torch.no_grad():
                    # forward pass
                    parse = parser(sentences, batch_num = bidx, mode = mode)
                    # calculate loss
                    loss = criterion(parse, targets)
                # calculate accuracies (both heads and labels need to match)
                accuracies = get_accuracies(parse, targets)

        except TokenizationError as tok_err:
            logging.error(f"[Error] {tok_err}. Skipped batch.")
            continue

        # store statistics
        for crit, val in criterion.stats.items():
            stats[crit].append(val)
        stats['loss'].append(float(loss.detach()))
        for acc_key, acc_val in accuracies.items():
            stats[f'acc_{acc_key}'].append(acc_val)
        stats['time'] = [np.sum(stats['time'])]
        stats['tokens'] = [np.sum(stats['tokens'])]

        # print batch statistics
        pct_complete = (1 - (num_remaining/len(dataset)))*100
        sys.stdout.write(
            f"\r[{mode.capitalize()} | Batch {bidx+1} | {pct_complete:.2f}%] "
            f"Acc: {' / '.join([f'{np.mean(v):.2f}' for s, v in sorted(stats.items()) if s.startswith('acc_')]) if decode else 'no-decode'}, "
            f"Loss: {' + '.join([f'{np.mean(v):.2f}' for s, v in sorted(stats.items()) if s.startswith('loss_')])} = "
            f"{np.mean(stats['loss']):.4f}"
        )
        sys.stdout.flush()

    # clear line
    print("\r", end='')

    return stats


def statistics(mode, stats, epoch_stats, ep_idx, epochs):
    # store epoch statistics
    for stat in epoch_stats:
        stats[f'{mode}/{stat}'].append(np.mean(epoch_stats[stat]))
        print(f"current epoch = {ep_idx} and max epochs = {epochs}")
    # print statistics
    logging.info(
        f"[Epoch {ep_idx+1}/{epochs}] {mode.capitalize()} completed with "
        f"AccGraph: {np.round(stats[f'{mode}/acc_graph'][-1], 4) if f'{mode}/acc_graph' in stats else 'None'}, "
        f"AccLabel: {np.round(stats[f'{mode}/acc_label'][-1], 4) if f'{mode}/acc_label' in stats else 'None'}, "
        f"AccWord: {np.round(stats[f'{mode}/acc_word'][-1], 4) if f'{mode}/acc_word' in stats else 'None'}, "
        f"DepthLoss: {np.round(stats[f'{mode}/loss_depth'][-1], 4) if f'{mode}/loss_depth' in stats else 'None'}, "
        f"DistLoss: {np.round(stats[f'{mode}/loss_dist'][-1], 4) if f'{mode}/loss_dist' in stats else 'None'}, "
        f"RootLoss: {np.round(stats[f'{mode}/loss_root'][-1], 4) if f'{mode}/loss_root' in stats else 'None'}, "
        f"LabelLoss: {np.round(stats[f'{mode}/loss_label'][-1], 4) if f'{mode}/loss_label' in stats else 'None'}, "
        f"Loss: {stats[f'{mode}/loss'][-1]:.4f}"
    )


def save_checkpoint(parser, optimizer, epoch, stats, path):
    # remove embedding model parameters
    parser_state = OrderedDict()
    for param, value in parser.state_dict().items():
        if param.startswith('_emb.'): continue
        parser_state[param] = value

    torch.save({
        'epoch': epoch,
        'stats': stats,
        'parser_state': parser_state,
        'optimizer': optimizer.state_dict()
    }, path)
    logging.info(f"Saved checkpoint to '{path}'.")


"""
Classes
"""

class EmbeddingModel(nn.Module):
    def __init__(self, lm_name, layers, cache=None, dataset_path = None, erasure_method = None, linguistic_feature = None, batch_size = 64, splits = None, hidden_state_path = None):
        super(EmbeddingModel, self).__init__()
        # load transformer
        self._tok = transformers.AutoTokenizer.from_pretrained(lm_name, use_fast=True, add_prefix_space=True)
        self._lm = transformers.AutoModel.from_pretrained(lm_name, return_dict=True)
        # load cache
        self._cache = cache  # {hash: torch.tensor (num_layers, sen_len, emb_dim)}
        # internal variables
        self._lm_name = lm_name
        self._lm_layers = layers
        # public variables
        self.emb_dim = self._lm.config.hidden_size
        self.num_layers = self._lm.config.num_hidden_layers
        self.erasers, self.one_hot_tags, padded_tags = load_hidden_states_train_eraser(dataset_path, lm_name, layers, erasure_method, linguistic_feature, hidden_state_path)
        self.padded_one_hot_tags = self.one_hot_tags.view(padded_tags.shape[0], padded_tags.shape[1], self.one_hot_tags.shape[1])
        self.erasure_method = erasure_method
        self.linguistic_feature = linguistic_feature
        self.batch_size = batch_size
        self.splits = splits


    def __repr__(self):
        return f'<{self._lm.__class__.__name__}: "{self._lm_name}", Layers {str(self._lm_layers)}{" , with cache" if self._cache is not None else ""}>'

    def forward(self, sentences, batch_num = 0, mode = 'train'):
        # try retrieving embeddings from cache
        emb_cache = self.retrieve(sentences)
        if emb_cache is not None:
            emb_layers, att_words = emb_cache
        else:
            # compute embeddings if not in cache
            tok_sentences = self.tokenize(sentences)
            model_inputs = {
                k: tok_sentences[k] for k in ['input_ids', 'token_type_ids', 'attention_mask']
                if k in tok_sentences
            }

            # perform embedding forward pass
            model_outputs = self._lm(**model_inputs, output_hidden_states=True)

            hidden_states = model_outputs.hidden_states  # tuple(num_layers * (batch_size, max_len, hidden_dim))


            # post-process embeddings from specified layers
            emb_layers, att_words = [], None
            for i,layer_idx in enumerate(self._lm_layers):

                
                emb_pieces = hidden_states[layer_idx] # batch_size, max_len, hidden_dim

               
                if self.splits:
                    # print("self.padded_one_hot_tags shape: ", self.padded_one_hot_tags.shape, flush=True)
                    # reformat tags into n_sentences * max_len
                    if mode == 'train':
                        current_padded_tags = self.padded_one_hot_tags[self.splits[mode]][batch_num * self.batch_size: (batch_num + 1) * self.batch_size, : emb_pieces.shape[1]]  # batch_size, max_len, n_classes
                    else:
                        current_padded_tags = self.padded_one_hot_tags[self.splits['dev']][batch_num * self.batch_size: (batch_num + 1) * self.batch_size, : emb_pieces.shape[1]]
                    # print("current padded tags .shape = ", current_padded_tags.shape, flush=True)
                    current_tags = current_padded_tags.reshape(-1, current_padded_tags.shape[2])
                    # current_tags = current_padded_tags.view(-1, current_padded_tags.shape[2]) # n_tokens (including padding) * n_classes
                else:
                    current_tags = self.one_hot_tags[batch_num * (emb_pieces.shape[0] * emb_pieces.shape[1]): (batch_num + 1) * (emb_pieces.shape[0] * emb_pieces.shape[1])] # batch_size
                

                #flat_erased_emb_pieces = self.erasers[i](emb_pieces.view(-1, self.emb_dim), current_tags) # emb pieces resized to n_tokens (including padding) * emb_dim
                flat_erased_emb_pieces = apply_eraser(emb_pieces.view(-1, self.emb_dim), self.erasers[i], current_tags, self.erasure_method)
                
                # resize to original shape
                emb_pieces = flat_erased_emb_pieces.view(emb_pieces.size(0), emb_pieces.size(1), emb_pieces.size(2)) # batch_size, max_len, hidden_dim

                # reduce WordPiece to words
                emb_words, att_words = self.reduce(sentences, tok_sentences, emb_pieces)
                # append to results
                emb_layers.append(emb_words)

            # store embeddings in cache (if cache is enabled)
            if self._cache is not None:
                self.cache(sentences, emb_layers)

        # reduce list of layers to single tuple if only one layer is returned
        emb_layers = emb_layers if len(self._lm_layers) != 1 else emb_layers[0]

        return emb_layers, att_words

    def retrieve(self, sentences):
        if self._cache is None:
            return None

        max_len = max([len(s) for s in sentences])
        emb_layers = [torch.zeros((len(sentences), max_len, self.emb_dim)) for _ in range(len(self._lm_layers))]
        att_words = torch.zeros((len(sentences), max_len), dtype=torch.bool)

        # iterate over sentences
        for sidx, sentence in enumerate(sentences):
            # retrieve sentence embedding using string hash
            sen_hash = hashlib.md5(' '.join(sentence).encode('utf-8')).hexdigest()
            # skip batch if not all sentences are in cache
            if sen_hash not in self._cache:
                return None

            # retrieve embeddings for each layer from cache
            for lidx in range(len(self._lm_layers)):
                emb_layers[lidx][sidx, :len(sentence), :] = self._cache[sen_hash][lidx]  # (sen_len, emb_dim)
            att_words[sidx, :len(sentence)] = True

        # move input to GPU (if available)
        if torch.cuda.is_available():
            emb_layers = [embs.to(torch.device('cuda')) for embs in emb_layers]
            att_words = att_words.to(torch.device('cuda'))

        return emb_layers, att_words

    def cache(self, sentences, emb_layers):
        # detach, duplicate and move embeddings to CPU
        emb_layers = [embs.detach().clone().cpu() for embs in emb_layers]

        # iterate over sentences
        for sidx, sentence in enumerate(sentences):
            # compute sentence hash
            sen_hash = hashlib.md5(' '.join(sentence).encode('utf-8')).hexdigest()

            # initialize cache entry with list over layers
            self._cache[sen_hash] = []
            # iterate over layers
            for lidx in range(len(self._lm_layers)):
                self._cache[sen_hash].append(emb_layers[lidx][sidx, :len(sentence), :])  # (sen_len, emb_dim)

    def tokenize(self, sentences):
        # tokenize batch: {input_ids: [[]], token_type_ids: [[]], attention_mask: [[]], special_tokens_mask: [[]]}
        tok_sentences = self._tok(
            sentences,
            is_split_into_words=True, padding=True, truncation=True,
            return_tensors='pt', return_special_tokens_mask=True, return_offsets_mapping=True
        )
        # move input to GPU (if available)
        if torch.cuda.is_available():
            tok_sentences = {k: v.to(torch.device('cuda')) for k, v in tok_sentences.items()}

        return tok_sentences

    def reduce(self, sentences, tok_sentences, emb_pieces):
        emb_words = torch.zeros_like(emb_pieces)
        # print("emb words shape: ", emb_words.shape)
        att_words = torch.zeros(emb_pieces.shape[:-1], dtype=torch.bool, device=emb_pieces.device)
        max_len = 0
        # iterate over sentences
        for sidx in range(emb_pieces.shape[0]):
            # get string tokens of current sentence
            tokens = self._tok.convert_ids_to_tokens(tok_sentences['input_ids'][sidx])
            offsets = tok_sentences['offset_mapping'][sidx]

            tidx = -1
            for widx, orig_word in enumerate(sentences[sidx]):
                # init aggregate word embedding
                emb_word = torch.zeros(emb_pieces.shape[-1], device=emb_pieces.device)  # (emb_dim,)
                num_tokens = 0
                coverage = 0
                while coverage < len(orig_word):
                    tidx += 1
                    if tidx >= len(emb_pieces[sidx, :]):
                        raise TokenizationError(
                            f"More words than pieces {tidx} >= {len(emb_pieces[sidx, :])}.\n"
                            f"UD (len={len(sentences[sidx])}): {sentences[sidx]}\n"
                            f"LM (len={len(tokens)}): {tokens}",
                            position=(sidx, tidx)
                        )
                    # skip if special tokens ([CLS], [SEQ], [PAD])
                    if tok_sentences['special_tokens_mask'][sidx, tidx] == 1: continue

                    token_span = offsets[tidx]  # (start_idx, end_idx + 1) within orig_word
                    # add WordPiece embedding to current word embedding sum
                    emb_word += emb_pieces[sidx, tidx]
                    num_tokens += 1
                    coverage = token_span[1]

                    # exit prematurely if next piece initiates new word (some LMs return less characters than in input)
                    if (tidx < len(offsets) - 1) and offsets[tidx + 1][0] == 0:
                        break

                # add mean of aggregate WordPiece embeddings and set attention to True
                # Perform division
                if num_tokens != 0:

                    # Ensuring emb_word is properly shaped for assignment
                    if (emb_word / num_tokens).shape != emb_words[sidx, widx].shape:
                        corrected_emb_word = (emb_word / num_tokens).unsqueeze(0)  # Adjusting dimensions if necessary
                        emb_words[sidx, widx] = corrected_emb_word
                    else:
                        emb_words[sidx, widx] = emb_word / num_tokens
                    
                else:
                    print("Division by zero error avoided")
                att_words[sidx, widx] = True

            # store new maximum sequence length
            max_len = len(sentences[sidx]) if len(sentences[sidx]) > max_len else max_len

        # reduce embedding and attention matrices to new maximum length
        emb_words = emb_words[:, :max_len, :]  # (batch_size, max_len, emb_dim)
        att_words = att_words[:, :max_len]  # (batch_size, max_len)

        return emb_words, att_words

#
# Graph Predictors
#


class RootedGraphPredictor(nn.Module):
	def __init__(self, embedding_dim, output_dim):
		super(RootedGraphPredictor, self).__init__()
		self._emb_dim = embedding_dim
		self._out_dim = output_dim
		# trainable parameters
		self._transform = nn.Linear(self._emb_dim, self._out_dim, bias=False)

	def forward(self, emb_sentences):
		dep_embeddings = self._transform(emb_sentences)
		batch_size, max_len, out_dim = dep_embeddings.size()

		# calculate differences
		dup_transformed = dep_embeddings.unsqueeze(2)
		dup_transformed = dup_transformed.expand(-1, -1, max_len, -1)
		dup_transposed = dup_transformed.transpose(1, 2)
		differences = dup_transformed - dup_transposed  # (batch_size, max_len, max_len, dep_dim)
		squared_diffs = differences.pow(2)
		distances = torch.sum(squared_diffs, -1)

		return dep_embeddings, distances

	def to_graph(self, roots, distances, mask):
		graphs = torch.ones_like(mask, dtype=torch.long) * -2  # (batch_size, max_len)

		# iterate over sentences
		for sidx in range(graphs.shape[0]):
			# get current sentence length
			sen_len = int(torch.sum(mask[sidx]))

			# set root node's head to -1
			sen_root = int(roots[sidx].detach())
			graphs[sidx, sen_root] = -1

			# gather initial nodes
			tree_nodes = [sen_root]
			free_nodes = [n for n in range(sen_len) if n != sen_root]

			# while there are free nodes, keep adding to graph
			while free_nodes:
				# look for minimum distance between tree and free nodes
				cur_tree_dists = distances[sidx, tree_nodes, :]  # (num_tree_nodes, max_len)
				cur_dists = cur_tree_dists[:, free_nodes]  # (num_tree_nodes, num_free_nodes)
				min_dist_idx = torch.argmin(cur_dists)  # returns argmin of flattened distances # returns tree node, free node
				# min_tree = tree_nodes[min_dist_idx // len(free_nodes)]  # tree node of minimum distance pair
				min_tree = tree_nodes[torch.div(min_dist_idx, len(free_nodes), rounding_mode='floor')]
				min_free = free_nodes[min_dist_idx % len(free_nodes)]  # free node of minimum distance pair

				# set head node of free node to tree node (point towards root)
				graphs[sidx, min_free] = min_tree

				# housekeeping
				tree_nodes.append(min_free)
				free_nodes.remove(min_free)

		return graphs


#
# Label Predictors
#


class LabelClassifier(nn.Module):
	def __init__(self, input_dim, num_labels, root_label):
		super(LabelClassifier, self).__init__()
		self._in_dim = input_dim
		self._num_labels = num_labels  # number of labels (e.g. 37)
		self._root_label = root_label  # index of root label
		# trainable parameters
		self._mlp = nn.Linear(self._in_dim, self._num_labels, bias=False)

	def forward(self, emb_sentences, att_sentences):
		# logits for all tokens in all sentences + padding -inf (batch_size, max_len, num_labels)
		logits = torch.ones(
			(att_sentences.shape[0], att_sentences.shape[1], self._num_labels),
			device=emb_sentences.device
		) * float('-inf')
		# get token embeddings of all sentences (total_tokens, emb_dim)
		emb_words = emb_sentences[att_sentences, :]
		# pass through MLP
		logits[att_sentences, :] = self._mlp(emb_words)  # (num_words, num_labels) -> (batch_size, max_len, num_labels)
		return logits

	def get_labels(self, lbl_logits):
		# gather word with highest root probability for each sentence
		roots = torch.argmax(lbl_logits[:, :, self._root_label], dim=-1)  # (batch_size, 1)
		# set root logits to -inf to prevent multiple roots
		lbl_logits_noroot = lbl_logits.detach().clone()
		lbl_logits_noroot[:, :, self._root_label] = torch.ones(
			(lbl_logits.shape[0], lbl_logits.shape[1]),
			device=lbl_logits.device
		) * float('-inf')
		# get predicted labels with maximum probability (padding should have -inf)
		labels = torch.argmax(lbl_logits_noroot, dim=-1)  # (batch_size, max_len)
		# add true root labels
		labels[torch.arange(lbl_logits.shape[0]), roots] = self._root_label
		# add -1 padding
		labels[(lbl_logits[:, :, 0] == float('-inf'))] = -1

		return roots, labels

class DepProbe(nn.Module):
    def __init__(self, emb_model, dep_dim, dep_rels):
        super(DepProbe, self).__init__()
        # internal variables
        self._root_label = dep_rels.index('root')
        
        # internal models
        self._emb = emb_model
        self._arc = RootedGraphPredictor(self._emb.emb_dim, dep_dim)
        self._lbl = LabelClassifier(self._emb.emb_dim, len(dep_rels), self._root_label)


    def __repr__(self):
        return \
            f'{self.__class__.__name__}:\n' \
            f'  {self._emb}\n' \
            f'  <{self._arc.__class__.__name__}: {self._arc._emb_dim} -> {self._arc._out_dim}>\n' \
            f'  <{self._lbl.__class__.__name__}: {self._lbl._in_dim} -> {self._lbl._num_labels}>'

    def get_trainable_parameters(self):
        return list(self._arc.parameters()) + list(self._lbl.parameters())

    def train(self, mode=True):
        super(DepProbe, self).train(mode)
        self._emb.eval()
        return self

    def forward(self, sentences, decode=True, batch_num=0, mode = 'train'):
        # embed sentences (batch_size, seq_length)
        # -> ([(batch_size, max_length, emb_dim) * 2], (batch_size, max_length))
        # -> ([emb_sentences_lay0, emb_sentences_lay1], att_sentences)
        with torch.no_grad():
            emb_layers, att_sentences = self._emb(sentences, batch_num, mode = mode)

        # calculate distances in dependency space
        # dep_embeddings: (batch_size, dep_dim)
        # distances: (batch_size, max_len, max_len)
        dep_embeddings, distances = self._arc(emb_layers[0].detach())

        # classify dependency relations
        lbl_logits = self._lbl(emb_layers[1].detach(), att_sentences.detach())

        # construct minimal return set
        results = {
            'dependency_embeddings': dep_embeddings,
            'distances': distances,
            'label_logits': lbl_logits
        }

        # decode labelled dependency graph
        if decode:
            # get roots and labels from logits
            roots, labels = self._lbl.get_labels(lbl_logits.detach())
            # construct MST starting at root
            graphs = self._arc.to_graph(roots.detach(), distances.detach(), att_sentences.detach())

            # add labels and graphs to results
            results['graphs'] = graphs
            results['labels'] = labels

        return results

class DepSpaceDataset:
    def __init__(self, ud, rels, idcs, batch_size, random = False):
        self._ud = ud
        self._rels = rels
        self._idcs = idcs
        self._batch_size = batch_size
        # iteration variables
        self._iter_idcs = None
        # cache variables
        self._cache_heads = {}
        self._cache_rels = {}
        self._cache_roots = {}
        self._cache_depths = {}
        self._cache_distances = {}
        self.random = random

    def __len__(self):
        return len(self._idcs)

    def __iter__(self):
        self._iter_idcs = set(self._idcs)
        return self

    def __next__(self):
        if len(self._iter_idcs) > 0:

            if self.random:
                # get random sample from UD
                batch_idcs = list(np.random.choice(list(self._iter_idcs), min(self._batch_size, len(self._iter_idcs)), replace=False))
            else:
                # get first n samples from UD
                batch_idcs = list(self._iter_idcs)[:min(self._batch_size, len(self._iter_idcs))]

            # gather sentences [['word', 'word', ...], ['word', 'word', ...]] (batch_size, var_lens)
            sentences = [s.to_words() for s in self._ud[batch_idcs]]
            max_len = max([len(s) for s in sentences])

            targets = self.get_targets(batch_idcs, max_len)

            self._iter_idcs -= set(batch_idcs)
            num_remaining = len(self._iter_idcs)

            return sentences, targets, num_remaining
        else:
            raise StopIteration

    def get_targets(self, idcs, max_len):
        targets = {
            'heads': torch.ones((len(idcs), max_len), dtype=torch.long) * -2,
            'rels': torch.ones((len(idcs), max_len), dtype=torch.long) * -1,
            'roots': torch.ones((len(idcs), max_len), dtype=torch.long) * -1,
            'depths': torch.ones((len(idcs), max_len)) * -1,
            'distances': torch.ones((len(idcs), max_len, max_len)) * -1
        }

        for bidx, sidx in enumerate(idcs):
            sen_len = len(self._ud[sidx].to_words())
            # compute heads and relations if not in cache
            if (sidx not in self._cache_heads) or (sidx not in self._cache_rels):
                heads, rels = self._ud[sidx].get_dependencies(include_subtypes=False)
                rels = [self._rels[r] for r in rels]  # map relation names to label indices
                self._cache_heads[sidx] = torch.tensor(heads, dtype=torch.long)
                self._cache_rels[sidx] = torch.tensor(rels, dtype=torch.long)
            targets['heads'][bidx, :sen_len] = self._cache_heads[sidx].clone()
            targets['rels'][bidx, :sen_len] = self._cache_rels[sidx].clone()

            # compute roots, depths and distances if not in cache
            if (sidx not in self._cache_roots) or (sidx not in self._cache_depths) or (sidx not in self._cache_distances):
                roots, depths, distances = self.get_dependency_structures(self._cache_heads[sidx])
                self._cache_roots[sidx] = roots
                self._cache_depths[sidx] = depths
                self._cache_distances[sidx] = distances
            targets['roots'][bidx, :sen_len] = self._cache_roots[sidx].clone()
            targets['depths'][bidx, :sen_len] = self._cache_depths[sidx].clone()
            targets['distances'][bidx, :sen_len, :sen_len] = self._cache_distances[sidx].clone()

        # move everything to GPU if available
        if torch.cuda.is_available():
            for tgt_key in ['heads', 'rels', 'roots', 'depths', 'distances']:
                targets[tgt_key] = targets[tgt_key].to(torch.device('cuda'))

        return targets

    def get_dependency_structures(self, heads):
        sen_len = heads.shape[0]
        # set root labels within sentence
        roots = (heads == -1).long()
        # init norms with padding value -1 (sen_len)
        depths = torch.ones_like(heads) * -1
        # init distances with padding values -1 (sen_len, sen_len)
        distances = torch.ones((sen_len, sen_len)) * -1

        # progress through tree
        cur_depth = 0  # start with depth 0
        cur_heads = [-1]  # start with root
        while cur_heads:
            cur_children = []
            # set distances between current heads, their children and the history
            for head in cur_heads:
                # gather current head's children
                children = [i for i in range(sen_len) if heads[i] == head]
                cur_children += children

                # skip further processing for root node
                if head < 0: continue

                # set distance to self to 0
                distances[head, head] = 0

                # set distance from current head to its children to 1
                distances[head, children] = 1
                distances[children, head] = 1

                # propagate existing distances to children
                for node in range(sen_len):
                    # init indices to propagate to with all children
                    prop_mask = list(children)
                    # skip uninitialized distances
                    if distances[head, node] < 1: continue
                    # if node is child, do not propagate its own distance to itself
                    if node in children:
                        prop_mask.remove(node)
                    # propagate current head's distance + 1 to all relevant child nodes
                    distances[prop_mask, node] = distances[head, node] + 1
                    distances[node, prop_mask] = distances[head, node] + 1

            # set depth of all child nodes at current depth
            depths[cur_children] = cur_depth

            cur_heads = cur_children
            cur_depth += 1

        return roots, depths, distances


"""
Custom errors
"""

class TokenizationError(Exception):
    def __init__(self, message, position=None):
        super().__init__(message)
        self.position = position
        

"""
Home-made functions
"""

def load_dataset(path, model):

    print(f"loading dataset from path: {path}", flush=True)
    ds = Dataset.load_from_disk(path).with_format(
    "torch", 
    )
    
    return ds['input_ids'], ds['tags']

def flatten_tensor_remove_padding(tensor, sentence_lengths):
    # Initialize an empty list to collect valid tokens
    valid_tokens = []
    
    # Iterate over each sentence and corresponding length
    for i, length in enumerate(sentence_lengths):
        # Extract only the first 'length' tokens from each sentence
        valid_tokens.append(tensor[i, :length, :])
    
    # Concatenate all valid tokens into a single tensor along the first dimension
    result = torch.cat(valid_tokens, dim=0)
    
    return result


def process_and_pad_sequences(input_ids, pos_tags, start_token=1, stop_token=2, pad_id=3, pad_pos=-1, mous=False, linguistic_feature=None):
    """
    Function that processes and pads sequences. It can handle an additional linguistic feature
    that affects the dimensions of pos_tags.
    """
    
    if len(input_ids) != pos_tags.shape[0]:
        raise ValueError("Length of input_ids and pos_tags do not match")

    if len(input_ids.shape) > 1:
        input_ids = input_ids.flatten()
    
    # Handle 3D pos_tags if linguistic_feature is 'distances'
    if linguistic_feature == "distances":
        original_pos_dim = pos_tags.shape[-1]
        pos_tags = pos_tags.reshape(-1, original_pos_dim)
    else:
        pos_tags = pos_tags.flatten()

    # Initialize lists to hold the sentences before padding and token counts
    sentences_input_ids = []
    sentences_pos_tags = []
    sentence_lengths = []
    
    # Start processing from the first start token
    start_idx = (input_ids == start_token).nonzero(as_tuple=True)[0]
    end_idx = (input_ids == stop_token).nonzero(as_tuple=True)[0]
    
    if len(start_idx) != len(end_idx):
        raise ValueError("Mismatch in the number of start and stop tokens")
    
    for start, end in zip(start_idx, end_idx):
        sentence_input_ids = input_ids[start:end + 1]
        sentence_pos_tags = pos_tags[start:end + 1]
        
        sentences_input_ids.append(sentence_input_ids)
        sentences_pos_tags.append(sentence_pos_tags)
        sentence_lengths.append(len(sentence_input_ids))

    max_tokens = max(sentence_lengths)
    
    padded_input_ids = []
    padded_pos_tags = []
    for sentence_input_ids, sentence_pos_tags in zip(sentences_input_ids, sentences_pos_tags):
        pad_length = max_tokens - len(sentence_input_ids)
        if pad_length > 0:
            sentence_input_ids = torch.cat([sentence_input_ids, torch.full((pad_length,), pad_id)])
            if linguistic_feature == "distances":
                padding_pos_tags = torch.full((pad_length, original_pos_dim), pad_pos)
            else:
                padding_pos_tags = torch.full((pad_length,), pad_pos)
            sentence_pos_tags = torch.cat([sentence_pos_tags, padding_pos_tags])

        padded_input_ids.append(sentence_input_ids)
        padded_pos_tags.append(sentence_pos_tags)

    padded_input_ids = torch.stack(padded_input_ids)
    padded_pos_tags = torch.stack(padded_pos_tags)

    assert padded_input_ids.shape[0] == padded_pos_tags.shape[0], "Mismatch in number of sentences"
    assert padded_input_ids.shape[1] == padded_pos_tags.shape[1], "Mismatch in number of tokens"

    if mous:
        assert padded_input_ids.shape == torch.Size([400, 24]), f"Shape of padded_input_ids is {padded_input_ids.shape} when it should be [400, 24]"

    return padded_input_ids, padded_pos_tags, sentence_lengths

# def process_and_pad_sequences(input_ids, pos_tags, start_token=1, stop_token=2, pad_id=3, pad_pos=-1, mous=False):
#     """
#     Function that processes and pads sequences
#     """
    
#     if len(input_ids) != len(pos_tags):
#         raise ValueError("Length of input_ids and pos_tags do not match")

#     if len(input_ids.shape) > 1:
#         input_ids = input_ids.flatten()
#         pos_tags = pos_tags.flatten()

#     # Initialize lists to hold the sentences before padding and token counts
#     sentences_input_ids = []
#     sentences_pos_tags = []
#     sentence_lengths = []
    
#     # Start processing from the first start token
#     start_idx = (input_ids == start_token).nonzero(as_tuple=True)[0]
#     end_idx = (input_ids == stop_token).nonzero(as_tuple=True)[0]
    
#     # Check if there's a mismatch in the number of start and stop tokens
#     if len(start_idx) != len(end_idx):
#         raise ValueError("Mismatch in the number of start and stop tokens")
    
#     for start, end in zip(start_idx, end_idx):
#         # Extract sentences between start and stop tokens, including the stop token
#         sentence_input_ids = input_ids[start:end + 1]
#         sentence_pos_tags = pos_tags[start:end + 1]
        
#         # Append the sentences to the lists
#         sentences_input_ids.append(sentence_input_ids)
#         sentences_pos_tags.append(sentence_pos_tags)
#         sentence_lengths.append(len(sentence_input_ids))  # Store the length before padding

#     # Find the maximum length of the sentences
#     max_tokens = max(sentence_lengths)
    
#     # Now pad all sentences to the max_tokens length
#     padded_input_ids = []
#     padded_pos_tags = []
#     for sentence_input_ids, sentence_pos_tags in zip(sentences_input_ids, sentences_pos_tags):
#         pad_length = max_tokens - len(sentence_input_ids)
#         if pad_length > 0:
#             sentence_input_ids = torch.cat([sentence_input_ids, torch.full((pad_length,), pad_id)])
#             sentence_pos_tags = torch.cat([sentence_pos_tags, torch.full((pad_length,), pad_pos)])
#         # Append the padded sentences to the lists
#         padded_input_ids.append(sentence_input_ids)
#         padded_pos_tags.append(sentence_pos_tags)

#     # Stack all padded sentences into a single tensor
#     padded_input_ids = torch.stack(padded_input_ids)
#     padded_pos_tags = torch.stack(padded_pos_tags)

#     assert padded_input_ids.shape[0] == padded_pos_tags.shape[0], "Number of sentences in input_ids and pos_tags do not match"
#     assert padded_input_ids.shape[1] == padded_pos_tags.shape[1], "Number of tokens in input_ids and pos_tags do not match"

#     if mous:
#         assert padded_input_ids.shape == torch.Size([400, 24]), f"Shape of padded_input_ids is {padded_input_ids.shape} when it should be [400, 24]"

#     return padded_input_ids, padded_pos_tags, sentence_lengths

# def generate_padded_hidden_states(padded_input_ids, model_name = "GroNLP/bert-base-dutch-cased", layers = None):
#     """
#     Function that generates padded hidden states
#     """

#     # Check if the model is the correct one
#     assert model_name == "GroNLP/bert-base-dutch-cased", "Only GroNLP/bert-base-dutch-cased is supported for now, change padding input ID if using another model"

#     # Load the model
#     model = transformers.AutoModel.from_pretrained(model_name, return_dict=True)

#     #Set the random seed for reproducibility
#     torch.manual_seed(42)

#     # Set the model to evaluation mode
#     model.eval()

#     # make attention mask for [PAD] tokens
#     attention_mask = (padded_input_ids != 3).int() 

#     # Generate hidden states
#     with torch.no_grad():
#         outputs = model(padded_input_ids, output_hidden_states=True, attention_mask=attention_mask)

#     hidden_states = outputs.hidden_states

#     return hidden_states


def generate_padded_hidden_states(padded_input_ids, model_name="GroNLP/bert-base-dutch-cased", layers=None, batch_size=10500):
    """
    Function that generates padded hidden states for a given model in batches.
    Args:
        padded_input_ids (torch.Tensor): Tensor of input ids padded to the same length.
        model_name (str): Name of the model to load.
        layers (list[int]): List of two layer indices whose hidden states are to be retained.
        batch_size (int): The size of the batch for processing.
    Returns:
        list[torch.Tensor]: List of two tensors, each containing the hidden states from one of the specified layers.
    """
    # Assert model and layers
    assert model_name == "GroNLP/bert-base-dutch-cased", "Only GroNLP/bert-base-dutch-cased is supported for now."
    assert len(layers) == 2, "Exactly two layers must be specified."


    # Load the model
    print(f"Loading model {model_name} for eraser training", flush=True)
    model = transformers.AutoModel.from_pretrained(model_name, return_dict=True)
    model.eval()

    # Initialize random seed for reproducibility
    torch.manual_seed(42)

    # Initialize lists to collect each layer's hidden states
    hidden_states_layer_1 = []
    hidden_states_layer_2 = []

    # Process in batches
    num_samples = padded_input_ids.shape[0]
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_ids = padded_input_ids[start_idx:end_idx]

        print(f"Processing batch {start_idx + 1} to {end_idx} of {num_samples}", flush=True)

        # Create attention mask (assuming padding id is 3)
        attention_mask = (batch_ids != 3).int()

        # Generate hidden states with no gradient calculation
        with torch.no_grad():
            outputs = model(batch_ids, output_hidden_states=True, attention_mask=attention_mask)
        
        # Extract and store the hidden states for specified layers
        batch_hidden_states = outputs.hidden_states
        hidden_states_layer_1.append(batch_hidden_states[layers[0]])
        hidden_states_layer_2.append(batch_hidden_states[layers[1]])

        del outputs, batch_hidden_states
        gc.collect()

    # Concatenate all batches for each layer
    final_hidden_states_layer_1 = torch.cat(hidden_states_layer_1, dim=0)
    final_hidden_states_layer_2 = torch.cat(hidden_states_layer_2, dim=0)

    return [final_hidden_states_layer_1, final_hidden_states_layer_2]



def apply_mapping(tensor, mapping):
    """
    Function that applies a mapping to a tensor
    """
    # Map each tensor value to its new value based on the mapping
    mapped_tensor = torch.tensor([mapping[val.item()] for val in tensor])
    return mapped_tensor


def create_mapping(tensor):
    """
    Function that creates a mapping for the unique values in a tensor
    """
    unique_vals = torch.unique(tensor)
    sorted_unique_vals = torch.sort(unique_vals).values
    mapping = {val.item(): idx for idx, val in enumerate(sorted_unique_vals)}
    return mapping

def load_padded_hidden_states(hidden_state_path, embedding_layers, n_batches = 2):
    """
    Function that loads hidden states for the embeddings layer
    """
    hidden_states =[]

    for layer in embedding_layers:
        batch_hidden_states = []
        for i in range(n_batches):
            current_hidden_state = torch.load(hidden_state_path + f"/hidden_states_layer_{layer}_batch_{i}.pt")
            batch_hidden_states.append(current_hidden_state)

            del current_hidden_state
            gc.collect()

        hidden_states.append(torch.cat(batch_hidden_states, dim=0))

    return hidden_states
            
    

def load_hidden_states_for_embeddings_layers(dataset_path, language_model, embedding_layers, linguistic_feature, hidden_state_path = None):
    
    # load language model
    language_model = transformers.AutoModel.from_pretrained(language_model, return_dict=True)

    if linguistic_feature == "dep_heads_and_labels":
        # load dataset
        input_ids, heads_tags = load_dataset(dataset_path[0], language_model)
        _ , labels_tags= load_dataset(dataset_path[1], language_model)


        # process and pad sequences
        padded_input_ids, padded_heads_tags, sentence_lengths = process_and_pad_sequences(input_ids, heads_tags)
        _, padded_labels_tags, _ = process_and_pad_sequences(input_ids, labels_tags)

        flat_padded_heads_tags = torch.flatten(padded_heads_tags)
        flat_padded_labels_tags = torch.flatten(padded_labels_tags)

        # map the tags to unique values and create a mapping so that the matrix is not uneccesarily large and contians no negative values
        flat_padded_heads_tags = apply_mapping(flat_padded_heads_tags, create_mapping(flat_padded_heads_tags))
        flat_padded_labels_tags = apply_mapping(flat_padded_labels_tags, create_mapping(flat_padded_labels_tags))

        flat_tags = (flat_padded_heads_tags, flat_padded_labels_tags)

        padded_tags = flat_tags[0].view(padded_heads_tags.shape[0], padded_heads_tags.shape[1]), flat_tags[1].view(padded_labels_tags.shape[0], padded_labels_tags.shape[1])

    elif linguistic_feature == "distances":
        # load dataset
        input_ids, tags = load_dataset(dataset_path[0], language_model)

        # process and pad sequences
        padded_input_ids, padded_tags, sentence_lengths = process_and_pad_sequences(input_ids, tags, linguistic_feature = linguistic_feature)

        flat_tags = padded_tags.view(-1, padded_tags.shape[2])

        
    else:
        # load dataset
        input_ids, tags = load_dataset(dataset_path[0], language_model)

        # process and pad sequences
        padded_input_ids, padded_tags, sentence_lengths = process_and_pad_sequences(input_ids, tags)

        flat_tags = padded_tags.flatten()

        # map the tags to unique values and create a mapping so that the matrix is not uneccesarily large and contians no negative values
        flat_tags = apply_mapping(flat_tags, create_mapping(flat_tags))

        # reformat the flar tags into a 2D tensor to use for applying the split indices later on
        padded_tags = flat_tags.view(padded_tags.shape[0], padded_tags.shape[1])

    # generate padded hidden states
    n_batches = math.ceil(padded_input_ids.shape[0] / 5250)
    print(f"Number of batches: {n_batches}")
    padded_hidden_states = load_padded_hidden_states(hidden_state_path, embedding_layers, n_batches = n_batches)
    # padded_hidden_states = generate_padded_hidden_states(padded_input_ids, layers = embedding_layers)

    n_features = padded_hidden_states[0].size(2)

    flattened_hidden_states = []
    for hidden_state in padded_hidden_states:
        # flatten the padded hidden states
        flattened_hidden_state = hidden_state.view(-1, n_features)
        flattened_hidden_states.append(flattened_hidden_state)
    
    del padded_hidden_states
    gc.collect()

    return flattened_hidden_states, flat_tags, padded_tags


def load_hidden_states_train_eraser(dataset_path, language_model, embedding_layers, erasure_method, linguistic_feature, hidden_state_path = None):
    """
    Function that loads hidden states for training the eraser
    """

    print("Loading hidden states for training the eraser")

    if linguistic_feature == "dep_heads_and_labels":
        hidden_states, tags, padded_tags = load_hidden_states_for_embeddings_layers(dataset_path, language_model, embedding_layers, linguistic_feature, hidden_state_path = hidden_state_path)
        one_hot_tags = make_one_hot_encoding(tags, dep_heads_and_labels = True)
        padded_tags = padded_tags[0]
    elif linguistic_feature == "distances":
        hidden_states, tags, padded_tags = load_hidden_states_for_embeddings_layers(dataset_path, language_model, embedding_layers, linguistic_feature, hidden_state_path = hidden_state_path)
        one_hot_tags = tags
    else:
        # Load hidden states for the embeddings layer
        hidden_states, tags, padded_tags = load_hidden_states_for_embeddings_layers(dataset_path, language_model, embedding_layers, linguistic_feature, hidden_state_path = hidden_state_path)
        # print unique tags
        # print("Unique tags:", torch.unique(tags))
        one_hot_tags = make_one_hot_encoding(tags)

    erasers = []
    for i in range(len(hidden_states)):
        
        # Train the eraser
        if erasure_method == "leace":
            eraser = train_leace(hidden_states[i], one_hot_tags)
        elif erasure_method == "oracle":
            eraser = train_oracle(hidden_states[i], one_hot_tags)
        elif erasure_method == "outregressor":
            eraser = train_outregressor(hidden_states[i], one_hot_tags)
        else:
            raise ValueError(f"Erasure method {erasure_method} not supported")
        erasers.append(eraser)

    del hidden_states
    gc.collect()
    
    return erasers, one_hot_tags, padded_tags


def torch_to_np(tensor):
    """
    Function that converts a torch tensor to a numpy array
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor

def np_to_torch(array):
    """
    Function that converts a numpy array to a torch tensor
    """
    if isinstance(array, np.ndarray):
        return torch.tensor(array)
    elif isinstance(array, torch.Tensor):
        return array


def verify_one_hot_encoding(tags_one_hot):
    """
    Function that verifies if the one-hot encoding is correct
    """
    # Check if tensor only contains 0s and 1s	
    tags_one_hot = np_to_torch(tags_one_hot)
    assert torch.all((tags_one_hot == 0) | (tags_one_hot == 1)), "One-hot encoding should only contain 0s and 1s"



def train_leace(hidden_states, tags_one_hot, dep_heads_and_labels = False):

    tags_one_hot = np_to_torch(tags_one_hot)
    
    eraser = LeaceEraser.fit(hidden_states, tags_one_hot)

    return eraser

def apply_leace(hidden_states, eraser):
    """
    Function that applies the LEACE eraser to the hidden states
    """
    hidden_states_clean = eraser(hidden_states)

    return hidden_states_clean


def make_one_hot_encoding(tags, dep_heads_and_labels = False):
    """
    Function that converts a tensor of tags to one-hot encoding
    """
    if dep_heads_and_labels:
        tags_one_hot = create_multi_hot_encoding(tags[0], tags[1])
    else:
        tags_one_hot = torch.nn.functional.one_hot(tags)
    
    verify_one_hot_encoding(tags_one_hot)

    return tags_one_hot


def train_oracle(hidden_states, tags_one_hot, dep_heads_and_labels = False):

    tags_one_hot = np_to_torch(tags_one_hot)

    eraser = OracleEraser.fit(hidden_states, tags_one_hot)

    return eraser

def apply_oracle(hidden_states, eraser, tags_one_hot):
    """
    Function that applies the oracle eraser to the hidden states
    """
    tags_one_hot = np_to_torch(tags_one_hot)

    hidden_states_clean = eraser(hidden_states, tags_one_hot)

    return hidden_states_clean

def apply_eraser(hidden_states, eraser, tags_one_hot, erasure_method):
    """
    Function that applies the eraser to the hidden states
    """

    if erasure_method == "leace":
        hidden_states_clean = apply_leace(hidden_states, eraser)
    elif erasure_method == "oracle":
        hidden_states_clean = apply_oracle(hidden_states, eraser, tags_one_hot)
    elif erasure_method == "outregressor":
        hidden_states_clean = outregressor(eraser, hidden_states, tags_one_hot)

    hidden_states_clean = np_to_torch(hidden_states_clean).float()

    return hidden_states_clean

def train_outregressor(hidden_states, tags_one_hot, dep_heads_and_labels = False):
    
    tags_one_hot = torch_to_np(tags_one_hot)
    
    model = LinearRegression()

    model.fit(tags_one_hot, hidden_states)
    
    return model

def outregressor(model, hidden_states, tags_one_hot):
    
    hidden_states_clean = hidden_states - model.predict(tags_one_hot)

    return hidden_states_clean

def apply_mapping(tensor, mapping):
    """
    Function that applies a mapping to a tensor
    """
    # Map each tensor value to its new value based on the mapping
    mapped_tensor = torch.tensor([mapping[val.item()] for val in tensor])
    return mapped_tensor


def create_mapping(tensor):
    """
    Function that creates a mapping for the unique values in a tensor
    """
    unique_vals = torch.unique(tensor)
    sorted_unique_vals = torch.sort(unique_vals).values
    mapping = {val.item(): idx for idx, val in enumerate(sorted_unique_vals)}
    return mapping


def create_multi_hot_encoding(heads, labels):
    """
    Function that creates a multi-hot encoding for the dependency heads and labels together
    """

    heads = np_to_torch(heads)
    labels = np_to_torch(labels)

    # Number of unique heads and labels
    num_heads = len(torch.unique(heads))
    num_labels = len(torch.unique(labels))

    # Create mappings for heads and labels
    head_mapping = create_mapping(heads)
    label_mapping = create_mapping(labels)

    # Apply mappings to tensors
    converted_heads = apply_mapping(heads, head_mapping)
    converted_labels = apply_mapping(labels, label_mapping)

    # Number of words
    num_tokens = len(heads)
    
    # Total number of features is the sum of unique heads and relations
    num_features = num_heads + num_labels
    
    # Initialize the tensor with zeros
    multi_hot_tensor = torch.zeros(num_tokens, num_features)
    
    # Encode dependency heads by setting the corresponding positions to 1
    for i, head in enumerate(converted_heads):
        multi_hot_tensor[i, head] = 1
    
    # Offset for relation indices in the tensor
    offset = num_heads
    
    # Encode relations by setting the corresponding positions to 1
    for i, label in enumerate(converted_labels):
        # Adjust index for the relation by adding offset
        multi_hot_tensor[i, label+ offset] = 1
    
    return multi_hot_tensor



def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='Universal Dependencies - Embedding Space Parsing')
    arg_parser.add_argument('ud_path', help='path to Universal Dependencies directory')
    arg_parser.add_argument('out_path', help='path to output directory')
    arg_parser.add_argument('dataset_path', nargs = '+', help='path to dataset')
    arg_parser.add_argument('erasure_method', help='method of erasure')
    arg_parser.add_argument('linguistic_feature', help='feature to be erased')
    # parser setup
    arg_parser.add_argument('-hsp','--hidden_state_path', help='path to hidden states')
    arg_parser.add_argument(
        '-lm', '--language_model', default='bert-base-multilingual-cased',
        help='language model name in the transformers library (default: bert-base-multilingual-cased')
    arg_parser.add_argument(
        '-el', '--embedding_layers', nargs='+', type=int, default=[6, 7],
        help='list of embedding layers (0: WordPiece -> 12: Layer 12, default: [6, 7])')
    arg_parser.add_argument(
        '-ec', '--embedding_cache',
        help='path to pre-computed embedding cache or set to "local" for in-memory caching (default: None)')
    arg_parser.add_argument(
        '-ds', '--dependency_size', type=int, default=128,
        help='dimensionality of dependency space transformation (default: 128)')
    arg_parser.add_argument(
        '-pt', '--parser_type', default='depprobe', choices=['structural', 'directed', 'depprobe', 'depprobe-mix'],
        help='parser type (default: depprobe)')
    arg_parser.add_argument(
        '-pd', '--parser_decode', default=False, action='store_true',
        help='set flag to decode parses during training (default: False)')
    # experiment setup
    arg_parser.add_argument(
        '-s', '--split', help='path to data split definition pickle (default: None - full UD)')
    arg_parser.add_argument(
        '-td', '--treebank_directory', default=False, action='store_true',
        help='set flag to load single treebank from directory instead of mix of treebanks from split (default: False)')
    arg_parser.add_argument(
        '-e', '--epochs', type=int, default=100, help='maximum number of epochs (default: 100)')
    arg_parser.add_argument(
        '-es', '--early_stop', type=int, default=5, help='maximum number of epochs without improvement (default: 5, -1 to disable)')
    arg_parser.add_argument(
        '-bs', '--batch_size', type=int, default=64, help='maximum number of sentences per batch (default: 64)')
    arg_parser.add_argument(
        '-lr', '--learning_rate', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    arg_parser.add_argument(
        '-rs', '--seed', type=int, default=42, help='seed for probabilistic components (default: 42)')
    return arg_parser.parse_args()


def main():
    args = parse_arguments()

    # check if output dir exists
    setup_output_directory(args.out_path)


    # setup logging
    setup_logging(os.path.join(args.out_path, 'train.log'))

    # set random seeds
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    transformers.set_seed(args.seed)

    # setup UD data
    ud, splits, rel_map = setup_data(args.ud_path, args.split, args.treebank_directory)


    train_data = DepSpaceDataset(ud, rel_map, splits['train'], args.batch_size, random = False)

    logging.info(f"Loaded training split with {len(train_data)} sentences.")
    # load dev split if early stopping is enabled
    if args.early_stop < 0:
        logging.info("Early stopping disabled. Not loading dev data.")
    else:
        eval_data = DepSpaceDataset(ud, rel_map, splits['dev'], args.batch_size)
        logging.info(f"Loaded dev split with {len(eval_data)} sentences.")

    print("setting up parser model")
    # setup parser model
    parser = setup_model(
        lm_name=args.language_model, dep_dim=args.dependency_size,
        parser_type=args.parser_type,
        emb_layers=args.embedding_layers,
        emb_cache=args.embedding_cache,
        dataset_path = args.dataset_path,
        erasure_method = args.erasure_method,
        linguistic_feature = args.linguistic_feature,
        batch_size = args.batch_size,
        splits = splits,
        hidden_state_path = args.hidden_state_path
    )

    print("setting up criterion")
    # setup loss
    criterion = setup_criterion(parser_type=args.parser_type)

    # setup optimizer
    optimizer = torch.optim.AdamW(params=parser.get_trainable_parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=0)
    logging.info(f"Optimizing using {optimizer.__class__.__name__} with learning rate {args.learning_rate}.")
    logging.info(f"Scheduler {scheduler.__class__.__name__} reduces learning rate by 0.1 after 1 epoch without improvement.")

    # main training loop
    stats = defaultdict(list)
    stats['time'].append(time.time())
    for ep_idx in range(args.epochs):
        # iterate over batches in training split
        cur_stats = run(
            parser, criterion, optimizer,
            train_data, mode='train', decode=args.parser_decode
        )

        # store and print statistics
        print(f"current epoch = {ep_idx}, max number of epochs = {args.epochs}")
        statistics('train', stats, cur_stats, ep_idx, args.epochs)

        # save most recent model
        path = os.path.join(args.out_path, 'newest.tar')
        save_checkpoint(parser, optimizer, ep_idx, stats, path)
        logging.info(f"Saved model from epoch {ep_idx + 1} to '{path}'.")

        # continue to next epoch if early stopping is disabled
        if args.early_stop < 0:
            continue

        # iterate over batches in dev split
        cur_stats = run(
            parser, criterion, None,
            eval_data, mode='eval', decode=True
        )
        stats['time'].append(time.time())

        # store and print statistics
        statistics('eval', stats, cur_stats, ep_idx, args.epochs)
        cur_eval_loss = stats['eval/loss'][-1]

        # save best model
        if cur_eval_loss <= min(stats['eval/loss']):
            path = os.path.join(args.out_path, 'best.tar')
            save_checkpoint(parser, optimizer, ep_idx, stats, path)
            logging.info(f"Saved model with best loss {cur_eval_loss:.4f} to '{path}'.")

        # update scheduler
        scheduler.step(cur_eval_loss)
        # check for early stopping
        if (ep_idx - stats['eval/loss'].index(min(stats['eval/loss']))) >= args.early_stop:
            logging.info(f"No improvement since {args.early_stop} epochs ({min(stats['eval/loss']):.4f} loss). Early stop.")
            break

        stats['time'].append(time.time())

    logging.info(f"Training completed after {ep_idx + 1} epochs and {stats['time'][-1] - stats['time'][0]:.2f} seconds.")
    logging.info(f"Total training time {sum(stats['train/time']):.2f} seconds across {sum(stats['train/tokens'])} ({sum(stats['train/time'])/sum(stats['train/tokens']):.2f} tokens/sec).")


if __name__ == '__main__':
    main()
    