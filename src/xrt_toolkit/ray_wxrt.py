import drjit as dr
import numpy as np
import numpy.typing as npt

import xrt_toolkit.drjit as xtk_drjit
import xrt_toolkit.ray_xrt as ray_xrt
import xrt_toolkit.util as xtk_util
from xrt_toolkit.array_module import NDArrayInfo

__all__ = [
    "RayWXRT",
]


class RayWXRT(ray_xrt.RayXRT):
    r"""
    Weighted X-Ray Transform in 2D or 3D.

    The Weightded X-Ray Transform (WXRT) of a function :math:`f: \bR^{D} \to \bR` is defined as

    .. math::

       \cP_{\bbw}[f](\bbn, \bbt)
       =
       \int_{\bR} f(\bbt + \bbn \alpha)
       \exp\left[ -\int_{-\infty}^{\alpha} w(\bbt + \bbn \beta) d\beta \right]
       d\alpha,

    where :math:`\bbn \in \bS^{D-1}` and :math:`\bbt \in \bbn^{\perp}`.
    :math:`\cP_{\bbw}[f]` hence denotes the set of weighted *line integrals* of :math:`f`.

    This implementation computes samples of the WXRT using a ray-marching method based on the `Dr.Jit <https://drjit.readthedocs.io/en/latest/reference.html>`_ compiler.
    It assumes :math:`(f,w)` is a pixelized image/volume where:

    * the lower-left element of :math:`(f,w)` are located at :math:`\bbo \in \bR^{D}`,
    * pixel dimensions are :math:`\bbDelta \in \bR_{+}^{D}`, i.e.

    .. math::

       \begin{align*}
           f(\bbr) & = \sum_{\bbq \subset \bN^{D}}
                             a_{\bbq}
                             1_{[\bbZero, \bbDelta]}(\bbr - \bbq \odot \bbDelta - \bbo),
                             \quad
                             a_{\bbq} \in \mathbb{R}, \\
           w(\bbr) & = \sum_{\bbq \subset \bN^{D}}
                             w_{\bbq}
                             1_{[\bbZero, \bbDelta]}(\bbr - \bbq \odot \bbDelta - \bbo),
                             \quad
                             w_{\bbq} \in \mathbb{R}.
       \end{align*}

    .. image:: ./dev/figures/wxray_parametrization.svg
       :alt: 2D weighted XRay Geometry
       :width: 50%
       :align: center

    Notes
    -----
    * Using `RayWXRT` on the CPU requires LLVM.
      See the `Dr.Jit documentation <https://drjit.readthedocs.io/en/latest/index.html>`_ for details.
    * Using `RayWXRT` on the GPU requires installing `xrt_toolkit` with GPU add-ons.
      See the README for details.
    """

    idtype = np.dtype(np.int32)
    fdtype = np.dtype(np.single)

    def __init__(
        self,
        origin: tuple[float],
        pitch: tuple[float],
        N: tuple[int],
        n_spec: npt.ArrayLike,
        t_spec: npt.ArrayLike,
        w_spec: npt.ArrayLike,
    ):
        r"""
        Parameters
        ----------
        origin: tuple[float]
            (D,) bottom-left coordinate :math:`\bbo \in \bR^{D}`.
        pitch: tuple[float]
            (D,) pixel size :math:`\bbDelta \in \bR_{+}^{D}`.
        N: tuple[int]
            (N1,...,ND) pixel count in each dimension.

            This parameter sets the dimensionality of the transform.
        n_spec: ndarray[float]
            (N_ray, D) ray directions :math:`\bbn \in \bS^{D-1}`.
        t_spec: ndarray[float]
            (N_ray, D) offset specifiers :math:`\bbt \in \bR^{D}`.
        w_spec: ndarray[float]
            (N1,...,ND) spatial decay weights :math:`w \in \mathbb{R}`.

        Notes
        -----
        * `RayWXRT` instances are **not arraymodule-agnostic**:
          they will only work with ndarrays belonging to the same array module as (`n_spec`, `t_spec`, `w_spec`).
        * `RayWXRT` is **not** precision-agnostic:
          it will only work on ndarrays in single-precision.
          A warning is emitted if inputs must be cast.
        """
        super().__init__(
            origin=origin,
            pitch=pitch,
            N=N,
            n_spec=n_spec,
            t_spec=t_spec,
        )

        ndi_n = NDArrayInfo.from_obj(self.cfg.n_spec)
        ndi_w = NDArrayInfo.from_obj(w_spec)
        assert ndi_n == ndi_w
        assert np.all(w_spec.shape == self.cfg.N)

        xp = ndi_w.module()
        w_spec = xp.require(  # allows zero-copy storage in `cfg_dr`.
            w_spec,
            dtype=self.fdtype,
            requirements="C",
        )

        # store `w_spec` in (cfg, cfg_dr)
        self.cfg = xtk_util.as_namedtuple(
            **self.cfg._asdict(),
            w_spec=w_spec,
        )
        drb = xtk_drjit.load_dr_variant(ndi_w)
        self.cfg_dr = xtk_util.as_namedtuple(
            **self.cfg_dr._asdict(),
            w=drb.Float(xtk_drjit.xp2dr(self.cfg.w_spec.ravel())),
        )

        # Cheap analytical Lipschitz upper bound given by
        #   \sigma_{\max}(P) <= \norm{P}{F},
        # with
        #   \norm{P}{F}^{2}
        #   <= (max cell weight)^{2} * #non-zero elements
        #    = (max cell weight)^{2} * N_ray * (maximum number of cells traversable by a ray)
        #    = (max cell weight)^{2} * N_ray * \norm{N}{2}
        #
        #    (max cell weight) =
        #        w_min > 0: \norm{pitch}{2}
        #        w_min < 0: cannot infer
        if w_spec.min() < 0:
            max_cell_weight = np.inf
        else:
            max_cell_weight = np.linalg.norm(self.cfg.pitch)
        self.lipschitz = max_cell_weight * np.sqrt(self.cfg.N_ray * np.linalg.norm(self.cfg.N))

    def apply(self, a: npt.ArrayLike) -> npt.ArrayLike:
        r"""
        Parameters
        ----------
        a: ndarray[float]
            (N1,...,ND) spatial weights :math:`a \in \bR^{N_{1} \times\cdots\times N_{D}}`.

        Returns
        -------
        b: ndarray[float32]
            (N_ray,) WXRT samples :math:`b = P_{w}[f](a)`.
        """
        a = xtk_util.cast_warn(a, self.fdtype)

        ndi = NDArrayInfo.from_obj(a)
        xp = ndi.module()

        drb = xtk_drjit.load_dr_variant(ndi)
        _I = a.ravel()  # (N1*...*ND,) contiguous
        I_dr = drb.Float(xtk_drjit.xp2dr(_I))

        xrt_apply = _get_wxrt_apply(drb, self.cfg.D)
        _P = xrt_apply(  # (N_ray,)
            o=self.cfg_dr.o,
            pitch=self.cfg_dr.pitch,
            N=self.cfg_dr.N,
            I=I_dr,
            w=self.cfg_dr.w,
            r=self.cfg_dr.r,
        )

        b = xp.asarray(_P, dtype=self.fdtype)  # (N_ray,)
        return b

    def adjoint(self, b: npt.ArrayLike) -> npt.ArrayLike:
        r"""
        Parameters
        ----------
        b: ndarray[float]
            (N_ray,) WXRT samples :math:`b = P_{w}[f](a)`.

        Returns
        -------
        a: ndarray[float32]
            (N1,...,ND) spatial weights :math:`a \in \bR^{N_{1} \times\cdots\times N_{D}}`.
        """
        b = xtk_util.cast_warn(b, self.fdtype)

        ndi = NDArrayInfo.from_obj(b)
        xp = ndi.module()

        drb = xtk_drjit.load_dr_variant(ndi)
        _P = b.ravel()  # (N_ray,) contiguous
        P_dr = drb.Float(xtk_drjit.xp2dr(_P))

        xrt_adjoint = _get_wxrt_adjoint(drb, self.cfg.D)
        _I = xrt_adjoint(
            o=self.cfg_dr.o,
            pitch=self.cfg_dr.pitch,
            N=self.cfg_dr.N,
            P=P_dr,
            w=self.cfg_dr.w,
            r=self.cfg_dr.r,
        )

        a = xp.asarray(_I, dtype=self.fdtype)  # (N1*...*ND,)
        a = a.reshape(self.cfg.N)  # (N1,...,ND)
        return a

    # Internal Helpers --------------------------------------------------------


def _get_wxrt_apply(drb: xtk_drjit.DrJitBackend, D: int):
    """
    Create DrJIT FW transform.
    """
    Arrayf = xtk_drjit.Arrayf_Factory(drb, D)
    Arrayu = xtk_drjit.Arrayu_Factory(drb, D)
    Rayf = xtk_drjit.Rayf_Factory(drb, D)
    BoundingBoxf = xtk_drjit.BoundingBoxf_Factory(drb, D)

    ray_step = xtk_drjit.get_ray_step(drb, D)

    def wxrt_apply(
        o: Arrayf,
        pitch: Arrayf,
        N: Arrayu,
        I: drb.Float,
        w: drb.Float,
        r: Rayf,
    ) -> drb.Float:
        # Weighted X-Ray Forward-Projection.
        #
        # Parameters
        #   o:     bottom-left coordinate of I[0,...,0]
        #   pitch: cell dimensions \in \bR_{+}
        #   N:     (D,) lattice size
        #   I:     (N1*...*ND,) cell weights \in \bR [C-order]
        #   w:     (N1*...*ND,) cell decay rates \in \bR [C-order]
        #   r:     (L,) ray descriptors
        # Returns
        #   P:     (L,) forward-projected samples \in \bR

        # Go to normalized coordinates
        ipitch = dr.rcp(pitch)
        r = Rayf(
            o=(r.o - o) * ipitch,
            d=dr.normalize(r.d * ipitch),
        )
        stride = Arrayu(N[1], 1) if (D == 2) else Arrayu(N[1] * N[2], N[2], 1)
        flat_index = lambda i: dr.dot(stride, Arrayu(i))  # Arrayf (int-valued) -> UInt32

        L = max(dr.shape(r.o)[1], dr.shape(r.d)[1])
        P = dr.zeros(drb.Float, L)  # Forward-Projection samples
        d_acc = dr.zeros(drb.Float, L)  # Accumulated decay

        # Move (intersecting) rays to volume surface
        bbox_vol = BoundingBoxf(Arrayf(0), Arrayf(N))
        active, a1, a2 = bbox_vol.ray_intersect(r)
        a_min = dr.minimum(a1, a2)
        r.o.assign(dr.select(active, r(a_min), r.o))

        r_next = ray_step(r)
        active &= bbox_vol.contains(r_next.o)
        loop = drb.Loop("WXRT FW", lambda: (r, r_next, active, P, d_acc))
        while loop(active):
            # Read (I, w) at current cell
            #   Careful to disable out-of-bound queries. (Due to FP-errors.)
            idx_I = dr.floor(0.5 * (r_next.o + r.o))
            mask = active & dr.all(0 <= idx_I) & dr.all(idx_I < N)
            weight = dr.gather(type(I), I, flat_index(idx_I), mask)
            decay = dr.gather(type(w), w, flat_index(idx_I), mask)

            # Compute constants
            length = dr.norm((r_next.o - r.o) * pitch)
            A = dr.exp(-d_acc)
            B = dr.select(
                dr.eq(decay, 0),
                length,
                (1 - dr.exp(-decay * length)) * dr.rcp(decay),
            )

            # Update line integral estimates
            P += weight * A * B
            d_acc += decay * length

            # Walk to next lattice intersection
            r.assign(r_next)
            r_next.assign(ray_step(r))
            active &= bbox_vol.contains(r_next.o)
        return P

    return wxrt_apply


def _get_wxrt_adjoint(drb: xtk_drjit.DrJitBackend, D: int):
    """
    Create DrJIT BW transform.
    """
    Arrayf = xtk_drjit.Arrayf_Factory(drb, D)
    Arrayu = xtk_drjit.Arrayu_Factory(drb, D)
    Rayf = xtk_drjit.Rayf_Factory(drb, D)
    BoundingBoxf = xtk_drjit.BoundingBoxf_Factory(drb, D)

    ray_step = xtk_drjit.get_ray_step(drb, D)

    def wxrt_adjoint(
        o: Arrayf,
        pitch: Arrayf,
        N: Arrayu,
        P: drb.Float,
        w: drb.Float,
        r: Rayf,
    ) -> drb.Float:
        # Weighted X-Ray Back-Projection.
        #
        # Parameters
        #   o:     bottom-left coordinate of I[0,...,0]
        #   pitch: cell dimensions \in \bR_{+}
        #   N:     (D,) lattice size
        #   P:     (L,) X-Ray samples \in \bR
        #   w:     (N1*...*ND,) cell decay rates \in \bR [C-order]
        #   r:     (L,) ray descriptors
        # Returns
        #   I:     (N1*...*ND,) back-projected samples \in \bR [C-order]

        # Go to normalized coordinates
        ipitch = dr.rcp(pitch)
        r = Rayf(
            o=(r.o - o) * ipitch,
            d=dr.normalize(r.d * ipitch),
        )
        stride = Arrayu(N[1], 1) if (D == 2) else Arrayu(N[1] * N[2], N[2], 1)
        flat_index = lambda i: dr.dot(stride, Arrayu(i))  # Array3f (int-valued) -> UInt32

        L = dr.shape(P)[0]
        I = dr.zeros(drb.Float, dr.prod(N)[0])  # noqa: E741 (Back-Projection samples)
        d_acc = dr.zeros(drb.Float, L)  # Accumulated decay

        # Move (intersecting) rays to volume surface
        bbox_vol = BoundingBoxf(Arrayf(0), Arrayf(N))
        active, a1, a2 = bbox_vol.ray_intersect(r)
        a_min = dr.minimum(a1, a2)
        r.o.assign(dr.select(active, r(a_min), r.o))

        r_next = ray_step(r)
        active &= bbox_vol.contains(r_next.o)
        active &= dr.neq(P, 0)
        loop = drb.Loop("WXRT BW", lambda: (r, r_next, active, d_acc))
        while loop(active):
            # Read (w,) at current cell
            #   Careful to disable out-of-bound queries. (Due to FP-errors.)
            idx_I = dr.floor(0.5 * (r_next.o + r.o))
            mask = active & dr.all(0 <= idx_I) & dr.all(idx_I < N)
            decay = dr.gather(type(w), w, flat_index(idx_I), mask)

            # Compute constants
            length = dr.norm((r_next.o - r.o) * pitch)
            A = dr.exp(-d_acc)
            B = dr.select(
                dr.eq(decay, 0),
                length,
                (1 - dr.exp(-decay * length)) * dr.rcp(decay),
            )

            # Update back-projections
            dr.scatter_reduce(dr.ReduceOp.Add, I, P * A * B, flat_index(idx_I), mask)
            d_acc += decay * length

            # Walk to next lattice intersection
            r.assign(r_next)
            r_next.assign(ray_step(r))
            active &= bbox_vol.contains(r_next.o)
        return I

    return wxrt_adjoint
