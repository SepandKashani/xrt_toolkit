import importlib

import drjit as dr
import numpy as np
import numpy.typing as npt

import xrt_toolkit.drjit as xtk_drjit
import xrt_toolkit.util as xtk_util
from xrt_toolkit.array_module import NDArrayInfo, to_NUMPY

__all__ = [
    "RayXRT",
]


class RayXRT:
    r"""
    X-Ray Transform in 2D or 3D.

    The X-Ray Transform (XRT) of a function :math:`f: \bR^{D} \to \bR` is defined as

    .. math::

       \cP[f](\bbn, \bbt)
       =
       \int_{\bR} f(\bbt + \bbn \alpha) d\alpha,

    where :math:`\bbn \in \bS^{D-1}` and :math:`\bbt \in \bbn^{\perp}`.
    :math:`\cP[f]` hence denotes the set of *line integrals* of :math:`f`.

    This implementation computes samples of the XRT using a ray-marching method based on the `Dr.Jit <https://drjit.readthedocs.io/en/latest/reference.html>`_ compiler.
    It assumes :math:`f` is a pixelized image/volume where:

    * the lower-left element of :math:`f` is located at :math:`\bbo \in \bR^{D}`,
    * pixel dimensions are :math:`\bbDelta \in \bR_{+}^{D}`, i.e.

    .. math::

       f(\bbr) = \sum_{bbq \subset \bN^{D}}
                       a_{\bbq}
                       1_{[\bbZero, \bbDelta]}(\bbr - \bbq \odot \bbDelta - \bbo),
       \quad
       a_{\bbq} \in \bR.

    .. image:: ./dev/figures/xray_parametrization.svg
       :alt: 2D XRay Geometry
       :width: 50%
       :align: center

    Notes
    -----
    * Using `RayXRT` on the CPU requires LLVM.
      See the `Dr.Jit documentation <https://drjit.readthedocs.io/en/latest/index.html>`_ for details.
    * Using `RayXRT` on the GPU requires installing `xrt_toolkit` with GPU add-ons.
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

        Notes
        -----
        * `RayXRT` instances are **not arraymodule-agnostic**:
          they will only work with ndarrays belonging to the same array module as (`n_spec`, `t_spec`).
        * `RayXRT` is **not** precision-agnostic:
          it will only work on ndarrays in single-precision.
          A warning is emitted if inputs must be cast.
        """
        N = xtk_util.broadcast_seq(N, None, self.idtype)
        assert (D := len(N)) in (2, 3)
        origin = xtk_util.broadcast_seq(origin, D, self.fdtype)
        pitch = xtk_util.broadcast_seq(pitch, D, self.fdtype)
        assert np.all(pitch > 0)
        assert np.all(N > 0)

        ndi_n = NDArrayInfo.from_obj(n_spec)
        ndi_t = NDArrayInfo.from_obj(t_spec)
        assert ndi_n == ndi_t
        assert (n_spec.shape == t_spec.shape) and (n_spec.ndim == 2)
        N_ray = len(n_spec)

        xp = ndi_n.module()
        n_spec = xp.require(n_spec, dtype=self.fdtype, requirements="F")
        t_spec = xp.require(t_spec, dtype=self.fdtype, requirements="F")

        self.cfg = xtk_util.as_namedtuple(
            D=D,
            # -------------------------
            origin=origin,
            pitch=pitch,
            N=N,
            # -------------------------
            n_spec=n_spec,
            t_spec=t_spec,
            N_ray=N_ray,
        )
        self.cfg_dr = self._init_dr_info(
            origin=origin,
            pitch=pitch,
            N=N,
            # -------------------------
            n_spec=n_spec,
            t_spec=t_spec,
        )

        # Cheap analytical Lipschitz upper bound given by
        #   \sigma_{\max}(P) <= \norm{P}{F},
        # with
        #   \norm{P}{F}^{2}
        #   <= (max cell weight)^{2} * #non-zero elements
        #   <= (max cell weight)^{2} * N_ray * (maximum number of cells traversable by a ray)
        #    = (max cell weight)^{2} * N_ray * \norm{N}{2}
        #
        #   (max cell weight) = \norm{pitch}{2}
        max_cell_weight = np.linalg.norm(pitch)
        self.lipschitz = max_cell_weight * np.sqrt(N_ray * np.linalg.norm(N))

        # Vectorize apply/adjoint calls
        sig_a = "(" + ",".join([f"n{d}" for d in range(self.cfg.D)]) + ")"
        sig_b = "(" + "p" + ")"
        sig_fw = f"{sig_a}->{sig_b}"
        sig_bw = f"{sig_b}->{sig_a}"
        self.apply = xp.vectorize(
            self.apply,
            otypes=[self.fdtype],
            signature=sig_fw,
        )
        self.adjoint = xp.vectorize(
            self.adjoint,
            otypes=[self.fdtype],
            signature=sig_bw,
        )

    def apply(self, a: npt.ArrayLike) -> npt.ArrayLike:
        r"""
        Parameters
        ----------
        a: ndarray[float]
            (N1,...,ND) spatial weights :math:`a \in \bR^{N_{1} \times\cdots\times N_{D}}`.

        Returns
        -------
        b: ndarray[float32]
            (N_ray,) XRT samples :math:`b = P[f](a)`.
        """
        a = xtk_util.cast_warn(a, self.fdtype)

        ndi = NDArrayInfo.from_obj(a)
        xp = ndi.module()

        drb = xtk_drjit.load_dr_variant(ndi)
        _I = a.ravel()  # (N1*...*ND,) contiguous
        I_dr = drb.Float(xtk_drjit.xp2dr(_I))

        xrt_apply = _get_xrt_apply(drb, self.cfg.D)
        _P = xrt_apply(  # (N_ray,)
            o=self.cfg_dr.o,
            pitch=self.cfg_dr.pitch,
            N=self.cfg_dr.N,
            I=I_dr,
            r=self.cfg_dr.r,
        )

        b = xp.asarray(_P, dtype=self.fdtype)  # (N_ray,)
        return b

    def adjoint(self, b: npt.ArrayLike) -> npt.ArrayLike:
        r"""
        Parameters
        ----------
        b: ndarray[float]
            (N_ray,) XRT samples :math:`b = P[f](a)`.

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

        xrt_adjoint = _get_xrt_adjoint(drb, self.cfg.D)
        _I = xrt_adjoint(
            o=self.cfg_dr.o,
            pitch=self.cfg_dr.pitch,
            N=self.cfg_dr.N,
            P=P_dr,
            r=self.cfg_dr.r,
        )

        a = xp.asarray(_I, dtype=self.fdtype)  # (N1*...*ND,)
        a = a.reshape(self.cfg.N)  # (N1,...,ND)
        return a

    def diagnostic_plot(
        self,
        ray_idx: npt.ArrayLike = None,
        show_grid: bool = False,
    ):
        r"""
        Plot ray trajectories.

        Parameters
        ----------
        ray_idx: ndarray[int]
            (Q,) indices of rays to plot. (Default: show all rays.)
        show_grid: bool
            If true, overlay the pixel grid.

        Returns
        -------
        fig: :py:class:`~matplotlib.figure.Figure`
            Diagnostic plot.

        Notes
        -----
        * Rays which do not intersect the volume are **not** shown.

        Examples
        --------

        .. plot::

           import numpy as np
           from xrt_toolkit import RayXRT

           op = RayXRT(
               origin=0,
               pitch=1,
               N=(5, 6),
               n_spec=np.array([[1   , 0   ],  # 3 rays ...
                                [0.5 , 0.5 ],
                                [0.75, 0.25]]),
               t_spec=np.array([[2.5, 3],  # ... all defined w.r.t volume center
                                [2.5, 3],
                                [2.5, 3]]),
           )
           fig = op.diagnostic_plot()
           fig.show()

        Notes
        -----
        Requires `Matplotlib <https://matplotlib.org/>`_ to be installed.
        """
        try:
            plt = importlib.import_module("matplotlib.pyplot")
            collections = importlib.import_module("matplotlib.collections")
            patches = importlib.import_module("matplotlib.patches")
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Install Matplotlib to use diagnostic_plot()")

        # Setup Figure ========================================================
        if self.cfg.D == 2:
            fig, ax = plt.subplots()
            data = [(ax, [0, 1], ["x", "y"])]
        else:  # D == 3 case
            fig, ax = plt.subplots(ncols=3)
            data = [
                (ax[0], [0, 1], ["x", "y"]),
                (ax[1], [0, 2], ["x", "z"]),
                (ax[2], [1, 2], ["y", "z"]),
            ]

        # Determine which rays intersect with BoundingBox =====================
        ndi = NDArrayInfo.from_obj(self.cfg.n_spec)
        drb = xtk_drjit.load_dr_variant(ndi)
        BBoxf = xtk_drjit.BoundingBoxf_Factory(drb, self.cfg.D)
        active, a1, a2 = BBoxf(
            pMin=self.cfg_dr.o,
            pMax=self.cfg_dr.o + self.cfg_dr.pitch * self.cfg_dr.N,
        ).ray_intersect(self.cfg_dr.r)
        dr.eval(active, a1, a2)

        # Then extract subset of interest (which intersect bbox)
        if ray_idx is None:
            ray_idx = slice(None)
        active, a1, a2 = map(lambda _: _.numpy()[ray_idx], [active, a1, a2])  # (Q,)
        a12 = np.stack([a1, a2], axis=-1)[active]  # (N_active, 2)
        ray_n = to_NUMPY(self.cfg.n_spec[ray_idx][active])  # (N_active, D)
        ray_t = to_NUMPY(self.cfg.t_spec[ray_idx][active])  # (N_active, D)

        for _ax, dim_idx, dim_label in data:
            # Subsample right dimensions ======================================
            origin = self.cfg.origin[dim_idx]
            N = self.cfg.N[dim_idx]
            pitch = self.cfg.pitch[dim_idx]
            _ray_n = ray_n[:, dim_idx]
            _ray_t = ray_t[:, dim_idx]

            # Helper variables ================================================
            bbox_dim = N * pitch

            # Draw BBox =======================================================
            rect = patches.Rectangle(
                xy=origin,
                width=bbox_dim[0],
                height=bbox_dim[1],
                facecolor="none",
                edgecolor="k",
                label="volume BBox",
            )
            _ax.add_patch(rect)

            # Draw Pitch =======================================================
            p_rect = patches.Rectangle(
                xy=origin + bbox_dim - pitch,
                width=pitch[0],
                height=pitch[1],
                facecolor="r",
                edgecolor="none",
                label="pitch size",
            )
            _ax.add_patch(p_rect)

            # Draw Origin =====================================================
            _ax.scatter(
                origin[0],
                origin[1],
                color="k",
                label="origin",
                marker="x",
            )

            # Draw Rays & Anchor Points =======================================
            # Each (2,2) sub-array in `coords` represents line start/end coordinates.
            coords = _ray_t.reshape(-1, 1, 2) + a12.reshape(-1, 2, 1) * _ray_n.reshape(-1, 1, 2)  # (N_active, 2, 2)
            lines = collections.LineCollection(
                coords,
                label=r"$t + \alpha n$",
                color="k",
                alpha=0.5,
                linewidth=1,
            )
            _ax.add_collection(lines)
            _ax.scatter(
                _ray_t[:, 0],
                _ray_t[:, 1],
                label=r"t",
                color="g",
                marker=".",
            )

            # Draw Overlay Grid ===============================================
            if show_grid:
                x_ticks = origin[0] + pitch[0] * np.arange(N[0])
                y_ticks = origin[1] + pitch[1] * np.arange(N[1])
                _ax.set_xticks(x_ticks)
                _ax.set_yticks(y_ticks)
                _ax.grid(
                    linestyle="--",
                    color="gray",
                )

            # Misc Details ====================================================
            pad_width = 0.1 * bbox_dim  # 10% axial pad
            _ax.set_xlabel(dim_label[0])
            _ax.set_ylabel(dim_label[1])
            _ax.set_xlim(origin[0] - pad_width[0], origin[0] + bbox_dim[0] + pad_width[0])
            _ax.set_ylim(origin[1] - pad_width[1], origin[1] + bbox_dim[1] + pad_width[1])
            _ax.legend(loc="lower right", bbox_to_anchor=(1, 1))
            _ax.set_aspect(1)

        fig.tight_layout()
        return fig

    # Internal Helpers --------------------------------------------------------
    @staticmethod
    def _init_dr_info(
        origin: np.ndarray,
        pitch: np.ndarray,
        N: np.ndarray,
        n_spec=npt.ArrayLike,
        t_spec=npt.ArrayLike,
    ):
        """
        Compute all RayXRT parameters.

        Returns
        -------
        info: namedtuple

          * o: (D,) Arrayf        [volume reference point]
          * pitch: (D,) Arrayf    [pixel pitch]
          * N: (D,) Arrayu        [pixel count]
          * r: (N_ray,) Rayf      [zero-copy view of (n_spec, t_spec)]
        """
        ndi = NDArrayInfo.from_obj(n_spec)
        drb = xtk_drjit.load_dr_variant(ndi)
        D = len(N)

        Arrayf = xtk_drjit.Arrayf_Factory(drb, D)
        Arrayu = xtk_drjit.Arrayu_Factory(drb, D)
        Rayf = xtk_drjit.Rayf_Factory(drb, D)

        info = xtk_util.as_namedtuple(  # drjit only accepts `int/float`, not NumPy dtypes
            o=Arrayf(*tuple(map(float, origin))),
            pitch=Arrayf(*tuple(map(float, pitch))),
            N=Arrayu(*tuple(map(int, N))),
            r=Rayf(
                o=Arrayf(*[xtk_drjit.xp2dr(_) for _ in t_spec.T]),
                d=Arrayf(*[xtk_drjit.xp2dr(_) for _ in n_spec.T]),
            ),
        )
        return info


def _get_xrt_apply(drb: xtk_drjit.DrJitBackend, D: int):
    """
    Create DrJIT FW transform.
    """
    Arrayf = xtk_drjit.Arrayf_Factory(drb, D)
    Arrayu = xtk_drjit.Arrayu_Factory(drb, D)
    Rayf = xtk_drjit.Rayf_Factory(drb, D)
    BoundingBoxf = xtk_drjit.BoundingBoxf_Factory(drb, D)

    ray_step = xtk_drjit.get_ray_step(drb, D)

    def xrt_apply(
        o: Arrayf,
        pitch: Arrayf,
        N: Arrayu,
        I: drb.Float,
        r: Rayf,
    ) -> drb.Float:
        # X-Ray Forward-Projection.
        #
        # Parameters
        #   o:     bottom-left coordinate of I[0,...,0]
        #   pitch: cell dimensions \in \bR_{+}
        #   N:     (D,) lattice size
        #   I:     (N1*...*ND,) cell weights \in \bR [C-order]
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

        # Move (intersecting) rays to volume surface
        bbox_vol = BoundingBoxf(Arrayf(0), Arrayf(N))
        active, a1, a2 = bbox_vol.ray_intersect(r)
        a_min = dr.minimum(a1, a2)
        r.o.assign(dr.select(active, r(a_min), r.o))

        r_next = ray_step(r)
        active &= bbox_vol.contains(r_next.o)
        loop = drb.Loop("XRT FW", lambda: (r, r_next, active, P))
        while loop(active):
            # Read (I,) at current cell
            #   Careful to disable out-of-bound queries. (Due to FP-errors.)
            idx_I = dr.floor(0.5 * (r_next.o + r.o))
            mask = active & dr.all(0 <= idx_I) & dr.all(idx_I < N)
            weight = dr.gather(type(I), I, flat_index(idx_I), mask)

            # Compute constants
            length = dr.norm((r_next.o - r.o) * pitch)

            # Update line integral estimates
            P += weight * length

            # Walk to next lattice intersection
            r.assign(r_next)
            r_next.assign(ray_step(r))
            active &= bbox_vol.contains(r_next.o)
        return P

    return xrt_apply


def _get_xrt_adjoint(drb: xtk_drjit.DrJitBackend, D: int):
    """
    Create DrJIT BW transform.
    """
    Arrayf = xtk_drjit.Arrayf_Factory(drb, D)
    Arrayu = xtk_drjit.Arrayu_Factory(drb, D)
    Rayf = xtk_drjit.Rayf_Factory(drb, D)
    BoundingBoxf = xtk_drjit.BoundingBoxf_Factory(drb, D)

    ray_step = xtk_drjit.get_ray_step(drb, D)

    def xrt_adjoint(
        o: Arrayf,
        pitch: Arrayf,
        N: Arrayu,
        P: drb.Float,
        r: Rayf,
    ) -> drb.Float:
        # X-Ray Back-Projection.
        #
        # Parameters
        #   o:     bottom-left coordinate of I[0,...,0]
        #   pitch: cell dimensions \in \bR_{+}
        #   N:     (D,) lattice size
        #   P:     (L,) X-Ray samples \in \bR
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

        I = dr.zeros(drb.Float, dr.prod(N)[0])  # noqa: E741 (Back-Projection samples)

        # Move (intersecting) rays to volume surface
        bbox_vol = BoundingBoxf(Arrayf(0), Arrayf(N))
        active, a1, a2 = bbox_vol.ray_intersect(r)
        a_min = dr.minimum(a1, a2)
        r.o.assign(dr.select(active, r(a_min), r.o))

        r_next = ray_step(r)
        active &= bbox_vol.contains(r_next.o)
        active &= dr.neq(P, 0)
        loop = drb.Loop("XRT BW", lambda: (r, r_next, active))
        while loop(active):
            #   Careful to disable out-of-bound queries. (Due to FP-errors.)
            idx_I = dr.floor(0.5 * (r_next.o + r.o))
            mask = active & dr.all(0 <= idx_I) & dr.all(idx_I < N)

            # Compute constants
            length = dr.norm((r_next.o - r.o) * pitch)

            # Update back-projections
            dr.scatter_reduce(dr.ReduceOp.Add, I, P * length, flat_index(idx_I), mask)

            # Walk to next lattice intersection
            r.assign(r_next)
            r_next.assign(ray_step(r))
            active &= bbox_vol.contains(r_next.o)
        return I

    return xrt_adjoint
