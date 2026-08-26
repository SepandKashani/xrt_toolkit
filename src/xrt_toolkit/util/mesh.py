import collections.abc as cabc
from dataclasses import dataclass

from .misc import broadcast_seq


@dataclass
class UniformSpec:
    r"""
    Multi-dimensional uniform mesh specifier.

    Defines points :math:`\bbx_{m} \in \bR^{D}` where each point lies on the
    regular lattice

    .. math::

       \bbx_{\bbm} = \bbx_{0} + \Delta_{\bbx} \odot \bbm,
       \qquad
       [\bbm]_{d} \in \{0,\ldots,M_{d}-1\}

    """

    start: tuple[float]
    step: tuple[float]
    num: tuple[int]

    def __init__(self, start, step, num):
        r"""
        Parameters
        ----------
        start: tuple[float]
            \bbx_{0} \in \bR^{D}
        step: tuple[float]
            \Delta_{\bbx} \in \bR_{+}^{D}
        num: tuple[int]
            (M1,...,MD) lattice size

        Scalars are broadcast to all dimensions.
        """
        start = broadcast_seq(start, None, float)

        step = broadcast_seq(step, None, float)
        assert all(s > 0 for s in step)

        num = broadcast_seq(num, None, int)
        assert all(n > 0 for n in num)

        D = max(map(len, [start, step, num]))

        self.start = broadcast_seq(start, D)
        self.step = broadcast_seq(step, D)
        self.num = broadcast_seq(num, D)

    @property
    def ndim(self) -> int:
        D = len(self.start)
        return D

    def __iter__(self) -> cabc.Iterator:
        for d in range(self.ndim):
            yield (self.start[d], self.step[d], self.num[d])

    @classmethod
    def centered(cls, step, num) -> "UniformSpec":
        r"""
        Initialize a UniformSpec centered at the origin.

        Parameters
        ----------
        step: tuple[float]
            \Delta_{\bbx} \in \bR_{+}^{D}
        num: tuple[int]
            (M1,...,MD) lattice size

        Scalars are broadcast to all dimensions.

        Returns
        -------
        u_spec: UniformSpec
        """
        offset_spec = cls(start=0, step=step, num=num)

        ll = tuple(
            -step * (num - 1) / 2
            for (step, num) in zip(offset_spec.step, offset_spec.num)
        )
        u_spec = cls(
            start=ll,
            step=offset_spec.step,
            num=offset_spec.num,
        )
        return u_spec


@dataclass
class DetectorSpec:
    r"""
    Physical dimensions of a 1D/2D pixelized detector.

    `DetectorSpec` does not encode the detector's position in space.
    """

    size: tuple[float]
    num_cell: tuple[int]

    def __init__(self, size, num_cell):
        r"""
        Parameters
        ----------
        size: tuple[float]
            Detector span (unitless) \in \bR_{+}^{D}

            A 1D size denotes detector width.
            A 2D size denotes detector (width, height).
        num_cell: tuple[int]
            (M1,...,MD) cell count per dimension

        Scalars are broadcast to all dimensions.
        """
        size = broadcast_seq(size, None, float)
        assert all(s > 0 for s in size)

        num_cell = broadcast_seq(num_cell, None, int)
        assert all(n > 0 for n in num_cell)

        D = max(map(len, [size, num_cell]))
        assert D in (1, 2)

        self.size = broadcast_seq(size, D)
        self.num_cell = broadcast_seq(num_cell, D)

    @property
    def cell_size(self) -> tuple[float]:
        c_size = tuple(self.size[d] / self.num_cell[d] for d in range(self.ndim))
        return c_size

    @property
    def ndim(self) -> int:
        D = len(self.size)
        return D

    def __iter__(self) -> cabc.Iterator:
        for d in range(self.ndim):
            yield (self.size[d], self.num_cell[d])
