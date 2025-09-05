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
