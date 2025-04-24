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
