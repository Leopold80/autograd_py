"""
定义矩阵运算符
"""

import numpy as np

from .core import Expression, Var


class Add(Expression):
    def __init__(self, parents):
        super(Add, self).__init__(parents=parents)

    def forward_impl(self) -> None:
        xs = (p.value for p in self.parents)
        self.value = sum(xs)

    def grad_impl(self, parent: Expression) -> np.matrix:
        return np.mat(np.eye(self.dim_asvector()))


Expression.__add__ = lambda self, other: Add(parents=[self, other])


class Sub(Expression):
    def __init__(self, parents):
        super(Sub, self).__init__(parents=parents)

    def forward_impl(self) -> None:
        self.value = self.parents[0].value - self.parents[1].value

    def grad_impl(self, parent: Expression) -> np.matrix:
        jaco = np.mat(np.eye(self.dim_asvector()))
        return jaco if parent is self.parents[0] else -jaco


Expression.__sub__ = lambda self, other: Sub(parents=[self, other])


class Mul(Expression):
    """矩阵元素对应相乘"""

    def __init__(self, parents):
        super(Mul, self).__init__(parents=parents)

    def forward_impl(self) -> None:
        self.value = np.multiply(*(p.value for p in self.parents))

    def grad_impl(self, parent: Expression) -> np.matrix:
        if parent is self.parents[0]:
            return np.mat(np.diag(self.parents[1].value.A1))
        else:
            return np.mat(np.diag(self.parents[0].value.A1))


Expression.__mul__ = Expression.__rmul__ = (
    lambda self, other:
    Mul(parents=[self, other if isinstance(other, Expression) else Var(np.mat(np.ones(self.value.shape) * other))])
)


class MatMul(Expression):
    def __init__(self, parents):
        assert len(parents) == 2
        super(MatMul, self).__init__(parents=parents)

    def forward_impl(self) -> None:
        self.value = self.parents[0].value * self.parents[1].value

    def grad_impl(self, parent: Expression) -> np.matrix:
        jaco = np.mat(np.zeros((self.dim_asvector(), parent.dim_asvector())))
        if parent is self.parents[0]:
            return MatMul.fill_diag(jaco, self.parents[1].value.T)
        else:
            jaco = MatMul.fill_diag(jaco, self.parents[0].value)
            row_sort = np.arange(self.dim_asvector()).reshape(self.shape()[::-1]).T.ravel()
            col_sort = np.arange(self.dim_asvector()).reshape(parent.shape()[::-1]).T.ravel()
            # return jaco[row_sort, :][:, col_sort]
            return jaco[row_sort, col_sort]

    @staticmethod
    def fill_diag(filled, filler):
        """将 filler 矩阵填充在 to_be_filled 的广义对角线上"""
        n = int(filled.shape[0] / filler.shape[0])
        r, c = filler.shape
        for i in range(n):
            filled[i * r:(i + 1) * r, i * c:(i + 1) * c] = filler
        return filled


Expression.__matmul__ = lambda self, other: MatMul(parents=[self, other])


class Div(Expression):
    def __init__(self, parents):
        super(Div, self).__init__(parents=parents)

    def forward_impl(self) -> None:
        self.value = self.parents[0].value / self.parents[1].value

    def grad_impl(self, parent: Expression) -> np.matrix:
        if parent is self.parents[0]:
            return np.mat(np.diag(1. / self.parents[1].value.A1))
        else:
            return np.mat(np.diag(-self.parents[0].value.A1 / self.parents[1].value.A1 ** 2))


Expression.__truediv__ = Expression.__rtruediv__ = (
    lambda self, other:
    Div(parents=[self, other if isinstance(other, Expression) else Var(np.mat(np.ones(self.value.shape) * other))])
)
