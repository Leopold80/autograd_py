from __future__ import annotations  # 为了一个类里面的成员能够做标注自身类的类型标注

from typing import Optional, List, Tuple

import numpy as np


class Expression:
    """所有表达式（算符和变量）的基类"""

    def __init__(self, parents: List[Expression]):
        self.value: Optional[None, np.matrix] = None  # 节点值
        self.jacobi: Optional[None, np.matrix] = None  # 雅可比矩阵（梯度）
        self.parents: List[Expression] = parents  # 父节点
        self.childs: List[Expression] = []  # 子节点
        # 将本节点添加到父节点的子节点
        for p in parents:
            p.childs.append(self)

    def dim_asvector(self) -> int:
        """矩阵作为向量的维度"""
        return self.value.size

    def shape(self) -> Tuple:
        """本节点矩阵的维度"""
        return self.value.shape

    def forward(self) -> None:
        """前向传播"""
        for p in self.parents:
            p.forward()
        self.forward_impl()

    def grad(self, numerator) -> np.matrix:
        """反向传播 dy/dx: x.grad(y)"""
        if self is not numerator:
            jacobies = (c.grad(numerator) * c.grad_impl(self) for c in self.childs)
            self.jacobi = sum(jacobies)
        else:
            self.jacobi = np.mat(np.eye(self.dim_asvector()))
        return self.jacobi

    def forward_impl(self) -> None:
        raise NotImplemented

    def grad_impl(self, parent: Expression) -> np.matrix:
        raise NotImplemented

    def clear_jacobi(self) -> None:
        """清空本节点的雅可比"""
        self.jacobi = None

    def reset_value(self, recursive=True) -> None:
        """重置本节点的值，若递归清空则清空本节点下游的所有节点值"""
        self.value = None
        if recursive:
            for c in self.childs:
                c.reset_value()

    def __add__(self, other) -> Expression:
        ...

    def __sub__(self, other) -> Expression:
        ...

    def __mul__(self, other) -> Expression:
        ...

    def __rmul__(self, other) -> Expression:
        ...

    def __truediv__(self, other) -> Expression:
        ...

    def __rtruediv__(self, other) -> Expression:
        ...

    def __matmul__(self, other) -> Expression:
        ...

    def __pow__(self, power, modulo=None) -> Expression:
        ...


class Var(Expression):
    """变量表达式"""

    def __init__(self, val: np.matrix):
        super(Var, self).__init__(parents=[])  # 表达式不可能有父节点
        self.value = val

    def set_value(self, val: np.matrix):
        self.reset_value()
        self.value = val

    def forward_impl(self) -> None: ...

    def grad_impl(self, parent) -> np.matrix:
        return np.mat(np.eye(self.dim_asvector()))
