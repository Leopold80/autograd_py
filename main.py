import numpy as np
from loguru import logger

import autojaco as jaco


def main():
    # a = jaco.Var(np.mat([1, 2, 3, 4]).reshape(2, 2))
    # b = jaco.Var(np.mat([1, 2, 3, 4]).reshape(2, 2))
    # c = a * b
    # c.forward()
    # print(c.value)
    # j = b.grad(c)
    # print(j)

    A = jaco.Var(np.mat([1, 2, 3, 4]).reshape(2, 2))
    B = jaco.Var(np.mat([1, 2, 3, 4]).reshape(2, 2))
    C = A * 2
    C.forward()
    logger.info("C: \n{}".format(C.value))
    Ja = A.grad(C)
    Jb = B.grad(C)
    logger.info("grad of A: \n{}".format(Ja))
    logger.info("grad of B: \n{}".format(Jb))


main()
