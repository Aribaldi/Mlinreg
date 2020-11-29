import math

#   Matrix is given in a form of list with list-elements, corresponding to
#   matrix row, e.g. A = [[1, 2], [3, 4]] results in
#   1 2
#   3 4
#   Vector is represented by list.


def check_squareness(A):
    """
    Makes sure that a matrix is square
        :param A: The matrix to be checked.
    """
    if len(A) != len(A[0]):
        raise ArithmeticError("Matrix must be square to inverse.")

def determinant(A, total=0):
    indices = list(range(len(A)))

    if len(A) == 2 and len(A[0]) == 2:
        val = A[0][0] * A[1][1] - A[1][0] * A[0][1]
        return val

    for fc in indices:
        As = copy_matrix(A)
        As = As[1:]
        height = len(As)
        builder = 0

        for i in range(height):
            As[i] = As[i][0:fc] + As[i][fc+1:]

        sign = (-1) ** (fc % 2)
        sub_det = determinant(As)
        total += A[0][fc] * sign * sub_det

    return total

def check_non_singular(A):
    det = determinant(A)
    if det != 0:
        return det
    else:
        raise ArithmeticError("Singular Matrix!")

def zeros_matrix(rows, cols):
    """
    Creates a matrix filled with zeros.
        :param rows: the number of rows the matrix should have
        :param cols: the number of columns the matrix should have
        :returns: list of lists that form the matrix.
    """
    M = []
    while len(M) < rows:
        M.append([])
        while len(M[-1]) < cols:
            M[-1].append(0.0)

    return M

def identity_matrix(n):
    """
    Creates and returns an identity matrix.
        :param n: the square size of the matrix
        :returns: a square identity matrix
    """
    I = zeros_matrix(n, n)
    for i in range(n):
        I[i][i] = 1.0

    return I

def copy_matrix(M):
    """
    Creates and returns a copy of a matrix.
        :param M: The matrix to be copied
        :return: The copy of the given matrix
    """
    rows = len(M)
    cols = len(M[0])

    MC = zeros_matrix(rows, cols)

    for i in range(rows):
        for j in range(rows):
            MC[i][j] = M[i][j]

    return MC

def print_matrix(M):
    """
        :param M: The matrix to be printed
    """
    for row in M:
        print([round(x,3)+0 for x in row])

def transpose(M):
    """
    Creates and returns a transpose of a matrix.
        :param M: The matrix to be transposed
        :return: the transpose of the given matrix
    """
    rows = len(M)
    cols = len(M[0])

    MT = zeros_matrix(cols, rows)

    for i in range(rows):
        for j in range(cols):
            MT[j][i] = M[i][j]

    return MT

def matrix_multiply(A,B):
    """
    Returns the product of the matrix A * B
        :param A: The first matrix - ORDER MATTERS!
        :param B: The second matrix
        :return: The product of the two matrices
    """
    rowsA = len(A)
    colsA = len(A[0])

    rowsB = len(B)
    colsB = len(B[0])

    if colsA != rowsB:
        raise ArithmeticError('Number of A columns must equal number of B rows.')

    C = zeros_matrix(rowsA, colsB)

    for i in range(rowsA):
        for j in range(colsB):
            total = 0
            for ii in range(colsA):
                total += A[i][ii] * B[ii][j]
            C[i][j] = total

    return C

def check_matrix_equality(A,B, tol=None):
    """
    Checks the equality of two matrices.
        :param A: The first matrix
        :param B: The second matrix
        :param tol: The decimal place tolerance of the check
        :return: The boolean result of the equality check
    """
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        return False

    for i in range(len(A)):
        for j in range(len(A[0])):
            if tol == None:
                if abs(A[i][j] - B[i][j]) > 1e-10:
                    return False
            else:
                if round(A[i][j],tol) != round(B[i][j],tol):
                    return False

    return True

def invert_matrix(A, tol=None):
    """
    Returns the inverse of the passed in matrix.
        :param A: The matrix to be inversed
        :return: The inverse of the matrix A
    """
    check_squareness(A)
    check_non_singular(A)

    n = len(A)
    AM = copy_matrix(A)
    I = identity_matrix(n)
    IM = copy_matrix(I)

    indices = list(range(n))
    for fd in range(n):
        fdScaler = 1.0 / AM[fd][fd]
        for j in range(n):
            AM[fd][j] *= fdScaler
            IM[fd][j] *= fdScaler
        for i in indices[0:fd] + indices[fd+1:]:
            crScaler = AM[i][fd]
            for j in range(n):
                AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
                IM[i][j] = IM[i][j] - crScaler * IM[fd][j]

    if check_matrix_equality(I,matrix_multiply(A,IM),tol):
        return IM
    else:
        raise ArithmeticError("Matrix inverse out of tolerance.")

def mult_vect_by_scalar(v, a):
    """
    Multiply vector by scalar.
    :param v: vector
    :param a: scalar
    """
    return [a * i for i in v]

def scalar_mult(v, u):
    """
    Scalar multiplication of two vectors.
    :param v: vector
    :param u: vector
    """
    return [v[i] * u[i] for i in range(len(v))]

def l1_norm(v):
    """
    L1-norm of a vector.
    :param v: vector
    """
    res = 0
    for e in v:
        res += abs(e)
    return res

def l2_norm(v):
    """
    L2-norm of a vector.
    :param v: vector
    """
    res = 0
    for e in v:
        res += e * e
    return math.sqrt(res)

def matrix_sum(A, B):
    ans = zeros_matrix(len(A), len(A[0]))

    for i in range(len(A)):
        for j in range(len(A[0])):
            ans[i][j] = A[i][j] + B[i][j]

    return ans

def matrix_by_scalar(A, scalar):
    ans = zeros_matrix(len(A), len(A[0]))

    for i in range(len(A)):
        for j in range(len(A[0])):
            ans[i][j] = A[i][j] * scalar

    return ans

def matrix_add_scalar(A, scalar):
    ans = zeros_matrix(len(A), len(A[0]))

    for i in range(len(A)):
        for j in range(len(A[0])):
            ans[i][j] = A[i][j] + scalar

    return ans

def add_one_for_bias(vector):
    b = vector[:]
    b[0:0] = [1.0]
    return b

def create_zeros_same_shape(A):
    return zeros_matrix(len(A), len(A[0]))

def elementwise_power(A, power):

    for i in range(len(A)):
        for j in range(len(A[0])):
            A[i][j] = A[i][j] ** power

    return A

def elementwise_product(A, B):
    ans = zeros_matrix(len(A), len(A[0]))

    for i in range(len(A)):
        for j in range(len(A[0])):
            ans[i][j] = A[i][j] * B[i][j]

    return ans


if __name__ == "__main__":
    print(mult_vect_by_scalar([1, -8, 10], 2))
    print(scalar_mult([5, -2, 4], [1, -8, 10]))
    print(l1_norm([1, 2, 4]))
    print(l2_norm([1, 2, 1]))

    a = [[43, 53], [62, 6]]
    b = [[47, 3], [6, 7]]
    print_matrix(matrix_multiply(a, b))
