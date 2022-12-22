# August 6, 2020: Revised December 2022
import numpy as np


# Define class and helper functions
class zerothOrderComposite:
    
    def __init__(self, p, n, x):
        '''Initialize dimension p, n, input vector x, and matrix A'''
        self.p = p
        self.n = n
        self.x = x
        np.random.seed(0)
        self.A = np.random.randn(self.p, self.n)
        # b = A * x
        self.b = self.A @ self.x 

    def helperNormalEquation(self, v):
        '''Compute (A * x - b) ^ T * A'''
        return (self.A @ v - self.b).T @ self.A

    def helperNormalEquationNorm(self, v):
        '''Compute || (A * x - b) ^ T * A ||_2 ^ 2'''
        return np.linalg.norm(self.helperNormalEquation(v), 2) ** 2

    def outerProduct(self, v):
        '''Compute v * v ^ T'''
        length = len(v)
        output = np.zeros((length, length))

        for i in range(length):
            output[:, i] = v[i] * v

        return output

    def helperZerothApproximation(self, v):
        '''Compute (A * x - b) ^ T * A[u * u ^ T - I]'''
        u = np.random.randn(self.n)
        B = self.outerProduct(u)
        output = self.helperNormalEquation(v) * (B - np.eye(self.n))

        return output 

    def helperZerothApproxNorm(self, v):
        '''Compute || (A * x - b) ^ T A[u * u ^ T - I] || ^ 2'''
        return np.linalg.norm(self.helperZerothApproximation(v), 2) ** 2

    def helperExpectation(self, v, N):
        '''Compute E_u || (A * x - b) ^ T A[u * u ^ T - I] ||_2 ^ 2 via sample average'''
        sumTotal = 0

        for i in range(N):
            sumTotal += self.helperZerothApproxNorm(v)

        return (1 / N) * sumTotal


if __name__ == '__main__':

    ''' 

    For simulation purposes, you can tweak the following five (5) parameters:

    1. p
    2 = n
    3 = init
    4 = inVector
    5 = numIteration    
        
    '''

    # Initialize dimension of A and x = init   
    p = 3
    n = 4
    init = [1, 2, 4, 8]
            
    # initialize class 
    zerothOrder = zerothOrderComposite(p, n, init)

    # initialize a randomly chosen vector
    inVector = [1, 2, 3, 3]

    # To approximate E_u || (A[inVector]-b)^TA[uu^T - I] ||^2 using sample average of numIteration size
    numIteration = 10 ** 3

    # Compute E_u || (A[inVector] - b) ^ T A[u * u ^ T - I] || ^ 2 and 4 * (n + 4.5)|| (A[inVector] - b) ^ T * A || ^ 2
    expectedNormSquared = zerothOrder.helperExpectation(inVector, numIteration)
    normSquared = 4 * (zerothOrder.n + 4.5) * zerothOrder.helperNormalEquationNorm(inVector)

    # Print Statements
    print(f'\nThe value of E_u || (A * x - b) ^ T A[u * u ^ T - I] ||_2 ^ 2: {expectedNormSquared}\n')
    print(f'The value of 4 * (n + 4.5) || (A * x - b) ^ T * A ||_2 ^ 2: {normSquared}')

    if expectedNormSquared <= normSquared:
        print('\nE_u || (A * x - b) ^ T * A[u * u ^ T - I] ||_2 ^ 2 <= 4 * (n + 4.5) || (A * x - b) ^ T * A||_2 ^ 2 ')
    else:
        print('E_u || (A * x - b) ^ T * A[u * u ^ T - I] ||_2 ^ 2 > 4 * (n + 4.5) || (A * x - b) ^ T * A||_2 ^ 2 ')
