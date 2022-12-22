'''OLD VERSION: See folder Python_code in root directory of git directory'''

# August 6, 2020
# Import Numerical Python
import numpy as np

# Define class and helper functions
class zerothOrderComposite:
    
    # Initialize dimension p, n, input vector x, and matrix A
    def __init__(self,p,n,x):
        self.p = p
        self.n = n
        self.x = x
        np.random.seed(0)
        self.A = np.random.randn(self.p,self.n)
        # b formed from A and x
        self.b = self.A @ self.x 
    
    # Compute (Ax-b)^TA
    def helperNormalEquation(self,v):
        return (self.A @ v - self.b).T @ self.A
    
    # Compute || (Ax-b)^TA ||^2
    def helperNormalEquationNorm(self,v):
        return np.linalg.norm(self.helperNormalEquation(v),2) ** 2
    
    # Compute vv^T
    def outerProduct(self,v):
        length = len(v)
        output = np.zeros((length,length))
        for i in range(length):
            output[:,i] = v[i] * v
        return output
    
    # Compute (Ax-b)^TA[uu^T - I]
    def helperZerothApproximation(self,v):
        u = np.random.randn(self.n)
        B = self.outerProduct(u)
        output = self.helperNormalEquation(v) * (B - np.eye(self.n))
        return output 
    
    # Compute || (Ax-b)^TA[uu^T - I] ||^2
    def helperZerothApproxNorm(self,v):
        return np.linalg.norm(self.helperZerothApproximation(v),2) ** 2
    
    # Compute E_u || (Ax-b)^TA[uu^T - I] ||^2 via sample average
    def helperExpectation(self,v,N):
        sumTotal = 0
        for i in range(N):
            sumTotal += self.helperZerothApproxNorm(v)
        return (1 / N) * sumTotal
    
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
init = [1,2,4,8]
        
# initialize class 
zerothOrder = zerothOrderComposite(p,n,init)

# initialize a randomly chosen vector
inVector = [1,2,3,3]

# To approximate E_u || (A[inVector]-b)^TA[uu^T - I] ||^2 using sample average of numIteration size
numIteration = 10**3

# Compute E_u || (A[inVector]-b)^TA[uu^T - I] ||^2 and 4(n+4.5)|| (A[inVector]-b)^TA ||^2
expectedNormSquared = zerothOrder.helperExpectation(inVector,numIteration)
normSquared = 4* (zerothOrder.n + 4.5) * zerothOrder.helperNormalEquationNorm(inVector)

# Print Statements
print(f'\nThe value of E_u || (Ax-b)^TA[uu^T-I] ||^2: {expectedNormSquared}\n')
print(f'The value of 4(n+4.5) || (Ax-b)^TA||^2: {normSquared}')

if expectedNormSquared <= normSquared:
    print('\nE_u || (Ax-b)^TA[uu^T-I] ||^2 <= 4(n+4.5) || (Ax-b)^TA||^2 ')
else:
    print('E_u || (Ax-b)^TA[uu^T-I] ||^2 > 4(n+4.5) || (Ax-b)^TA||^2 ')

