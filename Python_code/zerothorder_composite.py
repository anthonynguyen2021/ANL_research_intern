import numpy as np
import matplotlib.pyplot as plt


class zeroth_order_composite:

    def __init__(self, m, n, x):
        self.m = m
        self.n = n
        self.x = x
        np.random.seed(0)
        self.A = np.random.randn(self.m, self.n)
        self.b = self.A @ self.x

    def linear_fun(self,init):
        '''
        argument: 
            x: list[float]
        return:
            A * x - b
        '''
        return self.A @ init - self.b

    def ridge_regression_inner(self, init, lamda):
        '''Compute F(x) in Ridge regression where f = h o F'''
        output = np.zeros((self.m + self.n))

        for i in range(self.m):
            output[i] = np.dot(self.A[i], init) - self.b[i]

        for i in range(self.n):
            output[self.m+i] = np.sqrt(lamda) * init[i]

        return output

    def least_squares(self, init):
        '''f(x) := || A * x - b||_2 ^ 2 where A, b are defined in initialization of class zeroth_order_composite'''
        return np.linalg.norm(self.A @ init - self.b, 2) ** 2

    def ridge_regression(self, init, lamda):
        '''f(x) := || A * x - b||_2 ^ 2 + lambda ||x||_2 ^ 2'''
        return np.linalg.norm(self.A @ init - self.b, 2) ** 2 + lamda * np.linalg.norm(init, 2) ** 2

    def grad_least_squares(self, init):
        '''computes f'(x) = 2 * (A * x - b) ^ T * A'''
        return 2 * (self.A @ init - self.b).T @ self.A

    def grad_ridge_regression(self, init, lamda):
        '''returns f'(x) = 2 * (A * x - b) ^ T * A + 2 * lamda * x for Ridge Regression'''
        return 2 * (self.A @ init - self.b).T @ self.A + 2 * lamda * init

    def rosenbrock(self, init):
        '''Rosenbrock function evaluation given input init'''
        # n > 1
        output = 0

        for i in range(self.n - 1):
            output += 100 * (init[i+1] - init[i] ** 2) ** 2 + (1 - init[i]) ** 2

        return output

    def grad_rosenbrock(self, init):
        '''Compute f'(x) where f is the rosenbrock function'''
        grad_inner = np.zeros((2 * (self.n - 1), self.n))
        grad_inner[:, 0] = -20 * init[0] * self.elementary_basis(0, 2 * (self.n - 1)) - self.elementary_basis(self.n - 1, 2 * (self.n - 1))
        grad_inner[:, self.n - 1] = 10 * self.elementary_basis(self.n - 2, 2 * (self.n - 1))

        for i in range(1, self.n - 1):
            grad_inner[:, i] = -20 * init[i-1] * self.elementary_basis(i - 1, 2 * (self.n - 1)) + 10 * self.elementary_basis(i - 2, 2 * (self.n - 1)) - self.elementary_basis(self.n - 2 + i, 2 * (self.n - 1))

        outer_fun = np.zeros((2 * (self.n - 1)))

        for i in range(self.n - 1):
            outer_fun[i] = 10 * (init[i+1] - init[i] ** 2)
            outer_fun[self.n-1+i] = 1 - init[i]

        return outer_fun @ grad_inner

    def forward_difference(self, init, smooth, fun):
        '''Compute [f(x + \mu * u) - f(x)] / mu for least squares or Rosenbrock objective function'''
        temp = np.random.normal(0, 1, self.n)
        if fun == 'ls':
            forw_diff = (self.least_squares(init + smooth * temp) - self.least_squares(init)) / smooth
        else:
            forw_diff = (self.rosenbrock(init + smooth * temp) - self.rosenbrock(init)) / smooth

        return forw_diff, temp

    def forward_difference_ridge(self, init, smooth, lamda):
        '''Compute [f(x + \mu * u) - f(x)] / mu for ridge regression'''
        temp = np.random.normal(0, 1, self.n)
        forw_diff = (self.ridge_regression(init + smooth * temp, lamda) - self.ridge_regression(init, lamda) ) / smooth

        return forw_diff, temp

    def rosenbrock_inner(self, init):
        '''Recall Rosenbrock can be written as f(x) = h(F(x)); here we compute F(x)'''
        result = np.zeros((2 * (self.n - 1)))

        for i in range(self.n - 1):
            result[i] = 10 * (init[i+1] - init[i] ** 2)
            result[self.n-1+i] = 1 - init[i]

        return result

    def forward_difference_inner(self, init, smooth, fun):
        '''Compute (F(x + mu * u) - F(x)) / mu for two test cases'''
        temp = np.random.normal(0, 1, self.n)

        if fun == 'ls':
            finite_diff_vec = np.zeros((self.m))
            finite_diff_vec = (self.linear_fun(init + smooth * temp) - self.linear_fun(init)) / smooth
        else:
            finite_diff_vec = np.zeros((2 * (self.n - 1)))
            finite_diff_vec = (self.rosenbrock_inner(init + smooth * temp) - self.rosenbrock_inner(init)) / smooth

        return finite_diff_vec, temp

    def forward_difference_inner_ridge(self, init, smooth, lamda):
        '''Compute (F(x + mu * u) - F(x)) / mu for ridge regression'''
        temp = np.random.normal(0, 1, self.n)
        finite_diff_vec = np.zeros((self.m))
        finite_diff_vec = (self.ridge_regression_inner(init + smooth * temp, lamda) - self.ridge_regression_inner(init, lamda)) / smooth

        return finite_diff_vec, temp

    def forward_diff_scheme(self, init, smooth, step, fun):
        '''Computes x <- x - h * (f(x + mu * u) - f(x)) * u / mu'''
        forw_diff, temp = self.forward_difference(init, smooth, fun)
        return init - step * forw_diff * temp

    def forward_diff_scheme_ridge(self, init, smooth, step, lamda):
        forw_diff, temp = self.forward_difference_ridge(init, smooth, lamda)
        return init - step * forw_diff * temp

    def forward_diff_scheme_ridge2(self, init, smooth, step, lamda):
        finite_diff_vec, temp = self.forward_difference_inner_ridge(init, smooth, lamda)
        forw_diff = 2 * np.dot(self.ridge_regression_inner(init, lamda), finite_diff_vec)
        return init - step * forw_diff * temp

    def forward_diff_scheme_ridge3(self, init, smooth, step, lamda):
        finite_diff_vec, temp = self.forward_difference_inner_ridge(init, smooth, lamda)
        forw_diff = 2 * np.dot(self.ridge_regression_inner(init + smooth * temp,lamda), finite_diff_vec)
        return init - step * forw_diff * temp

    def forward_diff_scheme2(self, init, smooth, step, fun):
        '''Computes x <- x - h * 2 * F(x) * [F(x + mu * u) - F(x)] * u / mu'''
        finite_diff_vec, temp = self.forward_difference_inner(init, smooth, fun)

        if fun == 'ls':
            forw_diff = 2 * np.dot(self.linear_fun(init), finite_diff_vec)
        else:
            forw_diff = 2 * np.dot(self.rosenbrock_inner(init), finite_diff_vec)

        return init - step * forw_diff * temp

    def forward_diff_scheme3(self, init, smooth, step, fun):
        '''Computes x <- x - h * 2 * F(x + mu * u) * [F(x + mu * u) - F(x)] * u / mu'''
        finite_diff_vec, temp = self.forward_difference_inner(init, smooth, fun)

        if fun == 'ls':
            forw_diff = 2 * np.dot(self.linear_fun(init + smooth * temp), finite_diff_vec)
        else:
            forw_diff = 2 * np.dot(self.rosenbrock_inner(init + smooth * temp), finite_diff_vec)

        return init - step * forw_diff * temp

    def outer_product_basis(self, i, j):
        '''Return e_i * e_j ^ T'''
        elementary_i = np.zeros((self.n))
        elementary_i[i] = 1
        elementary_j = np.zeros((self.n))
        elementary_j[j] = 1

        return np.outer(elementary_i, elementary_j)

    def elementary_basis(self,i,size):
        '''Return vector e_i, a vector in R^size'''
        elem_vector = np.zeros((size))
        elem_vector[i] = 1
        return elem_vector
#%%
        
# Simulation #1: Change flag = True to run
   
# Runs Method RG_mu page 546 on the least squares problem 
        
# pointwise_opt = True means ||f(x_k) - f(x^*)|| optimality
# gradient_opt = True means || f'(x_k) || optimality
 
def sim1(pointwise_opt, gradient_opt):
          
    print('\nRunning Simulation #1')
    
    # Initialize 
    iterations = []     
    error = []
    error_grad = []
    
    # Pick Parameters
    x0 = np.array([2, 2, 1, 0]) # Can change the initial condition here; used to form b = Ax0
    m = 3
    n = len(x0)
    t = zeroth_order_composite(m, n, x0)
    L_gradf = 2 * np.linalg.norm(t.A.T @ t.A, 2)
    eps = 10 ** -2
    mu = (5 / (3 * (t.n + 4))) * np.sqrt(eps / (2 * L_gradf))  # 10 **-2
    #print((t.n / eps) * L_gradf * R**2)
    #R = 10**2 where ||x0 - x*|| <= R
    iteration = 10 ** 5 #(t.n / eps) * L_gradf * R**2 <- theoretical upperbound on # of iterations
    step = (4 * (t.n + 4) * L_gradf) **-1
    print(f'\nThe value x0 = {x0} \nValue of m = {m} \nValue of n = {n} \nRun the scheme for {iteration} iterations \nChoosing epsilon = {eps} \nWith smoothing parameter mu = {mu}')
    print(f'The lipschitz constant of f\' is = {L_gradf} \nStep size h = {step}')
    
    init = np.array([20, 40, 10 ** 4, 10 ** 4]) # Initial point in scheme
    print(f'With initial point: {init}')
    
    # Run Scheme
    for i in range(int(np.ceil(iteration))):
        init = t.forward_diff_scheme(init, mu, step, 'ls')
        iterations.append(i)
        error.append(np.linalg.norm(t.A @ init - t.b, 2))
        error_grad.append(np.linalg.norm(t.grad_least_squares(init)))
    
    # Graph and Print     
    print('\nObjective function: ||Ax-b||^2\n')
    if pointwise_opt:
        plt.loglog(iterations, error, label='Scheme 1')

    if gradient_opt:
        plt.loglog(iterations, error_grad, label='Scheme 1')

    plt.legend()
    plt.xlabel('Iterations, k')
    plt.ylabel('Error')
    plt.title('Least Squares')

    if pointwise_opt:
        print(f'\nThe last 10 iterations have the following error ||f(x_k) - f(x^*)||_2:\n\n {error[-10:]}')

    if gradient_opt:
        print(f'\nThe last 10 iterations have the following error ||f\'(x_k)||_2:\n\n {error_grad[-10:]}')
    
#%%   
    
# Simulation 2: Change flag = True to Run
 
# convergence depends on initial condition and the size of the L_f'  
        
# pointwise_opt = True means ||f(x_k) - f(x^*)|| optimality
# gradient_opt = True means || f'(x_k) || optimality

def sim2(pointwise_opt, gradient_opt):
        
    print('\nRunning Simulation #2')
    
    # Initialize
    iterations = []     
    error = []
    error_grad = []
    
    # Pick Parameters
    x0 = np.array([1, 1, 1, 12]) # Can change the initial condition here
    #x0 = np.array([10**2,1,1,10**4]) # seems to work with [1,1,1,1] but using different init yields convergence.
    m = 3
    n = len(x0)
    t = zeroth_order_composite(m, n, x0)
    iteration = 10 ** 5
    mu = 10 ** -2
    print(f'\nThe value x0 = {x0} \nValue of m = {m} \nValue of n = {n}') 
    print(f'\nRun the scheme for {iteration} iterations \nWith smoothing parameter mu = {mu}')
    
    # form hessian and take its matrix norm at x0
    hessian_f = np.zeros((t.n, t.n))

    for i in range(t.n - 1):
        hessian_f += 2 * (1 + 4 * x0[i] ** 2 - 2 * 10 * (x0[i+1]-x0[i])) * t.outer_product_basis(i, i)
        hessian_f += 2 * 100 * t.outer_product_basis(i + 1, i + 1)
        hessian_f += 2 * -20 * x0[i] * (t.outer_product_basis(i + 1, i) + t.outer_product_basis(i, i + 1))

    L_gradf = np.linalg.norm(hessian_f, 2)
    print(f'The 2-matrix norm of L_f at x0 is: {L_gradf}')
    
    step = (4 * (t.n + 4) * L_gradf) ** -1
    #init = np.array([1,1,1,1]) # Noisy error 
    init = x0  #np.array([2,2,2,2])
    print(f'\nWith initial point: {init} \nand stepsize h = {step}')
    
    # Run Scheme
    for i in range(iteration):
        init = t.forward_diff_scheme(init, mu, step, 'not_ls')
        iterations.append(i)
        error.append(np.abs(t.rosenbrock(init) - t.rosenbrock(np.ones(t.n))))
        error_grad.append(np.linalg.norm(t.grad_rosenbrock(init), 2))
            
    # Graph and print   
    print('\nObjective function: Rosenbrock function\n')
    if pointwise_opt:
        plt.loglog(iterations,error,label='Scheme 1')

    if gradient_opt:
        plt.loglog(iterations,error_grad, label='Scheme 1')

    plt.xlabel('Iterations, k')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Rosenbrock')

    if pointwise_opt:
        print(f'\nThe last 10 iterations have the following error ||f(x_k) - f(x^*)||_2:\n\n {error[-10:]}')

    if gradient_opt:
        print(f'\nThe last 10 iterations have the following error ||f\'(x_k)||_2:\n\n {error_grad[-10:]}')

#%%    
    
    
# Simulation #3: Change flag = True to run
# Performs better than Simulation #1
# Methods approx f'(x) ~ 2F(x)F'_mu(x)
        
# pointwise_opt = True means ||f(x_k) - f(x^*)|| optimality
# gradient_opt = True means || f'(x_k) || optimality
    
def sim3(pointwise_opt, gradient_opt):
         
    print('\nSimulation #3')
    
    # Initialize
    iterations = []     
    error = []
    error_grad = []
    
    # Pick Parameters
    x0 = np.array([2, 2, 1, 0]) # Can change the initial condition here; used to form b = Ax0
    m = 3
    n = len(x0)
    t = zeroth_order_composite(m, n, x0)
    L_gradf = 2 * np.linalg.norm(t.A.T @ t.A, 2)
    eps = 10 ** -2
    mu = (5 / (3 * (t.n + 4))) * np.sqrt(eps / (2 * L_gradf)) # 10 **-2
    #print((t.n / eps) * L_gradf * R**2)
    #R = 10**2 where ||x0 - x*|| <= R
    iteration = 10 ** 5 #(t.n / eps) * L_gradf * R**2 <- theoretical upperbound on # of iterations
    step = (4 * (t.n + 4) * L_gradf) ** -1
    print(f'\nThe value x0 = {x0} \nValue of m = {m} \nValue of n = {n} \nRun the scheme for {iteration} iterations \nChoosing epsilon = {eps} \nWith smoothing parameter mu = {mu}')
    print(f'The lipschitz constant of f\' is = {L_gradf} \nStep size h = {step}')
    
    #init = np.array([1,1,1,1]) # Initial point in scheme
    init = np.array([20, 40, 10 ** 4, 10 ** 4])
    print(f'With initial point: {init}')
    
    # Run Scheme
    for i in range(int(np.ceil(iteration))):
        init = t.forward_diff_scheme2(init, mu, step, 'ls')
        iterations.append(i)
        error.append(np.linalg.norm(t.A @ init - t.b, 2))
        error_grad.append(np.linalg.norm(t.grad_least_squares(init)))

    # Graph and print     
    print('\nObjective function: ||Ax-b||^2\n')
    if pointwise_opt:
        plt.loglog(iterations,error,label = 'Scheme 3')

    if gradient_opt:
        plt.loglog(iterations,error_grad, label = 'Scheme 3')

    plt.legend()
    plt.xlabel('Iterations, k')
    plt.ylabel('Error')
    plt.title('Least Squares')

    #plt.show()
    if pointwise_opt:
        print(f'\nThe last 10 iterations have the following error ||f(x_k) - f(x^*)||_2:\n\n {error[-10:]}')

    if gradient_opt:
        print(f'\nThe last 10 iterations have the following error ||f\'(x_k)||_2:\n\n {error_grad[-10:]}')

#%% 
    
    
# Simulation #4: Change flag = True to run
# Performs worse than Simulation #2. Error is O(1).
# Methods approx f'(x) ~ 2F(x)F'_mu(x)
        
# pointwise_opt = True means ||f(x_k) - f(x^*)|| optimality
# gradient_opt = True means || f'(x_k) || optimality
    
def sim4(pointwise_opt, gradient_opt):
          
    print('\nRunning Simulation #4')
    
    # Initialize
    iterations = []     
    error = []
    error_grad = []
    
    # Pick parameters
    x0 = np.array([1, 1, 1, 12]) # Can change the initial condition here
    #x0 = np.array([10**2,1,1,10**4]) # seems to work with [1,1,1,1] but using different init yields convergence.
    m = 3
    n = len(x0)
    t = zeroth_order_composite(m, n, x0)
    iteration = 10 ** 5
    mu = 10 ** -2
    print(f'\nThe value x0 = {x0} \nValue of m = {m} \nValue of n = {n}') 
    print(f'\nRun the scheme for {iteration} iterations \nWith smoothing parameter mu = {mu}')
    
    
    # form hessian and take its matrix norm at x0
    hessian_f = np.zeros((t.n, t.n))

    for i in range(t.n - 1):
        hessian_f += 2 * (1 + 4 * x0[i] ** 2 - 2 * 10 * (x0[i+1] - x0[i])) * t.outer_product_basis(i, i)
        hessian_f += 2 * 100 * t.outer_product_basis(i + 1, i + 1)
        hessian_f += 2 * -20 * x0[i] * (t.outer_product_basis(i + 1, i) + t.outer_product_basis(i, i + 1))

    L_gradf = np.linalg.norm(hessian_f, 2)
    print(f'\nThe 2-matrix norm of f\'\' at x0 = {x0} is: {L_gradf}')
    
    step = (4 * (t.n + 4) * L_gradf) ** -1
    #init = np.array([1,1,1,1]) # Noisy error 
    init = x0#np.array([2,2,2,2])
    print(f'\nWith initial point: {init} \nand stepsize h = {step}')
    
    # Run Scheme
    for i in range(int(np.ceil(iteration))):
        init = t.forward_diff_scheme2(init, mu, step, 'not_ls')
        iterations.append(i)
        error.append(np.abs(t.rosenbrock(init) - t.rosenbrock(np.ones(t.n))))
        error_grad.append(np.linalg.norm(t.grad_rosenbrock(init), 2))
            
    # Graph and Print
    print('\nObjective function: Rosenbrock function\n')

    if pointwise_opt:
        plt.loglog(iterations, error, label='Scheme 3')

    if gradient_opt:
        plt.loglog(iterations, error_grad, label='Scheme 3')

    plt.xlabel('Iterations, k')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Rosenbrock') 
    #plt.show()

    if pointwise_opt:
        print(f'\nThe last 10 iterations have the following error ||f(x_k) - f(x^*)||_2:\n\n {error[-10:]}')

    if gradient_opt:
        print(f'\nThe last 10 iterations have the following error ||f\'(x_k)||_2:\n\n {error_grad[-10:]}')

#%%    
    
    
# Simulation #5: Change flag = True to run
# Performs better than Simulation #1
# Methods approx f'(x) ~ 2F_mu(x)F'_mu(x)
        
# pointwise_opt = True means ||f(x_k) - f(x^*)|| optimality
# gradient_opt = True means || f'(x_k) || optimality
   
def sim5(pointwise_opt, gradient_opt):
     
    print('\nRunning Simulation #5')  
    
    # Initialize
    iterations = []     
    error = []
    error_grad = []
    
    # Pick parameters
    x0 = np.array([2, 2, 1, 0]) # Can change the initial condition here; used to form b = Ax0
    m = 3
    n = len(x0)
    t = zeroth_order_composite(m, n, x0)
    L_gradf = 2 * np.linalg.norm(t.A.T @ t.A, 2)
    eps = 10 ** -2
    mu = (5 / (3 * (t.n + 4))) * np.sqrt(eps / (2 * L_gradf)) # 10 **-2
    #print((t.n / eps) * L_gradf * R**2)
    #R = 10**2 where ||x0 - x*|| <= R
    iteration = 10 ** 5 #(t.n / eps) * L_gradf * R**2 <- theoretical upperbound on # of iterations
    step = (4 * (t.n + 4) * L_gradf) ** -1
    print(f'\nThe value x0 = {x0} \nValue of m = {m} \nValue of n = {n} \nRun the scheme for {iteration} iterations \nChoosing epsilon = {eps} \nWith smoothing parameter mu = {mu}')
    print(f'The lipschitz constant of f\' is = {L_gradf} \nStep size h = {step}')
    #init = np.array([1,1,1,1]) # Initial point in scheme
    init = np.array([20, 40, 10 ** 4, 10 ** 4])
    print(f'With initial point: {init} ')
    
    # Run scheme
    for i in range(int(np.ceil(iteration))):
        init = t.forward_diff_scheme3(init, mu, step, 'ls')
        iterations.append(i)
        error.append(np.linalg.norm(t.A @ init - t.b, 2))
        error_grad.append(np.linalg.norm(t.grad_least_squares(init)))
            
    # Plot and print
    print('\nObjective function: ||Ax-b||^2\n')
    if pointwise_opt:
        plt.loglog(iterations, error, label='Scheme 2')

    if gradient_opt:
        plt.loglog(iterations, error_grad, label='Scheme 2')

    plt.legend()
    plt.xlabel('Iterations, k')
    plt.ylabel('Error')
    plt.title('Least Squares')

    #plt.show()
    if pointwise_opt:
        print(f'\nThe last 10 iterations have the following error ||f(x_k) - f(x^*)||_2:\n\n {error[-10:]}')

    if gradient_opt:
        print(f'\nThe last 10 iterations have the following error ||f\'(x_k)||_2:\n\n {error_grad[-10:]}')
    
    
#%% 
    
    
# Simulation #6: Change flag = True to run
# Performs worse than Simulation #2,4. Error is O(1).
# Methods approx f'(x) ~ 2F_mu(x)F'_mu(x)

# pointwise_opt = True means ||f(x_k) - f(x^*)|| optimality
# gradient_opt = True means || f'(x_k) || optimality

def sim6(pointwise_opt, gradient_opt):      
    print('\nRunning Simulation #6')
        
    # initialize
    iterations = []     
    error = []
    error_grad = []
    
    # Pick parameters
    x0 = np.array([1, 1, 1, 12]) # Can change the initial condition here
    #x0 = np.array([10**2,1,1,10**4]) # seems to work with [1,1,1,1] but using different init yields convergence.
    m = 3
    n = len(x0)
    t = zeroth_order_composite(m, n, x0)
    iteration = 10 ** 5
    mu = 10 ** -2
    print('\nThe value x0 = ', x0,'\nValue of m = ', m, '\nValue of n = ', n, '\nRun the scheme for ', iteration, 'iterations', '\nWith smoothing parameter mu = ', mu)
    
    # form hessian and take its matrix norm at x0
    hessian_f = np.zeros((t.n, t.n))
    for i in range(t.n - 1):
        hessian_f += 2 * (1 + 4 * x0[i] ** 2 - 2 * 10 * (x0[i+1] - x0[i])) * t.outer_product_basis(i, i)
        hessian_f += 2 * 100 * t.outer_product_basis(i + 1, i + 1)
        hessian_f += 2 * -20 * x0[i] * (t.outer_product_basis(i + 1, i) + t.outer_product_basis(i, i + 1))

    L_gradf = np.linalg.norm(hessian_f, 2)
    print(f'\nThe 2-matrix norm of f\'\' at x0 = {x0} is: {L_gradf}')
    
    
    step = (4 * (t.n + 4) * L_gradf ) ** -1
    #init = np.array([1,1,1,1]) # Noisy error 
    #init = np.array([2,2,2,2])
    init = x0
    print(f'\nWith initial points: {init} \nand stepsize h = {step}')
    
    # Run scheme
    for i in range(int(np.ceil(iteration))):
        init = t.forward_diff_scheme3(init, mu, step, 'not_ls')
        iterations.append(i)
        error.append(np.abs(t.rosenbrock(init) - t.rosenbrock(np.ones(t.n))))
        error_grad.append(np.linalg.norm(t.grad_rosenbrock(init), 2))
        
    # Plot and print
    print('\nObjective function: Rosenbrock function\n')
    if pointwise_opt:
        plt.loglog(iterations, error, label='Scheme 2')

    if gradient_opt:
        plt.loglog(iterations, error_grad, label='Scheme 2')

    plt.xlabel('Iterations, k')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Rosenbrock')

    #plt.show()
    if pointwise_opt:
        print(f'\nThe last 10 iterations have the following error ||f(x_k) - f(x^*)||_2:\n\n {error[-10:]}')

    if gradient_opt:
        print(f'\nThe last 10 iterations have the following error ||f\'(x_k)||_2:\n\n {error_grad[-10:]}')               

#%%
        
# Simulation #7: Change flag = True to run
   
# Runs Method RG_mu page 546 on the ridge regression
        
# pointwise_opt = True means ||f(x_k) - f(x^*)|| optimality
# gradient_opt = True means || f'(x_k) || optimality
 
def sim7(pointwise_opt, gradient_opt, lamda):
          
    print('\nRunning Simulation #7')
    
    # Initialize 
    iterations = []     
    error = []
    error_grad = []
    
    # Pick Parameters
    x0 = np.array([2, 2, 1, 0]) # Can change the initial condition here; used to form b = Ax0
    m = 3
    n = len(x0)
    t = zeroth_order_composite(m, n, x0)
    L_gradf = 2 * np.linalg.norm(t.A.T @ t.A + lamda * np.eye(t.n), 2)
    eps = 10 ** -2
    mu = (5 / (3 * (t.n + 4))) * np.sqrt(eps / (2 * L_gradf))  # 10 **-2
    #print((t.n / eps) * L_gradf * R**2)
    #R = 10**2 where ||x0 - x*|| <= R
    iteration = 10 ** 5 #(t.n / eps) * L_gradf * R**2 <- theoretical upperbound on # of iterations
    step = (4 * (t.n + 4) * L_gradf) ** -1
    print(f'\nThe value x0 = {x0} \nRegularization coefficient lambda = {lamda} \nValue of m = {m} \nValue of n = {n} \nRun the scheme for {iteration} iterations \nChoosing epsilon = {eps} \nWith smoothing parameter mu = {mu}')
    print(f'The lipschitz constant of f\' is = {L_gradf} \nStep size h = {step}')
    
    init = np.array([20, 40, 10 ** 4, 10 ** 4]) # Initial point in scheme
    print(f'With initial point: {init}')

    xhat = np.linalg.inv(t.A.T @ t.A + lamda * np.eye(t.n)) @ (t.A.T @ t.b)
    print(f'The argmin of the ridgeregression problem is {xhat}')
    
    # Run Scheme
    for i in range(int(np.ceil(iteration))):
        init = t.forward_diff_scheme_ridge(init, mu, step, lamda)
        #init = t.forward_diff_scheme(init,mu,step,'ls')
        iterations.append(i)
        #error.append(np.linalg.norm(t.A @ init - t.b,2))
        error.append(np.linalg.norm(t.ridge_regression(init, lamda) - t.ridge_regression(xhat, lamda)))
        #error_grad.append(np.linalg.norm(t.grad_least_squares(init)))
        error_grad.append(np.linalg.norm(t.grad_ridge_regression(init, lamda)))
    
    # Graph and Print     
    print('\nObjective function: ||Ax-b||^2 + lambda ||x||^2 \n')
    if pointwise_opt:
        plt.loglog(iterations, error, label='Scheme 1')

    if gradient_opt:
        plt.loglog(iterations, error_grad, label='Scheme 1')

    plt.legend()
    plt.xlabel('Iterations, k')
    plt.ylabel('Error')
    plt.title(f'Ridge Regression with lambda = {lamda}')
    if pointwise_opt:
        print(f'\nThe last 10 iterations have the following error ||f(x_k) - f(x^*)||_2:\n\n {error[-10:]}')

    if gradient_opt:
        print(f'\nThe last 10 iterations have the following error ||f\'(x_k)||_2:\n\n {error_grad[-10:]}')   

#%%    
    
    
# Simulation #8: Change flag = True to run
# Methods approx f'(x) ~ 2F(x)F'_mu(x) for Ridge Regression
        
# pointwise_opt = True means ||f(x_k) - f(x^*)|| optimality
# gradient_opt = True means || f'(x_k) || optimality
    
def sim8(pointwise_opt, gradient_opt, lamda):
         
    print('\nSimulation #8')
    
    # Initialize
    iterations = []     
    error = []
    error_grad = []
    
    # Pick Parameters
    x0 = np.array([2, 2, 1, 0]) # Can change the initial condition here; used to form b = Ax0
    m = 3
    n = len(x0)
    t = zeroth_order_composite(m, n, x0)
    L_gradf = 2 * np.linalg.norm(t.A.T @ t.A + lamda * np.eye(t.n), 2)
    #L_gradf = 2 * np.linalg.norm(t.A.T @ t.A,2)
    eps = 10 ** -2
    mu = (5 / (3 * (t.n + 4))) * np.sqrt(eps / (2 * L_gradf))  # 10 ** -2
    #print((t.n / eps) * L_gradf * R**2)
    #R = 10**2 where ||x0 - x*|| <= R
    iteration = 10 ** 5 #(t.n / eps) * L_gradf * R**2 <- theoretical upperbound on # of iterations
    step = (4 * (t.n + 4) * L_gradf) ** -1
    print(f'\nThe value x0 = {x0} \nValue of m = {m} \nValue of n = {n} \nRun the scheme for {iteration} iterations \nChoosing epsilon = {eps} \nWith smoothing parameter mu = {mu}')
    print(f'The lipschitz constant of f\' is = {L_gradf} \nStep size h = {step}')
    
    #init = np.array([1,1,1,1]) # Initial point in scheme
    init = np.array([20, 40, 10 ** 4, 10 ** 4])
    print(f'With initial point: {init}')
    
    xhat = np.linalg.inv(t.A.T @ t.A + lamda * np.eye(t.n)) @ (t.A.T @ t.b)
    print(f'The argmin of the ridgeregression problem is {xhat}')
    
    # Run Scheme
    for i in range(int(np.ceil(iteration))):
        init = t.forward_diff_scheme_ridge2(init, mu, step, lamda)
        iterations.append(i)
        error.append(np.linalg.norm(t.ridge_regression(init, lamda) - t.ridge_regression(xhat, lamda)))
        error_grad.append(np.linalg.norm(t.grad_ridge_regression(init, lamda)))

    # Graph and print     
    print('\nObjective function: ||Ax-b||^2 + lambda ||x||^2 \n')
    if pointwise_opt:
        plt.loglog(iterations,error,label = 'Scheme 3')

    if gradient_opt:
        plt.loglog(iterations,error_grad, label = 'Scheme 3')

    plt.legend()
    plt.xlabel('Iterations, k')
    plt.ylabel('Error')
    plt.title(f'Ridge Regression with lambda = {lamda}')

    #plt.show()
    if pointwise_opt:
        print(f'\nThe last 10 iterations have the following error ||f(x_k) - f(x^*)||_2:\n\n {error[-10:]}')

    if gradient_opt:
        print(f'\nThe last 10 iterations have the following error ||f\'(x_k)||_2:\n\n {error_grad[-10:]}')

#%%    
    
    
# Simulation #9: Change flag = True to run
# Methods approx f'(x) ~ 2F_mu(x)F'_mu(x) for Ridge Regression
        
# pointwise_opt = True means ||f(x_k) - f(x^*)|| optimality
# gradient_opt = True means || f'(x_k) || optimality
    
def sim9(pointwise_opt, gradient_opt, lamda):
         
    print('\nSimulation #9')
    
    # Initialize
    iterations = []     
    error = []
    error_grad = []
    
    # Pick Parameters
    x0 = np.array([2, 2, 1, 0]) # Can change the initial condition here; used to form b = Ax0
    m = 3
    n = len(x0)
    t = zeroth_order_composite(m, n, x0)
    L_gradf = 2 * np.linalg.norm(t.A.T @ t.A + lamda * np.eye(t.n), 2)
    #L_gradf = 2 * np.linalg.norm(t.A.T @ t.A,2)
    eps = 10 ** -2
    mu = (5 / (3 * (t.n + 4))) * np.sqrt(eps / (2 * L_gradf))  # 10 ** -2
    #print((t.n / eps) * L_gradf * R**2)
    #R = 10**2 where ||x0 - x*|| <= R
    iteration = 10 ** 5 #(t.n / eps) * L_gradf * R**2 <- theoretical upperbound on # of iterations
    step = (4 * (t.n + 4) * L_gradf) ** -1
    print(f'\nThe value x0 = {x0} \nValue of m = {m} \nValue of n = {n} \nRun the scheme for {iteration} iterations \nChoosing epsilon = {eps} \nWith smoothing parameter mu = {mu}')
    print(f'The lipschitz constant of f\' is = {L_gradf} \nStep size h = {step}')
    
    #init = np.array([1,1,1,1]) # Initial point in scheme
    init = np.array([20, 40, 10 ** 4, 10 ** 4])
    print(f'With initial point: {init}')
    
    xhat = np.linalg.inv(t.A.T @ t.A + lamda * np.eye(t.n)) @ (t.A.T @ t.b)
    print(f'The argmin of the ridgeregression problem is {xhat}')
    
    # Run Scheme
    for i in range(int(np.ceil(iteration))):
        init = t.forward_diff_scheme_ridge3(init, mu, step, lamda)
        iterations.append(i)
        error.append(np.linalg.norm(t.ridge_regression(init,lamda) - t.ridge_regression(xhat, lamda)))
        error_grad.append(np.linalg.norm(t.grad_ridge_regression(init, lamda)))

    # Graph and print     
    print('\nObjective function: ||Ax-b||^2 + lambda ||x||^2 \n')

    if pointwise_opt:
        plt.loglog(iterations,error,label = 'Scheme 2')

    if gradient_opt:
        plt.loglog(iterations,error_grad, label = 'Scheme 2')

    plt.legend()
    plt.xlabel('Iterations, k')
    plt.ylabel('Error')
    plt.title(f'Ridge Regression with lambda = {lamda}')
 
    #plt.show()
    if pointwise_opt:
        print(f'\nThe last 10 iterations have the following error ||f(x_k) - f(x^*)||_2:\n\n {error[-10:]}')

    if gradient_opt:
        print(f'\nThe last 10 iterations have the following error ||f\'(x_k)||_2:\n\n {error_grad[-10:]}')

#%%

# Run Least Squares and plot them over each other
found = False
if found:

    print('\nRunning the Least Squares Problem on Scheme 1, 2, 3\n')
    sim1(True, False)
    sim3(True, False)    
    sim5(True, False)  
    plt.show()
    
#%%
    
# Run the Rosenbrock tests and plot them over each other
found = False
if found:

    print('\nRunning the Rosenbrock Problem on Scheme 1, 2, 3\n')
    sim2(False, True)
    sim4(False, True)    
    sim6(False, True)  
    plt.show()
    
#%%
    
# Run Ridge Regression tests and plot them over each other
    
found = True
if found:
    lamda = 10 ** 1
    print(f'\nRunning the Ridge Regression Problem on Scheme 1,2,3 using lambda = {lamda}\n')
    #sim7(True, False, lamda)
    #sim8(True, False, lamda)
    sim9(True, False, lamda)
    plt.show()
    