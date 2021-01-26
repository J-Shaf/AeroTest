from scipy import integrate,optimize
from math import pi,cos,sin,tan,acos,asin,atan
import matplotlib.pyplot as plt
import math
import numpy as np

p = 0.2
m = 0.01
t = 0.22

V = 23
alpha = 6

def z(theta):
    theta_0 = 0
    theta_1 = 2*pi
    theta_p = acos(1-(2*p))

    if theta >= theta_0 and theta <= theta_p:
        A = m/(p*p)
        B = p*(1-cos(theta))
        C = (0.5*(1-cos(theta)))**2
        return A*(B-C)

    elif theta > theta_p and theta <= theta_1:
        A = m/((1-p)**2)
        B = 1-(2*p)
        C = p*(1-cos(theta))
        D = (0.5*(1-cos(theta)))**2
        return A*(B+C-D)

    else:
        print(theta)
        raise BaseException

def z_prime(theta):
    theta_0 = 0
    theta_1 = 2*pi
    theta_p = acos(1-(2*p))

    if theta >= theta_0 and theta <= theta_p:
        A = m/(p*p)
        B = 2*p
        C = (1-cos(theta))
        return A*(B-C)

    elif theta > theta_p and theta <= theta_1:
        A = m/((1-p)**2)
        B = 2*p
        C = (1-cos(theta))
        return A*(B-C)

    else:
        raise BaseException

"""

def An(n):
    def func(theta):
        return z_prime(theta)*cos(n*theta)

    return (2/pi) * integrate.quad(func,0,pi)[0]

iters=4
As = [An(i) for i in range(0,iters)]

def gamma(theta,):


    n_0 = As[0] * ((1 + cos(theta)) / sin(theta))

    return 2 * V * (
    n_0 +  np.sum(
        [As[n]*sin(n*theta) 
        for n 
        in range(0,iters)
        ]
        ))

"""
def p_test(theta):
    return (alpha - z_prime(theta))**2 + 2*(alpha - z_prime(theta))


def solve():
    
    def to_solve(x):

        return integrate.quad(p_test,x,pi)[0] - integrate.quad(p_test,0,x)[0]


    answer = optimize.root_scalar(to_solve,bracket=(0,pi)).root
    answer_x_over_c = 0.5*(1-cos(a))
    print(answer_x_over_c)

solve()

"""
thetas = np.arange(0.01,pi,0.01)
gammas = [gamma(theta) for theta in thetas]

plt.plot(thetas,gammas)
plt.show()
"""