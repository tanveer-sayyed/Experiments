#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:47:21 2019

@author: tanveer


What is the work of a matrix? To multiply vectors!
In goes a vector x out-goes a vector Ax
"""

import numpy as np
import matplotlib.pyplot as plt

def Plot_Arrowed_Vectors(V):
    maxi = 0
    for i in range(len(V)):
        for j in range(len(V[i])):
            plt.quiver(*([0],[0]), V[i][j][0], V[i][j][1], angles='xy', scale_units='xy', scale=1, color=plt.cm.Paired((i+3)/10.))
        x = np.max(np.absolute(V[i]))
        if x > maxi:
            maxi = x
    plt.grid(b=True, which='major', linestyle= ':')
    plt.xticks(np.arange(-maxi, maxi + 1, 1))
    plt.yticks(np.arange(-maxi, maxi + 1, 1))
    plt.show()

# because eigenvector is a line we can choose any suitable point we want on that line!
def Scaling_Eigen_Vectors(eigenVectors):
    for i in range(len(eigenVectors)):
        eigenVectors[:,i] = eigenVectors[:,i] / np.absolute(min(eigenVectors[:,i])) # does NOT handle the zero-div error
    return eigenVectors

def Plot_All(A, eigenvectors):
    plt.figure(figsize=(18,3))
    plt.subplot(1,3,1)
    V = [np.array([[0, 0]]), [eigenVectors]] # padded [0,0] just to get the pink colour
    for i in range(len(V)):
        for j in range(len(V[i])):
            plt.quiver(*([0],[0]), V[i][j][0], V[i][j][1], angles='xy', scale_units='xy', scale=1, color=plt.cm.Paired((i+3)/10.))
    plt.grid(b=True, which='major', linestyle= ':')
    plt.xticks(np.arange(-5, 5 + 1, 1))
    plt.yticks(np.arange(-5, 5 + 1, 1))
    plt.title('Eigenvectors')
    plt.subplot(1,3,2)
    outVector = A.dot(eigenVectors[:,0])
    V = [[outVector], [eigenVectors[:,0]]]
    for i in range(len(V)):
        for j in range(len(V[i])):
            plt.quiver(*([0],[0]), V[i][j][0], V[i][j][1], angles='xy', scale_units='xy', scale=1, color=plt.cm.Paired((i+3)/10.))
    plt.grid(b=True, which='major', linestyle= ':')
    plt.xticks(np.arange(-5, 5 + 1, 1))
    plt.yticks(np.arange(-5, 5 + 1, 1))
    plt.title('matrix-A applied on vector-e1')
    plt.subplot(1,3,3)
    outVector = A.dot(eigenVectors[:,1])
    V = [[outVector], [eigenVectors[:,1]]]
    for i in range(len(V)):
        for j in range(len(V[i])):
            plt.quiver(*([0],[0]), V[i][j][0], V[i][j][1], angles='xy', scale_units='xy', scale=1, color=plt.cm.Paired((i+3)/10.))
    plt.grid(b=True, which='major', linestyle= ':')
    plt.xticks(np.arange(-5, 5 + 1, 1))
    plt.yticks(np.arange(-5, 5 + 1, 1))
    plt.title('matrix-A applied on vector-e2')

A = np.array([[1, 4], [2, 3]])
print('A :\n', A)
lambdas, eigenVectors = np.linalg.eig(A)
eigenVectors = Scaling_Eigen_Vectors(eigenVectors= eigenVectors)
print('lambdas: \n', lambdas)
print()
print('eigenvectors: \n', eigenVectors)

Plot_All(A, eigenVectors)
Λ = np.around(np.linalg.inv(eigenVectors).dot(A).dot(eigenVectors),
              decimals= 8)
print('diagonalized lambdas (Λ): \n', Λ)


n=2
np.around(np.linalg.inv(eigenVectors).dot(M).dot(eigenVectors),
              decimals= 8)

M = np.array([[1,1],[1,0]])
print('M^1: \n', M)
print('M^2: \n', M.dot(M))
print('M^3: \n', M.dot(M).dot(M))
print('M^4: \n', M.dot(M).dot(M).dot(M))
print('M^5: \n', M.dot(M).dot(M).dot(M).dot(M))
lambdas, eigenVectors = np.linalg.eig(M)
eigenVectors = Scaling_Eigen_Vectors(eigenVectors= eigenVectors)
print('lambdas of M^1: \n', lambdas)
print('eigenvectors of M^1: \n', eigenVectors)
lambdas, eigenVectors = np.linalg.eig(M.dot(M).dot(M).dot(M).dot(M))
eigenVectors = Scaling_Eigen_Vectors(eigenVectors= eigenVectors)
print('lambdas of M^5: \n', lambdas)
print('eigenvectors of M^5: \n', eigenVectors)


#Λ = np.around(np.linalg.inv(eigenVectors).dot(M).dot(eigenVectors), decimals= 8)

M = np.array([[1,1, 2],[1,0, 3], [2,4,5]])
lambdas, eigenVectors = np.linalg.eig(M)
eigenVectors = Scaling_Eigen_Vectors(eigenVectors= eigenVectors)
Λ = np.around(np.linalg.inv(eigenVectors).dot(M).dot(eigenVectors), decimals= 8)

print('With factorization: \n', eigenVectors.dot(Λ**20).dot(np.linalg.inv(eigenVectors)))
print('Without factorization: \n', M .dot(M).dot(M).dot(M).dot(M).dot(M).dot(M).dot(M).dot(M).dot(M)
                              .dot(M).dot(M).dot(M).dot(M).dot(M).dot(M).dot(M).dot(M).dot(M).dot(M))



Q = np.array([[0, -1], [1, 0]])
print('Asymmetry check: \n', Q == -np.transpose(Q) )
lambdas, eigenVectors = np.linalg.eig(Q)
eigenVectors = Scaling_Eigen_Vectors(eigenVectors= eigenVectors)
print('eigenvalues: \n', lambdas)
print()
print('eigenvectors: \n', eigenVectors)



Plot_Arrowed_Vectors([eigenVectors])


Ae1 = A.dot(eigenVectors[:,1])
Plot_Arrowed_Vectors([[Ae1], [eigenVectors[:,1]]])
Ae0 = A.dot(eigenVectors[:,0])
Plot_Arrowed_Vectors([[Ae0], [eigenVectors[:,0]]])


F = np.array([[1, 1], [2, 3]])
lambdas, eigenVectors = np.linalg.eig(F)
eigenVectors = Scaling_Eigen_Vectors(eigenVectors= eigenVectors)
Plot_Arrowed_Vectors([np.array([[0, 0]]), eigenVectors])


v = np.array([[2], [1]])
V = [A, [v.flatten()]]
Plot_Arrowed_Vectors( [[v.flatten()]] )
Plot_Arrowed_Vectors([A])
Plot_Arrowed_Vectors( V )


A = np.array([[5, 1], [3, 3]])
lambdas, eigenVectors = np.linalg.eig(A)
eigenVectors = Scaling_Eigen_Vectors(eigenVectors)
Av = A.dot(v)
Plot_Arrowed_Vectors([A])
Plot_Arrowed_Vectors([[eigenVectors]])
Plot_Arrowed_Vectors([A, [v.flatten()], [Av.flatten()]])
Plot_Arrowed_Vectors([[Av.flatten()]])


lambdas, eigenVectors = np.linalg.eig(A)
eigenVectors = Scaling_Eigen_Vectors(eigenVectors= eigenVectors)
Plot_Arrowed_Vectors([eigenVectors])

# We can see that the vector was modified when it was applied to matrix A.
# In other words transformation of iditial vector changed its direction.

# Applying the matrix didn’t change the direction of the vector.
"""
Imagine that the transformation of the initial vector by the matrix gives a new
vector with the exact same direction. This vector is called an eigenvector of A

This means that v is a eigenvector of A if v and Av are in the same direction or
to rephrase it if the vectors Av and v are parallel. The output vector is just a
scaled version of the input vector. This scalling factor is λ which is called the
eigenvalue of A.

Av = λv

We know that anything multiplied by 1 will give the same result.
similarly multiplying by [[1], [1]] would also yield the same result

"""



# Rescaled vectors:
"""As we saw it with numpy, if v
is an eigenvector of A, then any rescaled vector sv is also an eigenvector of A.
The eigenvalue of the rescaled vector is the same."""


import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from sklearn.decomposition import PCA

D = array([[1, 1],[2, 2],[3, 3],[4, 4],[5, 5], # Matrix D has all the
           [6, 6],[7, 7],[8, 8],[9, 9]])       # points on line x = y
# Adding noise:
E = np.zeros(np.shape(D))
E = D + np.random.rand(np.shape(D)[0], np.shape(D)[1])

from sklearn.preprocessing import MinMaxScaler
E = MinMaxScaler().fit_transform(E)
D = MinMaxScaler().fit_transform(D.astype(float))
# create the transform
pca = PCA(1)
# fit transform
pca.fit(E)
# access values and vectors
print('EigenVector(PCA component): ', pca.components_)
print('Explained variance: ', pca.explained_variance_)

M = np.mean(E)
C = E - M
V = np.cov(C)
lambdas, eigenVectors = np.linalg.eig(V)
Λ = np.around(np.linalg.inv(eigenVectors).dot(V).dot(eigenVectors), decimals= 8)

F = pca.transform(E)
E_projected = pca.inverse_transform(F)

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.scatter(D[:,0], D[:,1], color= 'lime', label = 'true data')
plt.scatter(E[:,0], E[:,1], color= 'black', label= 'after noise addition')
plt.xticks(np.arange(0, 1.1, .1))
plt.yticks(np.arange(0, 1.1, .1))
plt.legend()
plt.subplot(1,2,2)
plt.scatter(D[:,0], D[:,1], color= 'lime', label = 'true data')
plt.scatter(E_projected[:, 0], E_projected[:, 1], color = 'red', label= 'PCA projection of noisy data')
pca.components_[0] = pca.components_[0] / min(pca.components_[0])
plt.quiver(*([0,0]), pca.components_[0][0], pca.components_[0][1], angles='xy', scale_units='xy',
           scale=1, color='pink', alpha = 0.6, label= 'Principle Component(Rescaled Eigenvector)')
plt.xticks(np.arange(0, 1.1, .1))
plt.yticks(np.arange(0, 1.1, .1))
plt.legend()
plt.show()

