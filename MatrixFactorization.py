'''
Autoencoder availabel in https://github.com/Ananya-Bhattacharjee/ImputeDistances
'''
def matrix_fact_imputation(R):
    def mat_factorization(R, P, Q, K, steps=10000, alpha=0.002, beta=0.02):
        Q = Q.T
        for step in range(steps):
            print (step)
            for i in range(len(R)):
                for j in range(len(R[i])):
                    if R[i][j] >= 0 and i>=j:
                        eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                        for k in range(K):
                            P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                            Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
            eR = numpy.dot(P,Q)
            e = 0
            for i in range(len(R)):
                for j in range(len(R[i])):
                    if R[i][j] > 0:
                        e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                        for k in range(K):
                            e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
            if e < 0.000001:
                break
        return P, Q.T
    
    
    
    import numpy
    import time
    start_time = time.time()
    i=0
    j=0
    missing=0
    numg = R.shape[0]
    
    for i in range(0,R.shape[0]):
        for j in range(0,R.shape[0]):
            if numpy.isnan(R[i][j]) == True:
                R[i][j] = -1
    

    N = len(R)
    M = len(R[0])
    K = numg
    

    P = numpy.random.rand(N,K)
    Q = numpy.random.rand(M,K)

    nP, nQ = mat_factorization(R, P, Q, K)

    nR = numpy.dot(nP, nQ.T)

    print (nR)
    Result = numpy.zeros(shape=(numg, numg))

    
    for i in range(len(R)):
        for j in range(len(R[0])):
            if(R[i][j]==-1):
                Result[i][j]=nR[i][j]
            else:
                Result[i][j]=R[i][j]
    
    tempo = time.time() - start_time
    return Result,tempo
