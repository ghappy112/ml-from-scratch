# Copyright 2021, Gregory Happ, All rights reserved.
# ridge regression from scratch
# recommended methods for most users are .fit, .predict, .score, and .summary
# it is highly recommended to scale your independent variables. Further, it is also recommended to scale your dependent variable if you wish to use .summary

class ridge():
    
    def __init__(self, LAMBDA=1):
        self.LAMBDA = LAMBDA
        return None
    
    # multiply two matrices (used in .fit)
    def __matmul(self, A, B):
        nrows, ncols = len(A), len(B[0])
        product = [[0 for col in range(ncols)] for row in range(nrows)]
        for nrow in range(nrows):
            for ncol in range(ncols):
                new_element = 0
                for a, b in zip(A[nrow], [brow[ncol] for brow in B]):
                    new_element += a * b
                product[nrow][ncol] = new_element
        return product
        
    # transpose (used in .fit)
    def __transpose(self, A):
        return [[A[col][row] for col in range(len(A))] for row in range(len(A[0]))]

    # multiply every element in a list by a value (used in .__inverse)
    def __lmultiply(self, l, factor):
        return [factor * element for element in l]

    # subtract two lists (used in .__inverse)
    def __subtract(self, l1, l2):
        return [el1 - el2 for el1, el2 in zip(l1, l2)]
        
    # inverse (used in .fit)
    def __inverse(self, A):
        from copy import copy
        M = copy(A)
        Identity = [[1.0 if col == row else 0.0 for col in range(len(A[0]))] for row in range(len(A))] # create an identity matrix
        I = copy(Identity)
        while M != Identity: # we keep trying to solve it until the matrix is the identity matrix
            for i in range(len(M)): # loop through each row
                if M[i][i] != 1: # if the row doesn't have a 1 where it is supposed to, we need to do some operations:
                    counter = i
                    while M[i][i] != 1:
                        if counter >= len(M):
                            counter = 0
                        else:
                            if M[counter][i] != 0:
                                k = (M[i][i] - 1) / M[counter][i]
                                rowmul, Irowmul = self.__lmultiply(M[counter], k), self.__lmultiply(I[counter], k)
                                M[i], I[i] = self.__subtract(M[i], rowmul), self.__subtract(I[i], Irowmul)
                            else:
                                counter += 1
                for j in range(len(M)): # once we have a 1 in the correct index for that row, we make all other rows have 0's for that column
                    if i != j and M[j][i] != 0:
                        k = M[j][i]
                        rowmul, Irowmul = self.__lmultiply(M[i], k), self.__lmultiply(I[i], k)
                        M[j], I[j] = self.__subtract(M[j], rowmul), self.__subtract(I[j], Irowmul)    
        return I
        
    # add two matrices (used in .fit)
    def __matadd(self, A, B):
        nrows, ncols = len(A), len(B[0])
        sum = [[0 for col in range(ncols)] for row in range(nrows)]
        for nrow in range(nrows):
            for ncol in range(ncols):
                sum[nrow][ncol] = A[nrow][ncol] + B[nrow][ncol]
        return sum
        
    # multiply a matrix by a value (used in .coef_error)
    def __multiply(self, A, k):
        nrows, ncols = len(A), len(A[0])
        product = [[0 for col in range(ncols)] for row in range(nrows)]
        for row in range(nrows):
            for col in range(ncols):
                product[row][col] = A[row][col] * k
        return product
    
    # train model
    def fit(self, X, y):
        t = self.__transpose(X)
        XT_X = self.__matmul(t, X)
        lambda_I = [[self.LAMBDA if col == row else 0.0 for col in range(len(XT_X[0]))] for row in range(len(XT_X))] # create an identity matrix
        self.coef_ = [List[0] for List in self.__matmul(self.__matmul(self.__inverse(self.__matadd(XT_X, lambda_I)), t), [[Y] for Y in y])]
        self.intercept_ = sum(y) / len(y)
        for i in range(len(self.coef_)):
            column = [row[i] for row in X]
            self.intercept_ -= self.coef_[i] * (sum(column) / len(column))
            
    # predict
    def predict(self, X):
        result = []
        for List in X:
            prediction = self.intercept_
            for i in range(len(self.coef_)):
                prediction += self.coef_[i] * List[i]
            result.append(prediction)
        return result
    
    # get RSS (Residual Sum of Squares) (used in .RSE)
    def __getRSS(self, X, y):
        ymean = sum(y) / len(y)
        predictions = self.predict(X)
        RSS = 0.0
        for y_i, ypred_i in zip(y, predictions):
            RSS += (y_i - ypred_i)**2
        return RSS
    
    # get RSS (Residual Sum of Squares) and TSS (Total Sum of Squares) (used in .score, .F, and .F_Statistic)
    def __getRSSandTSS(self, X, y):
        ymean = sum(y) / len(y)
        predictions = self.predict(X)
        RSS = 0.0
        TSS = 0.0
        for y_i, ypred_i in zip(y, predictions):
            RSS += (y_i - ypred_i)**2
            TSS += (y_i - ymean)**2
        return RSS, TSS
    
    # R^2
    def score(self, X, y):
        RSS, TSS = self.__getRSSandTSS(X, y)
        return 1.0 - (RSS / TSS)
        
        # adjusted R^2
    def adjusted_r2(self, X, y):
        n = len(y)
        return 1.0 - (((1.0 - self.score(X, y))*(n - 1.0))/(n - len(self.coef_) - 1.0))
        
    # F-Value
    def F(self, X, y):
        RSS, TSS = self.__getRSSandTSS(X, y)
        return ((RSS - TSS) / -len(self.coef_)) / ((TSS) / len(y))

    # Residual Standard Error
    def RSE(self, X, y):
        return self.__getRSS(X, y) / (len(y) - len(self.coef_))

    # get Model Sum of Squares (used in .F_Statistic)
    def __getSSM(self, X, y):
        ymean = sum(y) / len(y)
        predictions = self.predict(X)
        SSM = 0.0
        for ypred_i in predictions:
            SSM += (ypred_i - ymean)**2
        return SSM

    # get critical value for F-Test
    def F_Statistic(self, X, y):
        RSS, TSS = self.__getRSSandTSS(X, y)
        SSM = self.__getSSM(X, y)
        k = len(self.coef_)
        DFM, DFE = len(y) - k, k - 1
        return (SSM/DFM)/(RSS/DFE)
    
    # get Variance (used in .__getStdev)
    def __getVariance(self, arr):
        mean = sum(arr)/len(arr)
        variances = []
        for i in arr:
            variances.append((i-mean)**2)
        return sum(variances)/len(variances)
    
    # get standard deviation (used in .F_Test)
    def __getStdev(self, arr):
        return self.__getVariance(arr)**0.5
    
    # get p-value (used in .F_Test)
    def __getP(self, x, mean, stdev):
        import math
        P = 0.5 * (1 + math.erf((x-mean)/(stdev*(2**0.5))))
        return P
    
    # F Test
    def F_Test(self, X, y):
        F = self.F(X, y)
        F_Statistic = self.F_Statistic(X, y)
        return 1.0 - self.__getP(F, F_Statistic, self.__getStdev(y))
    
    # get the standard errors of the coeficients
    def coef_error(self, X, y):
        predictions = self.predict(X)
        errors = []
        for y_i, ypred_i in zip(y, predictions):
            errors.append(y_i - ypred_i)
        Xs = [[1] + row for row in X]
        t = self.__transpose(Xs)
        arr =  self.__multiply(self.__inverse(self.__matmul(t, Xs)), self.__getVariance(errors)**2)
        result = []
        for l in arr:
            result.append(sum(l))
        return result[0], result[1:]
    
    # get the t-values for the coeficients
    def t(self, X, y):
        intercept_se, coef_se = self.coef_error(X, y)
        ts = [self.intercept_ / intercept_se]
        for coef, se in zip(self.coef_, coef_se):
            ts.append(coef / se)
        return ts[0], ts[1:]
    
    # get the p-values for the coeficients
    def p(self, X, y):
        intercept_t, coef_t = self.t(X, y)
        stdev = self.__getStdev(y)
        ps = [1.0 - self.__getP(intercept_t, 0, stdev)]
        for t in coef_t:
            ps.append(1.0 - self.__getP(t, 0, stdev))
        return ps[0], ps[1:]

    # get "stars" for statistical significance tables (used in .summary)
    def __stars(self, pval):
        if pval < 0.001:
            stars = "***"
        elif pval < 0.01:
            stars = "**"
        elif pval < 0.05:
            stars = "*"
        else:
            stars = ""
        return stars

    
    # display a summary of the results in an academic journal style format
    def summary(self, X, y, colnames=None, title=None, trim=91):
        intercept = self.intercept_
        coefs = self.coef_
        se = self.coef_error(X, y)
        p = self.p(X, y)
        N = len(y)
        r2 = self.score(X, y)
        adjr2 = self.adjusted_r2(X, y)
        RSE = self.RSE(X, y)
        F = self.F(X, y)
        F_Test = self.F_Test(X, y)
        df = N - len(coefs)
        STRING = ""
        STRING += "-"*trim + '\n'
        if title == None:
            title = " "*55 + "Ridge Regression (L2 Regularization)"
        STRING += title + '\n'
        STRING += "-"*trim + '\n'
        if colnames == None:
            colnames = []
            for x in range(1, len(coefs)+1):
                colnames.append("X" + str(x))
        for var in range(len(coefs)):
            stars = self.__stars(p[1][var])
            STRING += colnames[var] + " "*(54-len(colnames[var])) + " " + str(coefs[var])+stars + '\n'
            STRING += " "*55 + "("+str(se[1][var])+")" + '\n'
        stars = self.__stars(p[0])
        STRING += "Constant" + " "*47 + str(intercept)+stars + '\n'
        STRING += " "*55 + "("+str(se[0])+")" + '\n'
        STRING += "N" + " "*54 + str(N) + '\n'
        STRING += "R^2" + " "*52 + str(r2) + '\n'
        STRING += "Adjusted R^2" + " "*43 + str(adjr2) + '\n'
        STRING += "Residual Standard Error" + " "*32 + str(RSE) + "(df = " + str(df) + ")" + '\n'
        stars = self.__stars(F_Test)
        STRING += "F Statistic" + " "*44 + str(F)+stars + "(df = " + str(len(coefs)) + "; " + str(df) + ")" + '\n'
        STRING += "-"*trim + '\n'
        STRING += " "*55 + "***: p < 0.001" + '\n'
        STRING += " "*55 + "**: p < 0.01" + '\n'
        STRING += " "*55 + "*: p < 0.05" + '\n'
        return STRING
