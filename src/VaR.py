import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import pandas as pd
from scipy.stats import norm
import math


class ValueAtRisk:
    def __init__(self, interval, matrix, weights):
        # Initialize the basic parameters
        # ----Input-----
        # interval: significant interval in statistic, range from 0 to 1
        # matrix: stock price matrix, each row represents one day price for different tickers, two dimentions ndarray
        # weight: the weight for portfolio, one dimension array
        # ----output----
        if 0 < interval < 1:
            self.ci = interval
        else:
            raise Exception("Invalid confidence interval", interval)

        if isinstance(matrix, pd.DataFrame):
            matrix = matrix.values

        if matrix.ndim != 2:
            raise Exception("Only accept 2 dimensions matrix", matrix.ndim)

        if len(weights) != matrix.shape[1]:
            raise Exception("Weights Length doesn't match")

        self.input = matrix
        # simple return calculation
        # self.returnMatrix = np.diff(self.input,axis = 0)/self.input[1:]

        # log return calculation
        self.returnMatrix = np.diff(np.log(self.input), axis=0)
        if not isinstance(weights, np.ndarray):
            self.weights = np.array(weights)
        else:
            self.weights = weights

    def covMatrix(self):
        # return variance-covariance matrix using return matrix
        # ----Input-----
        # interval: significant interval in statistic, range from 0 to 1
        # matrix: stock price matrix, each row represents one day price for different tickers, two dimentions ndarray
        # weight: the weight for portfolio, one dimension array
        # ----output----
        # variance-covariance matrix
        return np.cov(self.returnMatrix.T)

    def calculateVariance(self, Approximation=False):
        # return variance ----Input----- Approximation: If true, using portfolio return to calculate variance. If
        # false, using cov-var matrix to calculate ----output---- portfolio variance
        if Approximation == True:
            self.variance = np.var(np.dot(self.returnMatrix, self.weights))
        else:
            self.variance = np.dot(np.dot(self.weights, np.cov(self.returnMatrix.T)), self.weights.T)
        return self.variance

    def var(self, marketValue=0, Approximation=False, window=252):
        # return parametric value at risk, the variance can be calculated by either cov matrix way or approximate
        # way, scale the one day VaR according to user specified time period ----Input----- marketValue: the market
        # value of portfolio, if the value is less or equal zero, function will return percentage result
        # approximation:  If true, using portfolio return to calculate variance. If false, using cov-var matrix to
        # calculate window: scale time period, default value is 252 which returns annualized VaR ----output---- Value
        # at Risk in dollar or percentage if input market value is zero
        if (self.returnMatrix.shape[1] != len(self.weights)):
            raise Exception("The weights and portfolio doesn't match")
        self.calculateVariance(Approximation)
        if (marketValue <= 0):
            return abs(norm.ppf(self.ci) * np.sqrt(self.variance)) * math.sqrt(window)
        else:
            return abs(norm.ppf(self.ci) * np.sqrt(self.variance)) * marketValue * math.sqrt(window)

    def setCI(self, interval):
        # set the confidence interval for value at risk
        # ----Input-----
        # interval: significant interval in statistic, range from 0 to 1
        # ----output----
        if interval > 0 and interval < 1:
            self.ci = interval
        else:
            raise Exception("Invalid confidence interval", interval)

    def setPortfolio(self, matrix):
        # Change the current portfolio's data and weights
        # ----Input-----
        # matrix: stock price matrix, each row represents one day price for different tickers, two dimensions ndarray
        # ----output----
        if isinstance(matrix, pd.DataFrame):
            matrix = matrix.values

        if matrix.ndim != 2:
            raise Exception("Only accept 2 dimensions matrix", matrix.ndim)

        self.input = matrix
        self.returnMatrix = np.diff(np.log(self.input), axis=0)

    def setWeights(self, weights):
        # set the weights for the portfolio
        # ----Input-----
        # interval: the weight for portfolio, one dimension array
        # ----output----
        if not isinstance(weights, np.ndarray):
            self.weights = np.array(weights)
        else:
            self.weights = weights


class PCAVaR(ValueAtRisk):
    def __init__(self, interval, matrix, universe, weights=np.ones((1))):
        # Initialize the parameters ----Input----- interval: significant interval in statistic, range from 0 to 1
        # matrix: stock price matrix, each row represents one day price for different tickers, two dimensions ndarray
        # universe: the stock universe to generate PCA components weight: the weight for portfolio, one dimension
        # array, default value is 1 which means there is only 1 stock in portfolio ----output----
        if len(matrix) != len(universe):
            raise Exception('The length of input data and the length of universe data should match')
        ValueAtRisk.__init__(self, interval, matrix, weights)
        if (isinstance(universe, pd.DataFrame)):
            universe = universe.values
        self.universe = universe
        self.universeReturnMatrix = np.diff(np.log(self.universe), axis=0)

    def getComponents(self, n_components=2):
        # Generate principle components
        # ----Input-----
        # n_components: the number of components user want to generate
        # ----output----
        # factor matrix
        if self.universe.shape[1] < n_components:
            raise Exception("Too many PCA Components")
        pca = PCA(n_components=n_components)
        pca.fit(self.universeReturnMatrix)
        self.betaMatrix = pca.components_
        self.factorMatrix = np.dot(self.universeReturnMatrix, self.betaMatrix.T)
        self.factorCovVarMat = np.cov(self.factorMatrix.T)
        return self.factorMatrix

    def betaRegression(self, returns):
        # Run linear regression on return series
        # ----Input-----
        # returns: return series, the return's date should match factor's date
        #          eg. the first return and the first row factors are in the same date
        # ----output----
        # regression coefficient
        reg = LinearRegression().fit(self.factorMatrix, returns)
        self.betaMatrix = reg.coef_.T
        return reg.coef_.T

    def var(self, marketValue=0, window=252, approximation=False):
        # Return value at risk for portfolio ----Input----- marketValue: the market value of portfolio, if the value
        # is less or equal zero, function will return percentage result approximation:  If true, using portfolio
        # return to run beta regression. If false, using each stock series to run beta regression window: scale time
        # period, default value is 252 which returns annualized VaR ----output---- Value at Risk in dollar or
        # percentage if input market value is zero

        if approximation:
            input = np.dot(self.input, self.weights).reshape((-1, 1))
        else:
            input = self.input

        colNum = input.shape[1]
        betas = []
        for i in range(colNum):
            singlePrice = input[:, i]
            singleReturn = np.diff(np.log(singlePrice), axis=0)
            betas.append(list(self.betaRegression(singleReturn)))
        self.betaMatrix = np.array(betas).T
        self.CovVarMat = np.dot(np.dot(self.betaMatrix.T, self.factorCovVarMat), self.betaMatrix)

        if approximation:
            self.variance = self.CovVarMat[0, 0]
        else:
            self.variance = np.dot(np.dot(self.weights, self.CovVarMat), self.weights.T)

        if marketValue <= 0:
            return abs(norm.ppf(self.ci) * np.sqrt(self.variance)) * math.sqrt(window)
        else:
            return marketValue * abs(norm.ppf(self.ci) * np.sqrt(self.variance)) * math.sqrt(window)


class HistoricalVaR(ValueAtRisk):
    def var(self, marketValue=0, window=0):
        # return historical VaR
        # ----Input-----
        # marketValue: the market value of portfolio, if the value less or equal zero, function will return percentage
        # window: look back period, if window is zero, it will use whole input price series
        # ----output----
        # Value at Risk in dollar or percentage if input market value is lee or equal zero
        self.portfolioReturn = np.dot(self.returnMatrix, self.weights)
        if window > len(self.portfolioReturn) + 1:
            raise Exception("invalid Window, cannot excess", len(self.portfolioReturn))

        if 0 < window < len(self.portfolioReturn):
            PercentageVaR = abs(
                np.percentile(self.portfolioReturn[-window:], 100 * (1 - self.ci), interpolation='nearest'))
        else:
            PercentageVaR = abs(np.percentile(self.portfolioReturn, 100 * (1 - self.ci), interpolation='nearest'))

        if marketValue <= 0:
            return PercentageVaR
        else:
            return PercentageVaR * marketValue
