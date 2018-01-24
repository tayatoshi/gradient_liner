# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def get_data(c,X,beta,error):
    """
    c 1*1
    X N*D
    beta D*1
    error N*1
    """
    Y =  c + np.dot(X,beta)+error
    return Y

class Gradient(object):
    def __init__(self,X,Y,eta):
        """
        self.X is variables
        self.Y is variables
        self.eta is learning rate
        self.iter is iteration of optimize
        """
        self.X = X
        self.Y = Y
        self.eta = eta
        self.diff = 0.00001

    def estimation(self,c_hat,beta_hat):
        """
        predict E[Y]
        """
        y_hat = c_hat + np.dot(self.X,beta_hat)
        return y_hat

    def residuals(self,y_hat):
        """
        calcurate residuals
        """
        res = np.dot((self.Y - y_hat).T,(self.Y - y_hat))
        return res

    def differential(self,c_hat,beta_hat):
        """
        calcuralte differenteials
        """
        diff_c = (self.residuals(self.estimation(c_hat + self.diff, beta_hat)) - self.residuals(self.estimation(c_hat, beta_hat)))/self.diff
        diff_beta = np.zeros(self.X.shape[1]).reshape(self.X.shape[1],1)
        for d in range(self.X.shape[1]):
            partial_beta_hat = beta_hat
            partial = np.zeros(self.X.shape[1]).reshape(self.X.shape[1],1)
            partial[d] = self.diff
            diff_beta[d,0] = (self.residuals(self.estimation(c_hat, beta_hat+partial)) - self.residuals(self.estimation(c_hat, beta_hat)))/self.diff
        return {'c' : diff_c, 'beta' : diff_beta}

    def fit(self,method = 0, Iter = 10**5):
        """
        start optimize
        method: 0 is OLS, 1 is MLE
        """
        c_hat_a = np.array(np.random.random()).reshape(1,1)
        beta_hat_a = (np.random.normal(1,1,self.X.shape[1])).reshape(self.X.shape[1],1)
        NUM = 0
        if method == 0:
            while NUM < Iter:
                if  NUM == 0:
                    c_hat_b = np.array(10).reshape(1,1)
                    beta_hat_b = np.ones(self.X.shape[1]).reshape(self.X.shape[1],1)
                else:
                    c_hat_b = c_hat_a
                    beta_hat_b = beta_hat_a
                Diff = self.differential(c_hat_b,beta_hat_b)
                c_hat_a = c_hat_b - self.eta * Diff['c']
                beta_hat_a= beta_hat_b - self.eta * Diff['beta']
                NUM = NUM + 1
                plt.scatter(NUM,c_hat_a,s=5,color='red')
                # if NUM < 50:
                #     print('time',NUM,' beta{},c{}'.format(beta_hat_a,c_hat_a))
                if NUM % 1000 == 0:
                    print('time',NUM,' beta{},c{}'.format(beta_hat_a,c_hat_a))
                if c_hat_a==c_hat_b and (beta_hat_a==beta_hat_b).sum()==self.X.shape[1]:
                    print('compleated at ', NUM)
                    break
        elif method == 1:
            print('1')
        else:
            print('ERROR : method is 0 or 1')
        return {'c':c_hat_a, 'beta':beta_hat_a}

X = np.random.normal(0,1,[200,3]).reshape(200,3)
error = np.random.normal(0,1,200).reshape(200,1)
beta = np.array([0.3,2,-0.5]).reshape(3,1)
c = np.array(5).reshape(1,1)

Y = c + np.dot(X,beta)+error

model = Gradient(X,Y,0.00001)

result = model.fit()

print('c:{}\nbeta:{}'.format(result['c'],result['beta']))
plt.show()
