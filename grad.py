# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

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

    def fit(self,method = 0, Iter = 10**8, retry = 5, trace = False):
        """
        start optimize
        method: 0 is OLS, 1 is MLE
        """
        c_hat_a = np.array(np.random.random()).reshape(1,1)
        beta_hat_a = (np.random.normal(1,1,self.X.shape[1])).reshape(self.X.shape[1],1)
        JUMP = 0
        same = 0
        Interim_c = None
        Interim_beta = None
        for j in range(retry):
            NUM = 0
            while NUM < Iter:
                if  NUM == 0:
                    c_hat_b = np.array(np.random.uniform(0,100)).reshape(1,1)
                    beta_hat_b = np.random.uniform(0,100,self.X.shape[1]).reshape(self.X.shape[1],1)
                    print('\ntimes:{}\ninit_c:{},init_beta:{}'.format(j, c_hat_b[0], beta_hat_b.reshape(self.X.shape[1])))
                else:
                    c_hat_b = c_hat_a
                    beta_hat_b = beta_hat_a
                Diff = self.differential(c_hat_b,beta_hat_b)
                c_hat_a = c_hat_b - self.eta * Diff['c']
                beta_hat_a= beta_hat_b - self.eta * Diff['beta']
                NUM = NUM + 1
                # plt.scatter(NUM,c_hat_a,s=5,color='red')
                BETA = 'beta_1 = '+ str(round(beta_hat_a[0,0],5))
                for i in range(1,self.X.shape[1]):
                    BETA = BETA + '\nbeta_' + str(i) + ' = ' + str(round(beta_hat_a[i,0],5))
                if trace == True and NUM % 1000 == 0:
                    print('iter',NUM,'\nbeta{},c{}\n'.format(BETA,c_hat_a[0,0]))
                if c_hat_a==c_hat_b and (beta_hat_a==beta_hat_b).sum()==self.X.shape[1]:
                    same = same + 1
                    if same == 20:
                        if Interim_c == None or self.residuals(self.estimation(c_hat_a, beta_hat_a)) < self.residuals(self.estimation(Interim_c,Interim_beta)):
                            Interim_c = c_hat_a
                            Interim_beta = beta_hat_a
                            NUM = 1
                            JUMP = 0
                            if Interim_c != None:
                                print("JUMP")
                        c_hat_a = c_hat_a + np.random.uniform(0,1)
                        beta_hat_a = beta_hat_a + np.random.uniform(0,1,self.X.shape[1]).reshape(self.X.shape[1],1)
                        # print('========{}==========='.format(BETA))
                        JUMP = JUMP + 1
                        same = 0
                        if JUMP == 10:
                            print('convergence at ', NUM,'residuals2=',self.residuals(self.estimation(Interim_c, Interim_beta))[0,0])
                            JUMP = 0
                            Interim_c = None
                            Interim_beta = None
                            break

            most_estimation = {'c':round(c_hat_a[0,0],5), 'beta':beta_hat_a,'BETA':BETA}
        print('completed!')
        print('=====RESULT=====')
        return most_estimation

X = pd.read_csv('X.csv',index_col = 0)
Y = pd.read_csv('Y.csv',index_col = 0)
X = np.array(X)
Y = np.array(Y)
model = Gradient(X,Y,0.00001)

result = model.fit()

print('c:{}\n{}'.format(result['c'],result['BETA']))
# plt.title('c = {},{}'.format(result['c'],result['BETA']))
# plt.show()
