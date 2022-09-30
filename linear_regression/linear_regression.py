import numpy as np
from utils.features import prepare_for_training
class LinearRegression:
    def __init__(self,data,labels,polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        (data_processed,
         features_mean,
         features_deviation)= prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True)
        self.data=data_processed
        self.labels=labels
        self.polynomial_degree=polynomial_degree
        self.sinusoid_degree=sinusoid_degree
        self.features_mean=features_mean
        self.features_deviation=features_deviation
        self.normalize_data=normalize_data
        num_features=self.data.shape[1]#需要的是列，指的是有多少特征
        self.theta=np.zeros((num_features),1)#num_features行，1列

    def train(self,alpha,num_iter=500):  #数据，学习率（步数较小是合理的），迭代次数
        #实现梯度下降
        cost_history=self.gradient_descent(alpha,num_iter)
        return self.theta ,cost_history

    def gradient_descent(self,alpha,nums_iter=500):
        cost_history=[]
        #每一次迭代都要更新参数
        for i in range(nums_iter):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data,self.labels))
        return cost_history

    def gradient_step(self,alpha):
        #梯度下降参数更新方法 矩阵运算
        num_examples=self.data.shape[0] #sample count
        prediciton= LinearRegression.hypothesis(self.data,self.theta)
        delta=prediciton-self.labels #预测值减去真实值
        theta=self.theta

        theta=theta-alpha*(1/num_examples)*(np.dot(delta.T,self.data)).T
        #矩阵转置=求和运算
        self.theta=theta


    def cost_function(self,data,labels):
        num_examples=data.shape[0]
        delta=LinearRegression.hypothesis(self.data,self.theta)-labels #预测值减去真实值
        cost=(1/2)*np.dot(delta.T,delta)
        print(cost)
        return cost[0][0]


    @staticmethod
    def hypothesis(data, theta):
        prediction = np.dot(data, theta)  # 数据*theta
        return prediction

    def get_cost(self, data, labels):
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data
                                              )[0]

        return self.cost_function(data_processed, labels)

    def predict(self, data):
        """
                    用训练的参数模型，与预测得到回归值结果
        """
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data
                                              )[0]

        predictions = LinearRegression.hypothesis(data_processed, self.theta)

        return predictions





