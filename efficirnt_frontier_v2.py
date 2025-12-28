"""
Credit to https://github.com/tejtw/TEJAPI_Python_Medium_Quant/blob/main/TEJAPI_Medium%E9%87%8F%E5%8C%96%E5%88%86%E6%9E%907.ipynb
Some part of the code are from TEJ 
"""

import ffn
import numpy as np
import plotly.graph_objects as go
import scipy.optimize as sco

global maxiter
maxiter = 10000

#The function is used to calculate the the std_dev of the profolio, which is the objective function we want to minimize
def objective_fun(portfolio_weight, cov_matrix):
    return np.sqrt(np.dot(portfolio_weight, np.dot(cov_matrix, portfolio_weight.T)))

#The function is used to calculate the returns and the covariance matrix of the profolio
def to_returns(prices):
    returns = prices / prices.shift(1) - 1
    returns = returns.iloc[1:, :]
    cov_matrix = np.cov(returns.T)
    return returns, cov_matrix

#We laverage the idea of Monte Carlo method to generate random weights for each stock in the profolio
#When the number of iteration is large enough, the edge of the random sample points will be close 
#to the optimal solution of efficient frontier
def random_sample(mean_returns, cov_matrix, maxiter=maxiter):
    result = np.zeros((2, maxiter))
    weightsRecordings = []
    for i in range(maxiter):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weightsRecordings.append(weights)
        returns = np.dot(weights.T, mean_returns)
        standard_deviation = objective_fun(weights, cov_matrix)
        result[0, i] = standard_deviation
        result[1, i] = returns
    return result, weightsRecordings

#The function is used to calculate the optimal solution of the efficient frontier with the given target returns
#This will be called in the function build_efficient_frontier
def risk_minimum_with_target(mean_returns, cov_matrix, target):
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                   {'type': 'eq', 'fun': lambda weights: np.dot(weights, mean_returns) - target})
    initial_weights = np.ones(len(mean_returns)) / len(mean_returns)
    bounds = [(0, 1)] * len(mean_returns)
    args = (cov_matrix, )
    result = sco.minimize(objective_fun,
                          initial_weights,
                          args=args, method='SLSQP',
                          constraints=constraints,
                          bounds=bounds,
                          options={'maxiter': maxiter})
    return result

#The function is used to build the efficient frontier with the given target returns
def build_efficient_frontier(mean_returns, cov_matrix, target_returns):
    efficient_frontier = []
    for target_return in target_returns:
        result = risk_minimum_with_target(mean_returns, cov_matrix, target_return)
        efficient_frontier.append((result.fun, target_return, result.x))
    return efficient_frontier

#The function is used to plot the efficient frontier
def plot_efficient_frontier(mean_returns, cov_matrix, optimal_result):
    result, weightsRecording = random_sample(mean_returns, cov_matrix, maxiter=maxiter)
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 100)
    print(mean_returns)
    efficient_frontier = build_efficient_frontier(mean_returns, cov_matrix, target_returns)
    # print(efficient_frontier)
    risk_list, return_list, weights_list = zip(*efficient_frontier)
    
    # Plotting the random sample
    graph_random_sample = go.Scatter(x=result[0, :],
                                     y=result[1, :],
                                     mode='markers',
                                     name = 'Random Sample',
                                     text=[', '.join([f'{w:.5f}' for w in weights]) for weights in weightsRecording],
                                     marker=dict(symbol='circle', size=5, color='blue')
                                     )
    
    graph_efficient_frontier = go.Scatter(x=risk_list, 
                                          y=return_list, 
                                          mode='lines', 
                                          name='Efficient Frontier',
                                          text=[', '.join([f'{w:.5f}' for w in weights]) for weights in weights_list],
                                          marker=dict(color='black')
                                          )
    
    graph_optimal_point = go.Scatter(x=[optimal_result.fun], 
                                     y=[np.dot(optimal_result.x, mean_returns)],
                                     mode='markers', 
                                     name='Optimal Point',
                                     text=[', '.join([f'{w:.5f}' for w in optimal_result.x])],
                                     marker=dict(symbol='star', size=20, color='red'))
    
    fig = go.Figure(data=[graph_efficient_frontier, graph_random_sample, graph_optimal_point])
    
    fig.update_layout(title='Efficient Frontier',
                      xaxis_title='Standard Deviation',
                      yaxis_title='Returns')
    fig.show()

#set the data, and the start and end date of the data
price = ffn.get('AAPL, ASML, GOOG, META, MSFT, NVDA, TSLA', start='2013-12-16', end='2023-12-16')
# price = ffn.get('KO, NSRGY, AAPL, GOOG, TSM, XOM, TM, NYA, MRVL, LMT', start='2013-12-16', end='2023-12-16')
returns, cov_matrix = to_returns(price)
total_returns = ffn.calc_total_return(price)
monthly_return = ffn.to_monthly(price)
mean_returns = total_returns / (len(monthly_return) / 12)
#calculate the lowest standard deviation and the corresponding returns on the efficient frontier
initial_weights = np.random.random(len(price.columns)) / len(price.columns)
constraint = ({'type': 'eq', 'fun': lambda weight: np.sum(weight) - 1})
bounds = [(0, 1)] * len(initial_weights)
result = sco.minimize(objective_fun,
                      initial_weights,
                      args=(cov_matrix,),
                      method='SLSQP',
                      constraints=constraint,
                      bounds=bounds,
                      options={'maxiter': maxiter})

print("Optimal Weights:")
for i, stock in enumerate(price.columns):
    print(f"{stock}: {result.x[i]:.8f}")
print(f"Optimal Standard Deviation: {result.fun:.8f}")
print(f"Optimal Returns: {np.dot(result.x, mean_returns):.8f}")

plot_efficient_frontier(mean_returns, cov_matrix, result)
# ffn.calc_stats(price).display()
