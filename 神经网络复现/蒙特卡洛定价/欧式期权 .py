import numpy as np
import matplotlib.pyplot as plt
import scipy


def european_option_mc(S0,K,r,sigma,T,option_type='call',num_simulations=100000,num_steps=252):
    """
    使用蒙特卡洛定价进行期权定价
    参数：S0:初始股价
          K:行权价
          r:无风险利率
          sigma:波动率
          T:到期时间
          option_type:期权类型 call为看涨，put为看跌
          num_simulations:模拟路径数量
          num_steps:时间步数
          
          返回：期权价格"""
    dt = T/num_steps
    #生成股价路径
    Z = np.random.normal(0,1,(num_simulations,num_steps))
    #初始化价格数组
    S = np.zeros((num_simulations,num_steps+1))
    S[:,0] = S0
    #模拟股票价格路径
    for t in range(1,num_steps+1):
        S[:,t] = S[:,t-1]*np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*Z[:,t-1])
    #计算期权到期收益
    if option_type.lower()=='call':
        payoff = np.maximum(S[:,-1]-K,0)
    else:
        payoff = np.maximum(K-S[:,-1],0)

    #计算期权价格(折现)
    option_price = np.exp(-r*T)*np.mean(payoff)
    std_error = np.std(payoff)/np.sqrt(num_simulations)
    return option_price,std_error

def main():
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1
    call_price_mc,call_error = european_option_mc(S0,K,r,sigma,T)
    put_price_mc,put_error = european_option_mc(S0,K,r,sigma,T,'put')
    print(f"蒙特卡洛模拟结果：")
    print(f"看涨期权价格：{call_price_mc:.4f},标准误差：{call_error:.4f}")

if __name__ == '__main__':
    main()