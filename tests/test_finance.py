import numpy as np
import pandas as pd
import pytest

from efficirnt_frontier_v2 import objective_fun, to_returns

def test_objective_function_math():
    """
    測試標準差計算函數 (Objective Function)
    數學原理：如果權重是 [1, 0]，共變異數矩陣是單位矩陣，標準差應該是 1
    """

    weights = np.array([1.0, 0.0])
    cov_matrix = np.array([[1.0, 0.0], 
                           [0.0, 1.0]])
    
    # 2. 執行你的函數
    result = objective_fun(weights, cov_matrix)
    
    # 3. 驗證結果 (Assert)
    expected_result = 1.0
    assert result == expected_result

def test_to_returns_shape():
    """
    測試報酬率轉換函數 (to_returns)
    測試輸入 3 天的股價，是否會回傳 2 天的報酬率 (因為第一天沒有前一天可比)
    """

    data = {
        'TESTT': [100, 101, 102],
        'TEST': [50, 52, 51]
    }
    prices = pd.DataFrame(data)
    
    returns, cov_matrix = to_returns(prices)
    

    assert returns.shape == (2, 2)

    assert cov_matrix.shape == (2, 2)