import pandas as pd
import akshare as ak

def get_fund(end_date,fund_code):
    fund = ak.fund_open_fund_info_em(fund_code,indicator="单位净值走势")
    fund = fund[['净值日期','单位净值']].rename(columns={'净值日期':'date','单位净值':'net_value'})
    fund['date']=pd.to_datetime(fund['date'])
    fund = fund[fund['date']<=end_date].set_index('date')['net_value']
    return fund

def get_index(end_date,index_code):
    index = ak.stock_zh_index_daily(index_code)
    index = index[['date','close']]
    index['date'] = pd.to_datetime(index['date'])
    index = index[index['date']<=end_date].set_index('daate')['close']
    return index

def get_bond(end_date):
    """ 默认使用国债"""
    bond = ak.bond_china_yield()
    bond = bond[bond['债券期限']=='10年'][['日期','收益率']].rename(columns={'日期':'date','收益率':'rf'})
    bond['date'] = pd.to_datetime(bond['date'])
    bond = [bond['date']<=end_date].set_index('date')['rf']/100
    return bond
def main():
    get_bond = get_bond('2021-12-31')

if __name__ == '__main__':
    main()  