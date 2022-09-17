#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import json
import dash
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State,dash_table,ctx
import dash_bootstrap_components as dbc
from datetime import datetime
import numpy as np
import inspect
from flask_caching import Cache
import yfinance as yf

def get_shares(empresa,df):
    symbol= empresa[:-3]
    df = df[(df.SgmtNm=='CASH')&
       (df.SctyCtgyNm=='SHARES')][['TckrSymb','CrpnNm','SpcfctnCd','MktCptlstn']]
    df['MktCptlstn'] = df['MktCptlstn'].astype('float64')
    company_name = df[(df.TckrSymb==symbol)]['CrpnNm'].values[0]
    df.set_index('TckrSymb',inplace=True)
    total_shares = df[df.CrpnNm==company_name]['MktCptlstn'].sum()
    shares = df[df.CrpnNm==company_name].loc[symbol]['MktCptlstn']
    shares_pct = shares/total_shares
    return {'Shares':shares,'Share_Pct':shares_pct}

def capex_da(arq, empresa):
    arq=arq[(arq.DENOM_CIA==empresa)&(arq.ORDEM_EXERC=='ÚLTIMO')]
    arq['DT_FIM_EXERC'] = pd.to_datetime(arq['DT_FIM_EXERC'], dayfirst=True)
    arq['DT_INI_EXERC'] = pd.to_datetime(arq['DT_INI_EXERC'], dayfirst=True)
    arq['Data Range'] = arq.DT_FIM_EXERC-arq.DT_INI_EXERC
    arq_parte= arq[arq['Data Range']<pd.Timedelta(days=300)]
    arq_parte['Ano'] = arq_parte['DT_FIM_EXERC'].map(lambda x: x.year)
    arq_parte=arq_parte[(arq_parte['Data Range']>(arq_parte[arq_parte['Ano']==2022]['Data Range'].min()+pd.Timedelta(days=20)))&
                        (arq_parte['Data Range']<
                        (arq_parte[arq_parte['Ano']==2022]['Data Range'].max()+pd.Timedelta(days=20)))]
    arq_ano= arq[arq['Data Range']>pd.Timedelta(days=300)]
    arq_ano=arq_ano[(arq_ano.ORDEM_EXERC=='ÚLTIMO')&(arq_ano.DENOM_CIA==empresa)]
    searchfor = ['Venda','venda','baixa','Baixa','Ganho','Perda','Lucro','Prov','Alien']
    arq_parte=arq_parte[((~arq_parte.DS_CONTA.str.contains('|'.join(searchfor)))&
        arq_parte.DS_CONTA.str.contains('mobilizado','mobilizado'))|
                     (arq_parte.DS_CONTA.str.startswith('Deprec'))|
                    (arq_parte.DS_CONTA.str.startswith('Amortiz'))]
    arq_ano=arq_ano[((~arq_ano.DS_CONTA.str.contains('|'.join(searchfor)))&
        arq_ano.DS_CONTA.str.contains('mobilizado','mobilizado'))|
                     (arq_ano.DS_CONTA.str.startswith('Deprec'))|
                    (arq_ano.DS_CONTA.str.startswith('Amortiz'))]
    searchfor = ['Depreciação','Depreciações','Venda','venda','baixa','Baixa','Ganho','Perda','Lucro','Prov','Alien']
    capex_parte0 = arq_parte[(~arq_parte.DS_CONTA.str.contains('|'.join(searchfor)))&
                    (arq_parte.Ano==2021)&         
                    (arq_parte.DS_CONTA.str.contains('mobilizado','mobilizado'))]['VL_CONTA'].sum()
    capex_parte1 = arq_parte[(~arq_parte.DS_CONTA.str.contains('|'.join(searchfor)))&
                    (arq_parte.Ano==2022)&         
                    (arq_parte.DS_CONTA.str.contains('mobilizado','mobilizado'))]['VL_CONTA'].sum()
    capex2022 =arq_ano[(~arq_ano.DS_CONTA.str.contains('|'.join(searchfor)))&
                    (arq_ano.DS_CONTA.str.contains('mobilizado','mobilizado'))]['VL_CONTA'].sum()
    da_parte0 =arq_parte[(arq_parte.CD_CONTA.str.startswith('6.01'))&
                     (arq_parte.Ano==2021)& 
                 (arq_parte.DS_CONTA.str.startswith('Deprec'))|
                (arq_parte.DS_CONTA.str.startswith('Amortiz'))]['VL_CONTA'].sum()
    da_parte1 =arq_parte[(arq_parte.CD_CONTA.str.startswith('6.01'))&
                         (arq_parte.Ano==2022)& 
                     (arq_parte.DS_CONTA.str.startswith('Deprec'))|
                    (arq_parte.DS_CONTA.str.startswith('Amortiz'))]['VL_CONTA'].sum()
    da2022 =arq_ano[(arq_ano.CD_CONTA.str.startswith('6.01'))&
                     (arq_ano.DS_CONTA.str.startswith('Deprec'))|
                    (arq_ano.DS_CONTA.str.startswith('Amortiz'))]['VL_CONTA'].sum()
    capex = capex2022-capex_parte0+capex_parte1
    da=da2022-da_parte0+da_parte1
    return capex,da

def get_tickers(tickers_b3,company_name):
    df = tickers_b3[(tickers_b3.SgmtNm=='CASH')&
       (tickers_b3.SctyCtgyNm=='SHARES')][['TckrSymb','CrpnNm']]
    df=df[df.CrpnNm==company_name]
    df.set_index('TckrSymb',inplace=True)
    return list(df.index)
        
def valuation(crescimento,wacc,cresc_lp,mg_ebit,tx_imposto,tx_invest,tx_cap_giro,anos,tickers_b3,divida,caixa,shares,sharePct,wc_rate,capex,da,df0):
    receita0 = df0['LTM'].loc['Receita de Venda de Bens e/ou Serviços']
    ebit0 = df0['LTM'].loc['Resultado Antes do Resultado Financeiro e dos Tributos']
    margem_ebit = ebit0/receita0
    tax =df0['LTM'].loc['Imposto de Renda e Contribuição Social sobre o Lucro'] 
    tax_rate = tax/ebit0
    inc_fixed_capital = (capex-da)
    pv_fcf = []
    last_wc=[wc_rate*receita0]
    for i in range(anos):
        receita = receita0*(1+crescimento)**(i+1)
        ebit = mg_ebit*receita
        imposto = tx_imposto*ebit
        nopat = ebit-imposto
        fixed_capital = tx_invest*receita
        wc = tx_cap_giro*receita                
        change_nwc = wc-last_wc[0]
        last_wc[0]=wc
        fcf = nopat -fixed_capital-change_nwc
        pv=fcf/(1+wacc)**(i+1)
        pv_fcf.append(pv)
    tv = pv_fcf[anos-1]*(1+cresc_lp)/(wacc-cresc_lp)
    pv_fcf.append(tv/(1+wacc)**(anos))
    tev = sum(pv_fcf)
    ev  = tev-divida+caixa
    share_price = ev/(shares/sharePct)
    if share_price<0:
        return 0
    return share_price

def valuation2(crescimento,wacc,cresc_lp,mg_ebit,tx_imposto,tx_invest,tx_cap_giro,anos,df0,capex,da,wc_rate):
    receita0 = df0['LTM'].loc['Receita de Venda de Bens e/ou Serviços']
    ebit0 = df0['LTM'].loc['Resultado Antes do Resultado Financeiro e dos Tributos']
    margem_ebit = ebit0/receita0
    tax =df0['LTM'].loc['Imposto de Renda e Contribuição Social sobre o Lucro'] 
    tax_rate = tax/ebit0
    inc_fixed_capital = (capex-da)
    pv_fcf = []
    df_dcf = pd.DataFrame(index=['Receita', 'LAJIR','IR/CS','NOPAT','Investimento Incremental',
                                 'Variação no Capital de Giro',
                                 'Fluxo de Caixa',
                                 'Fluxo de Caixa Descontado'])
    currentYear = datetime.now().year
    last_wc=[wc_rate*receita0]
    for i in range(anos):
        receita = receita0*(1+crescimento)**(i+1)
        ebit = mg_ebit*receita
        imposto = tx_imposto*ebit
        nopat = ebit-imposto
        fixed_capital = tx_invest*receita
        wc = tx_cap_giro*receita
        change_nwc = wc-last_wc[0]
        last_wc[0]=wc
        fcf = nopat -fixed_capital-change_nwc
        pv=fcf/(1+wacc)**(i+1)
        pv_fcf.append(pv)
        df_dcf[currentYear+i+1] = [receita,ebit,imposto,nopat,fixed_capital,change_nwc,fcf,pv]
    df_dcf = df_dcf.applymap("{:,.0f}".format)
    df_dcf = dbc.Table.from_dataframe(df_dcf, striped=True, bordered=True, hover=True,index=True)
    return df_dcf

def goalseek(func,param, goal,args):        
    args[param]=0
    a=inspect.getfullargspec(func).args
    step=0.001
    for i in np.arange(-1,50,step):      
        args[param]=i
        values = [args[x] for x in a]
        if  round(func(*values),2)== goal:
            return i
        elif round(func(*values),2)> goal:
            for nb in np.arange(-1,i,step/10)[::-1]:
                args[param]=nb
                values = [args[x] for x in a]
                if  round(func(*values),2)== goal:
                    return nb
                elif round(func(*values),2)< goal:
                    for n in np.arange(nb,50,step/100):
                        args[param]=n
                        values = [args[x] for x in a]
                        if  round(func(*values),2)== goal:
                            return n

# In[4]:


def format_number(format_,number):
    if number is None:
        return ''
    elif type(number)!='str':
        return str(format_.format(number))
    else:
        return format_.format(number)
def pegar_nome(empresa, df):
    r=[]
    for i in df['DENOM_CIA']:
                if empresa.upper() in i:
                        r.append(i)
    for i in r:
        if empresa.upper()==i:
            r=[i]
    if len(pd.Series(r).unique())>1:
        return f"Mais de um resultado.Tente novamente com nome mais específico. Opções:{pd.Series(r).unique()} "
    elif len(pd.Series(r).unique())==1:
            return r[0]
    return "N/A"
def ticker_empresa(empresa, df):
    r=[]
    e=[]
    df =  df[(df.SgmtNm=='CASH')&
       (df.SctyCtgyNm=='SHARES')]
    df.fillna('-',inplace=True)
    for i,x in enumerate(df['CrpnNm']):
                if empresa.upper() in x:
                        r.append(df['TckrSymb'].iloc[i]+'.SA')
                        e.append(x)
    for i,x in enumerate(e):
        if empresa.upper()==x:
                e=[x]
                
    if len(pd.Series(e).unique())>1:
        return f"Mais de um resultado.Tente novamente com nome mais específico. Opções:{pd.Series(e).unique()} "
    elif len(pd.Series(e).unique())==1:
            return {'Ticker':r[0],'Company_name':e[0]}
    return "N/A"
def filtrar_empresa(df1,empresa):
    datas = df1[df1.DENOM_CIA==empresa]['DT_FIM_EXERC'].unique()
    df1['Ano']=  df1['DT_FIM_EXERC'].apply(lambda x: x.strftime("%Y"))
    return df1[(df1.DENOM_CIA==empresa)&(df1.DT_FIM_EXERC.isin(datas))]
def filtrar_contas(df,contas):
    return df[df['DS_CONTA'].isin(contas)]

def _get(stock_info,key):
    if key in stock_info.keys():
        return stock_info[key]
    else:
        return "N/A"   
def info_acao(ticker):
    df=pd.DataFrame()
    stock =yf.Ticker(ticker)
    stock_info = stock.info
    hist = stock.history(period="max")
    stock_financials=stock.financials.iloc[:,0]
    name = _get(stock_info,'shortName')
    industry = _get(stock_info,'industry')
    beta = _get(stock_info,'beta')
    mkt_cap =_get(stock_info,'marketCap')
    pe=_get(stock_info,'trailingPE')
    enterprise_value=_get(stock_info,'enterpriseValue')
    debt =  _get(stock_info,'totalDebt')
    debt_equity = _get(stock_info,'debtToEquity')
    current_price = _get(stock_info,'currentPrice')
    roe= _get(stock_info,'returnOnEquity')
    cash = _get(stock_info,'totalCash')
    year_change= _get(stock_info,'52WeekChange')

    
    
    results={'Ticker':[ticker],'Company':[name],'Industry':[industry],
             'Beta':[beta],'Market Cap':[mkt_cap],'Enterprise Value':[enterprise_value],
          'Current Price' :[current_price],'Return on Equity':[roe], 'Total Debt':[debt],
          'Debt/Equity':[debt_equity] , 'Total Cash':[cash], '52 Week Change':[year_change],'P/L':[pe]}

    df_results=pd.DataFrame(results)
    df_results.fillna('-')
    df = pd.concat([df, df_results])
    df.set_index('Ticker',inplace=True)
    return (df,hist)
#-----------------------------------------------------------------------------------------------------------------------
def dados_da_empresa(empresa,ticker,df1,tickers_b3,df_a,df_p,df_fc,df_itr):
    empresa1=empresa
    ticker1=ticker
    #ABRINDO AS BASES E ARRUMANDO OS DADOS
    df_fc = df_fc[(df_fc.ORDEM_EXERC=='ÚLTIMO')&(df_fc.DENOM_CIA==empresa)&
                  (df_fc.CD_CONTA.str.startswith('6.01'))|(df_fc.CD_CONTA.str.startswith('6.02'))]
    df1['DT_FIM_EXERC'] = pd.to_datetime(df1['DT_FIM_EXERC'], dayfirst=True)
    df1['DT_INI_EXERC'] = pd.to_datetime(df1['DT_INI_EXERC'], dayfirst=True)
    df_itr['DT_FIM_EXERC'] = pd.to_datetime(df_itr['DT_FIM_EXERC'], dayfirst=True)
    df_itr['DT_INI_EXERC'] = pd.to_datetime(df_itr['DT_INI_EXERC'], dayfirst=True)
    dre = ['Receita de Venda de Bens e/ou Serviços','Custo dos Bens e/ou Serviços Vendidos',
           'Resultado Bruto','Despesas/Receitas Operacionais',
           'Resultado Antes do Resultado Financeiro e dos Tributos',
           'Resultado Financeiro','Resultado Antes dos Tributos sobre o Lucro',
           'Imposto de Renda e Contribuição Social sobre o Lucro','Resultado Líquido das Operações Continuadas']
    
    #CAPEX E D&A
    capex = capex_da(df_fc, empresa)[0]
    da =  capex_da(df_fc, empresa)[1]
    #DATAFRAME DA DRE; DEIXA APENAS OS DADOS DA EMPRESA SEPARADOS EM TRIMESTRES; TIRA REPETIÇÕES, FILTRA GRUPOS E CONSOLIDA
    df1=filtrar_empresa(df1,empresa)
    df1=filtrar_contas(df1,dre)
    df_itr=filtrar_empresa(df_itr,empresa)
    df_itr=filtrar_contas(df_itr,dre)
    df1=df1[['DENOM_CIA','DT_INI_EXERC','DT_FIM_EXERC','DS_CONTA','VL_CONTA','Ano']]
    df1['DS_CONTA']=pd.Categorical(df1['DS_CONTA'],['Receita de Venda de Bens e/ou Serviços',
                                                    'Custo dos Bens e/ou Serviços Vendidos',
                                                    'Resultado Bruto','Despesas/Receitas Operacionais',
                                                    'Resultado Antes do Resultado Financeiro e dos Tributos',
                                                    'Resultado Financeiro','Resultado Antes dos Tributos sobre o Lucro',
                                                    'Imposto de Renda e Contribuição Social sobre o Lucro',
                                                    'Resultado Líquido das Operações Continuadas'])
    df_itr=df_itr[['DENOM_CIA','DT_INI_EXERC','DT_FIM_EXERC','DS_CONTA','VL_CONTA','Ano']]
    df_itr['DS_CONTA']=pd.Categorical(df_itr['DS_CONTA'],['Receita de Venda de Bens e/ou Serviços',
                                                          'Custo dos Bens e/ou Serviços Vendidos',
                                                          'Resultado Bruto','Despesas/Receitas Operacionais',
                                                          'Resultado Antes do Resultado Financeiro e dos Tributos',
                                                          'Resultado Financeiro','Resultado Antes dos Tributos sobre o Lucro',
                                                          'Imposto de Renda e Contribuição Social sobre o Lucro',
                                                          'Resultado Líquido das Operações Continuadas'])
    df_itr['Data Range'] = df_itr.DT_FIM_EXERC-df_itr.DT_INI_EXERC
    df_itr= df_itr[df_itr['Data Range']<pd.Timedelta(days=99)]
    
    df1.drop_duplicates(inplace=True)

    #LTM
    df2 = df1.set_index(['Ano','DS_CONTA']).sort_index()
    df2=df2[~df2.index.duplicated(keep='first')]
    df2=df2.unstack('Ano')['VL_CONTA']
    df3 = df_itr.set_index(['Ano','DS_CONTA']).sort_index()
    df3_2021 = df3[df3.index.get_level_values('Ano')=='2021']
    df3_2021['VL_CONTA'] = df3_2021['VL_CONTA'].apply(lambda x : x*-1)
    df3 = pd.concat([df3_2021,df3.drop('2021')])
    df3=df3.groupby('DS_CONTA').sum()
    df = df2
    df['LTM'] =df2['2021'] +df3['VL_CONTA']
    
    #PESQUISA YFINANCE
    x=info_acao(ticker)
    dados_acao = x[0]
    hist=x[1]
    
    #INFO BALANÇO
    df_a.sort_values(by='DT_FIM_EXERC',inplace=True,ascending=False)  
    df_p.sort_values(by='DT_FIM_EXERC',inplace=True,ascending=False)
    df_a = df_a[df_a.DENOM_CIA==empresa]
    df_p=df_p[df_p.DENOM_CIA==empresa]
    caixa = df_a[(df_a.ORDEM_EXERC=='ÚLTIMO')&(df_a.DT_FIM_EXERC==df_a.DT_FIM_EXERC.max())&
                 (df_a.DS_CONTA=='Caixa e Equivalentes de Caixa')
                ]['VL_CONTA']
    caixa =caixa.values[0]
    current_assets = df_a[(df_a.ORDEM_EXERC=='ÚLTIMO')&(df_a.DT_FIM_EXERC==df_a.DT_FIM_EXERC.max())&
                 (df_a.DS_CONTA=='Ativo Circulante')
                ]['VL_CONTA'] 
    current_assets=current_assets.values[0]-caixa

    divida = df_p[(df_p.ORDEM_EXERC=='ÚLTIMO')&(df_p.DT_FIM_EXERC==df_p.DT_FIM_EXERC.max())&
                  (df_p.DS_CONTA=='Empréstimos e Financiamentos')&
                  (df_p.CD_CONTA.str.len()==7)
                 ]['VL_CONTA'].sum()
    current_liabilities =df_p[(df_p.ORDEM_EXERC=='ÚLTIMO')&
                  (df_p.DS_CONTA=='Passivo Circulante')&
                  (df_p.CD_CONTA.str.len()==4)
                 ]['VL_CONTA']
    current_liabilities=current_liabilities.values[0]
    capital_de_giro = current_assets-current_liabilities

    caixa_penultimo= df_a[(df_a.ORDEM_EXERC=='PENÚLTIMO')&
                 (df_a.DS_CONTA=='Caixa e Equivalentes de Caixa')
                ]['VL_CONTA']
    caixa_penultimo =caixa_penultimo.values[0]
    current_assets_pen = df_a[(df_a.ORDEM_EXERC=='PENÚLTIMO')&
                 (df_a.DS_CONTA=='Ativo Circulante')
                ]['VL_CONTA'] 
    current_assets_pen=current_assets_pen.values[0]-caixa_penultimo

    divida_pen = df_p[(df_p.ORDEM_EXERC=='PENÚLTIMO')&
                  (df_p.DS_CONTA=='Empréstimos e Financiamentos')&
                  (df_p.CD_CONTA.str.len()==7)
                 ]['VL_CONTA'].sum()
    current_liabilities_pen =df_p[(df_p.ORDEM_EXERC=='PENÚLTIMO')&
                  (df_p.DS_CONTA=='Passivo Circulante')&
                  (df_p.CD_CONTA.str.len()==4)
                 ]['VL_CONTA']
    capital_de_giro_pen = current_assets_pen-current_liabilities_pen
    current_price = dados_acao['Current Price'].values[0]
    receita0 = df['LTM'].loc['Receita de Venda de Bens e/ou Serviços']
    ebit0 = df['LTM'].loc['Resultado Antes do Resultado Financeiro e dos Tributos']
    margem_ebit = ebit0/receita0
    tax =-df['LTM'].loc['Imposto de Renda e Contribuição Social sobre o Lucro'] 
    ebt = df['LTM'].loc['Resultado Antes dos Tributos sobre o Lucro']
    tax_rate = tax/ebt   
    inc_fixed_capital = -capex-da
    inc_fixed_capital_rate = inc_fixed_capital/receita0
    wc_rate =capital_de_giro/receita0
    
    #COTAÇÃO
    fig = px.line(hist['Close'],title='Histórico de Cotação - '+ticker)
    fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',  'paper_bgcolor': 'rgba(0,0,0,0)'},yaxis_title = ticker,showlegend=False)
    df0=df.to_json(date_format='iso', orient='split')    
    df = df.applymap("{:,.0f}".format)
    df = dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True,index=True)
    
    #Guardar valores
    df0=json.dumps(df0)
    dicionario = {'current_price':current_price, 'empresa':empresa,'divida':divida, 'caixa':caixa,
                 'ticker1':ticker1,'df0':df0,'capex':capex,'da':da,'wc_rate':wc_rate} 
    return empresa+' - '+ticker,format_number("{:.2f}",current_price),format_number("{:.2f}",dados_acao['Beta'].values[0]),format_number("{:,.0f}",dados_acao['Market Cap'].values[0]),format_number("{:.2%}",dados_acao['Return on Equity'].values[0]),format_number("{:.2f}",dados_acao['P/L'].values[0]),format_number("{:,.0f}",divida),format_number("{:,.0f}",caixa),df,fig,round(margem_ebit,2),round(tax_rate,2),round(inc_fixed_capital_rate,2),round(wc_rate,2),dicionario


#### DASH

app =Dash(external_stylesheets=[dbc.themes.LUX])
server=app.server

CACHE_CONFIG = {
    'CACHE_TYPE': 'FileSystemCache',
    'CACHE_DIR': 'my_cache_directory'}
cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)
#----------------------------------------------------------------------------------------------------------------------------
url='https://raw.githubusercontent.com/breqz/reverse-dcf-b3/main/dfp_cia_aberta_DRE_con.csv'
df1=pd.read_csv(url,sep=',',dtype='str',low_memory=False)
df1['VL_CONTA'] = pd.to_numeric(df1['VL_CONTA'])
lista_empresas =  df1['DENOM_CIA'].unique()
keys = ['label']
options = {v: v for v in lista_empresas}
empresa_dropdown = dcc.Dropdown(options=options, 
                                value='Procure o nome da empresa desejada',
                               id='loading-drop-1',
                               style={'margin-left': '10px',"width": "350px", 'font-size': 11}, optionHeight=80)

load_button = dcc.Loading([html.Button("Carregar", id="loading-button", n_clicks=0,className='btn btn-outline-secondary btn-sm',
                                         style={'color':'secondary','textAlign':'center',
                                                'height':'30px',
                                                "background-color": "white",'verticalAlign':'center','margin-top':'30px'})])
load_button2 = dcc.Loading([html.Button("Carregar", id="loading-button-2", n_clicks=0,className='btn btn-outline-secondary btn-sm',
                                         style={'color':'secondary','textAlign':'center', 'height':'30px',
                                                'verticalAlign':'center','margin-top':'8px'})])

offcanvas1 = html.Div([
    html.Div([dbc.Button('?', color="secondary", className="me-1",outline=True,
               id='offcanvas-button',size='sm',style={"borderRadius": "10px",'margin-bottom':'10px'})],
             style={'height':'2px','verticalAlign':'center','display': 'inline-block','margin-left': '8px'}),
    dbc.Offcanvas(
    html.P(['Para referência, visite os dados do prof. A.Damodaran no ' , 
            html.A('link',href='https://pages.stern.nyu.edu/~adamodar/pc/datasets/capexemerg.xls',title='Download Excel')
           ]),
        id='offcanvas1',title='Ajuda',is_open=False,placement='bottom')
],style={'display': 'inline-block'})

offcanvas2 = html.Div([
    html.Div([dbc.Button('?', color="secondary", className="me-1",outline=True,
               id='offcanvas-button2',size='sm',style={"borderRadius": "10px",'margin-bottom':'10px'})],
             style={'height':'2px','verticalAlign':'center','display': 'inline-block','margin-left': '8px'}),
    dbc.Offcanvas(
    html.P(['Para referência, visite os dados do prof. A.Damodaran no ' , 
            html.A('link',href='https://pages.stern.nyu.edu/~adamodar/pc/datasets/wcdataemerg.xls',title='Download Excel')
           ]),
        id='offcanvas2',title='Ajuda',is_open=False,placement='bottom')
],style={'display': 'inline-block'})

offcanvas3 = html.Div([
    html.Div([dbc.Button('?', color="secondary", className="me-1",outline=True,
               id='offcanvas-button3',size='sm',style={"borderRadius": "10px",'margin-bottom':'10px'})],
             style={'height':'2px','verticalAlign':'center','display': 'inline-block','margin-left': '8px'}),
    dbc.Offcanvas(
    html.P(['Para referência, visite os dados do prof. A.Damodaran no ' , 
            html.A('link',href='https://pages.stern.nyu.edu/~adamodar/pc/datasets/waccemerg.xls',title='Download Excel')
           ]),
        id='offcanvas3',title='Ajuda',is_open=False,placement='bottom')
],style={'display': 'inline-block'})

dropdown = [dbc.Col(
    html.Div([
        'Escolha o ticker a ser usado',
        dcc.Dropdown(
    id='menu1',
    style={'width':'200px','color':'grey','background-color':'white'})]),
        style={'margin-left': '10px',"width": "250px"})]

premissa_wacc = html.Div([
    html.I('WACC',style={'margin-left': '10px'}),
    html.Br(),
    dcc.Input(id="wacc",value="Digite o número em formato decimal",style={'margin-left': '10px',"width": "300px"}),
    offcanvas3
])
premissa_anos = html.Div([
    html.I('Período de Projeção (anos)',style={'margin-left': '10px'}),
    html.Br(),
    dcc.Input(id="anos",style={'margin-left': '10px',"width": "300px"})
])
premissa_cresc = html.Div([
    html.I('Crescimento LP',style={'margin-left': '10px'}),
    html.Br(),
    dcc.Input(id="cresc_lp",value="Digite o número em formato decimal",style={'margin-left': '10px',"width": "300px"})
])

premissa_mg_ebit = html.Div([
    html.I('Margem Ebit',style={'margin-left': '10px'}),
    html.Br(),
    dcc.Input(id="mg-ebit",style={'margin-left': '10px',"width": "300px"})
])
premissa_imposto = html.Div([
    html.I('Taxa de Imposto',style={'margin-left': '10px'}),
    html.Br(),
    dcc.Input(id="tx-imposto",style={'margin-left': '10px',"width": "300px"})
])
premissa_investimento = html.Div([
    html.I('Taxa de Investimento Incremental (% da Receita)',style={'margin-left': '10px','display': 'inline-block'}),
    html.Br(),
    dcc.Input(id="tx-invest",style={'margin-left': '10px',"width": "300px"}),
    offcanvas1
])
premissa_cap_giro = html.Div([
    html.I('Capital de Giro (% da Receita)',style={'margin-left': '10px'}),
    html.Br(),
    dcc.Input(id="tx-cap-giro",style={'margin-left': '10px',"width": "300px"}),
    offcanvas2
])

price_card=   [
    dbc.CardHeader('Última Cotação',style={'textAlign': 'center'}),
    dbc.CardBody(html.H5('R$'+' -',id='price_card',style={'textAlign': 'center'}))]
beta_card=[
    dbc.CardHeader('Beta',style={'textAlign': 'center'}),
    dbc.CardBody(html.H5('-',id='beta_card',style={'textAlign': 'center'}))]
mktCap_card = [
    dbc.CardHeader('Valor de Mercado',style={'textAlign': 'center'}),
    dbc.CardBody(html.H5('R$ -',id='mktCap_card' ,style={'textAlign': 'center'}))]
roe_card = [
    dbc.CardHeader('ROE',style={'textAlign': 'center'}),
    dbc.CardBody(html.H5('-'+'%',id='roe_card',style={'textAlign': 'center'}))]
pe_card = [
    dbc.CardHeader('P/L',style={'textAlign': 'center'}),
    dbc.CardBody(html.H5('-',id='pe_card',style={'textAlign': 'center'}))]
debt_card =  [
    dbc.CardHeader('Dívida Total',style={'textAlign': 'center'}),
    dbc.CardBody(html.H5('R$ -',id= 'debt_card',style={'textAlign': 'center'}))]
cash_card = [
    dbc.CardHeader('Disponibilidades',style={'textAlign': 'center'}),
    dbc.CardBody(html.H5('R$ -',id='cash_card',style={'textAlign': 'center'}))]
result_card = [
    dbc.CardBody(html.H5('-',id='result_card',style={'textAlign': 'center','border':'none'}))]

#----------------------------------------------------------------------------------------------------------------------------
app.layout = html.Div([
    dbc.Row(dbc.Col(html.H2(children='REVERSE DCF'),width=6, align="center",style={'margin-left': '10px','margin-top':'10px'})),
    dbc.Row(dbc.Col(html.H4(children= 'Ferramenta rápida para análise de empresas listadas na B3'),
                    width=8, align="center",style={'margin-left': '10px','margin-top':'10px'})),
    html.Div(children = [html.Div([empresa_dropdown],
        style={"display": "inline-block","margin-top": "30px","verticalAlign": "middle"}
        ),            
        html.Div([load_button],
                 style={"marginLeft": "1%", "verticalAlign": "center","width": "50px",'textAlign':'center',
                        "height": "40px","display": "inline-block"}
                )]),
    dbc.Row(html.Br()),
    dcc.Loading(id='loading-dropdown',children=html.Div([],id='ticker-dropdown')),
    dbc.Row(html.Br()),
    dbc.Row(dbc.Col([dcc.Markdown('',id="loading-output-1",style={'marginLeft':'10px'}),
                    dcc.Markdown('',id="loading-output-2",style={'marginLeft':'10px'})],
                                style={"verticalAlign": "bottom",
                                        "size":12,
                                        "height": "80x","display": "inline-block"},md=12,width=12,align='center'),
            justify='center'),
    dbc.Row([
        dbc.Col(dbc.Card(price_card,color='light',style={'height':'100px'})),
        dbc.Col(dbc.Card(beta_card,color='light',style={'height':'100px'})),
        dbc.Col(dbc.Card(mktCap_card,color='light',style={'height':'100px'})),
        dbc.Col(dbc.Card(roe_card,color='light',style={'height':'100px'})),
        dbc.Col(dbc.Card(pe_card,color='light',style={'height':'100px'})),
        dbc.Col(dbc.Card(debt_card,color='light',style={'height':'100px'})),
        dbc.Col(dbc.Card(cash_card,color='light',style={'height':'100px'}))
    ]),
    dbc.Row(
        dbc.Col(dcc.Loading(id='loading-2',children=[html.Div([dcc.Graph(id='graf_preco')])]))
    ),
    dbc.Row(html.Br()),
    dbc.Row(
        dbc.Col(dbc.Table(id='ult_results',size='sm',responsive='sm'),
                style={'margin-left':'10px'})        
    ),
    dbc.Row(dbc.Col(html.H4(children='Premissas de Valuation'),width=6, align="center",style={'margin-left': '10px','margin-top':'10px'})),
    dbc.Row(dbc.Col(html.H6(children= 'Este método permite investidores compreenderem as expectativas do mercado, com intuito de identificar mudanças nas expectativas que gerem maiores retornos de investimento.'),
                    width=8, align="center",style={'margin-left': '10px','margin-top':'10px'})),
    dbc.Row([dbc.Col(premissa_wacc, align="center",style={"width": "50px",'margin-left': '10px','margin-top':'10px'}),            
           dbc.Col(premissa_cresc, align="center",style={"width": "50px",'margin-top':'10px',"display": "inline-block"}),
           dbc.Col(premissa_imposto, align="center",style={"width": "50px",'margin-top':'10px',"display": "inline-block"})
            ]),
    dbc.Row([dbc.Col(premissa_mg_ebit,align="center",style={"width": "50px",'margin-left': '10px','margin-top':'10px'}),
           dbc.Col(premissa_investimento,align="center",style={"width": "90px",'margin-top':'10px',"display": "inline-block"}),
           dbc.Col(premissa_cap_giro,align="center",style={"width": "50px",'margin-top':'10px',"display": "inline-block"})]),
    dbc.Row(dbc.Col(premissa_anos, align="center",style={"width": "50px",'margin-left':'10px',
                                                         'margin-top':'10px',"display": "inline-block"})),
    dbc.Row(load_button2,style={"verticalAlign": "center","width": "70px",'textAlign':'center',"marginLeft": "10px", 
                        "height": "60px"}),
    dbc.Row([dbc.Col(dbc.Card(result_card,color='light',style={'margin-left':'10px','height':'100px'}),width=3),
             dbc.Col(
                 dcc.Loading(dbc.Table(id='dcf',size='sm',responsive='sm',style={'display':'inline-block'})),width=9
                    )
            ])
           
,
    dcc.Store(id='tickers-list'),
    dcc.Store(id='co'),
    dcc.Store(id='dict')
]
)
@cache.memoize()

def global_df1():
    url='https://raw.githubusercontent.com/breqz/reverse-dcf-b3/main/dfp_cia_aberta_DRE_con.csv'
    df1=pd.read_csv(url,sep=',',dtype='str',low_memory=False)
    df1['VL_CONTA'] = pd.to_numeric(df1['VL_CONTA'])
    return df1

def global_tickers_b3():
    url='https://raw.githubusercontent.com/breqz/reverse-dcf-b3/main/InstrumentsConsolidatedFile.csv'
    tickers_b3=pd.read_csv(url,decimal=',',low_memory=False)
    tickers_b3['CrpnNm'].replace(r"^ +| +$", r"", regex=True,inplace=True)
    return tickers_b3

def global_df_a():
    url='https://raw.githubusercontent.com/breqz/reverse-dcf-b3/main/itr_cia_aberta_BPA_con_2022.csv'
    df_a=pd.read_csv(url,sep=',',dtype='str',low_memory=False)
    df_a['VL_CONTA'] = pd.to_numeric(df_a['VL_CONTA'])
    return df_a 

def global_df_p():
    url='https://raw.githubusercontent.com/breqz/reverse-dcf-b3/main/itr_cia_aberta_BPP_con_2022.csv'
    df_p=pd.read_csv(url,sep=',',dtype='str',low_memory=False)
    df_p['VL_CONTA'] = pd.to_numeric(df_p['VL_CONTA'])
    return df_p

def global_df_itr():
    url='https://raw.githubusercontent.com/breqz/reverse-dcf-b3/main/itr_cia_aberta_DRE_con_2022.csv'
    df_itr=pd.read_csv(url,sep=',',dtype='str',low_memory=False)
    df_itr['VL_CONTA'] = pd.to_numeric(df_itr['VL_CONTA'])
    return df_itr

def global_df_fc():
    arq=pd.DataFrame()
    url = 'https://raw.githubusercontent.com/breqz/reverse-dcf-b3/main/itr_cia_aberta_DFC_MI_con_2021-2022.csv'
    csv_df = pd.read_csv(url,sep=',',decimal=',',dtype='str')
    csv_df['VL_CONTA'] = pd.to_numeric(csv_df['VL_CONTA'])
    arq = pd.concat([arq,csv_df])
    url = 'https://raw.githubusercontent.com/breqz/reverse-dcf-b3/main/dfp_cia_aberta_DFC_MI_con_2021.csv'
    csv_df = pd.read_csv(url,sep=',',dtype='str')
    csv_df['VL_CONTA'] = pd.to_numeric(csv_df['VL_CONTA'])
    arq = pd.concat([arq,csv_df])     
    return arq
    
    
@app.callback(
    Output('loading-output-1','children'),
    Output('tickers-list','data'),
    Output('co','data'),
    state=dict(value = State("loading-drop-1",'value')),
    inputs=dict(n_clicks=Input("loading-button", "n_clicks"))
)
def carregar_empresa(value,n_clicks):
    if n_clicks>=1:        
        df1 = global_df1()
        tickers_b3 = global_tickers_b3()
        nm_empresa = value
        empresa = pegar_nome(nm_empresa, df1)
        if 'Mais de um resultado' in empresa or 'N/A' in empresa:
            return '###'+empresa,dash.no_update,dash.no_update
        tck = ticker_empresa(nm_empresa,tickers_b3)
        try:
            ticker = tck['Ticker']
        except:
            ticker=tck
        if 'Mais de um resultado' in ticker or 'N/A' in ticker:
            return '###'+ticker,dash.no_update,dash.no_update
        tickers_list=get_tickers(tickers_b3,tck['Company_name'])
        return '',json.dumps(tickers_list),json.dumps(empresa)
    else:
        from dash.exceptions import PreventUpdate
        raise PreventUpdate

@app.callback(
    Output('ticker-dropdown','children'),
    Input('tickers-list','data')
)

def inserir_dropdown(data):
    if data:
        dropdown = [dbc.Col(
        html.Div([
        'Escolha o ticker a ser usado',
            dcc.Dropdown(json.loads(data),
            id='menu',
            style={'width':'200px','color':'grey','background-color':'white'})]),
        style={'margin-left': '10px',"width": "250px"})]
        return dropdown
    else:
        from dash.exceptions import PreventUpdate
        raise PreventUpdate

@app.callback(
    Output('loading-output-2','children'),
    Output('price_card','children'),
    Output('beta_card','children'),
    Output('mktCap_card','children'),
    Output('roe_card','children'),
    Output('pe_card','children'),
    Output('debt_card','children'),
    Output('cash_card','children'),
    Output('ult_results','children'),
    Output('graf_preco','figure'),
    Output('mg-ebit','value'),
    Output('tx-imposto','value'),
    Output('tx-invest','value'),
    Output('tx-cap-giro','value'),
    Output('dict','data'),
    inputs=dict(value = Input("menu", "value"),
                n_clicks =Input("loading-button", "n_clicks")),
    state=dict(empresa= State("co", "data")
    )
)
def contexto(value,empresa,n_clicks):
    triggered_id = ctx.triggered_id
    if triggered_id=='loading-button' and n_clicks>1:
        return 0,0,0,0,0,0,0,0,0,{},0,0,0,0,0
    else:
        return info_empresa(value,empresa)    
    
def info_empresa(value,empresa):
    if not value:
        from dash.exceptions import PreventUpdate
        raise PreventUpdate
    df1 = global_df1()
    tickers_b3 = global_tickers_b3()
    df_a = global_df_a()
    df_p = global_df_p()
    df_fc = global_df_fc()
    df_itr = global_df_itr()
    symbol=value+'.SA'
    empresa=json.loads(empresa)
    return dados_da_empresa(empresa,symbol,df1,tickers_b3,df_a,df_p,df_fc,df_itr)

@app.callback(
    Output('result_card','children'),
    Output('dcf','children'),
    state=dict(wacc=State("wacc",'value'),
     cresc_lp=State("cresc_lp",'value'),
     anos=State("anos",'value'),
     mg_ebit=State("mg-ebit",'value'),
     tx_imposto=State("tx-imposto",'value'),
     tx_invest=State("tx-invest",'value'),
     tx_cap_giro=State("tx-cap-giro",'value'),
    dicionario=State('dict','data')),
    inputs=dict(n_clicks=Input("loading-button-2", "n_clicks"),
    n_clicks1 =Input("loading-button", "n_clicks"))
)
def contexto1(wacc,cresc_lp,anos,mg_ebit,tx_imposto,tx_invest,tx_cap_giro,n_clicks,dicionario,n_clicks1):
    triggered_id = ctx.triggered_id
    if triggered_id=='loading-button' and n_clicks1>1:
        return None,None
    else:
        return reverse_dcf(wacc,cresc_lp,anos,mg_ebit,tx_imposto,tx_invest,tx_cap_giro,n_clicks,dicionario)  

def reverse_dcf(wacc,cresc_lp,anos,mg_ebit,tx_imposto,tx_invest,tx_cap_giro,n_clicks,dicionario):
    if n_clicks!=None and n_clicks>0:
        dic = dicionario
        current_price = float(dic['current_price'])
        empresa=dic['empresa']
        divida=float(dic['divida'])
        caixa=float(dic['caixa'])
        ticker1=dic['ticker1']
        df0 = pd.read_json(json.loads(dic['df0']), orient='split')
        capex=float(dic['capex'])
        da=float(dic['da'])
        wc_rate=float(dic['wc_rate'])
        anos=int(anos)
        wacc=float(wacc)
        cresc_lp=float(cresc_lp)
        mg_ebit=float(mg_ebit)
        tx_imposto=float(tx_imposto)
        tx_invest=float(tx_invest)
        tx_cap_giro=float(tx_cap_giro)
        tickers_b3 = global_tickers_b3()        
        shares=get_shares(ticker1,tickers_b3)['Shares']
        sharePct=get_shares(ticker1,tickers_b3)['Share_Pct']
        result=goalseek(valuation,'crescimento',current_price,
                        {'wacc':wacc,'cresc_lp':cresc_lp,
                         'mg_ebit':mg_ebit,'tx_imposto':tx_imposto,
                         'tx_invest':tx_invest,'tx_cap_giro':tx_cap_giro,
                         'anos':anos,'tickers_b3':tickers_b3,
                         'divida':divida,'caixa':caixa,
                        'shares':shares,'sharePct':sharePct,
                        'wc_rate':wc_rate,
                        'capex':capex,'da':da,'df0':df0})
        if result:
            return f'É esperado que a {empresa} cresça sua receita constantemente à uma taxa de {"{:.2%}".format(round(result,4))} nos próximos {anos} anos',valuation2(result,wacc,cresc_lp,mg_ebit,tx_imposto,tx_invest,tx_cap_giro,anos,df0,capex,da,wc_rate)
        else:
            return 'Não foi possível encontrar um resultado, tente novamente alterando as premissas.',{}
    else:
        from dash.exceptions import PreventUpdate
        raise PreventUpdate   
@app.callback(
    Output("offcanvas1", "is_open"),
    Input("offcanvas-button", "n_clicks"),
    [State("offcanvas1", "is_open")]
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open

@app.callback(
    Output("offcanvas2", "is_open"),
    Input("offcanvas-button2", "n_clicks"),
    [State("offcanvas2", "is_open")]
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open
@app.callback(
    Output("offcanvas3", "is_open"),
    Input("offcanvas-button3", "n_clicks"),
    [State("offcanvas3", "is_open")]
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open

if __name__ == '__main__':
    app.run_server(debug=True)

