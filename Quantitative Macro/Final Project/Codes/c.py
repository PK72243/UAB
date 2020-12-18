import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import statsmodels.api as sm

# In this code I compute the consumption shocks (deviations from the Steady State)
countries=["BEL"]

pwt=pd.read_excel(io="pwt91.xlsx",sheet_name="Data")
pwt_c=pwt[["countrycode","country","year","rgdpe","pop","csh_c"]]
pwt_c=pwt_c[pwt_c["countrycode"].isin(countries)].dropna().reset_index(drop=True)
c=pd.DataFrame()
c["consumption_pc"]=pwt_c["csh_c"].multiply(pwt_c["rgdpe"]).divide(pwt_c["pop"])
index = pd.period_range(str(min(pwt_c["year"])), str(max(pwt_c["year"])), freq='Y')
c.set_index(index, inplace=True)
cycle, trend = sm.tsa.filters.hpfilter(c, 6.25) #I use HP filter for yearly data

fig, ax = plt.subplots()
ax.plot(pwt_c["year"],trend.values)
ax.plot(pwt_c["year"],c)
plt.show()
c_dev=np.log(c["consumption_pc"]/trend.values)
#np.save('c',c_dev)
#%% This part I use to show that variable used as a proxy for foreign shocks has bigger variance

rd=pd.read_excel(io="RD_WBD.xls",sheet_name="Data",skiprows=3)
fdi_wbd=pd.read_excel(io="FDI.xls",sheet_name="Data",skiprows=3)
rd_wbd=pd.concat([rd,fdi_wbd])
rd_wbd=pd.melt(rd_wbd,id_vars=rd_wbd.columns[:4],value_vars=rd_wbd.columns[4:],var_name="Year")
rd_wbd=rd_wbd[rd_wbd["Country Code"].isin(countries)].reset_index(drop=True)

var="Foreign direct investment, net inflows (% of GDP)"
rd_wbd=rd_wbd[rd_wbd["Indicator Name"].isin([var])].dropna()
fdi=pd.DataFrame(rd_wbd["value"]).set_index(pd.period_range(str(min(rd_wbd['Year'])),str(max(rd_wbd['Year'])),freq="Y"))



fig, ax = plt.subplots()
plt.hist(fdi["value"],normed=1,bins=10)
plt.show()
np.var(fdi["value"])
np.mean(fdi["value"])

var2="Charges for the use of intellectual property, receipts (BoP, current US$)"
rd_wbd2=pd.concat([rd,fdi_wbd])
rd_wbd2=pd.melt(rd_wbd2,id_vars=rd_wbd2.columns[:4],value_vars=rd_wbd2.columns[4:],var_name="Year")
rd_wbd2=rd_wbd2[rd_wbd2["Country Code"].isin(countries)].reset_index(drop=True)
rd_wbd2=rd_wbd2[rd_wbd2["Indicator Name"].isin([var2])].dropna()

index = pd.period_range(str(min(rd_wbd2["Year"])), str(max(rd_wbd2["Year"])), freq='Y')
rec=pd.DataFrame(rd_wbd2["value"])
rec.set_index(index, inplace=True)
cycle_rec, trend_rec = sm.tsa.filters.hpfilter(rec, 6.25)
rec_dev=np.log(rec["value"]/trend_rec.values)

np.var(rec_dev.values)
np.mean(rec_dev.values)
fig, ax = plt.subplots()
plt.hist(rec_dev.values,normed=1,bins=10)
plt.show()


pd_cor=pd.DataFrame(rec_dev.values).set_index(rd_wbd2["Year"])
pd_cor=pd.merge(pd_cor.reset_index(),rd_wbd2,left_on="Year",right_on="Year",how="inner").set_index("Year").drop(columns=rd_wbd2.columns[0:-2])
pd_cor.corr()

