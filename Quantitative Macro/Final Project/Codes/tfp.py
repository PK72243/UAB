import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import statsmodels.api as sm

#Choosing countries
countries=["BEL","NLD","DEU","DNK","AUT","ITA","FRA","ESP","PRT"]

#Reading file
pwt=pd.read_excel(io="pwt91.xlsx",sheet_name="Data")

#Some data wrangling
pwt_tfp=pwt[["countrycode","country","year","rtfpna","pop"]]
pwt_tfp=pwt_tfp[pwt_tfp["countrycode"].isin(countries)].dropna().reset_index(drop=True)
tfp_reg=np.log(pwt_tfp.drop(columns=["country","pop"]).pivot(index="year",columns="countrycode",values="rtfpna"))
tfp_reg_lag=np.log(pd.DataFrame(index=tfp_reg.index))
for col in countries:
	tfp_reg_lag[col]=tfp_reg[col].shift(periods=1)

#Here I creata an AR(1) model for each country	
models={}
pwt_tfp_shock=pd.DataFrame(index=tfp_reg.index[1:])
for country in countries:
	models[country]=sm.OLS(tfp_reg[country].to_numpy(),sm.add_constant(tfp_reg_lag[country].to_numpy()),missing="drop").fit()
	pwt_tfp_shock[country]=tfp_reg[country][1:]-models[country].predict()
pwt_tfp_shock=pd.melt(pwt_tfp_shock.reset_index(),id_vars="year",value_vars=pwt_tfp_shock.columns,var_name="countrycode",value_name="dTFP")
pwt_tfp=pd.merge(pwt_tfp,pwt_tfp_shock,how="outer",left_on=["countrycode","year"],right_on=["countrycode","year"])
pwt_tfp.iloc[pwt_tfp[pwt_tfp["year"]==np.min(pwt_tfp["year"])].index.values,pwt_tfp.columns.get_loc("dTFP")]=np.nan

# Graph
starting_year=1992
countries_bold=["BEL","NLD","DEU","FRA","ITA"]

fig,ax=plt.subplots()
for country in countries:
	if country in countries_bold:
		x=pwt_tfp[pwt_tfp["countrycode"]==country]["year"][starting_year-np.min(pwt_tfp["year"]):]
		y=pwt_tfp[pwt_tfp["countrycode"]==country]["dTFP"][starting_year-np.min(pwt_tfp["year"]):]
		ax.plot(x,y,label="{}".format(country),linewidth=1.8,)
	else:
		x=pwt_tfp[pwt_tfp["countrycode"]==country]["year"][starting_year-np.min(pwt_tfp["year"]):]
		y=pwt_tfp[pwt_tfp["countrycode"]==country]["dTFP"][starting_year-np.min(pwt_tfp["year"]):]
		ax.plot(x,y,label="{}".format(country),linewidth=0.3,color="gray")

ax.set_title("TFP shocks")
ax.set_ylabel("$\epsilon_{t}$")
ax.set_xlabel("Year")
plt.legend()
plt.show()	
# TFP correlation
pwt_cor=pwt_tfp.drop(columns=["country","rtfpna","pop"]).pivot(index="year",columns="countrycode",values="dTFP").loc[starting_year:]
pwt_cor.corr()
sn.heatmap(pwt_cor.corr(), annot=True)
plt.show()

#%%
#Reading data
rd_wbd=pd.read_excel(io="RD_WBD.xls",sheet_name="Data",skiprows=3)
fdi_wbd=pd.read_excel(io="FDI.xls",sheet_name="Data",skiprows=3)
# Some data wrangling
rd_wbd=pd.concat([rd_wbd,fdi_wbd])
rd_wbd=pd.melt(rd_wbd,id_vars=rd_wbd.columns[:4],value_vars=rd_wbd.columns[4:],var_name="Year")
rd_wbd=rd_wbd[rd_wbd["Country Code"].isin(countries)].reset_index(drop=True)
rd_wbd["Year"]=list(map(int,rd_wbd["Year"]))
rd_wbd=pd.merge(rd_wbd,pwt_tfp,how="outer",left_on=["Country Code","Year"],right_on=["countrycode","year"]).drop(columns=["countrycode","country","year"])
rd_wbd["value_per_capita"]=rd_wbd["value"].divide(rd_wbd["pop"])

rd_wbd_rd=rd_wbd[rd_wbd["Indicator Name"]=="Research and development expenditure (% of GDP)"].sort_values(["Country Code","Year"])
rd_wbd_rd["drd"]=rd_wbd_rd["value"].subtract(rd_wbd_rd["value"].shift(periods=1)).divide(rd_wbd_rd["value"])
rd_wbd_rd[["drd","dTFP"]].corr()

rd_wbd_pay=rd_wbd[rd_wbd["Indicator Name"]=="Charges for the use of intellectual property, payments (BoP, current US$)"].sort_values(["Country Code","Year"])
rd_wbd_pay["dPay"]=rd_wbd_pay["value"].subtract(rd_wbd_pay["value"].shift(periods=1)).divide(rd_wbd_pay["value"])
rd_wbd_pay[["dPay","dTFP"]].corr()

rd_wbd_fdi=rd_wbd[rd_wbd["Indicator Name"]=="Foreign direct investment, net inflows (% of GDP)"].sort_values(["Country Code","Year"])
rd_wbd_fdi["dFDI"]=rd_wbd_fdi["value"].subtract(rd_wbd_fdi["value"].shift(periods=1)).divide(rd_wbd_fdi["value"])
rd_wbd_fdi[["dFDI","dTFP"]].corr()

rd_wbd_rec=rd_wbd[rd_wbd["Indicator Name"]=="Charges for the use of intellectual property, receipts (BoP, current US$)"].sort_values(["Country Code","Year"])
rd_wbd_rec["drec"]=rd_wbd_rec["value"].subtract(rd_wbd_rec["value"].shift(periods=1)).divide(rd_wbd_rec["value"])
rd_wbd_rec[["drec","dTFP"]].corr()

#Graph function
def rd_graph(variable,bold=countries_bold,start=starting_year,value_per_capita=False):
	fig,ax=plt.subplots()
	for country in countries:
		min_year=int(np.min(rd_wbd[(rd_wbd["Country Code"]==country) & (rd_wbd["Indicator Name"]==variable)]["Year"]))
		if value_per_capita==False:
			y=rd_wbd[(rd_wbd["Country Code"]==country) & (rd_wbd["Indicator Name"]==variable)]["value"][starting_year-min_year:]
		else:
			y=rd_wbd[(rd_wbd["Country Code"]==country) & (rd_wbd["Indicator Name"]==variable)]["value_per_capita"][starting_year-min_year:]
		x=rd_wbd[(rd_wbd["Country Code"]==country) & (rd_wbd["Indicator Name"]==variable)]["Year"][starting_year-min_year:].values
		if country in countries_bold:
			ax.plot(x,y,label="{}".format(country),linewidth=1.8,)
		else:
			ax.plot(x,y,linewidth=0.3,color="gray")
	ax.set_title("{}".format(variable))
	ax.set_xlabel("Year")
	#plt.xlim(starting_year-1,2019)
	plt.legend()
	plt.show()	
	
rd_graph("High-technology exports (% of manufactured exports)")
rd_graph("Research and development expenditure (% of GDP)")
rd_graph("Charges for the use of intellectual property, receipts (BoP, current US$)")
rd_graph("Charges for the use of intellectual property, payments (BoP, current US$)")
rd_graph("Patent applications, residents",value_per_capita=True)
rd_graph("Patent applications, residents",value_per_capita=False)
rd_graph("Trademark applications, total",value_per_capita=True)
rd_graph("Scientific and technical journal articles",value_per_capita=True)
rd_graph("Foreign direct investment, net inflows (% of GDP)")

