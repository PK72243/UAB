import pandas as pd
import numpy as np
import statsmodels.api as sm
import itertools
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

	
CPS=pd.read_csv("cps_00002.csv",usecols=[0,2,6,7,8,9,10,11,12,13])
ATUS=pd.read_csv("atus_00001.csv",usecols=[0,1,4,5,6,7])
	
	
labour_force=[10,12,20,21,22]
CPS_employment=CPS.loc[CPS["EMPSTAT"].isin(labour_force),:]
employed=[10,12]
unemployed=[20,21,22]
CPS_employment.loc[CPS_employment["EMPSTAT"].isin(employed),"EMPSTAT"]=1
CPS_employment.loc[CPS_employment["EMPSTAT"].isin(unemployed),"EMPSTAT"]=0
	
wm = lambda x: np.average(x, weights=CPS_employment.loc[x.index, "WTFINL"])
employment_rate=CPS_employment.groupby(["YEAR","MONTH"]).agg(employment_rate=("EMPSTAT", wm))
	
	
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
for parameters in pdq:
    for parameters_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(employment_rate[:24],order=parameters
			,seasonal_order=parameters_seasonal,enforce_stationarity=True,enforce_invertibility=True)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(parameters,parameters_seasonal,results.aic))
        except: 
            continue
			
mod1 = sm.tsa.statespace.SARIMAX(employment_rate[:24],
	                                order=(0, 1, 0),
	                                seasonal_order=(1, 0, 0, 12),
									enforce_stationarity=True,enforce_invertibility=True)
results = mod1.fit()
#print(results.summary().tables[1])
pred_uc = pd.DataFrame(results.get_forecast(steps=8).predicted_mean,columns=["predicted_employment"])
pred_uc.index=[(2020,  1),
	            (2020,  2),
	            (2020,  3),
	            (2020,  4),
	            (2020,  5),
	            (2020,  6),
	            (2020,  7),
	            (2020,  8)]
employment_final=pd.concat([employment_rate[12:24],employment_rate[24:],pred_uc],axis=1,join="outer")
employment_final.columns=["Data 2019","Data 2020","Prediction"]
employment_final=employment_final.reset_index()
employment_final["level_0"]=[str(val) for val in employment_final["level_0"].values]
employment_final["level_1"]=[str(val) for val in employment_final["level_1"].values]
employment_final=employment_final.set_index([employment_final["level_0"]+"-"+employment_final["level_1"]]).drop(["level_0","level_1"],axis=1)
employment_final["Diff in PP"]=employment_final["Data 2020"].subtract(employment_final["Prediction"])
employment_final["Diff in PP(%)"]=(employment_final["Data 2020"].divide(employment_final["Prediction"])-1)*100
	
ticks1=["2019-1","2019-7","2020-1","2020-7"]
ticks2=["2020-1","2020-3","2020-5","2020-7"]
fig, ax = plt.subplots()
ax.plot(employment_final.index,employment_final["Data 2019"],"ro",
			 label="Data 2019",
			 color="blue", markersize=6)
ax.plot(employment_final.index,employment_final["Data 2020"],"ro",
			 label="Data 2020",
			 color="grey", markersize=6)
ax.plot(employment_final.index,employment_final["Prediction"],"ro",
			 label="Prediction",
			 color="red",markersize=6)
plt.xticks(ticks1)
plt.legend()
plt.show()
	
fig, ax = plt.subplots()
ax.plot(employment_final.index,employment_final["Data 2020"],"ro",
			 label="Data 2020",
			 color="grey", markersize=6)
ax.plot(employment_final.index,employment_final["Prediction"],"ro",
			 label="Prediction",
			 color="red",markersize=6)
plt.xticks(ticks2)
plt.legend()
plt.show()
	
fig, ax = plt.subplots()
ax.plot(employment_final.index,employment_final["Diff in PP"],"-",
			 color="black", linewidth=3)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in Employment Rate")
plt.show()
	
fig, ax = plt.subplots()
ax.plot(employment_final.index,employment_final["Diff in PP(%)"],"-",
			 color="black", linewidth=3)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in Employment Rate(%)")
plt.show()


###EDUCATION###
not_education=[0,1,999]
employment_education=CPS_employment.loc[~CPS_employment["EDUC"].isin(not_education),:]
bins=[0,72,90,120,130]
labels_education=["<HS","HS","College",">College"]
employment_education["Education"]=pd.cut(employment_education["EDUC"],bins,labels=labels_education)
	
wm_e = lambda x: np.average(x, weights=employment_education.loc[x.index, "WTFINL"])
employment_education_rate=employment_education.groupby(["YEAR","MONTH","Education"]).agg(employment_rate=("EMPSTAT", wm_e))#.reset_index(level="Education")
employment_education_rate=employment_education_rate[:128]
employment_education_rate=employment_education_rate.reset_index()
employment_education_rate_1=employment_education_rate.loc[employment_education_rate['Education'] == "<HS",:]
employment_education_rate_2=employment_education_rate.loc[employment_education_rate['Education'] == "HS",:]
employment_education_rate_3=employment_education_rate.loc[employment_education_rate['Education'] == "College",:]
employment_education_rate_4=employment_education_rate.loc[employment_education_rate['Education'] == ">College",:]
	
	
mod21 = sm.tsa.statespace.SARIMAX(employment_education_rate_1.iloc[:24,3],
	                                order=(0, 1, 0),
	                                seasonal_order=(1, 0, 0, 12),
									enforce_stationarity=True,enforce_invertibility=True)
results21 = mod21.fit()
#print(results21.summary().tables[1])
pred_uc_21 = pd.DataFrame(results21.get_forecast(steps=8).predicted_mean,columns=["predicted_employment"])
pred_uc_21["YEAR"]=2020
pred_uc_21["Education"]="<HS"
pred_uc_21["MONTH"]=None
for i in range(8):
	pred_uc_21.iloc[i,3]=i+1
	
mod22 = sm.tsa.statespace.SARIMAX(employment_education_rate_2.iloc[:24,3],
	                                order=(0, 1, 0),
	                                seasonal_order=(1, 0, 0, 12),
									enforce_stationarity=True,enforce_invertibility=True)
results22 = mod22.fit()
#print(results22.summary().tables[1])
pred_uc_22 = pd.DataFrame(results22.get_forecast(steps=8).predicted_mean,columns=["predicted_employment"])
pred_uc_22["YEAR"]=2020
pred_uc_22["Education"]="HS"
pred_uc_22["MONTH"]=None
for i in range(8):
	pred_uc_22.iloc[i,3]=i+1
	
mod23 = sm.tsa.statespace.SARIMAX(employment_education_rate_3.iloc[:24,3],
	                                order=(0, 1, 0),
	                                seasonal_order=(1, 0, 0, 12),
									enforce_stationarity=True,enforce_invertibility=True)
results23 = mod23.fit()
#print(results23.summary().tables[1])
pred_uc_23 = pd.DataFrame(results23.get_forecast(steps=8).predicted_mean,columns=["predicted_employment"])
pred_uc_23["YEAR"]=2020
pred_uc_23["Education"]="College"
pred_uc_23["MONTH"]=None
for i in range(8):
	pred_uc_23.iloc[i,3]=i+1
		
mod24 = sm.tsa.statespace.SARIMAX(employment_education_rate_4.iloc[:24,3],
	                                order=(0, 1, 0),
	                                seasonal_order=(1, 0, 0, 12),
									enforce_stationarity=True,enforce_invertibility=True)
results24 = mod24.fit()
#print(results24.summary().tables[1])
pred_uc_24 = pd.DataFrame(results24.get_forecast(steps=8).predicted_mean,columns=["predicted_employment"])
pred_uc_24["YEAR"]=2020
pred_uc_24["Education"]=">College"
pred_uc_24["MONTH"]=None
for i in range(8):
	pred_uc_24.iloc[i,3]=i+1
c=len(labels_education)*12
	
employment_education_final=pd.merge(employment_education_rate.iloc[c:,:],pred_uc_21,on=["YEAR","MONTH","Education"],how="outer")
employment_education_final=pd.merge(employment_education_final,pred_uc_22,on=["YEAR","MONTH","Education"],how="outer")
employment_education_final=pd.merge(employment_education_final,pred_uc_23,on=["YEAR","MONTH","Education"],how="outer")
employment_education_final=pd.merge(employment_education_final,pred_uc_24,on=["YEAR","MONTH","Education"],how="outer")
employment_education_final["predicted_employment"]=employment_education_final.iloc[c:,4].fillna(0)+employment_education_final.iloc[c:,5].fillna(0)+employment_education_final.iloc[c:,6].fillna(0)+employment_education_final.iloc[c:,7].fillna(0)
employment_education_final=employment_education_final.iloc[:,[0,1,2,3,8]]
	
employment_education_final["YEAR"]=[str(val) for val in employment_education_final["YEAR"].values]
employment_education_final["MONTH"]=[str(val) for val in employment_education_final["MONTH"].values]
employment_education_final=employment_education_final.set_index([employment_education_final["YEAR"]+"-"+employment_education_final["MONTH"]]).drop(["YEAR","MONTH"],axis=1)
employment_education_final["Diff in PP"]=employment_education_final["employment_rate"].subtract(employment_education_final["predicted_employment"])
employment_education_final["Diff in PP(%)"]=(employment_education_final["employment_rate"].divide(employment_education_final["predicted_employment"])-1)*100
	
	
fig, ax = plt.subplots()
ax.plot(employment_education_final.loc[employment_education_final["Education"]=="<HS","Diff in PP"].index,
			employment_education_final.loc[employment_education_final["Education"]=="<HS","Diff in PP"],"-",
		 label="<HS",
			 color="red", linewidth=2.2)
ax.plot(employment_education_final.loc[employment_education_final["Education"]=="HS","Diff in PP"].index,
			employment_education_final.loc[employment_education_final["Education"]=="HS","Diff in PP"],"-",
		 label="HS",
			 color="yellow", linewidth=2.2)
ax.plot(employment_education_final.loc[employment_education_final["Education"]=="College","Diff in PP"].index,
			employment_education_final.loc[employment_education_final["Education"]=="College","Diff in PP"],"-",
		 label="College",
			 color="green", linewidth=2.2)
ax.plot(employment_education_final.loc[employment_education_final["Education"]==">College","Diff in PP"].index,
			employment_education_final.loc[employment_education_final["Education"]==">College","Diff in PP"],"-",
		 label=">College",
			 color="blue", linewidth=2.2)
ax.plot(employment_final.index,employment_final["Diff in PP"],"-", label="Total",
			 color="black", linewidth=2.2)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in Employment Rate")
plt.legend()
plt.show()
	
fig, ax = plt.subplots()
ax.plot(employment_education_final.loc[employment_education_final["Education"]=="<HS","Diff in PP(%)"].index,
			employment_education_final.loc[employment_education_final["Education"]=="<HS","Diff in PP(%)"],"-",
		 label="<HS",
			 color="red", linewidth=2.2)
ax.plot(employment_education_final.loc[employment_education_final["Education"]=="HS","Diff in PP(%)"].index,
			employment_education_final.loc[employment_education_final["Education"]=="HS","Diff in PP(%)"],"-",
		 label="HS",
			 color="yellow", linewidth=2.2)
ax.plot(employment_education_final.loc[employment_education_final["Education"]=="College","Diff in PP(%)"].index,
			employment_education_final.loc[employment_education_final["Education"]=="College","Diff in PP(%)"],"-",
		 label="College",
			 color="green", linewidth=2.2)
ax.plot(employment_education_final.loc[employment_education_final["Education"]==">College","Diff in PP(%)"].index,
			employment_education_final.loc[employment_education_final["Education"]==">College","Diff in PP(%)"],"-",
		 label=">College",
			 color="blue", linewidth=2.2)
ax.plot(employment_final.index,employment_final["Diff in PP(%)"],"-",label="Total",
			 color="black", linewidth=2.2)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in Employment Rate(%)")
plt.legend()
plt.show()
	
####INDUSTRY###
ATUS_tele=ATUS.loc[ATUS["WRKHOMEABLE"].isin([0,1]),:]
ATUS_tele_ind=ATUS_tele.loc[~(ATUS_tele["IND"]==99999),:]
sum_weights=ATUS_tele_ind[["WT06","IND"]].groupby("IND").agg(np.sum)
ATUS_tele_ind=pd.merge(ATUS_tele_ind,sum_weights,on="IND",how="outer",suffixes=("","_sum"))
ATUS_tele_ind["weighted"]=ATUS_tele_ind["WT06"].multiply(ATUS_tele_ind["WRKHOMEABLE"]).divide(ATUS_tele_ind["WT06_sum"])
tele_ind_score=ATUS_tele_ind[["IND","weighted"]].groupby("IND").agg(np.sum)
ATUS_tele_ind=pd.merge(ATUS_tele_ind,tele_ind_score,
									  on="IND",suffixes=("","_telework"), how="inner").sort_values("weighted_telework")
ATUS_tele_ind["density"]=ATUS_tele_ind["WT06"].divide(ATUS_tele_ind["WT06"].sum())						  
ind_dense=ATUS_tele_ind[["density","IND"]].groupby("IND").agg(np.sum)									  
tele_ind=pd.merge(ind_dense,tele_ind_score,on="IND",how="inner").sort_values("weighted")	
tele_ind["cumulative"]=tele_ind["density"].cumsum()				  
bins_ind=[0,0.5,1.01]
labels_ind=["Physical","Dematerialized"]
tele_ind["Industry"]=pd.cut(tele_ind["cumulative"],bins_ind,labels=labels_ind)

employment_industry=CPS_employment.loc[CPS_employment["IND"]!=99999,:]	
employment_industry=pd.merge(employment_industry,tele_ind.reset_index()[["IND","Industry"]],on="IND",how="inner")



wm_i = lambda x: np.average(x, weights=employment_industry.loc[x.index, "WTFINL"])
employment_industry_rate=employment_industry.groupby(["YEAR","MONTH","Industry"]).agg(employment_rate=("EMPSTAT", wm_i))#.reset_index(level="industry")
employment_industry_rate=employment_industry_rate[:64]
employment_industry_rate=employment_industry_rate.reset_index()
employment_industry_rate_1=employment_industry_rate.loc[employment_industry_rate['Industry'] == "Physical",:]
employment_industry_rate_2=employment_industry_rate.loc[employment_industry_rate['Industry'] == "Dematerialized",:]
	

mod31 = sm.tsa.statespace.SARIMAX(employment_industry_rate_1.iloc[:24,3],
	                                order=(0, 1, 0),
	                                seasonal_order=(1, 0, 0, 12),
									enforce_stationarity=True,enforce_invertibility=True)
results31 = mod31.fit()
#print(results31.summary().tables[1])
pred_uc_31 = pd.DataFrame(results31.get_forecast(steps=8).predicted_mean,columns=["predicted_employment"])
pred_uc_31["YEAR"]=2020
pred_uc_31["Industry"]="Physical"
pred_uc_31["MONTH"]=None
for i in range(8):
	pred_uc_31.iloc[i,3]=i+1
	
mod32 = sm.tsa.statespace.SARIMAX(employment_industry_rate_2.iloc[:24,3],
	                                order=(0, 1, 0),
	                                seasonal_order=(1, 0, 0, 12),
									enforce_stationarity=True,enforce_invertibility=True)
results32 = mod32.fit()
#print(results32.summary().tables[1])
pred_uc_32 = pd.DataFrame(results32.get_forecast(steps=8).predicted_mean,columns=["predicted_employment"])
pred_uc_32["YEAR"]=2020
pred_uc_32["Industry"]="Dematerialized"
pred_uc_32["MONTH"]=None
for i in range(8):
	pred_uc_32.iloc[i,3]=i+1


c_i=len(labels_ind)*12
	
employment_industry_final=pd.merge(employment_industry_rate.iloc[c_i:,:],pred_uc_31,on=["YEAR","MONTH","Industry"],how="outer")
employment_industry_final=pd.merge(employment_industry_final,pred_uc_32,on=["YEAR","MONTH","Industry"],how="outer")
employment_industry_final["predicted_employment"]=employment_industry_final.iloc[c_i:,4].fillna(0)+employment_industry_final.iloc[c_i:,5].fillna(0)
employment_industry_final=employment_industry_final.iloc[:,[0,1,2,3,6]]
	
employment_industry_final["YEAR"]=[str(val) for val in employment_industry_final["YEAR"].values]
employment_industry_final["MONTH"]=[str(val) for val in employment_industry_final["MONTH"].values]
employment_industry_final=employment_industry_final.set_index([employment_industry_final["YEAR"]+"-"+employment_industry_final["MONTH"]]).drop(["YEAR","MONTH"],axis=1)
employment_industry_final["Diff in PP"]=employment_industry_final["employment_rate"].subtract(employment_industry_final["predicted_employment"])
employment_industry_final["Diff in PP(%)"]=(employment_industry_final["employment_rate"].divide(employment_industry_final["predicted_employment"])-1)*100
	

fig, ax = plt.subplots()
ax.plot(employment_industry_final.loc[employment_industry_final["Industry"]=="Physical","Diff in PP"].index,
			employment_industry_final.loc[employment_industry_final["Industry"]=="Physical","Diff in PP"],"-",
			label="Physical",
			color="red", linewidth=2.2)
ax.plot(employment_industry_final.loc[employment_industry_final["Industry"]=="Dematerialized","Diff in PP"].index,
			employment_industry_final.loc[employment_industry_final["Industry"]=="Dematerialized","Diff in PP"],"-",
			label="Dematerialized",
			color="blue", linewidth=2.2)
ax.plot(employment_final.index,employment_final["Diff in PP"],"-", label="Total",
			color="black", linewidth=2.2)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in Employment Rate")
plt.legend()
plt.show()
	
fig, ax = plt.subplots()
ax.plot(employment_industry_final.loc[employment_industry_final["Industry"]=="Physical","Diff in PP(%)"].index,
			employment_industry_final.loc[employment_industry_final["Industry"]=="Physical","Diff in PP(%)"],"-",
		 label="Physical",
			 color="red", linewidth=2.2)
ax.plot(employment_industry_final.loc[employment_industry_final["Industry"]=="Dematerialized","Diff in PP(%)"].index,
			employment_industry_final.loc[employment_industry_final["Industry"]=="Dematerialized","Diff in PP(%)"],"-",
		 label="Dematerialized",
			 color="blue", linewidth=2.2)
ax.plot(employment_final.index,employment_final["Diff in PP(%)"],"-",label="Total",
			 color="black", linewidth=2.2)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in Employment Rate(%)")
plt.legend()
plt.show()

###OCCUPATION###
ATUS_tele_occ=ATUS_tele.loc[~(ATUS_tele["OCC"]==99999),:]
sum_weights=ATUS_tele_occ[["WT06","OCC"]].groupby("OCC").agg(np.sum)
ATUS_tele_occ=pd.merge(ATUS_tele_occ,sum_weights,on="OCC",how="outer",suffixes=("","_sum"))
ATUS_tele_occ["weighted"]=ATUS_tele_occ["WT06"].multiply(ATUS_tele_occ["WRKHOMEABLE"]).divide(ATUS_tele_occ["WT06_sum"])
tele_occ_score=ATUS_tele_occ[["OCC","weighted"]].groupby("OCC").agg(np.sum)
ATUS_tele_occ=pd.merge(ATUS_tele_occ,tele_occ_score,
									  on="OCC",suffixes=("","_telework"), how="inner").sort_values("weighted_telework")
ATUS_tele_occ["density"]=ATUS_tele_occ["WT06"].divide(ATUS_tele_occ["WT06"].sum())						  
occ_dense=ATUS_tele_occ[["density","OCC"]].groupby("OCC").agg(np.sum)									  
tele_occ=pd.merge(occ_dense,tele_occ_score,on="OCC",how="inner").sort_values("weighted")	
tele_occ["cumulative"]=tele_occ["density"].cumsum()				  
bins_occ=[0,0.3334,0.6667,1.01]
labels_occ=["Physical","Middle","Dematerialized"]
tele_occ["Occupation"]=pd.cut(tele_occ["cumulative"],bins_occ,labels=labels_occ)

employment_occupation=CPS_employment.loc[CPS_employment["OCC"]!=99999,:]	
employment_occupation=pd.merge(employment_occupation,tele_occ.reset_index()[["OCC","Occupation"]],on="OCC",how="inner")

wm_i = lambda x: np.average(x, weights=employment_occupation.loc[x.index, "WTFINL"])
employment_occupation_rate=employment_occupation.groupby(["YEAR","MONTH","Occupation"]).agg(employment_rate=("EMPSTAT", wm_i))#.reset_index(level="occupation")
employment_occupation_rate=employment_occupation_rate[:96]
employment_occupation_rate=employment_occupation_rate.reset_index()
employment_occupation_rate_1=employment_occupation_rate.loc[employment_occupation_rate['Occupation'] == "Physical",:]
employment_occupation_rate_2=employment_occupation_rate.loc[employment_occupation_rate['Occupation'] == "Middle",:]
employment_occupation_rate_3=employment_occupation_rate.loc[employment_occupation_rate['Occupation'] == "Dematerialized",:]
	
mod41 = sm.tsa.statespace.SARIMAX(employment_occupation_rate_1.iloc[:24,3],
	                                order=(0, 1, 0),
	                                seasonal_order=(1, 0, 0, 12),
									enforce_stationarity=True,enforce_invertibility=True)
results41 = mod41.fit()
#print(results41.summary().tables[1])
pred_uc_41 = pd.DataFrame(results41.get_forecast(steps=8).predicted_mean,columns=["predicted_employment"])
pred_uc_41["YEAR"]=2020
pred_uc_41["Occupation"]="Physical"
pred_uc_41["MONTH"]=None
for i in range(8):
	pred_uc_41.iloc[i,3]=i+1
	
mod42 = sm.tsa.statespace.SARIMAX(employment_occupation_rate_2.iloc[:24,3],
	                                order=(0, 1, 0),
	                                seasonal_order=(1, 0, 0, 12),
									enforce_stationarity=True,enforce_invertibility=True)
results42 = mod42.fit()
#print(results42.summary().tables[1])
pred_uc_42 = pd.DataFrame(results42.get_forecast(steps=8).predicted_mean,columns=["predicted_employment"])
pred_uc_42["YEAR"]=2020
pred_uc_42["Occupation"]="Middle"
pred_uc_42["MONTH"]=None
for i in range(8):
	pred_uc_42.iloc[i,3]=i+1

mod43 = sm.tsa.statespace.SARIMAX(employment_occupation_rate_3.iloc[:24,3],
	                                order=(0, 1, 0),
	                                seasonal_order=(1, 0, 0, 12),
									enforce_stationarity=True,enforce_invertibility=True)
results43 = mod43.fit()
#print(results43.summary().tables[1])
pred_uc_43 = pd.DataFrame(results43.get_forecast(steps=8).predicted_mean,columns=["predicted_employment"])
pred_uc_43["YEAR"]=2020
pred_uc_43["Occupation"]="Dematerialized"
pred_uc_43["MONTH"]=None
for i in range(8):
	pred_uc_43.iloc[i,3]=i+1
	
c_o=len(labels_occ)*12
	
employment_occupation_final=pd.merge(employment_occupation_rate.iloc[c_o:,:],pred_uc_41,on=["YEAR","MONTH","Occupation"],how="outer")
employment_occupation_final=pd.merge(employment_occupation_final,pred_uc_42,on=["YEAR","MONTH","Occupation"],how="outer")
employment_occupation_final=pd.merge(employment_occupation_final,pred_uc_43,on=["YEAR","MONTH","Occupation"],how="outer")
employment_occupation_final["predicted_employment_total"]=employment_occupation_final.iloc[c_o:,4].fillna(0)+employment_occupation_final.iloc[c_o:,5].fillna(0)+employment_occupation_final.iloc[c_o:,6].fillna(0)
employment_occupation_final=employment_occupation_final.iloc[:,[0,1,2,3,7]]
	
employment_occupation_final["YEAR"]=[str(val) for val in employment_occupation_final["YEAR"].values]
employment_occupation_final["MONTH"]=[str(val) for val in employment_occupation_final["MONTH"].values]
employment_occupation_final=employment_occupation_final.set_index([employment_occupation_final["YEAR"]+"-"+employment_occupation_final["MONTH"]]).drop(["YEAR","MONTH"],axis=1)
employment_occupation_final["Diff in PP"]=employment_occupation_final["employment_rate"].subtract(employment_occupation_final["predicted_employment_total"])
employment_occupation_final["Diff in PP(%)"]=(employment_occupation_final["employment_rate"].divide(employment_occupation_final["predicted_employment_total"])-1)*100

fig, ax = plt.subplots()
ax.plot(employment_occupation_final.loc[employment_occupation_final["Occupation"]=="Physical","Diff in PP"].index,
			employment_occupation_final.loc[employment_occupation_final["Occupation"]=="Physical","Diff in PP"],"-",
			label="Physical",
			color="red", linewidth=2.2)
ax.plot(employment_occupation_final.loc[employment_occupation_final["Occupation"]=="Middle","Diff in PP"].index,
			employment_occupation_final.loc[employment_occupation_final["Occupation"]=="Middle","Diff in PP"],"-",
			label="Middle",
			color="gray", linewidth=2.2)
ax.plot(employment_occupation_final.loc[employment_occupation_final["Occupation"]=="Dematerialized","Diff in PP"].index,
			employment_occupation_final.loc[employment_occupation_final["Occupation"]=="Dematerialized","Diff in PP"],"-",
			label="Dematerialized",
			color="blue", linewidth=2.2)
ax.plot(employment_final.index,employment_final["Diff in PP"],"-", label="Total",
			color="black", linewidth=2.2)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in Employment Rate")
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(employment_occupation_final.loc[employment_occupation_final["Occupation"]=="Physical","Diff in PP(%)"].index,
			employment_occupation_final.loc[employment_occupation_final["Occupation"]=="Physical","Diff in PP(%)"],"-",
			label="Physical",
			color="red", linewidth=2.2)
ax.plot(employment_occupation_final.loc[employment_occupation_final["Occupation"]=="Middle","Diff in PP(%)"].index,
			employment_occupation_final.loc[employment_occupation_final["Occupation"]=="Middle","Diff in PP(%)"],"-",
			label="Middle",
			color="gray", linewidth=2.2)
ax.plot(employment_occupation_final.loc[employment_occupation_final["Occupation"]=="Dematerialized","Diff in PP(%)"].index,
			employment_occupation_final.loc[employment_occupation_final["Occupation"]=="Dematerialized","Diff in PP(%)"],"-",
			label="Dematerialized",
			color="blue", linewidth=2.2)
ax.plot(employment_final.index,employment_final["Diff in PP(%)"],"-", label="Total",
			color="black", linewidth=2.2)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in Employment Rate(%)")
plt.legend()
plt.show()
#############################################################
########################## AVG HOURS ########################
#############################################################
not_hours=[997,999]
CPS_hours=CPS.loc[~CPS["UHRSWORKT"].isin(not_hours),:]

wm = lambda x: np.average(x, weights=CPS_hours.loc[x.index, "WTFINL"])
hours_rate=CPS_hours.groupby(["YEAR","MONTH"]).agg(hours_rate=("UHRSWORKT", wm))

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
for parameters in pdq:
    for parameters_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(hours_rate[:24],order=parameters
			,seasonal_order=parameters_seasonal,enforce_stationarity=True,enforce_invertibility=True)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(parameters,parameters_seasonal,results.aic))
        except: 
            continue
			
mod100 = sm.tsa.statespace.SARIMAX(hours_rate[:24],
	                                order=(0, 1, 0),
	                                seasonal_order=(1, 0, 0, 12),
									enforce_stationarity=True,enforce_invertibility=True)
results100 = mod100.fit()
#print(results100.summary().tables[1])
pred_uc100 = pd.DataFrame(results100.get_forecast(steps=8).predicted_mean,columns=["predicted_hours"])
pred_uc100.index=[(2020,  1),
	            (2020,  2),
	            (2020,  3),
	            (2020,  4),
	            (2020,  5),
	            (2020,  6),
	            (2020,  7),
	            (2020,  8)]

hours_final=pd.concat([hours_rate[12:24],hours_rate[24:],pred_uc100],axis=1,join="outer")
hours_final.columns=["Data 2019","Data 2020","Prediction"]
hours_final=hours_final.reset_index()
hours_final["level_0"]=[str(val) for val in hours_final["level_0"].values]
hours_final["level_1"]=[str(val) for val in hours_final["level_1"].values]
hours_final=hours_final.set_index([hours_final["level_0"]+"-"+hours_final["level_1"]]).drop(["level_0","level_1"],axis=1)
hours_final["Diff in PP"]=hours_final["Data 2020"].subtract(hours_final["Prediction"])
hours_final["Diff in PP(%)"]=(hours_final["Data 2020"].divide(hours_final["Prediction"])-1)*100
	
fig, ax = plt.subplots()
ax.plot(hours_final.index,hours_final["Data 2019"],"ro",
			 label="Data 2019",
			 color="blue", markersize=6)
ax.plot(hours_final.index,hours_final["Data 2020"],"ro",
			 label="Data 2020",
			 color="grey", markersize=6)
ax.plot(hours_final.index,hours_final["Prediction"],"ro",
			 label="Prediction",
			 color="red",markersize=6)
plt.xticks(ticks1)
plt.legend()
plt.show()
	
fig, ax = plt.subplots()
ax.plot(hours_final.index,hours_final["Data 2020"],"ro",
			 label="Data 2020",
			 color="grey", markersize=6)
ax.plot(hours_final.index,hours_final["Prediction"],"ro",
			 label="Prediction",
			 color="red",markersize=6)
plt.xticks(ticks2)
plt.legend()
plt.show()
	
fig, ax = plt.subplots()
ax.plot(hours_final.index,hours_final["Diff in PP"],"-",
			 color="black", linewidth=3)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in Hours Worked")
plt.show()
	
fig, ax = plt.subplots()
ax.plot(hours_final.index,hours_final["Diff in PP(%)"],"-",
			 color="black", linewidth=3)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in Hours Worked(%)")
plt.show()

###Education##
hours_education=CPS_hours.loc[~CPS_hours["EDUC"].isin(not_education),:]
hours_education["Education"]=pd.cut(hours_education["EDUC"],bins,labels=labels_education)
	
wm_e = lambda x: np.average(x, weights=hours_education.loc[x.index, "WTFINL"])
hours_education_rate=hours_education.groupby(["YEAR","MONTH","Education"]).agg(hours_rate=("UHRSWORKT", wm_e))#.reset_index(level="Education")
hours_education_rate=hours_education_rate[:128]
hours_education_rate=hours_education_rate.reset_index()
hours_education_rate_1=hours_education_rate.loc[hours_education_rate['Education'] == "<HS",:]
hours_education_rate_2=hours_education_rate.loc[hours_education_rate['Education'] == "HS",:]
hours_education_rate_3=hours_education_rate.loc[hours_education_rate['Education'] == "College",:]
hours_education_rate_4=hours_education_rate.loc[hours_education_rate['Education'] == ">College",:]
	
mod121 = sm.tsa.statespace.SARIMAX(hours_education_rate_1.iloc[:24,3],
	                                order=(0, 1, 0),
	                                seasonal_order=(1, 0, 0, 12),
									enforce_stationarity=True,enforce_invertibility=True)
results121 = mod121.fit()
#print(results121.summary().tables[1])
pred_uc_121 = pd.DataFrame(results121.get_forecast(steps=8).predicted_mean,columns=["predicted_hours"])
pred_uc_121["YEAR"]=2020
pred_uc_121["Education"]="<HS"
pred_uc_121["MONTH"]=None
for i in range(8):
	pred_uc_121.iloc[i,3]=i+1
	
mod122 = sm.tsa.statespace.SARIMAX(hours_education_rate_2.iloc[:24,3],
	                                order=(0, 1, 0),
	                                seasonal_order=(1, 0, 0, 12),
									enforce_stationarity=True,enforce_invertibility=True)
results122 = mod122.fit()
#print(results122.summary().tables[1])
pred_uc_122 = pd.DataFrame(results122.get_forecast(steps=8).predicted_mean,columns=["predicted_hours"])
pred_uc_122["YEAR"]=2020
pred_uc_122["Education"]="HS"
pred_uc_122["MONTH"]=None
for i in range(8):
	pred_uc_122.iloc[i,3]=i+1
	
mod123 = sm.tsa.statespace.SARIMAX(hours_education_rate_3.iloc[:24,3],
                                order=(0, 1, 0),
                                seasonal_order=(1, 0, 0, 12),
								enforce_stationarity=True,enforce_invertibility=True)
results123 = mod123.fit()
#print(results123.summary().tables[1])
pred_uc_123 = pd.DataFrame(results123.get_forecast(steps=8).predicted_mean,columns=["predicted_hours"])
pred_uc_123["YEAR"]=2020
pred_uc_123["Education"]="College"
pred_uc_123["MONTH"]=None
for i in range(8):
	pred_uc_123.iloc[i,3]=i+1
		
mod124 = sm.tsa.statespace.SARIMAX(hours_education_rate_4.iloc[:24,3],
                                order=(0, 1, 0),
                                seasonal_order=(1, 0, 0, 12),
								enforce_stationarity=True,enforce_invertibility=True)
results124 = mod124.fit()
#print(results124.summary().tables[1])
pred_uc_124 = pd.DataFrame(results124.get_forecast(steps=8).predicted_mean,columns=["predicted_hours"])
pred_uc_124["YEAR"]=2020
pred_uc_124["Education"]=">College"
pred_uc_124["MONTH"]=None
for i in range(8):
	pred_uc_124.iloc[i,3]=i+1
c=len(labels_education)*12
	
hours_education_final=pd.merge(hours_education_rate.iloc[c:,:],pred_uc_121,on=["YEAR","MONTH","Education"],how="outer")
hours_education_final=pd.merge(hours_education_final,pred_uc_122,on=["YEAR","MONTH","Education"],how="outer")
hours_education_final=pd.merge(hours_education_final,pred_uc_123,on=["YEAR","MONTH","Education"],how="outer")
hours_education_final=pd.merge(hours_education_final,pred_uc_124,on=["YEAR","MONTH","Education"],how="outer")
hours_education_final["predicted_hours"]=hours_education_final.iloc[c:,4].fillna(0)+hours_education_final.iloc[c:,5].fillna(0)+hours_education_final.iloc[c:,6].fillna(0)+hours_education_final.iloc[c:,7].fillna(0)
hours_education_final=hours_education_final.iloc[:,[0,1,2,3,8]]
	
hours_education_final["YEAR"]=[str(val) for val in hours_education_final["YEAR"].values]
hours_education_final["MONTH"]=[str(val) for val in hours_education_final["MONTH"].values]
hours_education_final=hours_education_final.set_index([hours_education_final["YEAR"]+"-"+hours_education_final["MONTH"]]).drop(["YEAR","MONTH"],axis=1)
hours_education_final["Diff in PP"]=hours_education_final["hours_rate"].subtract(hours_education_final["predicted_hours"])
hours_education_final["Diff in PP(%)"]=(hours_education_final["hours_rate"].divide(hours_education_final["predicted_hours"])-1)*100
			

fig, ax = plt.subplots()
ax.plot(hours_education_final.loc[hours_education_final["Education"]=="<HS","Diff in PP"].index,
			hours_education_final.loc[hours_education_final["Education"]=="<HS","Diff in PP"],"-",
		 label="<HS",
			 color="red", linewidth=2.2)
ax.plot(hours_education_final.loc[hours_education_final["Education"]=="HS","Diff in PP"].index,
			hours_education_final.loc[hours_education_final["Education"]=="HS","Diff in PP"],"-",
		 label="HS",
			 color="yellow", linewidth=2.2)
ax.plot(hours_education_final.loc[hours_education_final["Education"]=="College","Diff in PP"].index,
			hours_education_final.loc[hours_education_final["Education"]=="College","Diff in PP"],"-",
		 label="College",
			 color="green", linewidth=2.2)
ax.plot(hours_education_final.loc[hours_education_final["Education"]==">College","Diff in PP"].index,
			hours_education_final.loc[hours_education_final["Education"]==">College","Diff in PP"],"-",
		 label=">College",
			 color="blue", linewidth=2.2)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in Hours Worked")
plt.legend()
plt.show()
	
fig, ax = plt.subplots()
ax.plot(hours_education_final.loc[hours_education_final["Education"]=="<HS","Diff in PP(%)"].index,
			hours_education_final.loc[hours_education_final["Education"]=="<HS","Diff in PP(%)"],"-",
		 label="<HS",
			 color="red", linewidth=2.2)
ax.plot(hours_education_final.loc[hours_education_final["Education"]=="HS","Diff in PP(%)"].index,
			hours_education_final.loc[hours_education_final["Education"]=="HS","Diff in PP(%)"],"-",
		 label="HS",
			 color="yellow", linewidth=2.2)
ax.plot(hours_education_final.loc[hours_education_final["Education"]=="College","Diff in PP(%)"].index,
			hours_education_final.loc[hours_education_final["Education"]=="College","Diff in PP(%)"],"-",
		 label="College",
			 color="green", linewidth=2.2)
ax.plot(hours_education_final.loc[hours_education_final["Education"]==">College","Diff in PP(%)"].index,
			hours_education_final.loc[hours_education_final["Education"]==">College","Diff in PP(%)"],"-",
		 label=">College",
			 color="blue", linewidth=2.2)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in Hours Worked(%)")
plt.legend()
plt.show()
	
###Industry###

hours_industry=CPS_hours.loc[CPS_hours["IND"]!=99999,:]	
hours_industry=pd.merge(hours_industry,tele_ind.reset_index()[["IND","Industry"]],on="IND",how="inner")

wm_i = lambda x: np.average(x, weights=hours_industry.loc[x.index, "WTFINL"])
hours_industry_rate=hours_industry.groupby(["YEAR","MONTH","Industry"]).agg(hours_rate=("UHRSWORKT", wm_i))#.reset_index(level="industry")
hours_industry_rate=hours_industry_rate[:64]
hours_industry_rate=hours_industry_rate.reset_index()
hours_industry_rate_1=hours_industry_rate.loc[hours_industry_rate['Industry'] == "Physical",:]
hours_industry_rate_2=hours_industry_rate.loc[hours_industry_rate['Industry'] == "Dematerialized",:]
	

mod131 = sm.tsa.statespace.SARIMAX(hours_industry_rate_1.iloc[:24,3],
	                                order=(0, 1, 0),
	                                seasonal_order=(1, 0, 0, 12),
									enforce_stationarity=True,enforce_invertibility=True)
results131 = mod131.fit()
#print(results131.summary().tables[1])
pred_uc_131 = pd.DataFrame(results131.get_forecast(steps=8).predicted_mean,columns=["predicted_hours"])
pred_uc_131["YEAR"]=2020
pred_uc_131["Industry"]="Physical"
pred_uc_131["MONTH"]=None
for i in range(8):
	pred_uc_131.iloc[i,3]=i+1
	
mod132 = sm.tsa.statespace.SARIMAX(hours_industry_rate_2.iloc[:24,3],
	                                order=(0, 1, 0),
	                                seasonal_order=(1, 0, 0, 12),
									enforce_stationarity=True,enforce_invertibility=True)
results132 = mod132.fit()
#print(results132.summary().tables[1])
pred_uc_132 = pd.DataFrame(results132.get_forecast(steps=8).predicted_mean,columns=["predicted_hours"])
pred_uc_132["YEAR"]=2020
pred_uc_132["Industry"]="Dematerialized"
pred_uc_132["MONTH"]=None
for i in range(8):
	pred_uc_132.iloc[i,3]=i+1


c_i=len(labels_ind)*12
	
hours_industry_final=pd.merge(hours_industry_rate.iloc[c_i:,:],pred_uc_131,on=["YEAR","MONTH","Industry"],how="outer")
hours_industry_final=pd.merge(hours_industry_final,pred_uc_132,on=["YEAR","MONTH","Industry"],how="outer")
hours_industry_final["predicted_hours"]=hours_industry_final.iloc[c_i:,4].fillna(0)+hours_industry_final.iloc[c_i:,5].fillna(0)
hours_industry_final=hours_industry_final.iloc[:,[0,1,2,3,6]]
	
hours_industry_final["YEAR"]=[str(val) for val in hours_industry_final["YEAR"].values]
hours_industry_final["MONTH"]=[str(val) for val in hours_industry_final["MONTH"].values]
hours_industry_final=hours_industry_final.set_index([hours_industry_final["YEAR"]+"-"+hours_industry_final["MONTH"]]).drop(["YEAR","MONTH"],axis=1)
hours_industry_final["Diff in PP"]=hours_industry_final["hours_rate"].subtract(hours_industry_final["predicted_hours"])
hours_industry_final["Diff in PP(%)"]=(hours_industry_final["hours_rate"].divide(hours_industry_final["predicted_hours"])-1)*100
	

fig, ax = plt.subplots()
ax.plot(hours_industry_final.loc[hours_industry_final["Industry"]=="Physical","Diff in PP"].index,
			hours_industry_final.loc[hours_industry_final["Industry"]=="Physical","Diff in PP"],"-",
			label="Physical",
			color="red", linewidth=2.2)
ax.plot(hours_industry_final.loc[hours_industry_final["Industry"]=="Dematerialized","Diff in PP"].index,
			hours_industry_final.loc[hours_industry_final["Industry"]=="Dematerialized","Diff in PP"],"-",
			label="Dematerialized",
			color="blue", linewidth=2.2)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in Hours Worked")
plt.legend()
plt.show()
	
fig, ax = plt.subplots()
ax.plot(hours_industry_final.loc[hours_industry_final["Industry"]=="Physical","Diff in PP(%)"].index,
			hours_industry_final.loc[hours_industry_final["Industry"]=="Physical","Diff in PP(%)"],"-",
		 label="Physical",
			 color="red", linewidth=2.2)
ax.plot(hours_industry_final.loc[hours_industry_final["Industry"]=="Dematerialized","Diff in PP(%)"].index,
			hours_industry_final.loc[hours_industry_final["Industry"]=="Dematerialized","Diff in PP(%)"],"-",
		 label="Dematerialized",
			 color="blue", linewidth=2.2)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in Hours Worked(%)")
plt.legend()
plt.show()

### OCCUPATION ###

hours_occupation=CPS_hours.loc[CPS_hours["OCC"]!=99999,:]	
hours_occupation=pd.merge(hours_occupation,tele_occ.reset_index()[["OCC","Occupation"]],on="OCC",how="inner")

wm_i = lambda x: np.average(x, weights=hours_occupation.loc[x.index, "WTFINL"])
hours_occupation_rate=hours_occupation.groupby(["YEAR","MONTH","Occupation"]).agg(hours_rate=("UHRSWORKT", wm_i))#.reset_index(level="occupation")
hours_occupation_rate=hours_occupation_rate[:96]
hours_occupation_rate=hours_occupation_rate.reset_index()
hours_occupation_rate_1=hours_occupation_rate.loc[hours_occupation_rate['Occupation'] == "Physical",:]
hours_occupation_rate_2=hours_occupation_rate.loc[hours_occupation_rate['Occupation'] == "Middle",:]
hours_occupation_rate_3=hours_occupation_rate.loc[hours_occupation_rate['Occupation'] == "Dematerialized",:]
	
mod141 = sm.tsa.statespace.SARIMAX(hours_occupation_rate_1.iloc[:24,3],
	                                order=(0, 1, 0),
	                                seasonal_order=(1, 0, 0, 12),
									enforce_stationarity=True,enforce_invertibility=True)
results141 = mod141.fit()
#print(results141.summary().tables[1])
pred_uc_141 = pd.DataFrame(results141.get_forecast(steps=8).predicted_mean,columns=["predicted_hours"])
pred_uc_141["YEAR"]=2020
pred_uc_141["Occupation"]="Physical"
pred_uc_141["MONTH"]=None
for i in range(8):
	pred_uc_141.iloc[i,3]=i+1
	
mod142 = sm.tsa.statespace.SARIMAX(hours_occupation_rate_2.iloc[:24,3],
	                                order=(0, 1, 0),
	                                seasonal_order=(1, 0, 0, 12),
									enforce_stationarity=True,enforce_invertibility=True)
results142 = mod142.fit()
#print(results142.summary().tables[1])
pred_uc_142 = pd.DataFrame(results142.get_forecast(steps=8).predicted_mean,columns=["predicted_hours"])
pred_uc_142["YEAR"]=2020
pred_uc_142["Occupation"]="Middle"
pred_uc_142["MONTH"]=None
for i in range(8):
	pred_uc_142.iloc[i,3]=i+1

mod143 = sm.tsa.statespace.SARIMAX(hours_occupation_rate_3.iloc[:24,3],
	                                order=(0, 1, 0),
	                                seasonal_order=(1, 0, 0, 12),
									enforce_stationarity=True,enforce_invertibility=True)
results143 = mod143.fit()
#print(results143.summary().tables[1])
pred_uc_143 = pd.DataFrame(results143.get_forecast(steps=8).predicted_mean,columns=["predicted_hours"])
pred_uc_143["YEAR"]=2020
pred_uc_143["Occupation"]="Dematerialized"
pred_uc_143["MONTH"]=None
for i in range(8):
	pred_uc_143.iloc[i,3]=i+1
	
c_o=len(labels_occ)*12
	
hours_occupation_final=pd.merge(hours_occupation_rate.iloc[c_o:,:],pred_uc_141,on=["YEAR","MONTH","Occupation"],how="outer")
hours_occupation_final=pd.merge(hours_occupation_final,pred_uc_142,on=["YEAR","MONTH","Occupation"],how="outer")
hours_occupation_final=pd.merge(hours_occupation_final,pred_uc_143,on=["YEAR","MONTH","Occupation"],how="outer")
hours_occupation_final["predicted_hours_total"]=hours_occupation_final.iloc[c_o:,4].fillna(0)+hours_occupation_final.iloc[c_o:,5].fillna(0)+hours_occupation_final.iloc[c_o:,6].fillna(0)
hours_occupation_final=hours_occupation_final.iloc[:,[0,1,2,3,7]]
	
hours_occupation_final["YEAR"]=[str(val) for val in hours_occupation_final["YEAR"].values]
hours_occupation_final["MONTH"]=[str(val) for val in hours_occupation_final["MONTH"].values]
hours_occupation_final=hours_occupation_final.set_index([hours_occupation_final["YEAR"]+"-"+hours_occupation_final["MONTH"]]).drop(["YEAR","MONTH"],axis=1)
hours_occupation_final["Diff in PP"]=hours_occupation_final["hours_rate"].subtract(hours_occupation_final["predicted_hours_total"])
hours_occupation_final["Diff in PP(%)"]=(hours_occupation_final["hours_rate"].divide(hours_occupation_final["predicted_hours_total"])-1)*100

fig, ax = plt.subplots()
ax.plot(hours_occupation_final.loc[hours_occupation_final["Occupation"]=="Physical","Diff in PP"].index,
			hours_occupation_final.loc[hours_occupation_final["Occupation"]=="Physical","Diff in PP"],"-",
			label="Physical",
			color="red", linewidth=2.2)
ax.plot(hours_occupation_final.loc[hours_occupation_final["Occupation"]=="Middle","Diff in PP"].index,
			hours_occupation_final.loc[hours_occupation_final["Occupation"]=="Middle","Diff in PP"],"-",
			label="Middle",
			color="gray", linewidth=2.2)
ax.plot(hours_occupation_final.loc[hours_occupation_final["Occupation"]=="Dematerialized","Diff in PP"].index,
			hours_occupation_final.loc[hours_occupation_final["Occupation"]=="Dematerialized","Diff in PP"],"-",
			label="Dematerialized",
			color="blue", linewidth=2.2)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in Hours Worked")
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(hours_occupation_final.loc[hours_occupation_final["Occupation"]=="Physical","Diff in PP(%)"].index,
			hours_occupation_final.loc[hours_occupation_final["Occupation"]=="Physical","Diff in PP(%)"],"-",
			label="Physical",
			color="red", linewidth=2.2)
ax.plot(hours_occupation_final.loc[hours_occupation_final["Occupation"]=="Middle","Diff in PP(%)"].index,
			hours_occupation_final.loc[hours_occupation_final["Occupation"]=="Middle","Diff in PP(%)"],"-",
			label="Middle",
			color="gray", linewidth=2.2)
ax.plot(hours_occupation_final.loc[hours_occupation_final["Occupation"]=="Dematerialized","Diff in PP(%)"].index,
			hours_occupation_final.loc[hours_occupation_final["Occupation"]=="Dematerialized","Diff in PP(%)"],"-",
			label="Dematerialized",
			color="blue", linewidth=2.2)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in Hours Worked(%)")
plt.legend()
plt.show()

##############################################
############ TOTAL hOURS #################
##############################################
total_hours=pd.concat([hours_final.iloc[:,:3],employment_final.iloc[:,:3]],axis=1,join="outer")
total_hours.columns=["Data 2019 h","Data 2020 h","Prediction h","Data 2019 e","Data 2020 e","Prediction e"]
total_hours["Data 2019 H"]=total_hours["Data 2019 h"].multiply(total_hours["Data 2019 e"])
total_hours["Data 2020 H"]=total_hours["Data 2020 h"].multiply(total_hours["Data 2020 e"])  
total_hours["Prediction H"]=total_hours["Prediction h"].multiply(total_hours["Prediction e"])
total_hours=total_hours.iloc[:,6:9]
total_hours["Diff in PP"]=total_hours["Data 2020 H"].subtract(total_hours["Prediction H"])
total_hours["Diff in PP(%)"]=(total_hours["Data 2020 H"].divide(total_hours["Prediction H"])-1)*100

	
fig, ax = plt.subplots()
ax.plot(total_hours.index,total_hours["Data 2019 H"],"ro",
			 label="Data 2019 H",
			 color="blue", markersize=6)
ax.plot(total_hours.index,total_hours["Data 2020 H"],"ro",
			 label="Data 2020 H",
			 color="grey", markersize=6)
ax.plot(total_hours.index,total_hours["Prediction H"],"ro",
			 label="Prediction H",
			 color="red",markersize=6)
plt.xticks(ticks1)
plt.ylabel("Total Hours")
plt.legend()
plt.show()
	
fig, ax = plt.subplots()
ax.plot(total_hours.index,total_hours["Data 2020 H"],"ro",
			 label="Data 2020 H",
			 color="grey", markersize=6)
ax.plot(total_hours.index,total_hours["Prediction H"],"ro",
			 label="Prediction H",
			 color="red",markersize=6)
plt.xticks(ticks2)
plt.ylabel("Total Hours")
plt.legend()
plt.show()
	
fig, ax = plt.subplots()
ax.plot(total_hours.index,total_hours["Diff in PP"],"-",
			 color="black", linewidth=3)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in Total Hours")
plt.show()
	
fig, ax = plt.subplots()
ax.plot(total_hours.index,total_hours["Diff in PP(%)"],"-",
			 color="black", linewidth=3)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in Total Hours(%)")
plt.show()

fig, ax = plt.subplots()
ax.plot(total_hours.index,total_hours["Diff in PP(%)"],"--",
			 color="black", linewidth=2,label="Total hours")
ax.plot(hours_final.index,hours_final["Diff in PP(%)"],"-",
			 color="blue", linewidth=1.2,label="Average hours")
ax.plot(employment_final.index,employment_final["Diff in PP(%)"],"-",
			 color="red", linewidth=1.2,label="Employment")
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference from the prediction(%)")
plt.legend()
plt.show()


##############################################
############ WEEKLY EARNINGS #################
##############################################

not_earnings=[9999.99]
CPS_wage=CPS.loc[~CPS["EARNWEEK"].isin(not_earnings),:]

wm = lambda x: np.average(x, weights=CPS_wage.loc[x.index, "WTFINL"])
wage_rate=CPS_wage.groupby(["YEAR","MONTH"]).agg(wage_rate=("EARNWEEK", wm))

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
for parameters in pdq:
    for parameters_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(wage_rate[:24],order=parameters
			,seasonal_order=parameters_seasonal,enforce_stationarity=True,enforce_invertibility=True)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(parameters,parameters_seasonal,results.aic))
        except: 
            continue
			
mod300 = sm.tsa.statespace.SARIMAX(wage_rate[:24],
	                                order=(1, 1, 0),
	                                seasonal_order=(0, 1, 0, 12),
									enforce_stationarity=True,enforce_invertibility=True)
results300 = mod300.fit()
#print(results300.summary().tables[1])
pred_uc300 = pd.DataFrame(results300.get_forecast(steps=8).predicted_mean,columns=["predicted_wage"])
pred_uc300.index=[(2020,  1),
	            (2020,  2),
	            (2020,  3),
	            (2020,  4),
	            (2020,  5),
	            (2020,  6),
	            (2020,  7),
	            (2020,  8)]

wage_final=pd.concat([wage_rate[12:24],wage_rate[24:],pred_uc300],axis=1,join="outer")
wage_final.columns=["Data 2019","Data 2020","Prediction"]
wage_final=wage_final.reset_index()
wage_final["level_0"]=[str(val) for val in wage_final["level_0"].values]
wage_final["level_1"]=[str(val) for val in wage_final["level_1"].values]
wage_final=wage_final.set_index([wage_final["level_0"]+"-"+wage_final["level_1"]]).drop(["level_0","level_1"],axis=1)
wage_final["Diff in PP"]=wage_final["Data 2020"].subtract(wage_final["Prediction"])
wage_final["Diff in PP(%)"]=(wage_final["Data 2020"].divide(wage_final["Prediction"])-1)*100
	
fig, ax = plt.subplots()
ax.plot(wage_final.index,wage_final["Data 2019"],"ro",
			 label="Data 2019",
			 color="blue", markersize=6)
ax.plot(wage_final.index,wage_final["Data 2020"],"ro",
			 label="Data 2020",
			 color="grey", markersize=6)
ax.plot(wage_final.index,wage_final["Prediction"],"ro",
			 label="Prediction",
			 color="red",markersize=6)
plt.xticks(ticks1)
plt.ylabel("Average Week Earnings")
plt.legend()
plt.show()
	
fig, ax = plt.subplots()
ax.plot(wage_final.index,wage_final["Data 2020"],"ro",
			 label="Data 2020",
			 color="grey", markersize=6)
ax.plot(wage_final.index,wage_final["Prediction"],"ro",
			 label="Prediction",
			 color="red",markersize=6)
plt.xticks(ticks2)
plt.ylabel("Average Week Earnings")
plt.legend()
plt.show()
	
fig, ax = plt.subplots()
ax.plot(wage_final.index,wage_final["Diff in PP"],"-",
			 color="black", linewidth=3)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in Week Earnings")
plt.show()
	
fig, ax = plt.subplots()
ax.plot(wage_final.index,wage_final["Diff in PP(%)"],"-",
			 color="black", linewidth=3)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in Week Earnings(%)")
plt.show()

###Education##
wage_education=CPS_wage.loc[~CPS_wage["EDUC"].isin(not_education),:]
wage_education["Education"]=pd.cut(wage_education["EDUC"],bins,labels=labels_education)
	
wm_e = lambda x: np.average(x, weights=wage_education.loc[x.index, "WTFINL"])
wage_education_rate=wage_education.groupby(["YEAR","MONTH","Education"]).agg(wage_rate=("EARNWEEK", wm_e))#.reset_index(level="Education")
wage_education_rate=wage_education_rate[:128]
wage_education_rate=wage_education_rate.reset_index()
wage_education_rate_1=wage_education_rate.loc[wage_education_rate['Education'] == "<HS",:]
wage_education_rate_2=wage_education_rate.loc[wage_education_rate['Education'] == "HS",:]
wage_education_rate_3=wage_education_rate.loc[wage_education_rate['Education'] == "College",:]
wage_education_rate_4=wage_education_rate.loc[wage_education_rate['Education'] == ">College",:]
	
mod321 = sm.tsa.statespace.SARIMAX(wage_education_rate_1.iloc[:24,3],
	                                order=(1, 1, 0),
	                                seasonal_order=(0, 1, 0, 12),
									enforce_stationarity=True,enforce_invertibility=True)
results321 = mod321.fit()
#print(results321.summary().tables[1])
pred_uc_321 = pd.DataFrame(results321.get_forecast(steps=8).predicted_mean,columns=["predicted_wage"])
pred_uc_321["YEAR"]=2020
pred_uc_321["Education"]="<HS"
pred_uc_321["MONTH"]=None
for i in range(8):
	pred_uc_321.iloc[i,3]=i+1
	
mod322 = sm.tsa.statespace.SARIMAX(wage_education_rate_2.iloc[:24,3],
	                                order=(1, 1, 0),
	                                seasonal_order=(0, 1, 0, 12),
									enforce_stationarity=True,enforce_invertibility=True)
results322 = mod322.fit()
#print(results322.summary().tables[1])
pred_uc_322 = pd.DataFrame(results322.get_forecast(steps=8).predicted_mean,columns=["predicted_wage"])
pred_uc_322["YEAR"]=2020
pred_uc_322["Education"]="HS"
pred_uc_322["MONTH"]=None
for i in range(8):
	pred_uc_322.iloc[i,3]=i+1
	
mod323 = sm.tsa.statespace.SARIMAX(wage_education_rate_3.iloc[:24,3],
                                order=(1, 1, 0),
								seasonal_order=(0, 1, 0, 12),
								enforce_stationarity=True,enforce_invertibility=True)
results323 = mod323.fit()
#print(results323.summary().tables[1])
pred_uc_323 = pd.DataFrame(results323.get_forecast(steps=8).predicted_mean,columns=["predicted_wage"])
pred_uc_323["YEAR"]=2020
pred_uc_323["Education"]="College"
pred_uc_323["MONTH"]=None
for i in range(8):
	pred_uc_323.iloc[i,3]=i+1
		
mod324 = sm.tsa.statespace.SARIMAX(wage_education_rate_4.iloc[:24,3],
                                order=(1, 1, 0),
	                                seasonal_order=(0, 1, 0, 12),
								enforce_stationarity=True,enforce_invertibility=True)
results324 = mod324.fit()
#print(results324.summary().tables[1])
pred_uc_324 = pd.DataFrame(results324.get_forecast(steps=8).predicted_mean,columns=["predicted_wage"])
pred_uc_324["YEAR"]=2020
pred_uc_324["Education"]=">College"
pred_uc_324["MONTH"]=None
for i in range(8):
	pred_uc_324.iloc[i,3]=i+1
c=len(labels_education)*12
	
wage_education_final=pd.merge(wage_education_rate.iloc[c:,:],pred_uc_321,on=["YEAR","MONTH","Education"],how="outer")
wage_education_final=pd.merge(wage_education_final,pred_uc_322,on=["YEAR","MONTH","Education"],how="outer")
wage_education_final=pd.merge(wage_education_final,pred_uc_323,on=["YEAR","MONTH","Education"],how="outer")
wage_education_final=pd.merge(wage_education_final,pred_uc_324,on=["YEAR","MONTH","Education"],how="outer")
wage_education_final["predicted_wage"]=wage_education_final.iloc[c:,4].fillna(0)+wage_education_final.iloc[c:,5].fillna(0)+wage_education_final.iloc[c:,6].fillna(0)+wage_education_final.iloc[c:,7].fillna(0)
wage_education_final=wage_education_final.iloc[:,[0,1,2,3,8]]
	
wage_education_final["YEAR"]=[str(val) for val in wage_education_final["YEAR"].values]
wage_education_final["MONTH"]=[str(val) for val in wage_education_final["MONTH"].values]
wage_education_final=wage_education_final.set_index([wage_education_final["YEAR"]+"-"+wage_education_final["MONTH"]]).drop(["YEAR","MONTH"],axis=1)
wage_education_final["Diff in PP"]=wage_education_final["wage_rate"].subtract(wage_education_final["predicted_wage"])
wage_education_final["Diff in PP(%)"]=(wage_education_final["wage_rate"].divide(wage_education_final["predicted_wage"])-1)*100
			

fig, ax = plt.subplots()
ax.plot(wage_education_final.loc[wage_education_final["Education"]=="<HS","Diff in PP"].index,
			wage_education_final.loc[wage_education_final["Education"]=="<HS","Diff in PP"],"-",
		 label="<HS",
			 color="red", linewidth=2.2)
ax.plot(wage_education_final.loc[wage_education_final["Education"]=="HS","Diff in PP"].index,
			wage_education_final.loc[wage_education_final["Education"]=="HS","Diff in PP"],"-",
		 label="HS",
			 color="yellow", linewidth=2.2)
ax.plot(wage_education_final.loc[wage_education_final["Education"]=="College","Diff in PP"].index,
			wage_education_final.loc[wage_education_final["Education"]=="College","Diff in PP"],"-",
		 label="College",
			 color="green", linewidth=2.2)
ax.plot(wage_education_final.loc[wage_education_final["Education"]==">College","Diff in PP"].index,
			wage_education_final.loc[wage_education_final["Education"]==">College","Diff in PP"],"-",
		 label=">College",
			 color="blue", linewidth=2.2)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in Week Earnings")
plt.legend()
plt.show()
	
fig, ax = plt.subplots()
ax.plot(wage_education_final.loc[wage_education_final["Education"]=="<HS","Diff in PP(%)"].index,
			wage_education_final.loc[wage_education_final["Education"]=="<HS","Diff in PP(%)"],"-",
		 label="<HS",
			 color="red", linewidth=2.2)
ax.plot(wage_education_final.loc[wage_education_final["Education"]=="HS","Diff in PP(%)"].index,
			wage_education_final.loc[wage_education_final["Education"]=="HS","Diff in PP(%)"],"-",
		 label="HS",
			 color="yellow", linewidth=2.2)
ax.plot(wage_education_final.loc[wage_education_final["Education"]=="College","Diff in PP(%)"].index,
			wage_education_final.loc[wage_education_final["Education"]=="College","Diff in PP(%)"],"-",
		 label="College",
			 color="green", linewidth=2.2)
ax.plot(wage_education_final.loc[wage_education_final["Education"]==">College","Diff in PP(%)"].index,
			wage_education_final.loc[wage_education_final["Education"]==">College","Diff in PP(%)"],"-",
		 label=">College",
			 color="blue", linewidth=2.2)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in Week Earnings(%)")
plt.legend()
plt.show()
	
###Industry###

wage_industry=CPS_wage.loc[CPS_wage["IND"]!=99999,:]	
wage_industry=pd.merge(wage_industry,tele_ind.reset_index()[["IND","Industry"]],on="IND",how="inner")

wm_i = lambda x: np.average(x, weights=wage_industry.loc[x.index, "WTFINL"])
wage_industry_rate=wage_industry.groupby(["YEAR","MONTH","Industry"]).agg(wage_rate=("EARNWEEK", wm_i))#.reset_index(level="industry")
wage_industry_rate=wage_industry_rate[:64]
wage_industry_rate=wage_industry_rate.reset_index()
wage_industry_rate_1=wage_industry_rate.loc[wage_industry_rate['Industry'] == "Physical",:]
wage_industry_rate_2=wage_industry_rate.loc[wage_industry_rate['Industry'] == "Dematerialized",:]
	

mod331 = sm.tsa.statespace.SARIMAX(wage_industry_rate_1.iloc[:24,3],
	                                order=(1, 1, 0),
	                                seasonal_order=(0, 1, 0, 12),
									enforce_stationarity=True,enforce_invertibility=True)
results331 = mod331.fit()
#print(results331.summary().tables[1])
pred_uc_331 = pd.DataFrame(results331.get_forecast(steps=8).predicted_mean,columns=["predicted_wage"])
pred_uc_331["YEAR"]=2020
pred_uc_331["Industry"]="Physical"
pred_uc_331["MONTH"]=None
for i in range(8):
	pred_uc_331.iloc[i,3]=i+1
	
mod332 = sm.tsa.statespace.SARIMAX(wage_industry_rate_2.iloc[:24,3],
	                                order=(1, 1, 0),
	                                seasonal_order=(0, 1, 0, 12),
									enforce_stationarity=True,enforce_invertibility=True)
results332 = mod332.fit()
#print(results332.summary().tables[1])
pred_uc_332 = pd.DataFrame(results332.get_forecast(steps=8).predicted_mean,columns=["predicted_wage"])
pred_uc_332["YEAR"]=2020
pred_uc_332["Industry"]="Dematerialized"
pred_uc_332["MONTH"]=None
for i in range(8):
	pred_uc_332.iloc[i,3]=i+1


c_i=len(labels_ind)*12
	
wage_industry_final=pd.merge(wage_industry_rate.iloc[c_i:,:],pred_uc_331,on=["YEAR","MONTH","Industry"],how="outer")
wage_industry_final=pd.merge(wage_industry_final,pred_uc_332,on=["YEAR","MONTH","Industry"],how="outer")
wage_industry_final["predicted_wage"]=wage_industry_final.iloc[c_i:,4].fillna(0)+wage_industry_final.iloc[c_i:,5].fillna(0)
wage_industry_final=wage_industry_final.iloc[:,[0,1,2,3,6]]
	
wage_industry_final["YEAR"]=[str(val) for val in wage_industry_final["YEAR"].values]
wage_industry_final["MONTH"]=[str(val) for val in wage_industry_final["MONTH"].values]
wage_industry_final=wage_industry_final.set_index([wage_industry_final["YEAR"]+"-"+wage_industry_final["MONTH"]]).drop(["YEAR","MONTH"],axis=1)
wage_industry_final["Diff in PP"]=wage_industry_final["wage_rate"].subtract(wage_industry_final["predicted_wage"])
wage_industry_final["Diff in PP(%)"]=(wage_industry_final["wage_rate"].divide(wage_industry_final["predicted_wage"])-1)*100
	

fig, ax = plt.subplots()
ax.plot(wage_industry_final.loc[wage_industry_final["Industry"]=="Physical","Diff in PP"].index,
			wage_industry_final.loc[wage_industry_final["Industry"]=="Physical","Diff in PP"],"-",
			label="Physical",
			color="red", linewidth=2.2)
ax.plot(wage_industry_final.loc[wage_industry_final["Industry"]=="Dematerialized","Diff in PP"].index,
			wage_industry_final.loc[wage_industry_final["Industry"]=="Dematerialized","Diff in PP"],"-",
			label="Dematerialized",
			color="blue", linewidth=2.2)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in Week Earnings")
plt.legend()
plt.show()
	
fig, ax = plt.subplots()
ax.plot(wage_industry_final.loc[wage_industry_final["Industry"]=="Physical","Diff in PP(%)"].index,
			wage_industry_final.loc[wage_industry_final["Industry"]=="Physical","Diff in PP(%)"],"-",
		 label="Physical",
			 color="red", linewidth=2.2)
ax.plot(wage_industry_final.loc[wage_industry_final["Industry"]=="Dematerialized","Diff in PP(%)"].index,
			wage_industry_final.loc[wage_industry_final["Industry"]=="Dematerialized","Diff in PP(%)"],"-",
		 label="Dematerialized",
			 color="blue", linewidth=2.2)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in Week Earnings(%)")
plt.legend()
plt.show()

### OCCUPATION ###

wage_occupation=CPS_wage.loc[CPS_wage["OCC"]!=99999,:]	
wage_occupation=pd.merge(wage_occupation,tele_occ.reset_index()[["OCC","Occupation"]],on="OCC",how="inner")

wm_i = lambda x: np.average(x, weights=wage_occupation.loc[x.index, "WTFINL"])
wage_occupation_rate=wage_occupation.groupby(["YEAR","MONTH","Occupation"]).agg(wage_rate=("EARNWEEK", wm_i))#.reset_index(level="occupation")
wage_occupation_rate=wage_occupation_rate[:96]
wage_occupation_rate=wage_occupation_rate.reset_index()
wage_occupation_rate_1=wage_occupation_rate.loc[wage_occupation_rate['Occupation'] == "Physical",:]
wage_occupation_rate_2=wage_occupation_rate.loc[wage_occupation_rate['Occupation'] == "Middle",:]
wage_occupation_rate_3=wage_occupation_rate.loc[wage_occupation_rate['Occupation'] == "Dematerialized",:]
	
mod341 = sm.tsa.statespace.SARIMAX(wage_occupation_rate_1.iloc[:24,3],
	                                order=(1, 1, 0),
	                                seasonal_order=(0, 1, 0, 12),
									enforce_stationarity=True,enforce_invertibility=True)
results341 = mod341.fit()
#print(results341.summary().tables[1])
pred_uc_341 = pd.DataFrame(results341.get_forecast(steps=8).predicted_mean,columns=["predicted_wage"])
pred_uc_341["YEAR"]=2020
pred_uc_341["Occupation"]="Physical"
pred_uc_341["MONTH"]=None
for i in range(8):
	pred_uc_341.iloc[i,3]=i+1
	
mod342 = sm.tsa.statespace.SARIMAX(wage_occupation_rate_2.iloc[:24,3],
	                                order=(1, 1, 0),
	                                seasonal_order=(0, 1, 0, 12),
									enforce_stationarity=True,enforce_invertibility=True)
results342 = mod342.fit()
#print(results342.summary().tables[1])
pred_uc_342 = pd.DataFrame(results342.get_forecast(steps=8).predicted_mean,columns=["predicted_wage"])
pred_uc_342["YEAR"]=2020
pred_uc_342["Occupation"]="Middle"
pred_uc_342["MONTH"]=None
for i in range(8):
	pred_uc_342.iloc[i,3]=i+1

mod343 = sm.tsa.statespace.SARIMAX(wage_occupation_rate_3.iloc[:24,3],
	                                order=(1, 1, 0),
	                                seasonal_order=(0, 1, 0, 12),
									enforce_stationarity=True,enforce_invertibility=True)
results343 = mod343.fit()
#print(results343.summary().tables[1])
pred_uc_343 = pd.DataFrame(results343.get_forecast(steps=8).predicted_mean,columns=["predicted_wage"])
pred_uc_343["YEAR"]=2020
pred_uc_343["Occupation"]="Dematerialized"
pred_uc_343["MONTH"]=None
for i in range(8):
	pred_uc_343.iloc[i,3]=i+1
	
c_o=len(labels_occ)*12
	
wage_occupation_final=pd.merge(wage_occupation_rate.iloc[c_o:,:],pred_uc_341,on=["YEAR","MONTH","Occupation"],how="outer")
wage_occupation_final=pd.merge(wage_occupation_final,pred_uc_342,on=["YEAR","MONTH","Occupation"],how="outer")
wage_occupation_final=pd.merge(wage_occupation_final,pred_uc_343,on=["YEAR","MONTH","Occupation"],how="outer")
wage_occupation_final["predicted_wage_total"]=wage_occupation_final.iloc[c_o:,4].fillna(0)+wage_occupation_final.iloc[c_o:,5].fillna(0)+wage_occupation_final.iloc[c_o:,6].fillna(0)
wage_occupation_final=wage_occupation_final.iloc[:,[0,1,2,3,7]]
	
wage_occupation_final["YEAR"]=[str(val) for val in wage_occupation_final["YEAR"].values]
wage_occupation_final["MONTH"]=[str(val) for val in wage_occupation_final["MONTH"].values]
wage_occupation_final=wage_occupation_final.set_index([wage_occupation_final["YEAR"]+"-"+wage_occupation_final["MONTH"]]).drop(["YEAR","MONTH"],axis=1)
wage_occupation_final["Diff in PP"]=wage_occupation_final["wage_rate"].subtract(wage_occupation_final["predicted_wage_total"])
wage_occupation_final["Diff in PP(%)"]=(wage_occupation_final["wage_rate"].divide(wage_occupation_final["predicted_wage_total"])-1)*100

fig, ax = plt.subplots()
ax.plot(wage_occupation_final.loc[wage_occupation_final["Occupation"]=="Physical","Diff in PP"].index,
			wage_occupation_final.loc[wage_occupation_final["Occupation"]=="Physical","Diff in PP"],"-",
			label="Physical",
			color="red", linewidth=2.2)
ax.plot(wage_occupation_final.loc[wage_occupation_final["Occupation"]=="Middle","Diff in PP"].index,
			wage_occupation_final.loc[wage_occupation_final["Occupation"]=="Middle","Diff in PP"],"-",
			label="Middle",
			color="gray", linewidth=2.2)
ax.plot(wage_occupation_final.loc[wage_occupation_final["Occupation"]=="Dematerialized","Diff in PP"].index,
			wage_occupation_final.loc[wage_occupation_final["Occupation"]=="Dematerialized","Diff in PP"],"-",
			label="Dematerialized",
			color="blue", linewidth=2.2)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in Week Earnings")
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(wage_occupation_final.loc[wage_occupation_final["Occupation"]=="Physical","Diff in PP(%)"].index,
			wage_occupation_final.loc[wage_occupation_final["Occupation"]=="Physical","Diff in PP(%)"],"-",
			label="Physical",
			color="red", linewidth=2.2)
ax.plot(wage_occupation_final.loc[wage_occupation_final["Occupation"]=="Middle","Diff in PP(%)"].index,
			wage_occupation_final.loc[wage_occupation_final["Occupation"]=="Middle","Diff in PP(%)"],"-",
			label="Middle",
			color="gray", linewidth=2.2)
ax.plot(wage_occupation_final.loc[wage_occupation_final["Occupation"]=="Dematerialized","Diff in PP(%)"].index,
			wage_occupation_final.loc[wage_occupation_final["Occupation"]=="Dematerialized","Diff in PP(%)"],"-",
			label="Dematerialized",
			color="blue", linewidth=2.2)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in Week Earnings(%)")
plt.legend()
plt.show()

##############################################################
######### POLAND #############################################
##############################################################
unemployment_pl=pd.read_excel("unemploymentPL.xlsx",skiprows=20,skipfooter=22)
unemployment_pl=unemployment_pl.iloc[2:5,:]
unemployment_pl.columns=["YEAR",1,2,3,4,5,6,7,8,9,10,11,12]
unemployment_pl=pd.melt(unemployment_pl,id_vars=["YEAR"],value_vars=[1,2,3,4,5,6,7,8,9,10,11,12],var_name="MONTH",value_name="unemployment_rate").sort_values(["YEAR","MONTH"])

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
for parameters in pdq:
	try:
		mod=ARIMA(unemployment_pl.iloc[:24,2],order=parameters)
		results=mod.fit()
		print('ARIMA{}- AIC:{}'.format(parameters,results.aic))
	except:
		continue
		   

mod1pl = ARIMA(unemployment_pl.iloc[:24,2],
	                                order=(0, 1, 1))
results1pl = mod1pl.fit()
#print(results1pl.summary().tables[1])
pred_uc1pl = pd.DataFrame(results1pl.forecast(steps=8)[0])
pred_uc1pl.index=[(2020,  1),
	            (2020,  2),
	            (2020,  3),
	            (2020,  4),
	            (2020,  5),
	            (2020,  6),
	            (2020,  7),
	            (2020,  8)]
unemployment_pl=unemployment_pl.set_index(["YEAR","MONTH"])
unemployment_final=pd.concat([unemployment_pl[12:24],unemployment_pl[24:],pred_uc1pl],axis=1,join="outer")
unemployment_final.columns=["Data 2019","Data 2020","Prediction"]
unemployment_final=unemployment_final.reset_index()
unemployment_final["level_0"]=[str('{:.0f}'.format(val)) for val in unemployment_final["level_0"].values]
unemployment_final["level_1"]=[str(val) for val in unemployment_final["level_1"].values]
unemployment_final=unemployment_final.set_index([unemployment_final["level_0"]+"-"+unemployment_final["level_1"]]).drop(["level_0","level_1"],axis=1)
unemployment_final["Diff in PP"]=unemployment_final["Data 2020"].subtract(unemployment_final["Prediction"])
unemployment_final["Diff in PP(%)"]=(unemployment_final["Data 2020"].divide(unemployment_final["Prediction"])-1)*100

	
fig, ax = plt.subplots()
ax.plot(unemployment_final.index,unemployment_final["Data 2019"],"ro",
			 label="Data 2019",
			 color="blue", markersize=6)
ax.plot(unemployment_final.index,unemployment_final["Data 2020"],"ro",
			 label="Data 2020",
			 color="grey", markersize=6)
ax.plot(unemployment_final.index,unemployment_final["Prediction"],"ro",
			 label="Prediction",
			 color="red",markersize=6)
plt.xticks(ticks1)
plt.ylabel("Unemployment rate")
plt.legend()
plt.show()
	
fig, ax = plt.subplots()
ax.plot(unemployment_final.index,unemployment_final["Data 2020"],"ro",
			 label="Data 2020",
			 color="grey", markersize=6)
ax.plot(unemployment_final.index,unemployment_final["Prediction"],"ro",
			 label="Prediction",
			 color="red",markersize=6)
plt.xticks(ticks2)
plt.ylabel("Unemployment rate")
plt.legend()
plt.show()
	
fig, ax = plt.subplots()
ax.plot(unemployment_final.index,unemployment_final["Diff in PP"],"-",
			 color="black", linewidth=3)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in unemployment rate")
plt.show()
	
fig, ax = plt.subplots()
ax.plot(unemployment_final.index,unemployment_final["Diff in PP(%)"],"-",
			 color="black", linewidth=3)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in unemployment rate(%)")
plt.show()


#### WAGES ####
wages=pd.read_excel("wagesPL.xlsx",skiprows=100,header=None,skipfooter=36,sheet_name="Dane miesięczne",usecols=[0,1,2])
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
for parameters in pdq:
	try:
		mod=ARIMA(wages.iloc[:24,2],order=parameters)
		results=mod.fit()
		print('ARIMA{}- AIC:{}'.format(parameters,results.aic))
	except:
		continue
		   

mod2pl = ARIMA(wages.iloc[:24,2], order=(1, 0, 1))
results2pl = mod2pl.fit()
#print(results2pl.summary().tables[1])
pred_uc2pl = pd.DataFrame(results2pl.forecast(steps=8)[0])
pred_uc2pl.index=[(2020,  "I"),
	            (2020,  "II"),
	            (2020,  "III"),
	            (2020,  "IV"),
	            (2020,  "V"),
	            (2020,  "VI"),
	            (2020,  "VII"),
	            (2020,  "VIII")]
wages.columns=["YEAR","MONTH","WAGE"]
wages=wages.set_index(["YEAR","MONTH"])
wages_final=pd.concat([wages[12:24],wages[24:],pred_uc2pl],axis=1,join="outer")
wages_final.columns=["Data 2019","Data 2020","Prediction"]
wages_final=wages_final.reset_index()
wages_final["level_0"]=[str('{:.0f}'.format(val)) for val in wages_final["level_0"].values]
wages_final["level_1"]=[str(val) for val in wages_final["level_1"].values]
wages_final=wages_final.set_index([wages_final["level_0"]+"-"+wages_final["level_1"]]).drop(["level_0","level_1"],axis=1)
wages_final["Diff in PP"]=wages_final["Data 2020"].subtract(wages_final["Prediction"])
wages_final["Diff in PP(%)"]=(wages_final["Data 2020"].divide(wages_final["Prediction"])-1)*100

		
fig, ax = plt.subplots()
ax.plot(wages_final.index,wages_final["Data 2019"],"ro",
			 label="Data 2019",
			 color="blue", markersize=6)
ax.plot(wages_final.index,wages_final["Data 2020"],"ro",
			 label="Data 2020",
			 color="grey", markersize=6)
ax.plot(wages_final.index,wages_final["Prediction"],"ro",
			 label="Prediction",
			 color="red",markersize=6)
plt.xticks(ticks1)
plt.ylabel("Wages")
plt.legend()
plt.show()
	
fig, ax = plt.subplots()
ax.plot(wages_final.index,wages_final["Data 2020"],"ro",
			 label="Data 2020",
			 color="grey", markersize=6)
ax.plot(wages_final.index,wages_final["Prediction"],"ro",
			 label="Prediction",
			 color="red",markersize=6)
plt.xticks(ticks2)
plt.ylabel("Wages")
plt.legend()
plt.show()
	
fig, ax = plt.subplots()
ax.plot(wages_final.index,wages_final["Diff in PP"],"-",
			 color="black", linewidth=3)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in wages")
plt.show()
	
fig, ax = plt.subplots()
ax.plot(wages_final.index,wages_final["Diff in PP(%)"],"-",
			 color="black", linewidth=3)
plt.axhline(y=0,linewidth=0.5, color="grey")
plt.xticks(ticks2)
plt.ylabel("Difference in wages wages(%)")
plt.show()