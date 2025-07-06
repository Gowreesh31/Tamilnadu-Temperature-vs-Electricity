import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import qrcode
from PIL import Image
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

print('தமிழ்நாடு புள்ளி விவரம்')
population=pd.read_csv('Population_sheet.csv')
gsdpGva=pd.read_csv('GSDP_GVA_Sheet.csv')
electricDist=pd.read_csv('Elec_Distri_Portfolio_Sheet.csv')
electricCons=pd.read_csv('Cat_Elect_Cons_Share_Sheet.csv')
capPotential=pd.read_csv('Cap_Potential_Sheet.csv')
lossTrend=pd.read_csv('At&C_Loss_Trends_Sheet.csv')
revenue=pd.read_csv('Average_Cost_&_Revenue_Sheet.csv')
geoCap=pd.read_csv('Geo_Cap_Gen_Sheet.csv')
electricDemand=pd.read_csv('Monthly_Peak_Demand_Sheet.csv')
perCapita=pd.read_csv('Per_Capita_Sheet.csv')
powerPurchase=pd.read_csv('Power_Purchase_Cost_Sheet.csv')
profitLoss=pd.read_csv('Profit_and_Loss_Sheet.csv')
sectoralGva=pd.read_csv('Sectoral_GVA_Sheet.csv')
tariffRates=pd.read_csv('Tariff_Rates_Sheet.csv')
stategdp=pd.read_csv('State_GDP.csv')
nationalgdp=pd.read_csv('National_GDP.csv')
tempData=pd.read_csv('temperature.csv')
temp2016=pd.read_excel('temperature 2016.xlsx')
comGDP=pd.read_csv('StateGDPMapIndia.csv')
APtemp=pd.read_csv('AP2018.csv')
APtemp=pd.concat([APtemp,pd.read_csv('AP2019.csv')],ignore_index=True)
APtemp=pd.concat([APtemp,pd.read_csv('AP2020.csv')],ignore_index=True)
APtemp=pd.concat([APtemp,pd.read_csv('AP2021.csv')],ignore_index=True)
APtemp=pd.concat([APtemp,pd.read_csv('AP2022.csv')],ignore_index=True)
APtemp=pd.concat([APtemp,pd.read_csv('AP2023.csv')],ignore_index=True)
tempTime=pd.concat([temp2016,pd.read_excel('temperature 2017.xlsx')],ignore_index=True)
tempTime=pd.concat([tempTime,pd.read_excel('temperature 2018.xlsx')],ignore_index=True)
tempTime=pd.concat([tempTime,pd.read_excel('temperature 2019.xlsx')],ignore_index=True)
tempTime=pd.concat([tempTime,pd.read_excel('temperature 2020.xlsx')],ignore_index=True)
tempTime=pd.concat([tempTime,pd.read_excel('temperature 2021.xlsx')],ignore_index=True)
tempTime=pd.concat([tempTime,pd.read_excel('temperature 2022.xlsx')],ignore_index=True)
tempTime=pd.concat([tempTime,pd.read_excel('temperature 2023.xlsx')],ignore_index=True)

def populationPercent():
    total=[0,0]
    total[0]=(100-5.96)
    total[1]=5.96
    label=['Others States','Tamil Nadu']
    plt.figure(figsize=(18.2,11.2),dpi=100)
    plt.pie(total,labels=label,autopct='%1.1f%%')
    plt.title('Population percent of TamilNadu in India')
    plt.savefig('population.jpg')
    plt.show()

#populationPercent()

def ruralUrbanComparison():
    total=[0,0]
    total[0]=float(population['State Urban Population Percentage (%)'].iloc[0])
    total[1]=float(population['State Rural Population Percentage (%)'].iloc[0])
    label=['Urban','Rural']
    plt.figure(figsize=(18.2,11.2),dpi=100)
    plt.pie(total,labels=label,autopct='%1.1f%%')
    plt.title('Urban Rural Population Comparison')
    plt.savefig('urbanRural.jpg')
    plt.show()
    
#ruralUrbanComparison()


def sectorWiseConsumption():
    
    for i in range(0,len(electricCons),7):
        list1=electricCons.iloc[i:i+7]
        s=list1['Year'].iloc[0]
        X=list(list1['Sector'])
        plt.figure(figsize=(18.2,11.2),dpi=100)
        Y=list(list1['Consumption (%)'])
        plt.pie(Y,labels=X,autopct='%.2f%%')
        t='Sector wise electricity consumption in '+s
        plt.title(t)
        t=t+'.jpg'
        plt.savefig(t)
        plt.show()
    
#sectorWiseConsumption()

def electricDem():
    list1=electricDemand['Peak Demand (in MW)']
    list1=list(list1)
    list1.reverse()
    plt.figure(figsize=(18.2,11.2),dpi=100)
    plt.title('Electricity Demand during 2012 - 2024')
    plt.xlabel('Years')
    plt.ylabel('Demand in MW')
    plt.plot(list1)
    plt.savefig('demand2015-2024.jpg')
    plt.show()
    
    list1=electricDemand['Peak Demand (in MW)']
    mat=np.zeros((10,12))
    month=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    year=['2024','2023','2022','2021','2020','2019','2018','2017','2016','2015']
    for i in range(10):
        for j in range(12):
            tmp=list1.iloc[(i*12)+j]
            mat[i][j]=round(tmp,1)

    plt.figure(figsize=(18.2,11.2),dpi=100)
    sns.heatmap(data=mat,annot=True,fmt='.1f',cmap='Reds',xticklabels=month,yticklabels=year)
    plt.title('Electricity Demand')
    plt.savefig('electricDemand.jpg')
    plt.show()    
    
electricDem()


def electricityProduction():

    src=[]
    for i in range(10):
        lst=[]
        for j in range(8):
            lst.append(geoCap['Generation (in MU)'].iloc[(i*8)+j])
            if i==0:
                src.append(geoCap['Source'].iloc[j])
        s=geoCap['Year'].iloc[(i*8)+j]
        s='Power production in '+s
        plt.figure(figsize=(18.2,11.2),dpi=100)
        plt.pie(lst,labels=src,autopct='%1.1f%%')
        plt.title(s)
        s=s+'.jpg'
        plt.savefig(s)
        plt.show()
            

#electricityProduction()

def comparison():

    list1=electricDemand['Peak Demand (in MW)']
    month=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    cons2k15=[]
    cons2k24=[]
    
    for i in range(12):
        cons2k24.append(list1.iloc[i])
        cons2k15.append(list1.iloc[(9*12)+i])

    axis=np.arange(12)
    plt.figure(figsize=(18.2,11.2),dpi=100)
    plt.bar(axis-0.2,cons2k24,0.4,label='2024')
    plt.bar(axis+0.2,cons2k15,0.4,label='2015')
    plt.xticks(axis,month)
    plt.xlabel('Months')
    plt.ylabel('Consumption in MW')
    plt.title('Comparison of power consumption in 2015 and 2024')
    plt.legend()
    plt.savefig('comparison2015-24.jpg')
    plt.show()
#comparison()


def profitAndLoss():

    list1=list(profitLoss['Profit and Loss (in Rs Crores)'])
    list2=profitLoss['Year']
    x=np.arange(len(list2))
    plt.figure(figsize=(18.2,11.2),dpi=100)
    plt.bar(x,list1)

    for index,value in enumerate(list1):
        plt.text(index,value-4,str(value),ha='center',va='top',fontsize=10)

    plt.title('Profit and Loss of TNEB')
    plt.xticks(x,list2,rotation=45)
    plt.savefig('profitAndLoss.jpg')
    plt.show()

profitAndLoss()

qr=qrcode.QRCode(version=3, box_size=20, border=10, error_correction=qrcode.constants.ERROR_CORRECT_H)
data='https://drive.google.com/drive/folders/103y_YJ52X9Ju7kM9IKTT34QDHEG94lsA?usp=drive_link'
qr.add_data(data)
qr.make(fit=True)
img=qr.make_image(fill_color='black',back_color='white')
img.save('qr_code.png')
img.show()

def predictDemand():

    
    SARIMAXmodel = SARIMAX(electricDemand['Peak Demand (in MW)'], order=(1,1,1), seasonal_order=(1,1,0,12))
    SARIMAXmodel = SARIMAXmodel.fit()

    electricDemand.index = pd.to_datetime(electricDemand.index)
    list1 = [2025, 2026, 2027, 2028, 2029, 2030]
    forecast_steps = len(list1)

    y_pred = SARIMAXmodel.get_forecast(steps=forecast_steps)
    y_pred_df = y_pred.conf_int(alpha=0.05)

    y_pred_df["Predictions"] = y_pred.predicted_mean

    y_pred_df.index = list1

    plt.plot(y_pred_df["Predictions"], color='Blue', label='SARIMA Predictions')
    plt.title('Electricity Prediction')
    plt.legend()
    plt.show()

    print(y_pred_df["Predictions"])

predictDemand()

def temperature():

    temp=tempData['Temperature']
    month=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    year=['2023','2022','2021','2020','2019','2018','2017','2016','2015']
    mat=np.zeros((9,12))
    arr=[31,28,31,30,31,30,31,31,30,31,30,31]
    lst1=[]
    print(APtemp)
    for i in range(6):
        for j in range(12):
            sum1=0
            for k in range(arr[j]):
                sum1+=APtemp['Maximum Temprature (in C)'][sum(arr[:j])+k]
            lst1.append(sum1/arr[j])
                
    
    for i in range(9):
        for j in range(12):
            mat[i][j]=temp.iloc[(i*12)+j]

    plt.figure(figsize=(18.2,11.2),dpi=100)
    sns.heatmap(data=mat,annot=True,fmt='.1f',cmap='Reds',xticklabels=month,yticklabels=year)
    plt.title('Temperature in Celsius')
    plt.savefig('temperatureHeatmap.jpg')
    plt.show()


    plt.figure(figsize=(18.2,11.2),dpi=100)
    plt.plot(temp)
    plt.plot(lst1)
    plt.title('TamilNadu and Andhra Pradesh Comparison')
    plt.legend(['Tamil Nadu','Andhra Pradesh'])
    plt.savefig('TemperatureComparison.jpg')
    plt.show()


    plt.figure(figsize=(18.2,11.2),dpi=100)
    plt.plot(list1,'--r')
    plt.plot(list2,'--b')
    plt.legend(['TamilNadu','Andhra Pradesh'])
    plt.title('Prediction for temperature')
    plt.plot(temp,linewidth=3)
    plt.plot(lst1,linewidth=3)
    plt.savefig('TemperaturePrediction.jpg')
    plt.show()
    
    

#temperature()
    
def standardization():
    list1 = (electricDemand['Peak Demand (in MW)'].iloc[:-12]).to_list()
    list2 = tempData['Temperature'].to_list()
    list1.reverse()

    dict1 = {'Demand': list1, 'Temperature': list2}
    df = pd.DataFrame(dict1)
    
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    df = df.tolist()
    labelList = []

    for i in range(108):
        labelList.append(tempData['Month'].iloc[i] + str(tempData['Year'].iloc[i]))

    for i in range(len(df)):
        list1[i] = df[i][0]
        list2[i] = df[i][1]

    plt.figure(figsize=(18.2, 11.2), dpi=100)
    plt.plot(list1, label='Electric Demand')
    plt.plot(list2, label='Temperature')
    
    X1 = np.arange(108)
    plt.xticks(X1[::12], labelList[::12], rotation=45)  # Show only every 12th label, rotated
    plt.legend()
    plt.title('Electricity Consumption VS Temperature')
    
    list1 = np.array(list1)
    list2 = np.array(list2)
    corrmat = np.corrcoef(list1, list2)
    
    plt.savefig('TemperatureVSElectricity.jpg')
    plt.show()

standardization()


def modelTemperature():

    tempDate=tempTime[['Date','Peak Demand (in MW)']].copy()
    tempDate.loc[:,'Date']=pd.to_datetime(tempDate['Date'])
    df1=tempDate.set_index('Date')
    monthlyTemp=df1.resample('M').mean()
    #plt.plot(monthlyTemp['Peak Demand (in MW)'])
    #plt.show()
    #plot_acf(monthlyTemp)
    #plot_pacf(monthlyTemp)
    #plt.show()
    

    model=SARIMAX(monthlyTemp,order=(1,1,1),seasonal_order=(1,1,1,12))
    results=model.fit()

    forecast=results.get_forecast(steps=12)
    forecastMean=forecast.predicted_mean
    forecast_ci=forecast.conf_int()

    plt.plot(monthlyTemp, label='Observed', color='blue')
    plt.plot(forecastMean, label='Forecast', color='red')
    plt.fill_between(forecast_ci.index, forecast_ci.iloc[:,0], forecast_ci.iloc[:,1], color='pink', alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Peak Demand (in MW)')
    plt.title('SARIMAX Forecast of Peak Demand')
    plt.legend()
    plt.show()
    
#modelTemperature()

def GDPanalysis():

    gdpYear=stategdp['Price (in Rs.Lakh Crore)'].to_list()
    gdpYearnational=nationalgdp['Price (in Rs.Lakh Crore)'].to_list()
    
    gdpYear=np.array(gdpYear[:len(gdpYear)-1])
    gdpYearnational=np.array(gdpYearnational[:len(gdpYearnational)-1])
    percent=(gdpYear*100)/gdpYearnational

    dict1={'State':gdpYear,'National':gdpYearnational}
    df=pd.DataFrame(dict1)
    plt.figure(figsize=(18.2,11.2),dpi=100)
    plt.title('TAMILNADU GDP')
    plt.plot(gdpYear)
    plt.savefig('TNGDP.jpg')
    plt.show()
    
    plt.figure(figsize=(18.2,11.2),dpi=100)
    plt.plot(gdpYearnational)
    plt.title('INDIA GDP')
    plt.show()
    
    scaler=StandardScaler()
    df=scaler.fit_transform(df)
    df=df.tolist()
    
    list1=np.zeros(len(df))
    list2=np.zeros(len(df))
    
    for i in range(len(df)):
        list1[i]=df[i][0]
        list2[i]=df[i][1]
        
    plt.figure(figsize=(18.2,11.2),dpi=100)
    plt.plot(list1)
    plt.plot(list2)
    plt.legend(['TamilNadu GDP','India GDP'])
    
    t1=np.arange(len(df))
    t2=np.arange(5)
    
    plt.xticks(t1,labels=t1)
    plt.yticks(t2,labels=t2)
    plt.savefig('GDPGrowth.jpg')
    plt.show()

    plt.figure(figsize=(18.2,11.2),dpi=100)
    plt.bar(t1-0.2,height=gdpYear,width=0.4)
    plt.bar(t1+0.2,height=gdpYearnational,width=0.4)
    plt.title('Comparison between India and TamilNadu GDP')
    plt.legend(['TamilNadu GDP','India GDP'])
    plt.savefig('ComparisonWithIndiaGDP.jpg')
    plt.show()


#GDPanalysis()


def GDPComparison():

    temp=comGDP['Price (in Rs.Lakh Crore)']
    arr1=np.zeros(39)
    lst1=[140,179,293]
    for i in range(3):
        for j in range(13):
            arr1[i*13+j]=temp.iloc[(lst1[i])+j]
    TN=arr1[:13]
    KA=arr1[13:26]
    MH=arr1[26:39]
    date=[i for i in range(2011,2024)]
    list1=np.arange(13)
    plt.figure(figsize=(18.2,11.2),dpi=100)
    plt.bar(list1-0.2,height=TN,width=0.2)
    plt.bar(list1,height=KA,width=0.2)
    plt.bar(list1+0.2,height=MH,width=0.2)
    plt.legend(['TamilNadu','Maharastra','Karnataka'])
    plt.xticks(list1,date)
    plt.xlabel('Year')
    plt.title('GDP Comparison with Other States')
    plt.savefig('ComparisonWithOtherStates.jpg')
    plt.show()
    
GDPComparison()
