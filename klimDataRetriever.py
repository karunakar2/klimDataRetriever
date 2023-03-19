import json
import os

import pandas as pd
import numpy as np

#the actual main function
def main(args):
    print('the module on command line has reduced functionality')
    print('this downloads a fresh copy of files onto your machine')
    print('the files have tables in wide format for your perusal')
    if len(args)<2:
        print('Only the starting month of the hydrological year can be specified')
        print('select from ','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
        print('Using default Jul for now')
        kData = klimDataRetriever(cached=False)
    else :
        kData = klimDataRetriever(hydroYearStartMonth=args[1],cached=False)
    
    
    print('Please check the csv files in the directory where this script is run from')

#backup in case main is called
if __name__ == '__main__':
    main(sys.argv[1:])

##--------------------------------------------------------------------------------------------------------    
class klimDataRetriever:
    _climDataHeader = ['year','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] #'hydYrAvg'
    _climateWindows = {
        'JJA' : ['Jun','Jul','Aug'],
        'SON' : ['Sep','Oct','Nov'],
        'DJF' : ['Dec','Jan','Feb'],
        'MAM' : ['Mar','Apr','May']
    }
    #self._climHydYrAvg = True
    _climHydYrAvg = {
        'year1' : ['Jul','Aug','Sep','Oct','Nov','Dec'],
        'year2' : ['Jan','Feb','Mar','Apr','May','Jun']
    }
    
    _climMacroDf = pd.DataFrame()
    
    _useCached = True
    _hydroYearMDString = '-07-01'
    
    _samUrl = 'http://www.nerc-bas.ac.uk/public/icd/gjma/newsam.1957.2007.txt'
    _ipoUrl = 'https://psl.noaa.gov/data/timeseries/IPOTPI/tpi.timeseries.ersstv5.filt.data'
    
    _debug = False
        
    def __init__(self, seasons:dict=None, hydroYearStartMonth:str=None, SAM:str='marshal', IPOfilt:bool = True, 
                 cached:bool=True, helpMe:bool=False, debug:bool=False):
        myMonths = self._climDataHeader[1:]

        if helpMe:
            self.citation()
            print('hydroYear needs a month keyword from ')
            print(myMonths)
            print('if not specified, it starts from July and ends by June of the following year')

            #seasons input
            print('The seasons need a dictionary which looks like')
            print(json.dumps(self._climateWindows))

        else:
            if debug:
                self._debug = True
                self.citation()

            if seasons != None:
                #do sanity check here before proceeding
                self._climateWindows = seasons

            if SAM != 'marshal': #default marshall
                try :
                    os.remove("aao.csv")
                except : pass

                print('switching SAM source')
                self._samUrl = 'https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/aao/monthly.aao.index.b79.current.ascii.table'

            if IPOfilt: #default true
                self._ipoUrl = 'https://psl.noaa.gov/data/timeseries/IPOTPI/tpi.timeseries.ersstv5.data'
            else :
                os.remove("ipo.csv")
                print('fetching filtered IPO dataset')


            if hydroYearStartMonth != None:
                
                if hydroYearStartMonth not in myMonths:
                    raise Exception(hydroYearStartMonth ," can't be used, try one from \n",(',').join(myMonths))

                self._climHydYrAvg['year1'] = myMonths[myMonths.index(hydroYearStartMonth):]
                self._climHydYrAvg['year2'] = myMonths[:myMonths.index(hydroYearStartMonth)]
                temp = f'{str(myMonths.index(hydroYearStartMonth) + 1)}-01'
                if len(temp) < 5:
                    temp = '0'+temp
                self._hydroYearMDString = '-'+temp
                if debug:
                    print(json.dumps(self._climHydYrAvg))
                    print(self._hydroYearMDString)
            if not cached:
                self._useCached = cached
    
    def citation(self):
        print('Cite this module as Karunakar, "HBRC (2021): A python module to fetch Climate indices-Ver $version_number"')
        print('Along with the link where the code is sourced from, this will help users to check if there are any updates')
        
        print('for others please refer to')
        print('SOI: https://crudata.uea.ac.uk/cru/data/soi/')
        print('IPO: https://psl.noaa.gov/data/timeseries/IPOTPI/')
        print('SAM : https://climatedataguide.ucar.edu/climate-data/marshall-southern-annular-mode-sam-index-station-based')
        print('SPSD : https://www.hbrc.govt.nz/assets/Document-Library/Publications-Database/4914-RM17-01-Hawkes-Bay-Seasonal-Forecasting.pdf')
        print('IOD: https://psl.noaa.gov/gcos_wgsp/Timeseries/DMI/')
        print('STR: HBRC (based on Cai et.al. 2011) || https://www.metoffice.gov.uk/hadobs/hadslp2/')
    
        
    def getSelectSeries(self, series : str ='hydYrAvg') -> pd.DataFrame():
        if not isinstance(series, str):
            raise Exception ('Expecting a string in series argument, recieved ', type(series))
        thisSeries = [series, 'datetime', 'year']
        if self._climMacroDf.empty:
            self.getClimData()

        c = self._climMacroDf #.copy()  #dont need a copy here
        df = c.loc[:, pd.IndexSlice[thisSeries, :]]
        df.columns = df.columns.droplevel()
        newCols = []
        for idx, thisCol in enumerate(df.columns):
            if thisCol == '':
                if type(df.iloc[1, idx]) is pd.Timestamp:
                    newCols.append('datetime')
                elif type(df.iloc[1, idx]) is np.int64:
                    newCols.append('year')
                else :
                    print(type(df.iloc[1, idx]))
                    newCols.append(thisCol)
            else:
                newCols.append(thisCol)
        df.columns = newCols
        return df
    
    def getClimData(self) -> pd.DataFrame():
        concatList = []
        hdrKeys = []
        try:
            temp1 = self.getSOIdata()
            concatList.append(temp1.set_index('year'))
            hdrKeys.append('soi')
        except Exception as er:
            if self._debug:
                print(er)
            print('Cannot fetch SOI data')
        try:
            temp2 = self.getIPOdata()
            concatList.append(temp2.set_index('year'))
            hdrKeys.append('ipo')
        except Exception as er:
            if self._debug:
                print(er)
            print('Cannot fetch IPO data')
        try:
            temp3 = self.getSPSDdata()
            concatList.append(temp3.set_index('year'))
            hdrKeys.append('sps')
        except Exception as er:
            if self._debug:
                print(er)
            print('Cannot fetch SPS data')
        try:
            temp4 = self.getAAOdata()
            concatList.append(temp4.set_index('year'))
            hdrKeys.append('aao')
        except Exception as er:
            if self._debug:
                print(er)
            print('Cannot fetch AAO data')
        try:
            temp5 = self.getIODdata()
            concatList.append(temp5.set_index('year'))
            hdrKeys.append('iod')
        except Exception as er:
            if self._debug:
                print(er)
            print('Cannot fetch IOD data')
        try:
            _,temp6,temp7 = self.getSTRdata()
            concatList.append(temp6.set_index('year'))
            hdrKeys.append('stri')
            concatList.append(temp7.set_index('year'))
            hdrKeys.append('strp')
        except Exception as er:
            if self._debug:
                print(er)
            print('Cannot fetch STR data')
        """
        display(temp1)
        display(temp2)
        display(temp3)
        display(temp4)
        display(temp5)
        display(temp6)
        display(temp7)
        """
        """
        df = (pd.concat([temp1.set_index('year'), temp2.set_index('year'), temp3.set_index('year'), 
                         temp4.set_index('year'), temp5.set_index('year'), temp6.set_index('year'), temp7.set_index('year')], 
                    axis=1, 
                    keys=['soi','ipo','sps','aao','iod','stri','strp'])
                .swaplevel(0,1,axis=1)
                .sort_index(axis=1, ascending=[True, False])
                )
        """
        df = (pd.concat(concatList, axis=1, 
                    keys=hdrKeys)
                .swaplevel(0,1,axis=1)
                .sort_index(axis=1, ascending=[True, False])
                )
        df.drop('datetime', axis=1, level=0, inplace=True)
        df.reset_index(inplace=True)
        #display(df['year'])

        df['datetime'] = df.apply(lambda row: np.datetime64(str(row['year'].values[0])+self._hydroYearMDString), axis=1)
        self._climMacroDf = df
        return df
    
    def getSOIdata(self) -> pd.DataFrame():
        try:
            if not self._useCached:
                raise Exception('please ignore this, discarding locally available file and fetching from source')
            c = pd.read_csv('soi.csv')
            c['datetime']= pd.to_datetime(c['datetime']) #make sure object is datetime
        except Exception as er:

            url = 'https://crudata.uea.ac.uk/cru/data/soi/soi.dat'
            c = pd.read_csv(url, delimiter= '\s+', header=None, na_values=-99.99)
            temp = self._climDataHeader.copy()
            temp.append('Annual')
            c.columns = temp
            #c.dropna(inplace=True)
            c['datetime'] = c.apply(lambda row: np.datetime64(str(int(row['year']))+self._hydroYearMDString), axis=1)

            if self._climHydYrAvg:
                jdAvg = c[self._climHydYrAvg['year1']].mean(axis=1) #july to dec avg #now it is first year
                jjAvg = c[self._climHydYrAvg['year2']].mean(axis=1) #following year
                jjAvg = np.append((jjAvg.values)[1:],[np.nan]) #shift the year
                hydYrAvg = (jdAvg*len(self._climHydYrAvg['year1']) + jjAvg*len(self._climHydYrAvg['year2']))/12  # average
                c = c.assign(hydYrAvg=hydYrAvg)

            for thisSeason in self._climateWindows.keys():
                if thisSeason == 'DJF':
                    temp1 = c[self._climateWindows[thisSeason]].copy()
                    temp1['Dec'] = temp1['Dec'].shift(1) #shift down
                    c = c.assign(temp=temp1[self._climateWindows[thisSeason]].mean(axis=1))
                else:
                    c = c.assign(temp=c[self._climateWindows[thisSeason]].mean(axis=1))
                c = c.rename(columns={"temp": thisSeason})
            c['state'] = c.apply(lambda row: 'ElNino' if row['hydYrAvg']<0 else 'LaNina', axis=1)

            c.to_csv('soi.csv',index=False)

        #print(c['hydYrAvg'].min(),c['hydYrAvg'].max())
        return c

    def getIODdata(self) -> pd.DataFrame():
        try:
            if not self._useCached:
                raise Exception('please ignore this, discarding locally available file and fetching from source')
            c = pd.read_csv('iod.csv')
            c['datetime']= pd.to_datetime(c['datetime']) #make sure object is datetime
        except Exception as er:
            url = 'https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data'
            c = pd.read_csv(url, delimiter= '\s+', header=None, skiprows = 1, skipfooter = 7, engine='python', na_values=-99)
            c.columns = self._climDataHeader.copy()
            c = c.assign(Annual=c.iloc[:,1:12].mean(axis=1))

            if self._climHydYrAvg:
                jdAvg = c[self._climHydYrAvg['year1']].mean(axis=1) #july to dec avg #now it is first year
                jjAvg = c[self._climHydYrAvg['year2']].mean(axis=1) #following year
                jjAvg = np.append((jjAvg.values)[1:],[np.nan]) #shift the year
                hydYrAvg = (jdAvg*len(self._climHydYrAvg['year1']) + jjAvg*len(self._climHydYrAvg['year2']))/12  # average
                c = c.assign(hydYrAvg=hydYrAvg)

            for thisSeason in self._climateWindows.keys():
                if thisSeason == 'DJF':
                    temp1 = c[self._climateWindows[thisSeason]].copy()
                    temp1['Dec'] = temp1['Dec'].shift(1) #shift down
                    c = c.assign(temp=temp1[self._climateWindows[thisSeason]].mean(axis=1))
                else:
                    c = c.assign(temp=c[self._climateWindows[thisSeason]].mean(axis=1))
                c = c.rename(columns={"temp": thisSeason})

            c['state'] = c.apply(lambda row: 'Neg' if row['hydYrAvg']<0 else 'Pos', axis=1)
            c['datetime'] = c.apply(lambda row: np.datetime64(str(row['year'])+self._hydroYearMDString), axis=1)

            c.to_csv('iod.csv',index=False)

        #print(c['Annual'].min(),c['Annual'].max())
        return c
    
    def getIPOdata(self) -> pd.DataFrame():
        try:
            if not self._useCached:
                raise Exception('please ignore this, discarding locally available file and fetching from source')
            c = pd.read_csv('ipo.csv')
            c['datetime']= pd.to_datetime(c['datetime']) #make sure object is datetime
        except Exception as er:
            #print(self._ipoUrl)
            #the latest data is always unfiltered
            c = pd.read_csv(self._ipoUrl, delimiter= '\s+', header=None, skiprows = 1, skipfooter = 11, engine='python', na_values=-99)
            c.columns = self._climDataHeader.copy()
            c = c.assign(Annual=c.iloc[:,1:12].mean(axis=1))

            if self._climHydYrAvg:
                jdAvg = c[self._climHydYrAvg['year1']].mean(axis=1) #july to dec avg #now it is first year
                jjAvg = c[self._climHydYrAvg['year2']].mean(axis=1) #following year
                jjAvg = np.append((jjAvg.values)[1:],[np.nan]) #shift the year
                hydYrAvg = (jdAvg*len(self._climHydYrAvg['year1']) + jjAvg*len(self._climHydYrAvg['year2']))/12  # average
                c = c.assign(hydYrAvg=hydYrAvg)

            for thisSeason in self._climateWindows.keys():
                if thisSeason == 'DJF':
                    temp1 = c[self._climateWindows[thisSeason]].copy()
                    temp1['Dec'] = temp1['Dec'].shift(1) #shift down
                    c = c.assign(temp=temp1[self._climateWindows[thisSeason]].mean(axis=1))
                else:
                    c = c.assign(temp=c[self._climateWindows[thisSeason]].mean(axis=1))
                c = c.rename(columns={"temp": thisSeason})

            c['state'] = c.apply(lambda row: 'Neg' if row['hydYrAvg']<0 else 'Pos', axis=1)
            c['datetime'] = c.apply(lambda row: np.datetime64(str(row['year'])+self._hydroYearMDString), axis=1)

            c.to_csv('ipo.csv',index=False)

        #print(c['Annual'].min(),c['Annual'].max())
        return c

    def getAAOdata(self) -> pd.DataFrame():
        try:
            if not self._useCached:
                raise Exception('please ignore this, discarding locally available file and fetching from source')
            c = pd.read_csv('aao.csv')
            c['datetime']= pd.to_datetime(c['datetime']) #make sure object is datetime
        except Exception as er:
            #the latest data is always unfiltered
            c = pd.read_csv(self._samUrl, delimiter= '\s+') #, header=None, skiprows = 1, skipfooter = 11, engine='python', na_values=-99)
            c.reset_index(inplace=True)
            c.rename(columns={'index':'year'}, inplace=True)
            c.columns = self._climDataHeader.copy()
            c = c.assign(Annual=c.iloc[:,1:12].mean(axis=1))

            if self._climHydYrAvg:
                jdAvg = c[self._climHydYrAvg['year1']].mean(axis=1) #july to dec avg #now it is first year
                jjAvg = c[self._climHydYrAvg['year2']].mean(axis=1) #following year
                jjAvg = np.append((jjAvg.values)[1:],[np.nan]) #shift the year
                hydYrAvg = (jdAvg*len(self._climHydYrAvg['year1']) + jjAvg*len(self._climHydYrAvg['year2']))/12  # average
                c = c.assign(hydYrAvg=hydYrAvg)

            for thisSeason in self._climateWindows.keys():
                if thisSeason == 'DJF':
                    temp1 = c[self._climateWindows[thisSeason]].copy()
                    temp1['Dec'] = temp1['Dec'].shift(1) #shift down
                    c = c.assign(temp=temp1[self._climateWindows[thisSeason]].mean(axis=1))
                else:
                    c = c.assign(temp=c[self._climateWindows[thisSeason]].mean(axis=1))
                c = c.rename(columns={"temp": thisSeason})

            c['state'] = c.apply(lambda row: 'Neg' if row['hydYrAvg']<0 else 'Pos', axis=1)
            c['datetime'] = c.apply(lambda row: np.datetime64(str(row['year'])+self._hydroYearMDString), axis=1)
            c.to_csv('aao.csv',index=False)

        #print(c['Annual'].min(),c['Annual'].max())
        return c
    
    def getSPSDdata(self) -> pd.DataFrame():
        try:
            if not self._useCached:
                raise Exception('please ignore this, discarding locally available file and fetching from source')
            c = pd.read_csv('sspd.csv')
            c['datetime']= pd.to_datetime(c['datetime']) #make sure object is datetime
        except Exception as er:
            print(er)

            #this link points to the ERSST V5
            baseUrl = 'http://apdrc.soest.hawaii.edu/erddap/griddap/hawaii_soest_2c74_d49d_5023.csv?anom'
            duration = '[(1900-01-01T00:00:00Z):1:(2020-02-01T00:00:00Z)]'

            #Northwest subtropical pacific
            latRange = '[(-46):1:(-30)]'
            lonRange = '[(160):1:(190)]'
            url = baseUrl + duration + latRange + lonRange
            temp = pd.read_csv(url,skiprows = [2]) #, parse_dates=date_cols) #, delimiter= '\s+', header=None, skiprows = 1, skipfooter = 11, engine='python', na_values=-99)
            nwstpDf = temp.groupby(['time']).mean()
            nwstpDf.dropna(inplace=True)
            #nwstpDf['NWSTP_SSTA'] = nwstpDf['anom']

            #Southeast extra tropical pacific
            latRange = '[(-66):1:(-32)]'
            lonRange = '[(190):1:(250)]'
            url = baseUrl + duration + latRange + lonRange
            temp = pd.read_csv(url,skiprows = [2]) #, parse_dates=date_cols) #, delimiter= '\s+', header=None, skiprows = 1, skipfooter = 11, engine='python', na_values=-99)
            seetpDf = temp.groupby(['time']).mean()
            seetpDf.dropna(inplace=True)
            #seetpDf['NWSTP_SSTA'] = seetpDf['anom']

            SSPD = nwstpDf.join(seetpDf, on='time', lsuffix='_nwstp', rsuffix='_seetp')
            #nwstpDf.set_index('time').join(seetpDf.set_index('time'))
            SSPD['delSSTA'] = SSPD['anom_nwstp']-SSPD['anom_seetp'] #+ is high temp around NZ
            SSPD.reset_index(inplace=True)
            SSPD.to_csv('sspd.csv')

            SSPD['time']= pd.to_datetime(SSPD['time'])

            c = pd.DataFrame(columns=['year'])
            c [self._climDataHeader[0]] = SSPD['time'].dt.year.unique()
            #display(c)
            #convert to wide format
            for mnth in range(1,13):
                redDf = (SSPD[SSPD['time'].dt.month == mnth])[['time','delSSTA']].copy()
                redDf['year'] = redDf['time'].dt.year
                redDf.drop(columns=['time'], inplace=True)
                #display(c.join(redDf,on='year',lsuffix='_',rsuffix='__'))
                c[self._climDataHeader[mnth]] = c.apply(lambda row: self._mySubFn(row, redDf), axis=1)

            c = c.assign(Annual=c.iloc[:,1:12].mean(axis=1))

            if self._climHydYrAvg:
                jdAvg = c[self._climHydYrAvg['year1']].mean(axis=1) #july to dec avg #now it is first year
                jjAvg = c[self._climHydYrAvg['year2']].mean(axis=1) #following year
                jjAvg = np.append((jjAvg.values)[1:],[np.nan]) #shift the year
                hydYrAvg = (jdAvg*len(self._climHydYrAvg['year1']) + jjAvg*len(self._climHydYrAvg['year2']))/12  # average
                c = c.assign(hydYrAvg=hydYrAvg)

            for thisSeason in self._climateWindows.keys():
                if thisSeason == 'DJF':
                    temp1 = c[self._climateWindows[thisSeason]].copy()
                    temp1['Dec'] = temp1['Dec'].shift(1) #shift down
                    c = c.assign(temp=temp1[self._climateWindows[thisSeason]].mean(axis=1))
                else:
                    c = c.assign(temp=c[self._climateWindows[thisSeason]].mean(axis=1))
                c = c.rename(columns={"temp": thisSeason})

            c['state'] = c.apply(lambda row: 'Neg' if row['hydYrAvg']<0 else 'Pos', axis=1)
            c['datetime'] = c.apply(lambda row: np.datetime64(str(int(row['year']))+self._hydroYearMDString), axis=1)

            c.to_csv('sspd.csv',index=False)

        return c

    def _mySubFn(self,row, redDf):
        try:
            return (redDf[redDf['year'] == row['year']])['delSSTA'].values[0]
        except:
            return np.nan

    #Sub tropical ridge can influence the dryness - Metservice & Andy sturman
    #Cai et.al has presented a methodology for Australia
    def getSTRdata(self,source='NCEP',geoBound={'lat':(-10,-50),'lon':(165,180)}):
        from ftplib import FTP
        import xarray as xr
        #import io
        import matplotlib.pyplot as plt

        ds = None
        if source == 'PSL' :
            #datasets from NOAA PSL
            #https://psl.noaa.gov/gcos_wgsp/Gridded/data.hadslp2.html
            dsFilName = 'slp.mnmean.real.nc'
            try:
                if self._useCached :
                    striDf = pd.read_csv('stri.csv')
                    striDf['datetime']= pd.to_datetime(c['datetime']) #make sure object is datetime
                    
                    strpDf = pd.read_csv('strp.csv')
                    strpDf['datetime']= pd.to_datetime(c['datetime']) #make sure object is datetime
                else : raise Exception('please ignore this, discarding locally available file and fetching from source')
            except Exception as er:
                ftp = FTP('ftp.cdc.noaa.gov')  # connect to host, default port
                print(ftp.login())              # user anonymous, passwd anonymous@
                print(ftp.cwd('/Datasets.other/hadslp2/'))
                ftp.retrbinary('RETR '+dsFilName, open(dsFilName, 'wb').write)
                ds = xr.open_dataset(dsFilName)
        else :
            #datasets from NCEP reanalysis
            dsFilName = 'slp.mon.mean.nc' #2
            try:
                if self._useCached :
                    ds = xr.open_dataset(dsFilName)
                else : raise Exception('please ignore this, discarding locally available file and fetching from source')
            except Exception as er:
                ftp = FTP('ftp2.psl.noaa.gov')  # connect to host, default port
                print(ftp.login())              # user anonymous, passwd anonymous@
                print(ftp.cwd('/Datasets/ncep.reanalysis.derived/surface/')) #2
                #print(ftp.retrlines('LIST'))
                ftp.retrbinary('RETR '+dsFilName, open(dsFilName, 'wb').write) #2
                ds = xr.open_dataset(dsFilName)

        if ds != None :
            redDs = ds.slp.sel(lat=slice(geoBound['lat'][0],geoBound['lat'][1]),lon=slice(geoBound['lon'][0],geoBound['lon'][1]))
            df = redDs.to_dataframe()
            df.reset_index(inplace=True)
            df.set_index('time',inplace=True)

            #fill missing dates, surprisingly varies for months
            #assuming the nan filling is done on lat long columns
            misDate = list(pd.date_range(start=str(df.index.min())[:10], end=str(df.index.max())[:10],
                        freq=pd.DateOffset(months=1)).difference(df.index.unique()))
            if len(misDate) > 0:
                print('patching missing dates')
                display(misDate)
                #since the area is a rectangle, a combo of lat longs works
                alLats = df.lat.unique()
                alLons = df.lon.unique()
                tempCombo = [(t, x, y) for t in misDate for x in alLats for y in alLons]
                myColumns = ['time','lat','lon']
                tempDf = pd.DataFrame(tempCombo, columns=myColumns)
                tempDf.set_index('time', inplace=True)
                df = df.append(tempDf) #.drop_duplicates()
                #display(df[df.index.isin(misDate) ])

            df.sort_index(ascending=False, inplace=True)
            print('mean SLP of',geoBound,df.slp.mean())
            df['delSlp'] = df.slp - df.slp.mean()

            #climatological positions and intensity of subtropical ridge
            redDf = df.groupby(by=['lat',df.index.month]).mean()
            redDf.drop(columns=['lon'],inplace=True)
            redDf.reset_index(inplace=True)

            maxPresX = []
            maxPLatX = []
            maxLabelsX = []
            for thisMonth in range(1,13):
                myDf = redDf[redDf['time']==thisMonth]
                myDf.set_index('lat',inplace=True)
                maxPresX.append(myDf.delSlp[myDf.slp == myDf.slp.max()].values[0])
                maxPLatX.append(myDf.index[myDf.slp == myDf.slp.max()].values[0])
                maxLabelsX.append(thisMonth)

            refDf = pd.DataFrame(data={'pres':maxPresX,'lat':maxPLatX,'month':maxLabelsX})
            refDf.set_index('month', inplace=True)
            refDf['mult'] = refDf['pres']*refDf['lat']
            #see plots in the individual notebook

            #STRI and STRP are cai et.al 2011 notation, building the dataframes
            myLats = df['lat'].unique()
            maxLocValDf = pd.DataFrame(index=df.index.unique())

            for thisLat in myLats:
                thisDf = df[df['lat'] == thisLat]
                redDf = thisDf.groupby(by=[thisDf.index]).mean()
                maxLocValDf[str(thisLat)] = redDf['delSlp']

            maxLocValDf['MaxLoc'] = maxLocValDf.idxmax(axis=1)
            maxLocValDf['MaxVal'] = maxLocValDf.max(axis=1)
            maxLocValDf["MaxLoc"] = pd.to_numeric(maxLocValDf["MaxLoc"], downcast="float")

            delIPDf = maxLocValDf[['MaxLoc', 'MaxVal']].copy()
            delIPDf['month'] = delIPDf.index.month
            delIPDf.reset_index(inplace=True)
            delIPDf.set_index('month', inplace=True)
            delIPDf = delIPDf.merge(refDf, on='month')
            delIPDf.set_index('time', inplace=True)
            delIPDf['delLoc'] = delIPDf['MaxLoc'] - delIPDf['lat']
            delIPDf['delPres'] = delIPDf['MaxVal'] - delIPDf['pres']

            ## Nan fill to the end of the year to make sure the following loop doesn't break
            misDate = list(pd.date_range(start=str(delIPDf.index.year.min())+"-01-01",
                                              end=str(delIPDf.index.year.max())+"-12-31",
                                                freq=pd.DateOffset(months=1)).difference(delIPDf.index))
            if len(misDate) > 0:
                print('nan patching missing dates')
                display(misDate)
                tempDf = pd.DataFrame(index=misDate)
                delIPDf = delIPDf.append(tempDf) #.drop_duplicates()
                #display(delIPDf)

            delIPDf['year'] = delIPDf.index.year
            striDf = pd.DataFrame(index=delIPDf.index.year.unique())
            strpDf = pd.DataFrame(index=delIPDf.index.year.unique())
            for thisMonth in delIPDf.index.month.unique()[::-1]: #the time goes back in the main df
                redDf = delIPDf[delIPDf.index.month == thisMonth]
                redDf.set_index('year',inplace=True)
                try :
                    striDf[thisMonth] = redDf['delPres']
                except :
                    print('pres', thisMonth)
                try:
                    strpDf[thisMonth] = redDf['delLoc']
                except:
                    print('loc', thisMonth)

            #display(striDf)
            monMapDict = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
            striDf.rename(columns=monMapDict,inplace=True)
            strpDf.rename(columns=monMapDict,inplace=True)
            striDf.index.name = 'year'
            strpDf.index.name = 'year'
            #striDf['year'] = striDf.index
            #striDf.rename(columns={'index':'year'}, inplace=True)
            striDf.reset_index(inplace=True)
            #strpDf['year'] = strpDf.index
            #strpDf.rename(columns={'index':'year'}, inplace=True)
            strpDf.reset_index(inplace=True)
            
            #get the seasonal averages
            striDf = striDf.assign(Annual=striDf.iloc[:,1:12].mean(axis=1))
            strpDf = strpDf.assign(Annual=strpDf.iloc[:,1:12].mean(axis=1))

            if self._climHydYrAvg:
                jdAvg = striDf[self._climHydYrAvg['year1']].mean(axis=1) #july to dec avg #now it is first year
                jjAvg = striDf[self._climHydYrAvg['year2']].mean(axis=1) #following year
                jjAvg = np.append((jjAvg.values)[1:],[np.nan]) #shift the year
                hydYrAvg = (jdAvg*len(self._climHydYrAvg['year1']) + jjAvg*len(self._climHydYrAvg['year2']))/12  # average
                striDf = striDf.assign(hydYrAvg=hydYrAvg)
                
                jdAvg = strpDf[self._climHydYrAvg['year1']].mean(axis=1) #july to dec avg #now it is first year
                jjAvg = strpDf[self._climHydYrAvg['year2']].mean(axis=1) #following year
                jjAvg = np.append((jjAvg.values)[1:],[np.nan]) #shift the year
                hydYrAvg = (jdAvg*len(self._climHydYrAvg['year1']) + jjAvg*len(self._climHydYrAvg['year2']))/12  # average
                strpDf = strpDf.assign(hydYrAvg=hydYrAvg)

            for thisSeason in self._climateWindows.keys():
                if thisSeason == 'DJF':
                    temp1 = striDf[self._climateWindows[thisSeason]].copy()
                    temp1['Dec'] = temp1['Dec'].shift(1) #shift down
                    striDf = striDf.assign(temp=temp1[self._climateWindows[thisSeason]].mean(axis=1))
                    
                    temp1 = strpDf[self._climateWindows[thisSeason]].copy()
                    temp1['Dec'] = temp1['Dec'].shift(1) #shift down
                    strpDf = strpDf.assign(temp=temp1[self._climateWindows[thisSeason]].mean(axis=1))
                else:
                    striDf = striDf.assign(temp=striDf[self._climateWindows[thisSeason]].mean(axis=1))
                    strpDf = strpDf.assign(temp=strpDf[self._climateWindows[thisSeason]].mean(axis=1))
                striDf = striDf.rename(columns={"temp": thisSeason})
                strpDf = strpDf.rename(columns={"temp": thisSeason})

            striDf['state'] = striDf.apply(lambda row: 'Neg' if row['hydYrAvg']<0 else 'Pos', axis=1)
            strpDf['state'] = strpDf.apply(lambda row: 'Neg' if row['hydYrAvg']<0 else 'Pos', axis=1)
            striDf['datetime'] = striDf.apply(lambda row: np.datetime64(str(row['year'])+self._hydroYearMDString), axis=1)
            strpDf['datetime'] = strpDf.apply(lambda row: np.datetime64(str(row['year'])+self._hydroYearMDString), axis=1)
            
            striDf.to_csv('stri.csv',index=False)
            strpDf.to_csv('strp.csv',index=False)
            refDf.to_csv('strClim.csv')

            return refDf, striDf, strpDf