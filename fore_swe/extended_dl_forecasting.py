import pandas as pd
import numpy as np

import os
import sys

import math

import random

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import haversine_distances

from sentence_transformers import SentenceTransformer

baseUrl = os.getenv("BASE_URL", "")
baseUrl = os.path.join(baseUrl, "data")

class DataLoader:    
    def normalizeCols(self,data, cols):
        temp_df = data.copy(deep = True)
        columns = [ col for col in data.columns if col in cols ]

        max_value  =  temp_df.loc[:,columns].max().max()
        min_value  =  temp_df.loc[:,columns].min().min() 

        temp_df.loc[:,columns] = (temp_df.loc[:, columns] - min_value)/(max_value - min_value)

        return temp_df
    
    def cleanDataJunk(self):

        lists = [self.precipitation_info['Date'].values,self.rmax_info['Date'].values,self.rmin_info['Date'].values,
                self.sph_info['Date'].values ,self.srad_info['Date'].values ,self.tb_19_info['Date'].values ,
                self.tb_37_info['Date'].values ,self.tb_diff_info['Date'].values ,self.tmin_info['Date'].values,
                self.tmax_info['Date'].values ,self.windspeed_info['Date'].values,self.swe_values['Date'].values, self.all_stations_day_length_info['Date'].values]
        
        sets = [set(lst) for lst in lists]
        
        included_values = set.intersection(*sets)

        self.precipitation_info = self.precipitation_info.loc[self.precipitation_info['Date'].isin(included_values),:]
        self.rmax_info = self.rmax_info.loc[self.rmax_info['Date'].isin(included_values),:]
        self.rmin_info = self.rmin_info.loc[self.rmin_info['Date'].isin(included_values),:]
        self.sph_info = self.sph_info.loc[self.sph_info['Date'].isin(included_values),:]
        self.srad_info = self.srad_info.loc[self.srad_info['Date'].isin(included_values),:]
        self.tb_19_info = self.tb_19_info.loc[self.tb_19_info['Date'].isin(included_values),:]
        self.tb_37_info = self.tb_37_info.loc[self.tb_37_info['Date'].isin(included_values),:]
        self.tb_diff_info = self.tb_diff_info.loc[self.tb_diff_info['Date'].isin(included_values),:]
        self.tmin_info = self.tmin_info.loc[self.tmin_info['Date'].isin(included_values),:]
        self.tmax_info = self.tmax_info.loc[self.tmax_info['Date'].isin(included_values),:]
        self.windspeed_info = self.windspeed_info.loc[self.windspeed_info['Date'].isin(included_values),:]
                        
        self.swe_values = self.swe_values.loc[self.swe_values['Date'].isin(included_values),:]

        self.all_stations_day_length_info = self.all_stations_day_length_info.loc[self.all_stations_day_length_info['Date'].isin(included_values),:]

        self.all_dates = np.array(self.precipitation_info['Date'].values)

    def getModelPrompts(self):
        model = SentenceTransformer("all-MiniLM-L6-v2")

        location_prompts = []

        for index,location_info in self.snotel_locations_info.iterrows():
            
            elev_prompt = model.encode([location_info['elevation prompt']])
            land_cover_prompt = model.encode([location_info['landcover prompt']])
            koppen_prompt = model.encode([location_info['koppen prompt']])
            southness_prompt = model.encode([location_info['southness prompt']])

            elev_prompt = np.array(elev_prompt).reshape(1,-1)
            land_cover_prompt = np.array(land_cover_prompt).reshape(1,-1)
            koppen_prompt = np.array(koppen_prompt).reshape(1,-1)
            southness_prompt = np.array(southness_prompt).reshape(1,-1)

            all_prompts = np.concatenate((elev_prompt,land_cover_prompt, koppen_prompt, southness_prompt),axis = 1)

            location_prompts.append(all_prompts)
        
        all_location_prompts = np.array(location_prompts).transpose(1,0,2)

        test_prompts = all_location_prompts[:,self.test_indices,:]
        train_prompts = all_location_prompts[:,self.train_indices,:]

        return (train_prompts,test_prompts, all_location_prompts)

    def getModelInputsAndOutputs(self):
        model_inputs = []
        model_outputs = []

        for index,location_info in self.snotel_locations_info.iterrows():
            elev = (location_info['Elevation'] - self.snotel_locations_info['Elevation'].min()) / (self.snotel_locations_info['Elevation'].max() - self.snotel_locations_info['Elevation'].min())
            lat = (location_info['Latitude'] - self.snotel_locations_info['Latitude'].min()) / (self.snotel_locations_info['Latitude'].max() - self.snotel_locations_info['Latitude'].min())
            longitude = (location_info['Longitude'] - self.snotel_locations_info['Longitude'].min()) / (self.snotel_locations_info['Longitude'].max() - self.snotel_locations_info['Longitude'].min())
            southness = (location_info['Southness'] - self.snotel_locations_info['Southness'].min()) / (self.snotel_locations_info['Southness'].max() - self.snotel_locations_info['Southness'].min())
            
            spatial_feature = np.array([lat,longitude,elev,southness])

            day_number = self.precipitation_info.loc[:, self.precipitation_info.columns.isin(['day'])].to_numpy()
            
            # precipitation_info = self.precipitation_info.loc[:, self.precipitation_info.columns.isin([location_info['Station Name']])].to_numpy()
            rain_info = self.rain_info.loc[:, self.rain_info.columns.isin([location_info['Station Name']])].to_numpy()
            snow_info = self.snow_info.loc[:, self.snow_info.columns.isin([location_info['Station Name']])].to_numpy()
            rmax_info = self.rmax_info.loc[:, self.rmax_info.columns.isin([location_info['Station Name']])].to_numpy()
            rmin_info = self.rmin_info.loc[:, self.rmin_info.columns.isin([location_info['Station Name']])].to_numpy()
            sph_info = self.sph_info.loc[:, self.sph_info.columns.isin([location_info['Station Name']])].to_numpy()
            srad_info = self.srad_info.loc[:, self.srad_info.columns.isin([location_info['Station Name']])].to_numpy()
            tb_19_info = self.tb_19_info.loc[:, self.tb_19_info.columns.isin([location_info['Station Name']])].to_numpy()
            tb_37_info = self.tb_37_info.loc[:, self.tb_37_info.columns.isin([location_info['Station Name']])].to_numpy()
            tb_diff_info = self.tb_diff_info.loc[:, self.tb_diff_info.columns.isin([location_info['Station Name']])].to_numpy()
            tmin_info = self.tmin_info.loc[:, self.tmin_info.columns.isin([location_info['Station Name']])].to_numpy()
            tmax_info = self.tmax_info.loc[:, self.tmax_info.columns.isin([location_info['Station Name']])].to_numpy()
            windspeed_info = self.windspeed_info.loc[:, self.windspeed_info.columns.isin([location_info['Station Name']])].to_numpy()
            swe_info = self.input_swe_info.loc[:, self.input_swe_info.columns.isin([location_info['Station Name']])].to_numpy()

            day_length_info = self.all_stations_day_length_info.loc[:, self.all_stations_day_length_info.columns.isin([location_info['Station Name']])].to_numpy()

            merged_values = np.concatenate((day_number, day_length_info, rain_info,snow_info, rmax_info, rmin_info, sph_info, srad_info, 
                                            tb_19_info, tb_37_info, tb_diff_info, tmin_info, tmax_info, windspeed_info, swe_info),axis = 1)
            
            merged_values = np.insert(merged_values,[0],spatial_feature, axis= 1)    
            location_swe_values = self.swe_values.loc[:,self.swe_values.columns.isin([location_info['Station Name']])].to_numpy()

            model_outputs.append(location_swe_values)
            model_inputs.append(merged_values)
        
        model_inputs = np.array(model_inputs).transpose(1,0,2)
        model_outputs = np.array(model_outputs).transpose(1,0,2)

        return (model_inputs,model_outputs)
    
    def completeInputsAndOutputs(self):
        num_of_locations = self.model_inputs.shape[1]
        
        if len(self.test_indices) == 0:
            self.test_indices = random.sample(range(0, num_of_locations), 102)
        
        self.all_indices = np.arange(0,num_of_locations,1)
        self.train_indices = list(set(self.all_indices).difference(set(self.test_indices)))

        hist_added_inputs = []
        updated_outputs = []

        start_index = -1
        
        filtered_active_date, filtered_active_date_indices = self.getFilteredDateIndices(start_year = 1991, start_month = 12, end_month = 5)
        
        start_index = np.where(filtered_active_date == "1995-12-01")[0][0]

        filtered_active_date = np.array(filtered_active_date)[(start_index+1):]
        filtered_active_date_indices = np.array(filtered_active_date_indices)[(start_index+1):]

        self.test_yr_indices_1 = np.where(np.logical_and(filtered_active_date >= "{}-12-01".format(self.year_1) , filtered_active_date<= "{}-05-30".format(self.year_1+1)))[0]
        self.test_yr_indices_2 = np.where(np.logical_and(filtered_active_date >= "{}-12-01".format(self.year_2) , filtered_active_date<= "{}-05-30".format(self.year_2+1)))[0]
        self.test_yr_indices_3 = np.where(np.logical_and(filtered_active_date >= "{}-12-01".format(self.year_3) , filtered_active_date<= "{}-05-30".format(self.year_3+1)))[0]
        self.test_yr_indices_4 = np.where(np.logical_and(filtered_active_date >= "{}-12-01".format(self.year_4) , filtered_active_date<= "{}-05-30".format(self.year_4+1)))[0]
        self.test_yr_indices_5 = np.where(np.logical_and(filtered_active_date >= "{}-12-01".format(self.year_5) , filtered_active_date<= "{}-05-30".format(self.year_5+1)))[0]
        
        self.test_yr_indices_1 = filtered_active_date_indices[self.test_yr_indices_1]
        self.test_yr_indices_2 = filtered_active_date_indices[self.test_yr_indices_2]
        self.test_yr_indices_3 = filtered_active_date_indices[self.test_yr_indices_3]
        self.test_yr_indices_4 = filtered_active_date_indices[self.test_yr_indices_4]
        self.test_yr_indices_5 = filtered_active_date_indices[self.test_yr_indices_5]

        all_test_indices = list(self.test_yr_indices_1) + list(self.test_yr_indices_2) + list(self.test_yr_indices_3) + list(self.test_yr_indices_4) + list(self.test_yr_indices_5)
        all_training_indices = sorted(set(list(filtered_active_date_indices)).difference(set(list(all_test_indices)))) 

        self.training_data_indices = all_training_indices 

        return

       
    def getFilteredDateIndices(self, start_year = 1991, start_month = 10, start_day = 1, end_month = 9):
        filtered_dates = []
        filtered_dates_indices = []
        
        years = [value for value in range(start_year,2020)]

        for year in years:
            
            start_date = '{}-{:02d}-{:02d}'.format(year, start_month, start_day)
            end_date = '{}-{:02d}-30'.format(year + 1 if end_month < start_month else year, end_month)

            indices = np.where(np.logical_and(self.all_dates >= start_date, self.all_dates <= end_date))[0]
            
            filtered_dates.append(self.all_dates[indices])
            filtered_dates_indices.append(indices)
    
        filtered_dates = np.concatenate(filtered_dates)
        filtered_dates_indices = np.concatenate(filtered_dates_indices)
            
        return (filtered_dates,filtered_dates_indices)
    
    def generateBatches(self,data,batch_size):
        start_index = 0
        end_index = batch_size
    
        batch_data = []
    
        while end_index <= len(data):
            batch_data.append(data[start_index:end_index])
        
            start_index = end_index
            end_index += batch_size
    
        return batch_data
    
    def getTestingIndices(self):
        return self.test_indices
    
    def getTrainingIndices(self):
        return self.train_indices
    
    def getTrainingDataIndices(self):
        return self.training_data_indices
    
    def getTestingDataIndices(self):
        return (self.test_yr_indices_1, self.test_yr_indices_2, self.test_yr_indices_3, self.test_yr_indices_4, self.test_yr_indices_5)
    
    def getHashedValuesMap(self):
        return self.location_hash_values_map
    
    def prepareInputsAndOutputs(self):
        self.model_inputs, self.model_outputs = self.getModelInputsAndOutputs()
    
    def addWaterYearDayNumber(self, observation_info, start_year = 2001, start_month = 10):
        filtered_dates, filtered_dates_indices = self.getFilteredDateIndices()

        date_day_value_map = {}
        current_year = start_year
        start_month = 10

        count = 1

        for individual_date in filtered_dates:
            splitted_date = individual_date.split('-')

            year = splitted_date[0]
            month = splitted_date[1]

            if (current_year != int(year,10)) and (int(month,10) == start_month):
                count = 1
                current_year = int(year,10)

            date_day_value_map[individual_date] = count
            count = count +1

        observation_info['day'] = observation_info['Date'].map(date_day_value_map)
        column_pop = observation_info.pop('day')
        observation_info.insert(1,'day', column_pop)

        observation_info['day'] = (observation_info['day'] - observation_info['day'].min()) /(observation_info['day'].max() - observation_info['day'].min())

        return observation_info.loc[observation_info['Date'].isin(filtered_dates),:]
    
    def addDayInfo(self):

        self.precipitation_info = self.addWaterYearDayNumber(self.precipitation_info,1991,10)
        self.rmax_info = self.addWaterYearDayNumber(self.rmax_info,1991,10)
        self.rmin_info = self.addWaterYearDayNumber(self.rmin_info,1991,10)
        self.sph_info = self.addWaterYearDayNumber(self.sph_info,1991,10)
        self.srad_info = self.addWaterYearDayNumber(self.srad_info,1991,10)
        self.tb_19_info = self.addWaterYearDayNumber(self.tb_19_info,1991,10)
        self.tb_37_info = self.addWaterYearDayNumber(self.tb_37_info,1991,10)
        self.tb_diff_info = self.addWaterYearDayNumber(self.tb_diff_info,1991,10)
        self.tmin_info = self.addWaterYearDayNumber(self.tmin_info,1991,10)
        self.tmax_info = self.addWaterYearDayNumber(self.tmax_info,1991,10)
        self.windspeed_info = self.addWaterYearDayNumber(self.windspeed_info,1991,10)
        self.snow_info = self.addWaterYearDayNumber(self.rain_info,1991,10)
        self.rain_info = self.addWaterYearDayNumber(self.snow_info,1991,10)


    def getSlopeValue(self, all_data, period):
        all_data_copy = all_data.copy()
        selected_columns = [column for column in all_data_copy.columns if column not in ['Date','day']]

        for column in selected_columns:
            all_data_copy[column] = all_data[column].rolling(window=period, min_periods=1,center = False).apply(
                                    lambda x: (x[-1] - x[0]) / period if len(x) >= 1 else 0, raw=True)

        return all_data_copy
               
    def getSlopeTrends(self):
        self.VE_37_daily_slope = self.getSlopeValue(self.VE_37_daily,10)
        self.VE_19_daily_slope = self.getSlopeValue(self.VE_19_daily,10)
        self.Max_temp_daily_slope = self.getSlopeValue(self.Max_temp_daily,10)
        self.Min_temp_daily_slope = self.getSlopeValue(self.Min_temp_daily,10)
        self.Obs_temp_daily_slope = self.getSlopeValue(self.Obs_temp_daily,10)
        self.TB_diff_daily_slope = self.getSlopeValue(self.TB_diff_daily, 10)

    def getHaversineDistance(self, locations_info):
        all_locations_coordinates = locations_info.loc[:,['Latitude', 'Longitude']].to_numpy() 

        all_locations_radians = np.deg2rad(all_locations_coordinates)
        all_locations_radians = all_locations_radians.reshape(-1,2)

        distance = haversine_distances(all_locations_radians)

        earth_radius_in_km = 6371

        print("Distances", distance * earth_radius_in_km)
        print("Distances shape", (distance * earth_radius_in_km).shape)

        return distance * earth_radius_in_km
    
    def getAlgularityOfLocations(self,locations_info):
        all_locations_info = locations_info.loc[:, ['Latitude','Longitude', 'Elevation']].to_numpy()

        feet_eq_in_meters= 0.3048
        earth_radius_in_km = 6371

        all_locations_coordinates = []

        for location_info in all_locations_info:
            x = (earth_radius_in_km + location_info[2] * feet_eq_in_meters) * np.cos(np.deg2rad(location_info[0])) * np.cos(np.deg2rad(location_info[1]))
            y = (earth_radius_in_km + location_info[2] * feet_eq_in_meters) * np.cos(np.deg2rad(location_info[0])) * np.sin(np.deg2rad(location_info[1]))
            z = (earth_radius_in_km + location_info[2] * feet_eq_in_meters) * np.sin(np.deg2rad(location_info[0]))

            all_locations_coordinates.append([x,y,x])
        
        all_locations_coordinates = np.array(all_locations_coordinates)

        dot_product_matrix = np.dot(all_locations_coordinates, all_locations_coordinates.T)
        magnitudes = np.linalg.norm(all_locations_coordinates,axis = 1)

        magnitude_matrix = np.outer(magnitudes,magnitudes)

        cos_angle = dot_product_matrix/magnitude_matrix
        angularity_matrix_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))

        angle_in_degrees = np.degrees(angularity_matrix_rad)

        print("Angularities", angle_in_degrees)
        print("Angularities shape", angle_in_degrees.shape)

        return angle_in_degrees

    def getDistanceAndAngularity(self):
        locations_info = self.snotel_locations_info.loc[:,['Station Name', 'Elevation', 'Latitude', 'Longitude']]

        distance_matrix = self.getHaversineDistance(locations_info)
        angle_matrix = self.getAlgularityOfLocations(locations_info)

        return distance_matrix, angle_matrix

    def readAndStoreData(self):

        #Loads the data for daily observations.
        self.precipitation_info = pd.read_csv('{}/precipitation_info.csv'.format(baseUrl))
        self.rmax_info = pd.read_csv('{}/rmax_info.csv'.format(baseUrl))
        self.rmin_info = pd.read_csv('{}/rmin_info.csv'.format(baseUrl))
        self.sph_info = pd.read_csv('{}/sph_info.csv'.format(baseUrl))
        self.srad_info = pd.read_csv('{}/srad_info.csv'.format(baseUrl))
        self.tb_19_info = pd.read_csv('{}/tb_19_info.csv'.format(baseUrl))
        self.tb_37_info = pd.read_csv('{}/tb_37_info.csv'.format(baseUrl))
        self.tb_diff_info = pd.read_csv('{}/tb_diff_info.csv'.format(baseUrl))
        self.tmin_info = pd.read_csv('{}/tmin_info.csv'.format(baseUrl))
        self.tmax_info = pd.read_csv('{}/tmax_info.csv'.format(baseUrl))
        self.windspeed_info = pd.read_csv('{}/windspeed_info.csv'.format(baseUrl))
        self.rain_info = pd.read_csv('{}/rain_info.csv'.format(baseUrl))
        self.snow_info = pd.read_csv('{}/snow_info.csv'.format(baseUrl))

        self.all_stations_day_length_info = pd.read_csv('{}/all_stations_day_length.csv'.format(baseUrl))

        #Loads the target swe values.
        self.swe_values = pd.read_csv('{}/swe_values.csv'.format(baseUrl))

        # Loads the locations info.
        self.snotel_locations_info = pd.read_csv('{}/snotel_location_info_2.csv'.format(baseUrl))
    
    def extractRainSnowPartition(self):
        alpha = -10.04
        beta  =  1.41
        gamma = 0.09

        locations = list(self.tmax_info.columns)

        rain_info = pd.DataFrame()
        snow_info = pd.DataFrame()

        for index in range(1,len(locations)):
            location = locations[index]

            precipitations = list(self.precipitation_info[location].values)

            avg_temp = list((self.tmax_info[location].values + self.tmin_info[location].values)/2)
            avg_rh = list((self.rmax_info[location].values + self.rmin_info[location].values)/2)

            eps = sys.float_info.epsilon
            
            denominator = 1 + eps**(alpha + beta * np.array(avg_temp) + gamma * np.array(avg_rh))
            probability_snow = 1/denominator

            snow_values = probability_snow * np.array(precipitations)
            rain_values = (1 - probability_snow) * np.array(precipitations)

            rain_info[location] = rain_values
            snow_info[location] = snow_values
        
        rain_info.insert(0,'Date', self.precipitation_info['Date'].values)
        snow_info.insert(0,'Date', self.precipitation_info['Date'].values)

        self.rain_info = rain_info
        self.snow_info = snow_info

        self.rain_info.to_csv('{}/rain_info.csv'.format(baseUrl), index = 0)
        self.snow_info.to_csv('{}/snow_info.csv'.format(baseUrl), index = 0)
    
    def getMaxMinSWE(self):
        max_values = np.max(self.model_outputs[self.training_data_indices,:,:], axis = 1)
        highest_swe = np.max(max_values.reshape(-1))

        lowest_swe = 0

        return (highest_swe + 500, lowest_swe)

    def normalizeData(self):

        #Loads the data for daily observations.
        self.precipitation_info = self.normalizeCols(self.precipitation_info, list(self.snotel_locations_info['Station Name'].values))
        self.rmax_info = self.normalizeCols(self.rmax_info, list(self.snotel_locations_info['Station Name'].values))
        self.rmin_info = self.normalizeCols(self.rmin_info, list(self.snotel_locations_info['Station Name'].values))
        self.sph_info = self.normalizeCols(self.sph_info, list(self.snotel_locations_info['Station Name'].values))
        self.srad_info = self.normalizeCols(self.srad_info, list(self.snotel_locations_info['Station Name'].values))
        self.tb_19_info = self.normalizeCols(self.tb_19_info, list(self.snotel_locations_info['Station Name'].values))
        self.tb_37_info = self.normalizeCols(self.tb_37_info, list(self.snotel_locations_info['Station Name'].values))
        self.tb_diff_info = self.normalizeCols(self.tb_diff_info, list(self.snotel_locations_info['Station Name'].values))
        self.tmin_info = self.normalizeCols(self.tmin_info, list(self.snotel_locations_info['Station Name'].values))
        self.tmax_info = self.normalizeCols(self.tmax_info, list(self.snotel_locations_info['Station Name'].values))
        self.windspeed_info = self.normalizeCols(self.windspeed_info, list(self.snotel_locations_info['Station Name'].values))
        self.rain_info = self.normalizeCols(self.rain_info, list(self.snotel_locations_info['Station Name'].values))
        self.snow_info = self.normalizeCols(self.snow_info, list(self.snotel_locations_info['Station Name'].values))
        self.input_swe_info = self.normalizeCols(self.swe_values, list(self.snotel_locations_info['Station Name'].values))
        self.all_stations_day_length_info = self.normalizeCols(self.all_stations_day_length_info, list(self.snotel_locations_info['Station Name'].values))

    def __init__(self,test_indices = [],n_hyperplanes = 8, temp_window = 3):
        self.test_indices = test_indices
        
        self.readAndStoreData()

        self.n_hyperplanes = n_hyperplanes
        self.location_hash_values_map = {}
        self.location_sp_feature_map = {}

        self.temp_window = temp_window
        
        # Years for testing        
        self.year_1 = 2014
        self.year_2 = 2015
        self.year_3 = 2016
        self.year_4 = 2017
        self.year_5 = 2018
