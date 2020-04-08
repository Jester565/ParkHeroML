from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import argparse
import os.path
import re
import sys
import urllib
import json
from pytz import timezone
import dateutil.parser
from datetime import timedelta, datetime, time
import requests

from time import sleep

import pymysql
import csv

#'FastPass' for FastPass training
#'RideTime' for RideTime training
modelMode = os.environ['model_mode']
numSteps = int(os.environ['train_steps'])
dateRange = int(os.environ['date_range'])
#percentage of the data that should be used in training (0.5 for 50% data for training)
TRAIN_PERCENT = 1

#Create temporary tables for each park day that combines information throughout the database
dayTempTableQuery = '''CREATE TEMPORARY TABLE DayTempTable (date DATE, parkID INT, DOY INT, DOW TINYINT, OpenHour TINYINT, HoursOpen TINYINT, MagicHours TINYINT, BlockLevel TINYINT, PRIMARY KEY(date, parkID))
    SELECT ps.date AS date,
    ps.parkID AS parkID,
    YEAR(ps.date) AS Year,
    MONTH(ps.date) - 1 AS Month,
    MONTH(ps.date) * 31 + DAYOFMONTH(ps.date) AS DOY, 
    DAYOFWEEK(ps.date) - 1 AS DOW, 
    HOUR(ps.openTime) AS OpenHour, 
    TRUNCATE(HOUR(ps.closeTime) + ((HOUR(ps.closeTime) - HOUR(ps.openTime))/ABS(HOUR(ps.closeTime) - HOUR(ps.openTime)) - 1) * -12, 0) - HOUR(ps.openTime) AS HoursOpen, 
    IF(ps.magicHourStartTime IS NULL, 0, 1) AS MagicHours, 
    rs.blockLevel AS BlockLevel, 
    ps.crowdLevel AS CrowdLevel,
    IF(h.name IS NULL, 0, 1) AS IsHoliday
    FROM ParkSchedules ps 
    INNER JOIN Parks p ON ps.parkID=p.id 
    INNER JOIN ResortSchedules rs ON ps.date=rs.date AND p.resortID=rs.resortID
    LEFT JOIN Holidays h ON ps.date=h.date
    WHERE ps.parkID=p.id AND p.resortID=rs.resortID AND ps.date=rs.date'''

#Modifiable query to get each row for training
baseQuery = '''SELECT 
    rt.rideID AS RideID,
    dt.Year AS Year,
    dt.Month AS Month,
    dt.DOW AS DOW,
    dt.OpenHour AS OpenHour,
    dt.HoursOpen AS HoursOpen,
    dt.MagicHours AS MagicHours,
    dt.BlockLevel AS BlockLevel,
    dt.CrowdLevel AS CrowdLevel,
    HOUR(SUBTIME(rt.dateTime, ps.openTime)) AS HoursSinceOpen, 
    hw.feelsLikeF AS Temp, 
    dt.IsHoliday AS IsHoliday,
    {}
    FROM RideTimes rt 
    INNER JOIN Rides r ON rt.rideID=r.id
    INNER JOIN Parks p ON p.id=r.parkID
    INNER JOIN DayTempTable dt ON dt.date=DATE(DATE_SUB(rt.dateTime, INTERVAL 4 HOUR)) AND p.id=dt.parkID
    INNER JOIN HourlyWeather hw ON hw.resortID=p.resortID AND DATE_SUB(DATE_SUB(rt.dateTime, INTERVAL MINUTE(rt.dateTime) MINUTE), INTERVAL SECOND(rt.dateTime) SECOND)=hw.dateTime
    INNER JOIN ParkSchedules ps ON dt.date=ps.date AND ps.parkID=p.id
    WHERE {} ORDER BY dt.date'''

#Query to get each row for FastPass training (must run after DayTempTable created)
fastPassQuery = baseQuery.format('''TIME_TO_SEC(SUBTIME(rt.fastPassTime, ps.openTime)) / 60 AS FastPassTime''', '''rt.fastPassTime IS NOT NULL''')

#Query to get each row for RideTime training (must run after DayTempTable created)
rideTimesQuery = baseQuery.format('''rt.waitMins''', '''rt.waitMins IS NOT NULL''')

#Query to get each row for RideTime training (must run after DayTempTable created)
def zipToCoords(zip):
    params = {
        "sensor": False,
        "address": zip,
        "key": googleKey
    }
    headers = {
        "accept": "application/json"
    }
    response = requests.get("https://maps.googleapis.com/maps/api/geocode/json", params, headers=headers)
    jsonData = json.loads(response.text)
    location = jsonData["results"][0]["geometry"]["location"]
    coords = str(location["lat"])
    coords += ","
    coords += str(location["lng"])
    return coords

#Get the forecast from DarkSky
#  coords: { lat, lng } (latitude, longitude) to get forecast for
#  dt: Date & hour to get forecast for
def getForecast(coords, dt):
    secondsSinceEpoch = str(int(dt.timestamp()))
    print("Time since epoch: ", secondsSinceEpoch)
    params = {
        "exclude": "currently,minutley,daily,alerts,flags"
    }
    headers = {
        "accept": "application/json"
    }
    response = requests.get("https://api.darksky.net/forecast/" + darkskySecret + "/" + coords + "," + secondsSinceEpoch + "?extend=100", params, headers=headers)
    jsonData = json.loads(response.text)
    return jsonData

#Get & parse forecast to more ML friendly format
def getHourlyForecast(coords, dt):
    jsonData = getForecast(coords, dt)
    hourlyForecasts = []
    for hourlyForecast in jsonData["hourly"]["data"]:
        feelsLikeF = hourlyForecast["apparentTemperature"]
        intensity = 0
        if "precipIntensity" in hourlyForecast:
            intensity = hourlyForecast["precipIntensity"]
        #Get & parse forecast to more ML friendly format
        rainStatus = 0
        if (intensity is not None):
            if (intensity <= 0.001):
                rainStatus = 0
            elif (intensity < 2.5):
                rainStatus = 1
            elif (intensity < 7.6):
                rainStatus = 2
            elif (intensity < 50):
                rainStatus = 3
            else:
                rainStatus = 4
            rainStatus = rainStatus * 4 + 2
        hourlyForecasts.append({"feelsLikeF": feelsLikeF, "rainLevel": rainStatus})
    return hourlyForecasts

#Create dataset from rows that contain tensorflow-friendly values
def toDataset(castedColArrs):
    dsDict = {
        "RideID": castedColArrs[0],
        "Year": castedColArrs[1],
        "Month": castedColArrs[2],
        "DOW": castedColArrs[3],
        "OpenHour": castedColArrs[4],
        "HoursOpen": castedColArrs[5],
        "MagicHours": castedColArrs[6],
        "BlockLevel": castedColArrs[7],
        "CrowdLevel": castedColArrs[8],
        "HoursSinceOpen": castedColArrs[9],
        "Temp": castedColArrs[10],
        "IsHoliday": castedColArrs[11]
    }
    if (len(castedColArrs) > 12):
        #Returned for training and testing, result is given in last column
        return tf.data.Dataset.from_tensor_slices((dsDict, castedColArrs[12]))
    else:
        #Returned for predictions, we don't have a value cause we dont know the future... yet
        return tf.data.Dataset.from_tensor_slices(dsDict)

def train(cursor, hiddenLayers, nSteps, trainPercent, query):
    cursor.execute(query)
    rows = cursor.fetchall()
    colArrs = []
    uniqueVals = []
    for row in rows:
        for i, val in enumerate(row):
            if (i >= len(colArrs)):
                colArrs.append([])
                uniqueVals.append({})
            colArrs[i].append(int(val))
            uniqueVals[i][int(val)] = True
   
    colArrSize = len(colArrs[0])
    trainSize = int(colArrSize * trainPercent)
    trainColArrs = []
    testArrs = []
    testColArrs = []
    #convert MySQL values to tensorflow-friendly values
    for arr in colArrs:
        trainArr = arr[:trainSize]
        trainColArrs.append(tf.cast(trainArr, dtype=tf.int32))
        if (trainPercent < 1):
            testArr = arr[trainSize:]
            testArrs.append(testArr)
            testColArrs.append(tf.cast(testArr, dtype=tf.int32))

    ds = toDataset(trainColArrs)
    #if not all rows go to training, use the rest for evaluation
    if (trainPercent < 1):
        testDS = toDataset(testColArrs)
    
    def inputTrain():
        return (
            ds.shuffle(2000000).batch(128).repeat().make_one_shot_iterator().get_next())

    def inputTest():
        #Don't put repeat in here lmao
        return (
            testDS.batch(1).make_one_shot_iterator().get_next())
    
    #Tell TensorFlow how to read each value
    # bucketized: value falls in given categories
    # categorical: there are only x categories
    # numeric: increasing & decreasing value means something
    featureColumns = [
        tf.feature_column.indicator_column(tf.feature_column.bucketized_column(
            source_column = tf.feature_column.numeric_column("RideID"),
            boundaries = sorted(list(uniqueVals[0].keys())))),
        tf.feature_column.indicator_column(tf.feature_column.bucketized_column(
            source_column = tf.feature_column.numeric_column("Year"),
            boundaries = sorted(list(uniqueVals[1].keys())))),
        tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(
            key='Month',
            num_buckets=12)),
        tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(
            key='DOW',
            num_buckets=7)),
        tf.feature_column.numeric_column(key="OpenHour"),
        tf.feature_column.numeric_column(key="HoursOpen"),
        tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(
            key='MagicHours',
            num_buckets=2)),
        tf.feature_column.indicator_column(tf.feature_column.bucketized_column(
            source_column = tf.feature_column.numeric_column("BlockLevel"),
            boundaries = sorted(list(uniqueVals[7].keys())))),
        tf.feature_column.numeric_column(key="CrowdLevel"),
        tf.feature_column.numeric_column(key="HoursSinceOpen"),
        tf.feature_column.numeric_column(key="Temp"),
        tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(
            key='IsHoliday',
            num_buckets=2)),
    ]
    
    model = tf.estimator.DNNRegressor(hidden_units=hiddenLayers, feature_columns=featureColumns)
    print("Training starting")
    model.train(input_fn=inputTrain, steps=nSteps)
    print("Training finished")
    if trainPercent < 1:
        print("Testing starting")
        res = model.evaluate(input_fn=inputTest)
        print("Testing result: ", res)
    return model


def predict(cursor, model, rideQuery, genResultRow, dateRange):
    #Get zipcode for forecasts of each resort (Disneyland & California Adventures are 2 parks belonging to same resort)
    getZipQuery = "SELECT id, timezone, zipcode FROM Resorts"
    cursor.execute(getZipQuery)
    
    colArrs = [[] for i in range(12)]
    colExtras = [[] for i in range(3)]
    for (resortID, resortTZ, zipcode) in cursor.fetchall():
        now = datetime.now(tz=timezone(resortTZ))
        now = now.replace(hour=0, minute=0, second=0, microsecond=0)  #round to start of day
        coords = zipToCoords(zipcode)
        #add a days worth of forecasts
        yesterday=now - timedelta(days=1)
        #get array of dates within the date range
        dateList = []
        for i in range(dateRange):
            dateList.append((now + timedelta(days=i)).strftime('%Y-%m-%d'))
        #get every resort's park's schedule in the dateList
        parksQuery = ("SELECT dt.parkID,"
            "dt.Year AS Year,"
            "dt.Month AS Month,"
            "dt.DOW AS DOW,"
            "dt.OpenHour AS OpenHour,"
            "dt.HoursOpen AS HoursOpen,"
            "dt.MagicHours AS MagicHours,"
            "dt.BlockLevel AS BlockLevel,"
            "dt.CrowdLevel AS CrowdLevel,"
            "dt.IsHoliday AS IsHoliday,"
            "MONTH(dt.date) * 31 + DAYOFMONTH(dt.date) AS DOY,"
            "dt.date AS date "
            "FROM DayTempTable dt "
            "WHERE dt.date IN (%s)")
        format_strings = ','.join(['%s'] * len(dateList))
        cursor.execute(parksQuery % format_strings, tuple(dateList))
        
        #Should run for every park (ie California Adventures, Disneyland)
        for (parkID, year, month, dow, openHour, hoursOpen, magicHours, blockLevel, crowdLevel, isHoliday, doy, parkDate) in cursor.fetchall():
            #TODO: Handle weather for entire resort instead of by park
            print("PARKID: ", parkID)
            weatherNow = now
            hourlyForecasts = []
            while len(hourlyForecasts)<=(openHour+hoursOpen):
                #add a days worth of forecasts
                hourlyForecasts.extend(getHourlyForecast(coords, weatherNow))
                #increment by a day so the next forecast we get is for the next day
                weatherNow += timedelta(days=1)
            cursor.execute(rideQuery.format(str(parkID), yesterday.strftime("%Y-%m-%d")))

            #Create row for each ride's open hour
            for (rideID) in cursor.fetchall():
                rideID = rideID[0]
                hoursSinceOpen=0
                dt = datetime.combine(parkDate, time(openHour, 0))
                while hoursSinceOpen<=hoursOpen:
                    hour = openHour + hoursSinceOpen
                    feelsLikeF = hourlyForecasts[hour]["feelsLikeF"]
                    rainStatus = hourlyForecasts[hour]["rainLevel"]
                    colArrs[0].append(int(rideID))
                    colArrs[1].append(int(year))
                    colArrs[2].append(int(month))
                    colArrs[3].append(int(dow))
                    colArrs[4].append(int(openHour))
                    colArrs[5].append(int(hoursOpen))
                    colArrs[6].append(int(magicHours))
                    colArrs[7].append(int(blockLevel))
                    colArrs[8].append(int(crowdLevel))
                    colArrs[9].append(int(hoursSinceOpen))
                    colArrs[10].append(int(feelsLikeF))
                    colArrs[11].append(int(isHoliday))
                    colExtras[0].append(int(doy))
                    colExtras[1].append(rainStatus)
                    colExtras[2].append(dt.strftime('%Y-%m-%d %H:%M:%S'))
                    hoursSinceOpen+=1
                    dt += timedelta(hours=1)
    
    #Make dataset for data to predict
    predictColArrs = [] 
    for arr in colArrs:
        predictColArrs.append(tf.cast(arr, dtype=tf.int32))
        
    predictDS = toDataset(predictColArrs)

    def inputPrediction():
        #Don't put repeat in here lmao
        return (
            predictDS.batch(1).make_one_shot_iterator().get_next())

    print(len(colArrs[0]), " predictions starting")
    predictions = model.predict(input_fn=inputPrediction)
    print("Predictions finished")

    results = []
    #Convert database table format
    for i, prediction in enumerate(predictions):
        results.append(genResultRow(i, colArrs, colExtras, prediction))
    return results

def genFastPassResultRow(i, colArrs, colExtras, prediction):
    predictionTime = dateutil.parser.parse(str(colExtras[2][i])) + timedelta(hours=colArrs[4][i]) + timedelta(minutes=int(prediction["predictions"][0]))
    return [colArrs[0][i], colExtras[0][i], colArrs[3][i], colArrs[4][i], colArrs[5][i], colArrs[6][i], colArrs[7][i], colArrs[8][i], colArrs[9][i], colArrs[10][i], colExtras[1][i], -1, colArrs[1][i], predictionTime.strftime('%Y-%m-%d %H:%M:%S'), colExtras[2][i]]

def genRideTimeResultRow(i, colArrs, colExtras, prediction):
    return [colArrs[0][i], colExtras[0][i], colArrs[3][i], colArrs[4][i], colArrs[5][i], colArrs[6][i], colArrs[7][i], colArrs[8][i], colArrs[9][i], colArrs[10][i], colExtras[1][i], int(prediction["predictions"][0]), colArrs[1][i], None, colExtras[2][i]]

#Get database connection config  
dbHost = os.environ['db_host']
dbUser = os.environ['db_user']
dbPwd = os.environ['db_pwd']
dbDb = os.environ['db_db']
dbPort = os.environ['db_port']

print("Initializing")
cnx = pymysql.connect(user=dbUser, password=dbPwd,
    host=dbHost,
    database=dbDb,
    port=int(dbPort))
cursor = cnx.cursor()

#Get google_key to get coordinates for zipcode
if ('google_key' in os.environ):
    googleKey = os.environ['google_key']

#Get darksky_secret for forecasts 
if ('darksky_secret' in os.environ):
    darkskySecret = os.environ['darksky_secret']

cursor.execute(dayTempTableQuery)

model = None
if modelMode == "FastPass":
    model = train(cursor, [90], numSteps, TRAIN_PERCENT, fastPassQuery)
    fpRideQuery = "SELECT DISTINCT(rt.rideID) FROM RideTimes rt, Rides r WHERE r.parkID={} AND rt.rideID=r.id AND DATE(rt.dateTime)>=\"{}\" AND rt.waitMins is not null AND rt.fastPassTime is not null"
    results = predict(cursor, model, fpRideQuery, genFastPassResultRow, dateRange)
    for result in results:
        cursor.execute('INSERT INTO BatchResults VALUES ({0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, "{13}", "{14}") ON DUPLICATE KEY UPDATE fastPassTime="{13}"; '.format(*result))
        sleep(0.25)
elif modelMode == "RideTime":
    model = train(cursor, [90], numSteps, TRAIN_PERCENT, rideTimesQuery)
    rtRideQuery = "SELECT DISTINCT(rt.rideID) FROM RideTimes rt, Rides r WHERE r.parkID={} AND rt.rideID=r.id AND DATE(rt.dateTime)>=\"{}\" AND rt.waitMins is not null"
    results = predict(cursor, model, rtRideQuery, genRideTimeResultRow, dateRange)
    for result in results:
        cursor.execute('INSERT INTO BatchResults VALUES ({0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, NULL, "{14}") ON DUPLICATE KEY UPDATE waitMins={11}; '.format(*result))
        sleep(0.25)
else:
    print("MODEL MODE NOT RECOGNIZED!")

print("Commiting")
cnx.commit()
print("DONE!")
