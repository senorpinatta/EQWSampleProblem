from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import Row
import numpy as np
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":

    sc = SparkContext(appName="Hello Spark")

    sqlContext = SQLContext(sc)

    df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('DataSample.csv')

    df_no_dupes = df.dropDuplicates(['TimeSt','Latitude','Longitude'])

    poiList = [(53.546167000000004, -113.48573400000001), (45.521629, -73.566024), (45.22483, -63.232729000000006)]

    def distance1(row):
        return float(np.sqrt((row.Latitude-poiList[0][0])**2+(row.Longitude-poiList[0][1])**2))

    def distance2(row):
        return float(np.sqrt((row.Latitude-poiList[1][0])**2+(row.Longitude-poiList[1][1])**2))

    def distance3(row):
        return float(np.sqrt((row.Latitude-poiList[2][0])**2+(row.Longitude-poiList[2][1])**2))


    d1rdd = df_no_dupes.rdd.map(lambda x: distance1(x))
    d1rdd.collect()
    d2rdd = df_no_dupes.rdd.map(lambda x: distance2(x))
    d2rdd.collect()
    d3rdd = df_no_dupes.rdd.map(lambda x: distance3(x))
    d3rdd.collect()

    d1List = [i for i in d1rdd.collect()]
    d2List = [i for i in d2rdd.collect()]
    d3List = [i for i in d3rdd.collect()]

    labelCol = []
    distanceCol = []
    radius1, radius2, radius3 = 0, 0, 0
    d1Count, d2Count, d3Count = 0, 0, 0
    d1Avg, d2Avg, d3Avg = 0, 0, 0

    def magic(labelCol, distanceCol):
        global radius1, radius2, radius3, d1Count, d2Count, d3Count, d1Avg, d2Avg, d3Avg
        for x in range(0,19998):
            d1 = d1List[x]
            d2 = d2List[x]
            d3 = d3List[x]
            if d1 < d2 and d1 < d3:
                if d1 > radius1:
                    radius1 = d1
                labelCol.append("POI1")
                distanceCol.append(d1)
                d1Count += 1
                d1Avg += d1
            elif d2 < d1 and d2 < d3:
                if d2 > radius2:
                    radius2 = d2
                labelCol.append("POI2")
                distanceCol.append(d2)
                d2Count += 1
                d2Avg += d1
            else:
                if d3 > radius3:
                    radius3 = d3
                labelCol.append("POI3")
                distanceCol.append(d3)
                d3Count += 1
                d3Avg += d1
        d1Avg = d1Avg/d1Count
        d2Avg = d2Avg/d2Count
        d3Avg = d3Avg/d3Count

    magic(labelCol, distanceCol)

    poi1SD, poi2SD, poi3SD = 0, 0, 0
    def SDCalc():
        global poi1SD, poi2SD, poi3SD
        for x in range(0, 19998):
            if labelCol[x] == "POI1":
                poi1SD += (distanceCol[x]-d1Avg)**2
            elif labelCol[x] == "POI2":
                poi2SD += (distanceCol[x]-d2Avg)**2
            else:
                poi3SD += (distanceCol[x]-d3Avg)**2
        poi1SD = np.sqrt(poi1SD/(d1Count))
        poi2SD = np.sqrt(poi2SD/(d2Count))
        poi3SD = np.sqrt(poi3SD/(d3Count))

    SDCalc()

    area1 = np.pi*(radius1)**2
    area2 = np.pi*(radius2)**2
    area3 = np.pi*(radius3)**2

    density1 = d1Count/area1
    density2 = d2Count/area2
    density3 = d3Count/area3

    area1 = np.pi * (radius1**2)
    area2 = np.pi * (radius2**2)
    area3 = np.pi * (radius3**2)
    
    density1 = d1Count/area1
    density2 = d1Count/area2
    density3 = d1Count/area3

    circle1 = plt.Circle((poiList[0][0], poiList[0][1]), radius=radius1, color='r')
    circle2 = plt.Circle((poiList[1][0], poiList[1][1]), radius=radius2, color='g')
    circle3 = plt.Circle((poiList[2][0], poiList[2][1]), radius=radius3, color='b')
    
    ax = plt.gca()
    ax.cla()
    
    ax.add_patch(circle3)
    ax.add_patch(circle1)
    ax.add_patch(circle2)

    print("POI_radii: ", radius1, radius2, radius3, "\nPOI_count: ", d1Count, d2Count, d3Count, "\nPOI_Avg: ", d1Avg, d2Avg, d3Avg, "\nPOI_STD: ", poi1SD, poi2SD, poi3SD, "\nPOI_Area: ", area1, area2, area3, "\nPOI_density: ", density1, density2, density3, file=open("output.txt", "w"))
    
    plt.axis('scaled')
    plt.show()

    sc.stop()