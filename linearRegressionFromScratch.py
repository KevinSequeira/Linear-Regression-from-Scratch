# ===============================================================================
# Project Details
# ===============================================================================
# Author    : Kevin Sequeira
# Date      : 17-October-2018
# Project   : Linear Regression from Scratch
# ===============================================================================
# Set the Working Directory for the Project
# ===============================================================================
print('\n' * 100);
import os;
os.chdir('D:/Programming/Machine Learning/Machine Learning with Python/Linear Regression/');
print('Current Working Directory:');
print(os.getcwd());
print();
# ===============================================================================
# Import all the necessary Packages
# ===============================================================================
from subprocess import call;
call("pip install pandas", shell = True);
import pandas as pan;
call("pip install datetime", shell = True);
from datetime import datetime as dt;
call("pip install numpy", shell = True);
import numpy as np;
call("pip install random", shell = True);
from random import randrange;
from random import seed;
call("pip install scipy", shell = True);
from scipy import stats;
call("pip install sklearn", shell = True);
from sklearn.linear_model import LinearRegression;
# ===============================================================================
# Define the Functions for the Linear Regression Project
# ===============================================================================
def loadData():
    return pan.read_csv('bayAreaHomeSales.csv');

def exploreData(dataFile):
    print('Dataset Columns: ');
    print(dataFile.columns);
    print('Dataset Shape: ', dataFile.shape);
    print('Data Information: ');
    print(dataFile.info());
    print();

def dropData(dataFile, attributes):
    dataFile.drop(columns = attributes, inplace = True);
    exploreData(dataFile);
    return dataFile;

def transformAttributeType(dataFile):
    dataFile['zindexvalue'] = dataFile['zindexvalue'].str.replace(',', '');
    dataFile['zindexvalue'] = pan.to_numeric(dataFile['zindexvalue']);

    dataFile['lastsolddate'] = dataFile['lastsolddate'].astype(str);
    dataFile['lastsolddate'] = pan.to_datetime(dataFile['lastsolddate'], format = '%m/%d/%Y');

    dataFile['lastsoldprice'] = dataFile['lastsoldprice'];
    return dataFile;

def createNewAttributes(dataFile):
    dataFile['pricepersqft'] = dataFile['lastsoldprice'] / dataFile['finishedsqft'];
    return dataFile;

def transformAttributeValues(dataFile):
    neighborhoodFrequency = dataFile.groupby('neighborhood').count()['address'];
    meanPricePerSqFt = dataFile.groupby('neighborhood').mean()['pricepersqft'];
    priceByNeighborhood = pan.concat([neighborhoodFrequency, meanPricePerSqFt], axis = 1);
    priceByNeighborhood.columns = ['frequency', 'pricepersqft'];
    print(priceByNeighborhood.describe());
    print();

    lowPriceNeighborhoods = priceByNeighborhood[priceByNeighborhood['pricepersqft'] < priceByNeighborhood.describe()['pricepersqft'][50%];
    lowPriceNeighborhoods = lowPriceNeighborhoods.index;
    highPriceLowFreqNeighborhoods = priceByNeighborhood[priceByNeighborhood['pricepersqft'] >= priceByNeighborhood.describe()['pricepersqft'][50%]];
    highPriceLowFreqNeighborhoods = highPriceLowFreqNeighborhoods[highPriceLowFreqNeighborhoods['frequency'] < priceByNeighborhood.describe()['frequency'][50%]];
    highPriceLowFreqNeighborhoods = highPriceLowFreqNeighborhoods.index;
    highPriceHighFreqNeighborhoods = priceByNeighborhood[priceByNeighborhood['pricepersqft'] >= priceByNeighborhood.describe()['pricepersqft'][50%]];
    highPriceHighFreqNeighborhoods = highPriceHighFreqNeighborhoods[highPriceHighFreqNeighborhoods['frequency'] >= priceByNeighborhood.describe()['frequency'][50%]]];
    highPriceHighFreqNeighborhoods = highPriceHighFreqNeighborhoods.index;

    def groupByHood(hood):
        if hood in lowPriceNeighborhoods:
            return 'lowprice';
        elif hood in highPriceLowFreqNeighborhoods:
            return 'highpricelowdemand';
        else:
            return 'highpricehighdemand';

    dataFile['neighborhood'] = dataFile['neighborhood'].apply(lambda x: groupByHood(x));
    # dataFile.ix[dataFile['neighborhood'].isin(list(lowPriceNeighborhoods), 'neighborhood'] = '1.0'
    mapping = {'lowprice': 1.0, 'highpricelowdemand': 2.0, 'highpricehighdemand': 3.0};
    dataFile = dataFile.replace({'neighborhood': mapping});
    # print(dataFile['neighborhood']);
    # print();
    dataFile['neighborhood'] = pan.to_numeric(dataFile['neighborhood']);

    def groupByUsecode(usecode):
        if usecode in useCodes:
            return useCodes.index(usecode) + 1;

    # useCodes = dataFile['usecode'].values.tolist();
    useCodes = dataFile['usecode'].value_counts().index.tolist()
    dataFile['usecode'] = dataFile['usecode'].apply(groupByUsecode);
    return dataFile;

def transformData(dataFile):
    dataFile = transformAttributeType(dataFile);
    dataFile = createNewAttributes(dataFile);
    dataFile = transformAttributeValues(dataFile);
    exploreData(dataFile);
    return dataFile;

def createDummyVariables(dataFile, attributes):
    for attribute in attributes:
        dummyList = pan.get_dummies(dataFile[attribute]);
        dataFile = pan.concat([dataFile, dummyList], axis = 1);
        dataFile.drop(columns = attribute, inplace = True);
    exploreData(dataFile);
    return dataFile;

def correlationMatrix(dataFile):
    corrMatrix = dataFile.corr();
    print('Numeric Column Correlation with Last Sold Price: ');
    print(corrMatrix['lastsoldprice'].sort_values(ascending = False));
    print();

def normalizeData(dataFile):
    dataFile = (dataFile - dataFile.mean()) / dataFile.std();
    print(dataFile.describe());
    return dataFile;

def createTargetData(dataFile):
    dataFileTarget = pan.DataFrame(dataFile['lastsoldprice'], columns = ['lastsoldprice']);
    dataFile = dropData(dataFile,
                        attributes = ['lastsoldprice']);
    print(dataFile.info(), '\n');
    print(dataFileTarget.info(), '\n');
    return dataFile, dataFileTarget;

def divideDataTrainTest(dataFile, dataFileTarget, trainPercent, testPercent):
    dataFile = np.array(dataFile);
    dataFileTarget = np.array(dataFileTarget);
    dataFileCopy = dataFile.copy();
    dataFileTargetCopy = dataFileTarget.copy();
    recordCount = int(len(dataFile) * trainPercent / 100);
    testFile = list();
    testTarget = list();
    while len(dataFileCopy) >= recordCount:
        index = randrange(len(dataFileCopy));
        testFile.append(dataFileCopy[index]);
        testTarget.append(dataFileTarget[index]);
        dataFileCopy = np.delete(dataFileCopy, index, 0);
        dataFileTargetCopy = np.delete(dataFileTargetCopy, index, 0);
    dataFile = dataFileCopy.copy();
    dataFile = np.hstack([np.ones((len(dataFile), 1), float), dataFile]);
    dataFileTarget = dataFileTargetCopy.copy();
    testFile = np.array(testFile);
    testFile = np.hstack([np.ones((len(testFile), 1), float), testFile]);
    testTarget = np.array(testTarget).reshape(len(testTarget), 1);
    del dataFileCopy, dataFileTargetCopy;
    print('Training Data Dimensions: ', dataFile.shape);
    print('Training Target Dimensions: ', dataFileTarget.shape);
    print('Test Data Dimensions: ', testFile.shape);
    print('Test Target Dimensions: ', testTarget.shape);
    return dataFile, dataFileTarget, testFile, testTarget;

def costFunction(dataFile, dataTarget, coeffMatrix):
    dataLength = len(dataFile);
    costValue = np.sum(((dataFile.dot(coeffMatrix.T)) - dataTarget) ** 2) / (2 * dataLength);
    return costValue;

def gradientDescent(dataFile, dataTarget, coeffMatrix, alpha, iterations):
    costHistory = [0] * iterations;
    dataLength = len(dataTarget);
    for iteration in range(iterations):
        loss = dataFile.dot(coeffMatrix.T) - dataTarget;
        gradient = dataFile.T.dot(loss) / dataLength;
        coeffMatrix = coeffMatrix - (alpha * gradient.T);
        costValue = costFunction(dataFile, dataTarget, coeffMatrix);
        costHistory[iteration] = costValue;
    print('Model Coefficients: ');
    print(coeffMatrix);
    print('Model Cost: ', costHistory[-1], '\n');
    return coeffMatrix, costHistory;

def developLinearModel(dataFile, dataTarget):
    coeffMatrix = np.zeros((1, len(dataFile[0])));
    print(coeffMatrix);
    print();
    initialCost = costFunction(dataFile, dataTarget, coeffMatrix);
    print('Initial Cost: ', initialCost, '\n');
    coeffMatrix, costHistory = gradientDescent(dataFile, dataTarget, coeffMatrix, 0.2, 20000);
    return coeffMatrix;

def predict(testData, coeffMatrix):
    predictedTarget = testData.dot(coeffMatrix.T);
    return predictedTarget;

def calculateRMSE(testTarget, predictedTarget):
    modelRMSE = np.sqrt(sum((testTarget - predictedTarget) ** 2) / len(testTarget));
    return modelRMSE;

def modelRSquared(testTarget, predictedTarget):
    targetMean = np.mean(testTarget);
    totalSSE = np.sum((testTarget - targetMean) ** 2);
    residualSSE = np.sum((testTarget - predictedTarget) ** 2);
    rSquared = 1 - (residualSSE / totalSSE);
    return rSquared;

def calculatePValueChiSquare(testTarget, predictedTarget):
    chiSquare = np.sum(((predictedTarget - testTarget) ** 2) / predictedTarget);
    pValueChiSquare = 1 - stats.chi2.cdf(chiSquare, 1);
    print('Model Chi-Square: ', chiSquare, '\n');
    return pValueChiSquare;

def calculatePValueFStatistic(testTarget, predictedTarget, degFreeRegression, degFreeResidual):
    meanRegressionSSE = np.sum((predictedTarget - np.mean(predictedTarget)) ** 2) / degFreeRegression;
    meanResidualSSE = np.sum((testTarget - predictedTarget) ** 2) / degFreeResidual;
    fStatistic = meanRegressionSSE / meanResidualSSE;
    print('Model F-Statistic: ', fStatistic, '\n');
    pValueFStatistic = 1 - stats.f.cdf(fStatistic, degFreeRegression, degFreeResidual);
    return pValueFStatistic;
# ===============================================================================
# Code for the Main Program
# ===============================================================================
# seed(111);
seed(111);
pan.set_option('display.max_columns', 20)
housingData = loadData();
exploreData(housingData);
housingData = dropData(housingData,
                       attributes = ['Unnamed: 0',
                                     'info',
                                     'z_address',
                                     'zestimate',
                                     'zipcode',
                                     'zpid']);
housingData = transformData(housingData);
housingData = dropData(housingData,
                       attributes = ['address',
                                     'lastsolddate',
                                     'latitude',
                                     'longitude',
                                     'pricepersqft',
                                     'usecode']);
print(housingData.describe(), '\n');
correlationMatrix(housingData);
# housingData = createDummyVariables(housingData,
#                                    attributes = ['neighborhood',
#                                                  'usecode']);
housingData = normalizeData(housingData);
trainingData, trainingTarget = createTargetData(housingData);
trainingData, trainingTarget, testData, testTarget = divideDataTrainTest(trainingData, trainingTarget, 75, 25);
linearModel = developLinearModel(trainingData, trainingTarget);
predictedTarget = predict(testData, linearModel);
modelRMSE = calculateRMSE(testTarget, predictedTarget);
print('Model RMSE: ', modelRMSE, '\n');
modelR2 = modelRSquared(testTarget, predictedTarget);
print('Model R-Squared: ', modelR2, '\n');
modelPValueChiSqTest = calculatePValueChiSquare(testTarget, predictedTarget);
print('Model P-Value for Chi-Square Test: ', modelPValueChiSqTest, '\n');
modelPValueFStatistic = calculatePValueFStatistic(testTarget, predictedTarget,
                                                  len(trainingData[0]),
                                                  len(trainingData) - len(trainingData[0]) - 1);
print('Model P-Value for F-Statistic: ', modelPValueFStatistic, '\n');
# ===============================================================================
# Code for the Main Program using Scikit Learn
# ===============================================================================
linearRegression = LinearRegression();
linearRegression.fit(trainingData, trainingTarget);
predictedTarget = linearRegression.predict(testData);
print('Scikit Learn Model R-Squared: ', linearRegression.score(testData, testTarget));
print();
