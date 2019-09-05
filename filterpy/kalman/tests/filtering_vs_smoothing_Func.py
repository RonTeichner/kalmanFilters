from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter, update, predict, batch_filter
import scipy.io as sio

def dbm2var(x_dbm):
    return np.power(10, np.divide(x_dbm - 30, 10))

def volt2dbm(x_volt):
    return 10*np.log10(np.power(x_volt, 2)) + 30

def volt2db(x_volt):
    return 10*np.log10(np.power(x_volt, 2))

def watt2dbm(x_volt):
    return 10*np.log10(x_volt) + 30

def watt2db(x_volt):
    return 10*np.log10(x_volt)

def simFiltering2SmoothingErrorPower(desiredInputMeasNoise_dbm, desiredOutputMeasNoise_dbm, F, Q, H, nBatch, enablePlot):

    R = np.diag([dbm2var(desiredInputMeasNoise_dbm), dbm2var(desiredOutputMeasNoise_dbm)])

    xDim = F.shape[0]
    measDim = H.shape[0]

    Q_dbm = 10*np.log10(np.diag(Q)) + 30
    R_dbm = 10*np.log10(np.diag(R)) + 30

    nSamples = 10000

    meanErrF2ErrSPower_Linear = np.zeros(nBatch)
    meanErrS2ErrFPower_Linear = np.zeros(nBatch)
    outputMeasSnr_Linear = np.zeros(nBatch)
    for batchIdx in range(nBatch):
        if batchIdx % 10 == 0:
            print('starting iter %d' % batchIdx)
        # generate state vector and measurement:
        xGroundTruth = np.zeros((nSamples, xDim))
        xPredictGroundTruth = np.zeros((nSamples, xDim))
        xGroundTruth[0] = np.multiply(np.power(np.diag(Q), 0.5), np.random.randn(1, xDim))
        y = np.zeros((nSamples, measDim))
        yGroundTruth = np.zeros((nSamples, measDim))
        yGroundTruth[0] = np.dot(H, xGroundTruth[0])
        y[0] = yGroundTruth[0] + np.multiply(np.power(np.diag(R), 0.5), np.random.randn(1, measDim))

        for n in range(1, nSamples):
            processNoise = np.multiply(np.power(np.diag(Q), 0.5), np.random.randn(1, xDim))
            xPredictGroundTruth[n] = np.dot(F, xGroundTruth[n-1])
            xGroundTruth[n] = xPredictGroundTruth[n] + processNoise

            measurementNoise = np.multiply(np.power(np.diag(R), 0.5), np.random.randn(1, measDim))
            yGroundTruth[n] = np.dot(H, xGroundTruth[n])
            y[n] = yGroundTruth[n] + measurementNoise

        xGroundTruth_dbm = 10 * np.log10(np.power(xGroundTruth, 2)) + 30
        yGroundTruth_dbm = 10 * np.log10(np.power(yGroundTruth, 2)) + 30
        y_dbm = 10 * np.log10(np.power(y, 2)) + 30

        inputNoisePower_dbm = 10*np.log10(np.diag(R)[0]) + 30
        outputNoisePower_dbm = 10*np.log10(np.diag(R)[1]) + 30
        inputMeanSigPower_dbm = 10*np.log10(np.mean(np.power(yGroundTruth[:,0],2))) + 30
        outputMeanSigPower_dbm = 10*np.log10(np.mean(np.power(yGroundTruth[:,1],2))) + 30
        inputMeasSnr_db = inputMeanSigPower_dbm - inputNoisePower_dbm
        outputMeanSnr_db = outputMeanSigPower_dbm - outputNoisePower_dbm

        outputMeasSnr_Linear[batchIdx] = np.mean(np.divide(np.power(yGroundTruth[:, 1], 2), np.diag(R)[1]))

        processNoiseMeanPower_dbm = 10*np.log10(np.mean(np.power(xGroundTruth - xPredictGroundTruth, 2), axis=0)) + 30
        autonomousStateMeanPower_dbm = 10*np.log10(np.mean(np.power(xPredictGroundTruth, 2), axis=0)) + 30
        processMeanSnr_db = autonomousStateMeanPower_dbm - processNoiseMeanPower_dbm

        if enablePlot and batchIdx == 0:
            print('input meas snr: %d [db]; output meas snr: %d [db]' %(inputMeasSnr_db, outputMeanSnr_db))
            print('process mean snr [db]: ', np.array_str(processMeanSnr_db, precision=0, suppress_small=True))

        if batchIdx == 0:
            kf = KalmanFilter(dim_x=xDim, dim_z=measDim)
            kf.F = F
            kf.H = H
            kf.R = R
            kf.Q = Q

        kf.x = np.zeros((xDim, 1))
        kf.P = np.diag(Q)[0] * np.eye(xDim)

        #kf.test_matrix_dimensions()

        muF, covF, _, _ = kf.batch_filter(y)
        muS, covS, _, _ = kf.rts_smoother(muF, covF)

        errF = xGroundTruth - muF[:,:,0]
        errS = xGroundTruth - muS[:,:,0]

        errF_rel_lin = np.abs(np.divide(errF, xGroundTruth))
        errS_rel_lin = np.abs(np.divide(errS, xGroundTruth))

        errF_W = np.power(errF, 2)
        errS_W = np.power(errS, 2)
        errF_dbm = watt2dbm(errF_W)
        errS_dbm = watt2dbm(errS_W)

        errF2errS_db = errF_dbm - errS_dbm

        nCroppedSamples = 1000
        a = int(nSamples/2 - nCroppedSamples/2)
        b = a + 100
        tVec = np.arange(a,b)

        # cropping:
        feature = -10

        errF = errF[a:b, feature]
        errS = errS[a:b, feature]
        errF_W = errF_W[a:b, feature]
        errS_W = errS_W[a:b, feature]
        errF_rel_lin = errF_rel_lin[a:b, feature]
        errS_rel_lin = errS_rel_lin[a:b, feature]
        muF = muF[a:b, feature]
        muS = muS[a:b, feature]
        errF2errS_db = errF2errS_db[a:b, feature]

        meanErrF2ErrSPower_Linear[batchIdx] = np.divide(np.median(errF_W), np.median(errS_W))
        meanErrS2ErrFPower_Linear[batchIdx] = np.divide(np.median(errS_W), np.median(errF_W))
        # wrong: meanErrF2ErrSPower_Linear[batchIdx] = np.mean(np.divide(errS_W, errF_W))
        #meanErrF2ErrS_dbm = 10*np.log10(meanErrF2ErrSPower_Linear[batchIdx]) + 30

        if enablePlot and batchIdx == int(nBatch/2):
            plt.subplot(3,1,1)
            plt.plot(tVec, xGroundTruth[a:b, feature], label='groundTruth')
            #plt.plot(tVec, y[a:b, -1], label='measurement')
            plt.plot(tVec, muF, label='filtering')
            plt.plot(tVec, muS, label='smoothing')
            plt.title('snr: %d' %outputMeanSnr_db)
            plt.legend()
            plt.subplot(3,1,2)
            plt.plot(tVec, (errF_W), label='filtering error')
            plt.plot(tVec, (errS_W), label='smoothing error')
            plt.title('estimation relative errors')
            plt.ylabel('W')
            plt.legend()
            plt.subplot(3,1,3)
            plt.plot(tVec, np.divide(errF_W , errS_W))
            #plt.ylabel('W')
            plt.title('filtering 2 smoothing error power; mean: %d [db]' % watt2db(meanErrF2ErrSPower_Linear[batchIdx]))
            plt.show()

    return meanErrF2ErrSPower_Linear, meanErrS2ErrFPower_Linear, outputMeasSnr_Linear