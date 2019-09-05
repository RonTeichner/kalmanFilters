from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter, update, predict, batch_filter
import scipy.io as sio
from tempfile import TemporaryFile
from filtering_vs_smoothing_Func import *

mat_contents = sio.loadmat('../../../../traffic/mcRun01/sDataset.mat')
sDatasetRnn = mat_contents['sDatasetRnn']
partialF = (sDatasetRnn['kalmanMatrices'][0,0]['F'][0,0])
partialQ = ((sDatasetRnn['kalmanMatrices'][0,0]['Q'][0,0]))[0]
#kalmanR = ((sDatasetRnn['kalmanMatrices'][0,0]['R'][0,0]))[0]
partialH = ((sDatasetRnn['kalmanMatrices'][0,0]['H'][0,0]))[0]

F = partialF[0]
Q = partialQ
H = partialH

F[-2,-1] = 1
F[-10,-1] = 1
F[-12,-1] = 1

#F = 0.01*np.random.randn(F.shape[0], F.shape[1])

nMonteCarlo = 100

enablePlot = True

#desiredInputMeasNoise_dbm = 100
a = -10
#desiredOutputMeasNoiseVec_dbm = np.arange(a, a+1)#np.arange(-35, 5)
desiredOutputMeasNoiseVec_dbm = np.arange(-35, 15, 1)
nItrs = desiredOutputMeasNoiseVec_dbm.size

meanErrF2ErrS_db = np.zeros(nItrs)
meanErrS2ErrF_db = np.zeros(nItrs)
outputMeasSnrPower_db = np.zeros(nItrs)
for itr in range(nItrs):
    print('starting general itr %d out of %d' %(itr, nItrs))
    desiredInputMeasNoise_dbm = desiredOutputMeasNoiseVec_dbm[itr] + 6
    meanErrF2ErrS_linear, meanErrS2ErrF_linear, outputMeasSnrPower_Linear = simFiltering2SmoothingErrorPower(desiredInputMeasNoise_dbm, desiredOutputMeasNoiseVec_dbm[itr], F, Q, H, nMonteCarlo, enablePlot)
    meanErrF2ErrS_db[itr] = 10*np.log10(np.mean(meanErrF2ErrS_linear))
    meanErrS2ErrF_db[itr] = 10*np.log10(np.mean(meanErrS2ErrF_linear))
    outputMeasSnrPower_db[itr] = 10*np.log10(np.mean(outputMeasSnrPower_Linear))
    #print('meanErrF2ErrS_db:', np.array_str(10*np.log10(meanErrF2ErrS_linear), precision=0, suppress_small=True))
    #print('outputMeasSnr_db:', np.array_str(10*np.log10(outputMeasSnrPower_Linear), precision=0, suppress_small=True))

sortedIdx = np.argsort(outputMeasSnrPower_db)
outputMeasSnrPower_db = outputMeasSnrPower_db[sortedIdx]
meanErrF2ErrS_db = meanErrF2ErrS_db[sortedIdx]
meanErrS2ErrF_db = meanErrS2ErrF_db[sortedIdx]

plt.subplot(2,1,1)
plt.plot(outputMeasSnrPower_db, meanErrF2ErrS_db)
plt.xlabel('meas snr [db]')
plt.ylabel('[db]')
plt.title('filtering2smoothing error (power ratio)')

plt.subplot(2,1,2)
plt.plot(outputMeasSnrPower_db, meanErrS2ErrF_db)
plt.xlabel('meas snr [db]')
plt.ylabel('[db]')
plt.title('smoothing2filtering error (power ratio)')
plt.show()

np.save('./meanErrF2ErrS_db_file.npy', meanErrF2ErrS_db)
np.save('./outputMeasSnrPower_db_file.npy', outputMeasSnrPower_db)