import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pdb
import traceback
""" All the Code Below Is For the Second Generation Packetizer """

progAmpNames = ['program{}_amplitude'.format(progIdx) for progIdx in range(4)]
progPWNames = ['program{}_pw'.format(progIdx) for progIdx in range(4)]
groupConfigNames = ['TherapyConfigGroup{}'.format(grpIdx) for grpIdx in range(4)]
metaMatrixColumns = [
    'dataSize', 'dataTypeSequence',
    'systemTick', 'timestamp', 'microloss', 'macroloss', 'bothloss',
    'jsonPacketIdx', 'chanSamplesLen', 'dataSizePerChannel', 'SampleRate',
    'PacketGenTime', 'PacketRxUnixTime',
    'firstSampleTick', 'lastSampleTick', 'globalSequence', 'skipPacket', 'packetNeedsFixing']

def init_numpy_array(input_json, num_cols, data_type):
    num_rows = len(input_json[0][data_type])
    return np.zeros((num_rows, num_cols))


def strip_prog_name(x):
    return int(x.split('program')[-1].split('_')[0])

def unpack_meta_matrix_time(
        meta_matrix, intersample_tick_count, usePacketGenTime=True):
    # Initialize variables for looping:
    master_time = meta_matrix[0, 3]
    prev_master_time = meta_matrix[0, 3]
    # packet GEN time
    pGenT = meta_matrix[0, 11]
    prev_pGenT = meta_matrix[0, 11]
    # packet RX time
    pRxT = meta_matrix[0, 12]
    prev_pRxT = meta_matrix[0, 12]
    #
    old_system_tick = 0
    running_tick_counter = 0
    firstSampleTick = np.zeros(meta_matrix.shape[0])
    lastSampleTick = np.zeros(meta_matrix.shape[0])
    unresolvedMacrolossIdx = []
    resolvedMacroloss = True
    print('Unpacking meta matrix time')
    for packet_number, i in enumerate(meta_matrix):
        # packet_number = meta_matrix[sorted_packet_number, 7]
        if i[16]:
            # if instructed to toss
            continue
        master_time = i[3]  # Resets the master time
        pGenT = i[11]  # Resets the packet gen time
        pRxT = i[12]  # Resets the packet rx time
        if i[5] or (not resolvedMacroloss):
            # We just suffered a macro packet loss...
            print('meta matrix: adding time to the running counter')
            if usePacketGenTime:
                # msk = meta_matrix[:, 11] > 0
                # plt.plot(meta_matrix[msk, 11]); plt.show()
                # plt.plot(np.diff(meta_matrix[:, 3])); plt.show()
                # plt.plot(np.diff(meta_matrix[:, 12])); plt.show()
                if ~((pGenT < 0) or (prev_pGenT<0)):
                    gap = pGenT - prev_pGenT  # in msec
                else:
                    gap = pRxT - prev_pRxT # in msec
                #
                n_rollovers = np.floor(gap / ((2 ** 16) * 1e-1))
                # did we just roll over?
                gapRemainder = (gap - n_rollovers * (2 ** 16) * 1e-1)
                # are we very close to a new rollover?
                altGapRemainder = np.abs(gap - (n_rollovers + 1) * (2 ** 16) * 1e-1)
                if not min(gapRemainder, altGapRemainder) < 250:
                    resolvedMacroloss = True
                    print('     resolved')
                    running_tick_counter += n_rollovers * 2 ** 16
                else:
                    unresolvedMacrolossIdx.append(packet_number)
                    resolvedMacroloss = False
                    print('   unresolved')
            else:
                min_gap = (master_time - (prev_master_time + 1))
                max_gap = min_gap + 2
                min_n_rollovers = np.floor(min_gap / ((2 ** 16) * 1e-4))
                max_n_rollovers = np.floor(max_gap / ((2 ** 16) * 1e-4))
                if min_n_rollovers == max_n_rollovers:
                    resolvedMacroloss = True
                    print('     resolved')
                    running_tick_counter += min_n_rollovers * 2 ** 16
                else:
                    unresolvedMacrolossIdx.append(packet_number)
                    resolvedMacroloss = False
                    print('   unresolved')
            # old_system_tick = 0
            # running_tick_counter = 0
        if resolvedMacroloss:
            curr_system_tick = i[2]
            running_tick_counter += (curr_system_tick - old_system_tick) % (2 ** 16)
            # i[9] = packet['Header']['dataSize'] / (2 * len(packet['ChannelSamples']))
            backtrack_time = running_tick_counter - ((i[9] - 1) * intersample_tick_count)
            #
            firstSampleTick[packet_number] = backtrack_time
            lastSampleTick[packet_number] = running_tick_counter
            # Update counters for next loop
            old_system_tick = curr_system_tick
            prev_master_time = master_time
            prev_pGenT = pGenT
            prev_pRxT = pRxT
    #
    for packet_number in unresolvedMacrolossIdx[::-1]:
        curr_system_tick = meta_matrix[packet_number, 2]
        next_system_tick = meta_matrix[packet_number + 1, 2]
        sys_tick_increment = (next_system_tick - curr_system_tick) % (2 ** 16)
        lastSampleTick[packet_number] = lastSampleTick[packet_number + 1] - sys_tick_increment
        firstSampleTick[packet_number] = (
            lastSampleTick[packet_number] -
            ((meta_matrix[packet_number, 9] - 1) * intersample_tick_count))
    meta_matrix[:, 13] = firstSampleTick
    meta_matrix[:, 14] = lastSampleTick
    return meta_matrix


def correct_meta_matrix_consecutive_sys_tick(
        meta_matrix, intersampleTickCount=None,
        frameDuration=None, nominalFrameSize=None, 
        forceRemoveEqualConsecutiveDataTypeSequence=False, verbose=False):
    # correct issue described on page 64 of summit user manual
    metaMatrix = pd.DataFrame(meta_matrix, columns=metaMatrixColumns)
    sysTickDiff = metaMatrix['systemTick'].diff()
    tsDiff = metaMatrix['PacketGenTime'].diff()
    metaMatrix['rolloverGroup'] = ((sysTickDiff < 0) | (tsDiff > (2 ** 16 * 1e-1))).cumsum()
    # metaMatrix['rolloverGroup'] = (sysTickDiff < 0).cumsum()
    # TODO access deviceLog to find what the nominal frame size actually is
    if frameDuration is None:
        # frameDuration is the duration in msec
        # frameSize is the # of samples in the frame
        if nominalFrameSize is None:
            nominalFrameSize = metaMatrix['dataSizePerChannel'].value_counts().idxmax()
            print('nominalFrameSize = {}'.format(nominalFrameSize))
        frameDuration = nominalFrameSize * intersampleTickCount * 1e-1
    for name, group in metaMatrix.groupby('rolloverGroup'):
        # duplicateSysTick = group.duplicated('systemTick')
        duplicateSysTick = (group['systemTick'].diff() == 0)
        if duplicateSysTick.any():
            # duplicateIdxs = duplicateSysTick.index[np.flatnonzero(duplicateSysTick)]
            duplicateIdxs = duplicateSysTick.loc[duplicateSysTick].index
            alreadyVisited = pd.Series(False, index=group.index)
            for duplicateIdx in duplicateIdxs:
                sysTickVal = group.loc[duplicateIdx, 'systemTick']
                allOccurences = group.loc[group['systemTick'] == sysTickVal, :].copy()
                # mark these packets for deletion?
                if len(allOccurences['dataTypeSequence'].unique()) == 1:
                    if forceRemoveEqualConsecutiveDataTypeSequence:
                        meta_matrix[allOccurences.index, -2] = True # "skipPacket"
                else:
                    # allOccurences['PacketGenTime'] - metaMatrix['PacketRxUnixTime'].min()
                    # allOccurences['PacketRxUnixTime'] - metaMatrix['PacketRxUnixTime'].min()
                    # allOccurences['timestamp']- metaMatrix['timestamp'].min()
                    # try:
                    #     assert len(allOccurences) == 2
                    # except Exception:
                    #     print('WARNING! More than 2 consecutive identical system ticks found')
                    #     pdb.set_trace()
                    #  'dataTypeSequence' rolls over, correct for it
                    # dtsRolledOver = allOccurences['dataTypeSequence'].diff().fillna(1) < -100
                    # addToDTS = allOccurences['dataTypeSequence'] ** 0 - 1
                    # for subI, dtsRolled in dtsRolledOver.iteritems():
                    #     if dtsRolled:
                    #         addToDTS.loc[subI:] += 255
                    # allOccurences.loc[:, 'dataTypeSequence'] += addToDTS
                    # assert (allOccurences.loc[:, 'dataTypeSequence'].diff().fillna(1) >= 0).all()
                    # specialCase = (
                    #     (allOccurences['dataTypeSequence'] == 255).any() &
                    #     (allOccurences['dataTypeSequence'] == 0).any()
                    # )
                    # 
                    # if specialCase:
                    #     idxNeedsChanging = allOccurences['dataTypeSequence'].astype(np.int).idxmax()
                    # else:
                    #     idxNeedsChanging = allOccurences['dataTypeSequence'].astype(np.int).idxmin()
                    # pdb.set_trace()
                    if not alreadyVisited.loc[allOccurences.index].all():
                        addToSysTick = 10 * frameDuration * np.arange(-allOccurences.shape[0] + 1, 0)
                        #
                        #
                        for j, idxNeedsChanging in enumerate(allOccurences.index[:-1]):
                            meta_matrix[idxNeedsChanging, -1] = True
                            newSysTick = meta_matrix[idxNeedsChanging, 2] + addToSysTick[j]
                            # print('newSysTick = {}'.format(newSysTick))
                            if newSysTick < 0:
                                newSysTick += 2 ** 16
                            meta_matrix[idxNeedsChanging, 2] = newSysTick
                        alreadyVisited.loc[allOccurences.index] = True
    return meta_matrix


def correct_meta_matrix_time_displacement(
        meta_matrix, intersample_tick_count,
        verbose=False, plotting=False):
    tdMeta = pd.DataFrame(
        meta_matrix[:, [1, 2, 3, 4, 5, 6, 9, 13, 14]],
        columns=[
            'dataTypeSequence', 'systemTick', 'masterClock',
            'microloss', 'macroloss', 'bothloss',
            'packetSize', 'firstSampleTick', 'lastSampleTick']
        )
    # how far is this packet from the preceding one
    tdMeta['sampleGap'] = tdMeta['firstSampleTick'].values - tdMeta['lastSampleTick'].shift(1).values
    # how much does this packet overlap the next one?
    tdMeta['sampleOverlap'] = tdMeta['lastSampleTick'].values - tdMeta['firstSampleTick'].shift(-1).values
    #
    tdMeta['displacementDifference'] = (tdMeta['sampleGap'].values + tdMeta['sampleOverlap'].values) / 2
    tdMeta['packetsNotLost'] = ~(tdMeta['microloss'].astype(bool) | tdMeta['macroloss'].astype(bool))
    #
    tdMeta['packetsOverlapFuture'] = tdMeta['sampleOverlap'] > 0
    #
    packetsNeedFixing = tdMeta.index[tdMeta['packetsOverlapFuture']]
    if plotting and packetsNeedFixing.any():
        ax = sns.distplot(
            tdMeta.loc[packetsNeedFixing, 'sampleGap'].dropna(),
            label='sampleGap'
            )
        ax = sns.distplot(
            tdMeta.loc[packetsNeedFixing, 'sampleOverlap'].dropna(),
            label='sampleOverlap'
            )
        ax = sns.distplot(
            tdMeta.loc[packetsNeedFixing, 'displacementDifference'].dropna(),
            label='displacementDifference'
            )
        plt.legend()
        plt.show()
    # calc correction:
    nextGoodSysTicks = tdMeta.loc[packetsNeedFixing + 1, 'systemTick']
    nextGoodPacketSizes = tdMeta.loc[packetsNeedFixing + 1, 'packetSize']
    correctedSysTicks = (
        nextGoodSysTicks.to_numpy() -
        nextGoodPacketSizes.to_numpy() * intersample_tick_count)
    tdMeta.loc[packetsNeedFixing, 'systemTick'] = correctedSysTicks
    #print(tdMeta.loc[packetsNeedFixing, 'systemTick'])
    #print('difference in systicks is\n{}'.format(tdMeta.loc[packetsNeedFixing, 'systemTick'] - correctedSysTicks))
    #print('original sample gap was\n{}'.format(tdMeta.loc[packetsNeedFixing, 'sampleGap']))
    #tdMeta.loc[packetsNeedFixing + 1, 'lastSampleTick']
    # correctiveValues = tdMeta.loc[packetsNeedFixing, 'sampleGap'].fillna(method = 'bfill') * 10
    # tdMeta.loc[packetsNeedFixing, 'systemTick'] = tdMeta.loc[
    #     packetsNeedFixing, 'systemTick'] - (round(correctiveValues) - intersample_tick_count)
    meta_matrix[:, 2] = tdMeta['systemTick'].values
    meta_matrix[packetsNeedFixing, -1] = True
    # meta_matrix[:, 11] = tdMeta['packetsOverlapFuture'].to_numpy()
    return meta_matrix


def process_meta_data(
        meta_matrix, sampleRateLookupDict=None, frameSize=None, frameDuration=None,
        intersampleTickCount=None, plotting=False, fixPacketGenTime=True,
        forceRemoveEqualConsecutiveDataTypeSequence=True,
        input_json=None):
    metaDF = pd.DataFrame(meta_matrix, columns=metaMatrixColumns)
    metaDF.loc[:, 'packetNeedsFixing'] = metaDF.loc[:, 'packetNeedsFixing'].astype('bool')
    metaDF.loc[:, 'skipPacket'] = metaDF.loc[:, 'skipPacket'].astype('bool')
    ########
    metaDF.sort_values(
        by=['PacketRxUnixTime', 'dataTypeSequence', 'globalSequence'],
        kind='mergesort', inplace=True)
    metaDF.reset_index(drop=True, inplace=True)
    ######## fixPacketGenTime
    if fixPacketGenTime:
        noGenTimeMask = metaDF['PacketGenTime'] < 0
        metaDF.loc[noGenTimeMask, 'PacketGenTime'] = np.nan
    #
    if plotting:
        rxTDiff = metaDF['PacketRxUnixTime'].diff()
        fig, ax = plt.subplots(5, 1, figsize=(30, 15))
        twinAx = []
        # ax[0] checks for need to sort
        p0, = ax[0].plot(
            rxTDiff.index, rxTDiff,
            label='RX time diff (msec)')
        rxTAx = ax[0].twinx()
        twinAx.append(rxTAx)
        twinP0, = rxTAx.plot(
            metaDF.index, metaDF['PacketRxUnixTime'], 'c',
            label='RX time (msec)')
        rxTAx.plot(
            metaDF.index[noGenTimeMask],
            metaDF.loc[noGenTimeMask, 'PacketRxUnixTime'],
            'y*', label='Packet Gen Time < 0 (missing)')
        rxTAx.set_ylabel('RX time (msec)')
        rxTAx.legend(loc='lower right')
        rxTAx.yaxis.get_label().set_color(twinP0.get_color())
        # ax[1] deals with PacketGenTime
        actualLatency = metaDF['PacketRxUnixTime'] - metaDF['PacketGenTime']
        p1, = ax[1].plot(
            metaDF.index, actualLatency,
            label='Rx - Gen')
        ax[1].axhline(actualLatency.median(), c='r')
        ax[1].text(
                1, 0, 'mean = {:.2f} msec (std = {:.2f} msec)'.format(
                    actualLatency.mean(), actualLatency.std()),
                va='bottom', ha='right',
                transform=ax[1].transAxes)
        # ax[2] deals with dataTypeSeq
        dtsDiff = metaDF['dataTypeSequence'].diff()
        dtsAx = ax[2].twinx()
        twinAx.append(dtsAx)
        twinP2, = dtsAx.plot(
            metaDF.index, metaDF['dataTypeSequence'], 'c',
            label='dataTypeSequence')
        p2, = ax[2].plot(
            dtsDiff.index, dtsDiff, '-o',
            label='dataTypeSequence diff (excluding 255)')
        ax[2].set_title('dataTypeSequence')
        ax[2].set_ylim(
            dtsDiff[dtsDiff > -255].min() - 1,
            dtsDiff[dtsDiff > -255].max() + 1,
            )
        # ax[3] deals with globalSeq
        # ax[3].plot(metaDF.index, metaDF['globalSequence'])
        # gsDiff = metaDF['globalSequence'].diff()
        # gsDiffAx = ax[3].twinx()
        # gsDiffAx.plot(gsDiff.index, gsDiff, 'c')
        # plt.show()
        # ax[3] deals with timestamp
        tsDiff = metaDF['timestamp'].diff()
        p3, = ax[3].plot(
            tsDiff.index,
            tsDiff, label='timestamp diff')
        tsAx = ax[3].twinx()
        twinAx.append(tsAx)
        twinP3, = tsAx.plot(
            metaDF.index, metaDF['timestamp'], 'c',
            label='timestamp')
    else:
        fig, ax, twinAx = None, None, None
    ######## fixPacketGenTime
    if fixPacketGenTime:
        latency = metaDF['PacketRxUnixTime'] - metaDF['PacketGenTime']
        latency = latency.fillna(method='bfill').fillna(method='ffill')
        latency = (
            latency
            .rolling(50, min_periods=10, center=True)
            .apply(lambda x : np.nanmean(x), raw=True))
        metaDF.loc[noGenTimeMask, 'PacketGenTime'] = (
            metaDF.loc[noGenTimeMask, 'PacketRxUnixTime'] -
            latency.loc[noGenTimeMask])
    ########
    # drop packets due to irregular timestamp
    #######
    # tsDropMask = (metaDF['timestamp'].diff() < 0)
    # metaDF.loc[:, 'skipPacket'] = tsDropMask.to_numpy()
    # TODO: calc running median timestamp, remove outliers
    metaDF.loc[:, 'skipPacket'] = (metaDF['timestamp'].diff() < 0)
    if plotting:
        tsAx.plot(
            metaDF.index[metaDF['skipPacket']],
            metaDF.loc[metaDF['skipPacket'], 'timestamp'],
            'r*', label='dropped bc. of timestamp')
    if metaDF['skipPacket'].any():
        print('Deleting {} packets because of irregular timestamps'.format(metaDF['skipPacket'].sum()))
        metaDF = metaDF.loc[~metaDF['skipPacket'], :]
        metaDF.reset_index(drop=True, inplace=True)
    #
    if intersampleTickCount is None:
        mostCommonSampleRateCode = metaDF['SampleRate'].value_counts().idxmax()
        fs = float(sampleRateLookupDict[mostCommonSampleRateCode])
        intersampleTickCount = (fs ** -1) / (100e-6)
    if frameSize is None:
        frameSize = (frameDuration) / (intersampleTickCount * 1e-1)
        # pdb.set_trace()
    ######## correct meta matrix consecutive sys tick
    metaDF.iloc[:, :] = correct_meta_matrix_consecutive_sys_tick(
        metaDF.to_numpy(),
        intersampleTickCount=int(intersampleTickCount),
        nominalFrameSize=frameSize,
        forceRemoveEqualConsecutiveDataTypeSequence=forceRemoveEqualConsecutiveDataTypeSequence)
    if metaDF['skipPacket'].any():
        print('Deleting {} packets because of irregular dataTypeSequence'.format(metaDF['skipPacket'].sum()))
        metaDF = metaDF.loc[~metaDF['skipPacket'], :]
        metaDF.reset_index(drop=True, inplace=True)
    ######## Check against repeated packets
    equalConsecDTS = metaDF['dataTypeSequence'].diff().fillna(1) == 0
    # metaDF.loc[505:508, :].T
    # metaDF.loc[343:346, 'PacketRxUnixTime'] - metaDF['PacketRxUnixTime'].min()
    if equalConsecDTS.any():
        print('found equal consecutive datatypesequence!')
        for rowIdx in equalConsecDTS.loc[equalConsecDTS].index:
            currPacketIdx = metaDF.loc[rowIdx, 'jsonPacketIdx']
            prevPacketIdx = metaDF.loc[rowIdx - 1, 'jsonPacketIdx']
            if 'TimeDomainData' in input_json[0].keys():
                nChan = int(metaDF.loc[rowIdx, 'chanSamplesLen'])
                currData = np.concatenate(
                    [
                        input_json[0]['TimeDomainData'][int(currPacketIdx)]['ChannelSamples'][j]['Value']
                        for j in range(nChan)
                    ])
                prevData = np.concatenate(
                    [
                        input_json[0]['TimeDomainData'][int(prevPacketIdx)]['ChannelSamples'][j]['Value']
                        for j in range(nChan)
                    ])
            elif 'AccelData' in input_json[0].keys():
                currData = np.concatenate(
                    [
                        input_json[0]['AccelData'][int(currPacketIdx)][j]
                        for j in ['XSamples', 'YSamples', 'ZSamples']
                    ])
                prevData = np.concatenate(
                    [
                        input_json[0]['AccelData'][int(prevPacketIdx)][j]
                        for j in ['XSamples', 'YSamples', 'ZSamples']
                    ])
            tossThisPacket = False
            if currData.shape == prevData.shape:
                if np.allclose(prevData, currData):
                    print("Warning: repeated packet found")
                    tossThisPacket = True
            if forceRemoveEqualConsecutiveDataTypeSequence:
                tossThisPacket = True
            metaDF.loc[rowIdx, 'skipPacket'] = tossThisPacket
        if metaDF['skipPacket'].any():
            print('Deleting {} packets because of irregular dataTypeSequence'.format(metaDF['skipPacket'].sum()))
            metaDF = metaDF.loc[~metaDF['skipPacket'], :]
            metaDF.reset_index(drop=True, inplace=True)
    packetsNeedFixing = metaDF.loc[metaDF['packetNeedsFixing'].astype('bool'), :].index
    if plotting:
        dtsAx.plot(
            metaDF.index[packetsNeedFixing],
            metaDF['dataTypeSequence'].iloc[packetsNeedFixing],
            'g*', label='equal consecutive system ticks')
    ########
    metaDF.iloc[:, :] = code_micro_and_macro_packet_loss(metaDF.to_numpy())
    ########
    metaDF.iloc[:, :] = unpack_meta_matrix_time(metaDF.to_numpy(), intersampleTickCount)
    ########
    # discard gross inconsistencies between systemTick and packetRxTime
    metaDF.loc[:, 'skipPacket'] = (
        (metaDF['lastSampleTick'].diff() * 1e-1) - metaDF['PacketRxUnixTime'].diff()
        ).abs() > 1e3
    if metaDF['skipPacket'].any():
        print('Deleting {} packets because of irregular reported frame durations'.format(metaDF['skipPacket'].sum()))
        metaDF = metaDF.loc[~metaDF['skipPacket'], :]
        metaDF.reset_index(drop=True, inplace=True)
        ########
        metaDF.iloc[:, :] = code_micro_and_macro_packet_loss(metaDF.to_numpy())
        ########
        metaDF.iloc[:, :] = unpack_meta_matrix_time(metaDF.to_numpy(), intersampleTickCount)
    if plotting:
        # ax[4] deals with packetGenTime vs lastSampleTick
        lastSampleTickInMsec = metaDF.loc[~metaDF['skipPacket'], 'lastSampleTick'] * 1e-1
        residAx = ax[4].twinx()
        twinP4, = residAx.plot(
            metaDF.loc[~metaDF['skipPacket'], 'PacketGenTime'],
            lastSampleTickInMsec, 'co', label='system tick')
        try:
            pCoeffs, regrStats = np.polynomial.polynomial.polyfit(
                metaDF.loc[~metaDF['skipPacket'], 'PacketGenTime'],
                lastSampleTickInMsec, 1, full=True)
        except:
            traceback.print_exc()
            pdb.set_trace()
        ssTot = ((lastSampleTickInMsec - lastSampleTickInMsec.mean()) ** 2).sum()
        rSq = 1 - regrStats[0] / ssTot
        sysTickHat =  np.polynomial.polynomial.polyval(
            metaDF.loc[~metaDF['skipPacket'], 'PacketGenTime'], pCoeffs)
        sysTickResiduals = lastSampleTickInMsec - sysTickHat
        regrStatement = 'sysTick = {:6.4e}*packetGenTime{:+6.4e}; R^2 = {:.3f}; std of residuals = {:.3f} msec'.format(
            pCoeffs[1], pCoeffs[0], rSq[0], np.std(sysTickResiduals))
        residAx.plot(
            metaDF.loc[~metaDF['skipPacket'], 'PacketGenTime'],
           sysTickHat, 'r-', label='system tick (hat) | packetGen time')
        p4, = ax[4].plot(
            metaDF.loc[~metaDF['skipPacket'], 'PacketGenTime'],
            sysTickResiduals, '-', label='residuals')
        ax[4].text(
            1, 0, regrStatement,
            va='bottom', ha='right',
            transform=residAx.transAxes)
        #
        ax[0].set_ylabel('RX diff (msec)')
        ax[0].set_title('RX Time')
        ax[0].yaxis.get_label().set_color(p0.get_color())
        ax[0].legend(loc='upper right')
        #
        ax[1].set_title('RxTime - GenTime')
        ax[1].set_ylabel('(msec)')
        ax[1].yaxis.get_label().set_color(p1.get_color())
        ax[1].legend()
        #
        ax[2].legend(loc='upper right')
        ax[2].set_ylabel('(count)')
        ax[2].yaxis.get_label().set_color(p2.get_color())
        dtsAx.set_ylabel('(count)')
        dtsAx.yaxis.get_label().set_color(twinP2.get_color())
        dtsAx.legend(loc='lower right')
        #
        ax[3].set_title('timestamp')
        ax[3].set_xlabel('Packet #')
        ax[3].legend(loc='upper right')
        ax[3].set_ylabel('sec')
        ax[3].yaxis.get_label().set_color(p3.get_color())
        tsAx.set_ylabel('sec')
        tsAx.yaxis.get_label().set_color(twinP3.get_color())
        tsAx.legend(loc='lower right')
        #
        ax[4].set_title('systemTick (msec)')
        ax[4].set_ylabel('msec')
        ax[4].set_xlabel('PacketGenTime (msec)')
        residAx.set_ylabel('msec')
        ax[4].yaxis.get_label().set_color(p4.get_color())
        ax[4].legend(loc='upper right')
        residAx.legend(loc='lower right')
        residAx.yaxis.get_label().set_color(twinP4.get_color())
    return metaDF.to_numpy(), fig, ax, twinAx


def extract_accel_meta_data(
        input_json, sampleRateLookupDict=None,
        intersampleTickCount=None, plotting=False, fixPacketGenTime=True):
    meta_matrix = init_numpy_array(input_json, 18, 'AccelData')
    for index, packet in enumerate(input_json[0]['AccelData']):
        meta_matrix[index, 0] = packet['Header']['dataSize']
        meta_matrix[index, 1] = packet['Header']['dataTypeSequence']
        meta_matrix[index, 2] = packet['Header']['systemTick']
        meta_matrix[index, 3] = packet['Header']['timestamp']['seconds']
        # 4 will hold microloss
        # 5 will hold macroloss
        # 6 will hold coincidence of losses
        meta_matrix[index, 7] = index
        #  Each accelerometer packet has samples for 3 axes
        meta_matrix[index, 8] = 3
        assert ((len(packet['XSamples']) == len(packet['YSamples'])) & (len(packet['XSamples']) == len(packet['ZSamples'])))
        meta_matrix[index, 9] = len(packet['XSamples']) # Number of samples in each channel always 8 for accel data
        meta_matrix[index, 10] = packet['SampleRate']
        meta_matrix[index, 11] = packet['PacketGenTime']
        meta_matrix[index, 12] = packet['PacketRxUnixTime']
        # 13 will hold first sampleTick
        # 14 will hold last sampleTick
        meta_matrix[index, 15] = packet['Header']['globalSequence']
        meta_matrix[index, 16] = False # 16 will hold instruction to skip
        meta_matrix[index, 17] = False # 17 will mark whether the packet needed a systick adjustment
    meta_matrix, fig, ax, twinAx = process_meta_data(
        meta_matrix, sampleRateLookupDict=sampleRateLookupDict,
        frameSize=8, # accel packets have 8 samples per packet
        intersampleTickCount=intersampleTickCount, plotting=plotting,
        fixPacketGenTime=fixPacketGenTime, input_json=input_json)
    return meta_matrix, fig, ax, twinAx


def extract_td_meta_data(
        input_json, sampleRateLookupDict=None, frameDuration=None,
        intersampleTickCount=None, plotting=False, fixPacketGenTime=True):
    meta_matrix = init_numpy_array(input_json, 18, 'TimeDomainData')
    for index, packet in enumerate(input_json[0]['TimeDomainData']):
        meta_matrix[index, 0] = packet['Header']['dataSize']
        meta_matrix[index, 1] = packet['Header']['dataTypeSequence']
        meta_matrix[index, 2] = packet['Header']['systemTick']
        meta_matrix[index, 3] = packet['Header']['timestamp']['seconds']
        # 4 will hold microloss
        # 5 will hold macroloss
        # 6 will hold coincidence of losses
        meta_matrix[index, 7] = index
        meta_matrix[index, 8] = len(packet['ChannelSamples'])
        meta_matrix[index, 9] = packet['Header']['dataSize'] / (2 * len(packet['ChannelSamples']))
        meta_matrix[index, 10] = packet['SampleRate']
        meta_matrix[index, 11] = packet['PacketGenTime']
        meta_matrix[index, 12] = packet['PacketRxUnixTime']
        # 13 will hold first sampleTick
        # 14 will hold last sampleTick
        meta_matrix[index, 15] = packet['Header']['globalSequence']
        meta_matrix[index, 16] = False # 16 will hold instruction to skip
        meta_matrix[index, 17] = False # 17 will mark whether the packet needed a systick adjustment
    meta_matrix, fig, ax, twinAx = process_meta_data(
        meta_matrix, sampleRateLookupDict=sampleRateLookupDict, frameDuration=frameDuration,
        intersampleTickCount=intersampleTickCount, plotting=plotting,
        fixPacketGenTime=fixPacketGenTime, input_json=input_json)
    return meta_matrix, fig, ax, twinAx


def extract_time_sync_meta_data(input_json):
    timeSync = input_json['TimeSyncData']
    timeSyncData = pd.DataFrame(
        columns=[
            'HostUnixTime', 'PacketGenTime', 'LatencyMilliseconds',
            'dataSize', 'dataType', 'dataTypeSequence', 'globalSequence',
            'systemTick', 'timestamp', 'microloss', 'macroloss', 'bothloss',
            'firstSampleTick', 'lastSampleTick', 'skipPacket'
            ]
        )
    for index, packet in enumerate(timeSync):
        if packet['LatencyMilliseconds'] > 0:
            entryData = {
                'PacketRxUnixTime': packet['PacketRxUnixTime'],
                'PacketGenTime': packet['PacketGenTime'],
                'LatencyMilliseconds': packet['LatencyMilliseconds'],
                'dataSize': packet['Header']['dataSize'],
                'dataType': packet['Header']['dataType'],
                'dataTypeSequence': packet['Header']['dataTypeSequence'],
                'globalSequence': packet['Header']['globalSequence'],
                'systemTick': packet['Header']['systemTick'],
                'timestamp': packet['Header']['timestamp']['seconds'],
                'jsonPacketIdx': index,
                'microloss': 0, 'macroloss': 0, 'bothloss': 0,
                'skipPacket': False,
                'packetNeedsFixing': False,
                # 'microseconds': 0,
                # 'time_master': np.nan
                }
            entrySeries = pd.Series(entryData)
            timeSyncData = timeSyncData.append(
                entrySeries, ignore_index=True, sort=True)
    #
    timeSyncData = timeSyncData.reindex(columns=metaMatrixColumns + ['LatencyMilliseconds'])
    sortedIndices = timeSyncData.sort_values(
        by=['PacketRxUnixTime', 'dataTypeSequence', 'globalSequence'],
        kind='mergesort').index.to_numpy()
    timeSyncData = timeSyncData.iloc[sortedIndices, :].reset_index(drop=True)
    #
    timeSyncData.loc[:, metaMatrixColumns] = correct_meta_matrix_consecutive_sys_tick(
        timeSyncData.loc[:, metaMatrixColumns].to_numpy(), frameDuration=1e3)
    timeSyncData.loc[:, metaMatrixColumns] = code_micro_and_macro_packet_loss(
        timeSyncData.loc[:, metaMatrixColumns].to_numpy())
    timeSyncData.loc[:, metaMatrixColumns] = unpack_meta_matrix_time(
        timeSyncData.loc[:, metaMatrixColumns].to_numpy(), 0)
    #
    timeSyncData.loc[:, 'HostUnixTime'] = timeSyncData.loc[:, 'PacketRxUnixTime']
    timeSyncData.loc[:, 'microseconds'] = timeSyncData.loc[:, 'lastSampleTick'] * 100
    timeSyncData.loc[:, 'time_master'] = timeSyncData['timestamp'].iloc[0]
    timeSyncData['time_master'] = pd.to_datetime(
        timeSyncData['time_master'], unit='s', origin=pd.Timestamp('2000-03-01'))
    timeSyncData['microseconds'] = pd.to_timedelta(
        timeSyncData['microseconds'], unit='us')
    timeSyncData['actual_time'] = timeSyncData['time_master'] + (
        timeSyncData['microseconds'])
    return timeSyncData


def extract_stim_meta_data(input_json):
    stimLog = input_json

    stimStatus = pd.DataFrame(
        columns=['HostUnixTime', 'therapyStatus', 'activeGroup', 'frequency'] +
                ['amplitudeChange', 'pwChange'] + progAmpNames + progPWNames
        )

    activeGroup = np.nan
    lastUpdate = {'program': 0, 'amplitude': 0}

    for entry in stimLog:
        if 'RecordInfo' in entry.keys():
            entryData = {'HostUnixTime': entry['RecordInfo']['HostUnixTime']}

        if 'therapyStatusData' in entry.keys():
            entryData.update(entry['therapyStatusData'])
            if 'activeGroup' in entry['therapyStatusData'].keys():
                activeGroup = entry['therapyStatusData']['activeGroup']

        activeGroupSettings = 'TherapyConfigGroup{}'.format(activeGroup)
        thisAmplitude = None
        thisPW = None
        if activeGroupSettings in entry.keys():
            if 'RateInHz' in entry[activeGroupSettings]:
                entryData.update(
                    {'frequency': entry[activeGroupSettings]['RateInHz']}
                    )
            ampChange = False
            pwChange = False
            for progIdx in range(4):
                programName = 'program{}'.format(progIdx)
                if programName in entry[activeGroupSettings].keys():
                    if 'amplitude' in entry[activeGroupSettings][programName]:
                        ampChange = True
                        thisAmplitude = entry[activeGroupSettings][programName][
                            'AmplitudeInMilliamps']
                        entryData.update(
                            {programName + '_amplitude': thisAmplitude}
                        )
                    if 'pulseWidth' in entry[activeGroupSettings][programName]:
                        pwChange = True
                        thisPW = entry[activeGroupSettings][programName][
                            'PulseWidthInMicroseconds']
                        entryData.update(
                            {
                                programName + '_pw': thisPW
                            }
                        )
        #  was there an amplitude change?
        entryData.update({'amplitudeChange': ampChange})
        #  was there a pw change?
        entryData.update({'pwChange': pwChange})

        entrySeries = pd.Series(entryData)
        stimStatus = stimStatus.append(
            entrySeries, ignore_index=True, sort=True)

    stimStatus.fillna(method='ffill', axis=0, inplace=True)
    return stimStatus


def extract_stim_meta_data_events(input_json, trialSegment=None):
    stimLog = input_json

    eventsList = []
    
    lastUpdate = {
        'activeGroup': np.nan, 'therapyStatus': np.nan,
        'program': np.nan, 'RateInHz': np.nan}
    groupRates = {i: {'RateInHz': np.nan, 'ratePeriod': np.nan} for i in range(4)}
    lastUpdate.update({progName: 0 for progName in progAmpNames})
    for idx, entry in enumerate(stimLog):
        theseEvents = []
        if 'RecordInfo' in entry.keys():
            hostUnixTime = entry['RecordInfo']['HostUnixTime']
            if (trialSegment is not None) and len(eventsList) == 0:
                theseEvents.append(pd.DataFrame({
                    'HostUnixTime': hostUnixTime,
                    'ins_property': 'trialSegment', 'ins_value': trialSegment},
                    index=[0]))

        if 'therapyStatusData' in entry.keys():
            for key in ['activeGroup', 'therapyStatus']:
                if key in entry['therapyStatusData'].keys():
                    value = entry['therapyStatusData'][key]
                    theseEvents.append(pd.DataFrame({
                        'HostUnixTime': hostUnixTime,
                        'ins_property': key, 'ins_value': value},
                        index=[0]))
                    lastUpdate[key] = value
        
        for grpIdx, confGrpName in enumerate(groupConfigNames):
            if confGrpName in entry.keys():
                if 'RateInHz' in entry[confGrpName].keys():
                    groupRates[grpIdx]['RateInHz'] = entry[confGrpName]['RateInHz']
                    groupRates[grpIdx]['ratePeriod'] = entry[confGrpName]['ratePeriod']
        
        thisConfGroupName = (
            'TherapyConfigGroup{}'.format(lastUpdate['activeGroup']))
        if thisConfGroupName in entry.keys():
            updateOrder = [
                'RateInHz', 'ratePeriod',
                'program0', 'program1',
                'program2', 'program3']
            for key in updateOrder:
                if key in entry[thisConfGroupName].keys():
                    value = entry[thisConfGroupName][key]
                    if 'program' in key:
                        #  program level update
                        progIdx = int(key[-1])
                        theseEvents.append(pd.DataFrame({
                            'HostUnixTime': hostUnixTime,
                            'ins_property': 'program', 'ins_value': progIdx},
                            index=[0]))
                        for progKey, progValue in value.items():
                            if progKey in ['amplitude', 'pulseWidth']:
                                theseEvents.append(pd.DataFrame({
                                    'HostUnixTime': hostUnixTime,
                                    'ins_property': progKey, 'ins_value': progValue},
                                    index=[0]))
                                lastUpdate[key + '_' + progKey] = progValue
                    else:
                        #  group level update
                        theseEvents.append(pd.DataFrame({
                            'HostUnixTime': hostUnixTime,
                            'ins_property': key, 'ins_value': value},
                            index=[0]))
                        lastUpdate[key] = value
        if len(theseEvents):
            theseEventsDF = pd.concat(theseEvents, ignore_index=True, sort=True)
            #
            recordedGroupChange = theseEventsDF['ins_property'].str.contains('activeGroup').any()
            recordedRateChange = theseEventsDF['ins_property'].str.contains('RateInHz').any()
            if recordedGroupChange and not recordedRateChange:
                for k, v in groupRates[lastUpdate['activeGroup']].items():
                    fillerDF = pd.DataFrame({
                        'HostUnixTime': hostUnixTime,
                        'ins_property': k, 'ins_value': v},
                        index=[0])
                    theseEventsDF = pd.concat(
                        [theseEventsDF, fillerDF],
                        ignore_index=True, sort=True)
            eventsList.append(theseEventsDF)
    return pd.concat(eventsList, ignore_index=True, sort=True)


def code_micro_and_macro_packet_loss(meta_matrix):
    meta_matrix[np.where((np.diff(meta_matrix[:, 1]) % (2 ** 8)) > 1)[0] + 1, 4] = 1  # Top packet of microloss; nonconsecutive dataTypeSequence
    # meta_matrix[np.where((np.diff(meta_matrix[:, 3]) >= ((2 ** 16) * .0001)))[0] + 1, 5] = 1  # Top packet of macroloss; more than 6.5535 seconds apart
    # TODO: the low res clock cannot resolve differences > 1 sec, so call anything > 6 a macroloss
    # see UCSF code for how to handle this
    # np.diff(meta_matrix[:, 12]).max()
    # meta_matrix[np.where((np.diff(meta_matrix[:, 3]) >= 6))[0] + 1, 5] = 1  # Top packet of macroloss; more than 6. seconds apart
    # pdb.set_trace()
    meta_matrix[np.where((np.diff(meta_matrix[:, 11]) >= (2 ** 16 * 1e-1)))[0] + 1, 5] = 1  # Top packet of macroloss; more than 6. seconds apart
    meta_matrix[:, 6] = ((meta_matrix[:, 4]).astype(int) & (meta_matrix[:, 5]).astype(int))  # Code coincidence of micro and macro loss
    return meta_matrix


def calculate_statistics(meta_matrix, intersample_tick_count):
    # Calculate total number of actual data points that were received
    num_real_points = meta_matrix[:, 9].sum()

    # Calculate the total number of large roll-overs (>= 6.5536 seconds)
    num_macro_rollovers = meta_matrix[:, 5].sum()

    # Calculate the packet number before and after the occurrence of a small packet losse
    micro_loss_stack = np.dstack((np.where(meta_matrix[:, 4] == 1)[0] - 1, np.where(meta_matrix[:, 4] == 1)[0]))[0]

    # Remove small packet losses that coincided with large packet losses
    micro_loss_stack = micro_loss_stack[
        np.isin(micro_loss_stack[:, 1], np.where(meta_matrix[:, 5] == 1)[0], invert=True)]

    # Allocate array for calculating micropacket loss
    loss_array = np.zeros(len(micro_loss_stack))

    # Loop over meta data to extract and calculate number of data points lost to small packet loss.
    for index, packet in enumerate(micro_loss_stack):
        loss_array[index] = (((meta_matrix[packet[1], 2] - (meta_matrix[packet[1], 9] * intersample_tick_count)) -
                              meta_matrix[packet[0], 2]) % (2 ** 16)) / intersample_tick_count

    # Sum the total number of lost data points due to small packet loss.
    loss_as_scalar = np.around(loss_array).sum()

    return num_real_points, num_macro_rollovers, loss_as_scalar


def unpacker_td(meta_matrix, input_json, intersample_tick_count):
    # First we verify that num of channels and sampling rate does not change
    if np.diff(meta_matrix[:, 8]).sum():
        raise ValueError('Number of Active Channels Changes Throughout the Recording')
    if np.diff(meta_matrix[:, 10]).sum():
        raise ValueError('Sampling Rate Changes Throughout the Recording')
    # Initialize array to hold output data
    keepMask = ~meta_matrix[:, 16].astype(np.bool)
    final_array = np.zeros(
        (
            int(meta_matrix[keepMask, 9].sum()),
            int(4 + meta_matrix[0, 8]))
            # 3 + meta_matrix[0, 8].astype(int))
            )
    # Initialize variables for looping:
    array_bottom = 0
    reference_time = meta_matrix[0, 3]
    for packet_number, i in enumerate(meta_matrix):
        if i[16]:
            continue
        running_tick_counter = i[14]
        backtrack_time = running_tick_counter - ((i[9] - 1) * intersample_tick_count)
        # Populate master clock time into array
        final_array[int(array_bottom):int(array_bottom + i[9]), 0] = np.array([reference_time] * int(i[9]))
        # Linspace microsecond clock and populate into array
        final_array[int(array_bottom):int(array_bottom + i[9]), 1] = np.linspace(
            backtrack_time, running_tick_counter, int(i[9]))
        # Populate coarse clock time into array
        final_array[int(array_bottom):int(array_bottom + i[9]), -2] = np.array([reference_time] * int(i[9]))
        # Put packet number into array for debugging
        final_array[int(array_bottom):int(array_bottom + i[9]), -1] = np.array([packet_number] * int(i[9]))
        # Unpack timedomain data from original packets into array
        for j in range(0, int(i[8])):
            final_array[int(array_bottom):int(array_bottom + i[9]), j + 2] = \
            input_json[0]['TimeDomainData'][int(i[7])]['ChannelSamples'][j]['Value']
        # Update counters for next loop
        array_bottom += i[9]
    # Convert systemTick into microseconds
    final_array[:, 1] = final_array[:, 1] * 100
    return final_array


def unpacker_accel(meta_matrix, input_json, intersample_tick_count):
    # First we verify that num of channels and sampling rate does not change
    if np.diff(meta_matrix[:, 8]).sum():
        raise ValueError('Number of Active Channels Changes Throughout the Recording')
    if np.diff(meta_matrix[:, 10]).sum():
        raise ValueError('Sampling Rate Changes Throughout the Recording')
    # Initialize array to hold output data
    keepMask = ~meta_matrix[:, 16].astype(np.bool)
    final_array = np.zeros(
        (
            int(meta_matrix[keepMask, 9].sum()),
            int(4 + meta_matrix[0, 8]))
            # 3 + meta_matrix[0, 8].astype(int))
            )
    # Initialize variables for looping:
    array_bottom = 0
    reference_time = meta_matrix[0, 3]
    for packet_number, i in enumerate(meta_matrix):
        if i[16]:
            continue
        running_tick_counter = i[14]
        backtrack_time = running_tick_counter - ((i[9] - 1) * intersample_tick_count)
        # Populate master clock time into array
        final_array[int(array_bottom):int(array_bottom + i[9]), 0] = np.array([reference_time] * int(i[9]))
        # Linspace microsecond clock and populate into array
        final_array[int(array_bottom):int(array_bottom + i[9]), 1] = np.linspace(
            backtrack_time, running_tick_counter, int(i[9]))
        # Populate coarse clock time into array
        final_array[int(array_bottom):int(array_bottom + i[9]), -2] = np.array([reference_time] * int(i[9]))
        # Put packet number into array for debugging
        final_array[int(array_bottom):int(array_bottom + i[9]), -1] = np.array([packet_number] * int(i[9]))
        # Unpack timedomain data from original packets into array
        for accel_index, accel_channel in enumerate(['XSamples', 'YSamples', 'ZSamples']):
            final_array[int(array_bottom):int(array_bottom + i[9]), accel_index + 2] = \
            input_json[0]['AccelData'][int(i[7])][accel_channel]
        # Update counters for next loop
        array_bottom += i[9]
    # Convert systemTick into microseconds
    final_array[:, 1] = final_array[:, 1] * 100
    return final_array

def save_to_disk(data_matrix, filename_str, time_format, data_type, num_cols=None):
    # TODO: We need to find a different way of dynamically naming columns. Current method won't work.
    if num_cols is None:
        num_cols = data_matrix.shape[1]
    if data_type == 'accel':
        channel_names = ['accel_' + x for x in ['x', 'y', 'z']] # None of his channel name stuff works.
    else:
        channel_names = ['channel_' + str(x) for x in range(0, num_cols)]
    column_names = ['time_master', 'microseconds'] + channel_names + ['coarseClock', 'packetIdx']
    df = pd.DataFrame(data_matrix, columns=column_names)
    if time_format == 'full':
        df.time_master = pd.to_datetime(df.time_master, unit='s', origin=pd.Timestamp('2000-03-01'))
        df.microseconds = pd.to_timedelta(df.microseconds, unit='us')
        df['actual_time'] = df.time_master + df.microseconds
    else:
        df['actual_time'] = df.time_master + (df.microseconds / 1E6)
    df.to_csv(filename_str, index=False)
    return df


def print_session_statistics():
    # TODO: Implement a printing function to show statistics to the user at the end of processing
    return


# Note: This function is deprecated!!!!!! Please see the 'silent' version below.
def packet_time_calculator_verbose(input_packets):
    current_run_time = 0
    backtrack_time = 0
    old_run_time = 0
    old_packet_time = input_packets[0]['Header']['systemTick']

    current_packet_number = input_packets[0]['Header']['dataTypeSequence']
    old_packet_number = input_packets[0]['Header']['dataTypeSequence']
    packet_counter = 0
    lost_packet_array = []

    voltage_array = {x['Key']: np.empty(0) for x in input_packets[0]['ChannelSamples']}
    timestamp_array = np.empty(0)

    for i in input_packets:
        num_points = i['Header']['dataSize'] // 4
        print('Num Data Points: {}'.format(num_points))
        current_run_time += ((i['Header']['systemTick'] - old_packet_time) % (2 ** 16))
        print('Current Run Time: {}'.format(current_run_time))
        backtrack_time = current_run_time - (
                (num_points - 1) * 10)  # 100usec * 10 = 1msec (period for recording at 1000Hz)
        print('Backtrack Time {}'.format(backtrack_time))
        print('Upcoming Packet Number: {}'.format(i['Header']['dataTypeSequence']))
        current_packet_number = (i['Header']['dataTypeSequence'] - old_packet_number) % (2 ** 8)
        print('Packet Delta: {}'.format(current_packet_number))
        print('Old Packet Number: {}\n'.format(old_packet_number))
        if (current_packet_number > 1):
            print("^We just lost a packet...^\n")
            lost_packet_array.append(packet_counter)
            print('Old Packet Time: {}'.format(old_run_time))
            lower_bound_time = old_run_time + 10
            print('Missing Packet Lower Bound Time: {}'.format(lower_bound_time))
            missing_data_count = ((backtrack_time - lower_bound_time) // 10)
            print('Missing Data Count: {}'.format(missing_data_count))
            timestamp_array = np.append(timestamp_array,
                                        np.linspace(lower_bound_time, (backtrack_time), missing_data_count,
                                                    endpoint=False))
            for j in i['ChannelSamples']:
                voltage_array[j['Key']] = np.append(voltage_array[j['Key']], np.array([0] * missing_data_count))

        timestamp_array = np.append(timestamp_array, np.linspace(backtrack_time, current_run_time, num_points))
        for j in i['ChannelSamples']:
            voltage_array[j['Key']] = np.append(voltage_array[j['Key']], j['Value'])

        old_run_time = current_run_time
        old_packet_number = i['Header']['dataTypeSequence']
        old_packet_time = i['Header']['systemTick']
        packet_counter += 1

    return ((timestamp_array - timestamp_array[0]), voltage_array, lost_packet_array)


def packet_time_calculator_silent(input_td_data, timing_dict, td_packet_str='TimeDomainData'):
    td_packets = input_td_data[td_packet_str]
    num_points_divisor = 2 * len(td_packets[0]['ChannelSamples'])

    current_run_time = 0
    old_run_time = 0
    old_packet_time = td_packets[0]['Header']['systemTick']

    old_packet_number = td_packets[0]['Header']['dataTypeSequence']
    packet_counter = 0
    lost_packet_array = {}

    voltage_array = {x['Key']: np.empty(0) for x in td_packets[0]['ChannelSamples']}
    timestamp_array = np.empty(0)

    timing_multiplier = timing_dict[td_packets[0]['SampleRate']]  # Assume uniform timing in TD data

    for i in td_packets:
        num_points = i['Header']['dataSize'] // num_points_divisor
        #             print('Num Data Points: {}'.format(num_points))
        current_run_time += ((i['Header']['systemTick'] - old_packet_time) % (2 ** 16))
        #             print('Current Run Time: {}'.format(current_run_time))
        backtrack_time = current_run_time - ((num_points - 1) * timing_multiplier)
        #             print('Backtrack Time {}'.format(backtrack_time))
        #             print('Upcoming Packet Number: {}'.format(i['Header']['dataTypeSequence']))
        packet_delta = (i['Header']['dataTypeSequence'] - old_packet_number) % (2 ** 8)
        #             print('Packet Delta: {}'.format(current_packet_number))
        #             print('Old Packet Number: {}\n'.format(old_packet_number))
        if packet_delta > 1:
            #                 print("^We just lost a packet...^\n")
            #                 print('Old Packet Time: {}'.format(old_run_time))
            lower_bound_time = old_run_time + timing_multiplier
            #                 print('Missing Packet Lower Bound Time: {}'.format(lower_bound_time))
            missing_data_count = ((backtrack_time - lower_bound_time) // timing_multiplier)
            lost_packet_array[packet_counter] = [packet_delta, missing_data_count]
            #                 print('Missing Data Count: {}'.format(missing_data_count))
            timestamp_array = np.append(timestamp_array,
                                        np.linspace(lower_bound_time, backtrack_time, missing_data_count,
                                                    endpoint=False))
            for j in i['ChannelSamples']:
                voltage_array[j['Key']] = np.append(voltage_array[j['Key']], np.array([0] * missing_data_count))

        timestamp_array = np.append(timestamp_array, np.linspace(backtrack_time, current_run_time, num_points))
        for j in i['ChannelSamples']:
            voltage_array[j['Key']] = np.append(voltage_array[j['Key']], j['Value'])

        old_run_time = current_run_time
        old_packet_number = i['Header']['dataTypeSequence']
        old_packet_time = i['Header']['systemTick']
        packet_counter += 1

    return [((timestamp_array - timestamp_array[0]) * 0.0001), voltage_array, lost_packet_array]
