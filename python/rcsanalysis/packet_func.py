import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pdb
""" All the Code Below Is For the Second Generation Packetizer """

progAmpNames = ['program{}_amplitude'.format(progIdx) for progIdx in range(4)]
progPWNames = ['program{}_pw'.format(progIdx) for progIdx in range(4)]


def init_numpy_array(input_json, num_cols, data_type):
    num_rows = len(input_json[0][data_type])
    return np.zeros((num_rows, num_cols))


def strip_prog_name(x):
    return int(x.split('program')[-1].split('_')[0])

def unpack_meta_matrix_time(meta_matrix, intersample_tick_count):
    # Initialize variables for looping:

    master_time = meta_matrix[0, 3]
    old_system_tick = 0
    running_ms_counter = 0

    firstSampleTime = np.zeros(meta_matrix.shape[0])
    lastSampleTime = np.zeros(meta_matrix.shape[0])
    
    for packet_number, i in enumerate(meta_matrix):
        if i[5]:
            # We just suffered a macro packet loss...
            old_system_tick = 0
            running_ms_counter = 0
            master_time = i[3]  # Resets the master time

        running_ms_counter += ((i[2] - old_system_tick) % (2 ** 16))
        backtrack_time = running_ms_counter - ((i[9] - 1) * intersample_tick_count)

        firstSampleTime[packet_number] = backtrack_time / 10
        lastSampleTime[packet_number] = running_ms_counter / 10

        # Update counters for next loop
        old_system_tick = i[2]

    return firstSampleTime, lastSampleTime


def correct_meta_matrix_consecutive_sys_tick(
        meta_matrix, frameLength=500, verbose=True):
    # correct issue described on page 64 of summit user manual
    metaMatrix = pd.DataFrame(
        meta_matrix[:, [1, 2, -1]],
        columns=['dataTypeSequence', 'systemTick', 'packetIdx'])
    metaMatrix['rolloverGroup'] = (
        metaMatrix['systemTick'].diff() < 0).cumsum()

    for name, group in metaMatrix.groupby('rolloverGroup'):
        duplicateSysTick = group.duplicated('systemTick')
        if duplicateSysTick.any():
            duplicateIdxs = duplicateSysTick.index[np.flatnonzero(duplicateSysTick)]
            for duplicateIdx in duplicateIdxs:
                sysTickVal = group.loc[duplicateIdx, 'systemTick']
                allOccurences = group.loc[group['systemTick'] == sysTickVal, :]
                if verbose:
                    print(allOccurences)
                atMax = allOccurences['dataTypeSequence'] == 255
                dtSequence = allOccurences['dataTypeSequence'].copy()
                dtSequence.loc[atMax] = -1
                idxNeedsChanging = dtSequence.idxmin()
                
                #  if 2815 in group['packetIdx']:
                #      print('at packet_func')
                #      pdb.set_trace()
                #TODO access deviceLog to find what the frame size actually is
                
                meta_matrix[idxNeedsChanging, 2] = meta_matrix[idxNeedsChanging, 2] - 500
    
    return meta_matrix


def correct_meta_matrix_time_displacement(
    meta_matrix, intersample_tick_count, verbose=False, plotting=False):
    
    tdMeta = pd.DataFrame(
        meta_matrix[:, [1, 2, 3, 4, 5, 6]],
        columns=[
            'dataTypeSequence', 'systemTick', 'masterClock',
            'microloss', 'macroloss', 'bothloss']
        )

    firstSampleTime, lastSampleTime = unpack_meta_matrix_time(
        meta_matrix, intersample_tick_count)
    tdMeta['firstSampleTime'] = firstSampleTime
    tdMeta['lastSampleTime'] = lastSampleTime

    # how far is this packet from the preceding one
    tdMeta['sampleGap'] = tdMeta['firstSampleTime'].values - tdMeta['lastSampleTime'].shift(1).values
    # how much does this packet overlap the next one?
    tdMeta['sampleOverlap'] = tdMeta['lastSampleTime'].values - tdMeta['firstSampleTime'].shift(-1).values

    tdMeta['displacementDifference'] = tdMeta['sampleGap'].values - tdMeta['sampleOverlap'].values
    tdMeta['packetsNotLost'] = ~(tdMeta['microloss'].astype(bool) | tdMeta['macroloss'].astype(bool))
    tdMeta['packetsOverlapFuture'] = tdMeta['sampleOverlap'] > 0

    if plotting:
        ax = sns.distplot(
            tdMeta.loc[tdMeta['packetsNotLost'] & tdMeta['packetsOverlapFuture'], 'sampleGap'].dropna(),
            label='sampleGap'
            )
        ax = sns.distplot(
            tdMeta.loc[
                tdMeta['packetsNotLost'] & tdMeta['packetsOverlapFuture'],
                'sampleOverlap'
                ].dropna(),
            label='sampleOverlap'
            )
        ax = sns.distplot(
            tdMeta.loc[
                tdMeta['packetsNotLost'] & tdMeta['packetsOverlapFuture'],
                'displacementDifference'].dropna(),
            label='displacementDifference'
            )
        plt.legend()
        plt.show()

    packetsNeedFixing = tdMeta.index[
        tdMeta['packetsNotLost'] & tdMeta['packetsOverlapFuture']]
    correctiveValues = tdMeta.loc[packetsNeedFixing, 'sampleGap'].fillna(method = 'bfill') * 10
    #  pdb.set_trace()
    tdMeta.loc[packetsNeedFixing, 'systemTick'] = tdMeta.loc[
        packetsNeedFixing, 'systemTick'] - (round(correctiveValues) - intersample_tick_count)
    meta_matrix[:, 2] = tdMeta['systemTick'].values
    
    return meta_matrix, packetsNeedFixing


def extract_td_meta_data(input_json):
    meta_matrix = init_numpy_array(input_json, 11, 'TimeDomainData')
    for index, packet in enumerate(input_json[0]['TimeDomainData']):
        meta_matrix[index, 0] = packet['Header']['dataSize']
        meta_matrix[index, 1] = packet['Header']['dataTypeSequence']
        meta_matrix[index, 2] = packet['Header']['systemTick']
        meta_matrix[index, 3] = packet['Header']['timestamp']['seconds']
        meta_matrix[index, 7] = index
        meta_matrix[index, 8] = len(packet['ChannelSamples'])
        meta_matrix[index, 9] = packet['Header']['dataSize'] / (2 * len(packet['ChannelSamples']))
        meta_matrix[index, 10] = packet['SampleRate']
    
    meta_matrix = correct_meta_matrix_consecutive_sys_tick(
        meta_matrix, frameLength=500)
    
    return meta_matrix


def extract_time_sync_meta_data(input_json):
    timeSync = input_json['TimeSyncData']

    timeSyncData = pd.DataFrame(
        columns=[
            'HostUnixTime', 'PacketGenTime', 'LatencyMilliseconds',
            'dataSize', 'dataType', 'dataTypeSequence', 'globalSequence',
            'systemTick', 'timestamp', 'microloss', 'macroloss', 'bothloss',
            'microseconds', 'time_master'
            ]
        )
    for index, packet in enumerate(timeSync):
        if packet['LatencyMilliseconds'] > 0:
            entryData = {
                'HostUnixTime': packet['PacketRxUnixTime'],
                'PacketGenTime': packet['PacketGenTime'],
                'LatencyMilliseconds': packet['LatencyMilliseconds'],
                'dataSize': packet['Header']['dataSize'],
                'dataType': packet['Header']['dataType'],
                'dataTypeSequence': packet['Header']['dataTypeSequence'],
                'globalSequence': packet['Header']['globalSequence'],
                'systemTick': packet['Header']['systemTick'],
                'timestamp': packet['Header']['timestamp']['seconds'],
                'packetIdx': index,
                'microloss': 0, 'macroloss': 0, 'bothloss': 0,
                'microseconds': 0, 'time_master': np.nan
                }
            entrySeries = pd.Series(entryData)
            timeSyncData = timeSyncData.append(
                entrySeries, ignore_index=True, sort=True)
    timeSyncData = timeSyncData[[
            'dataSize', 'dataTypeSequence', 'systemTick', 'timestamp',
            'microloss', 'macroloss', 'bothloss',
            'packetIdx', 'HostUnixTime',
            'PacketGenTime', 'LatencyMilliseconds',
            'dataType', 'globalSequence', 'time_master'
        ]]
    
    timeSyncData.iloc[:, :] = code_micro_and_macro_packet_loss(
        timeSyncData.values)
    #  pdb.set_trace()
    old_system_tick = 0
    running_ms_counter = 0
    master_time = timeSyncData['timestamp'].iloc[0]
    for index, packet in timeSyncData.iterrows():
        
        if packet['macroloss']:
            # We just suffered a macro packet loss...
            old_system_tick = 0
            running_ms_counter = 0
            master_time = packet['timestamp']  # Resets the master time

        running_ms_counter += (
            (packet['systemTick'] - old_system_tick) % (2 ** 16))
        
        old_system_tick = packet['systemTick']
        timeSyncData.loc[index, 'microseconds'] = running_ms_counter * 100
        timeSyncData.loc[index, 'time_master'] = master_time
        
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

        configGroupName = (
            'TherapyConfigGroup{}'.format(lastUpdate['activeGroup']))
        if configGroupName in entry.keys():            
            updateOrder = [
                'RateInHz', 'ratePeriod',
                'program0', 'program1',
                'program2', 'program3']
            for key in updateOrder:
                if key in entry[configGroupName].keys():
                    value = entry[configGroupName][key]
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
                                lastUpdate[key + '_'+progKey] = progValue
                    else:
                        #  group level update
                        theseEvents.append(pd.DataFrame({
                            'HostUnixTime': hostUnixTime,
                            'ins_property': key, 'ins_value': value},
                            index=[0]))
                        lastUpdate[key] = value
        eventsList = eventsList + theseEvents
    return pd.concat(eventsList, ignore_index=True, sort=True)

def extract_accel_meta_data(input_json):
    meta_matrix = init_numpy_array(input_json, 11, 'AccelData')
    for index, packet in enumerate(input_json[0]['AccelData']):
        meta_matrix[index, 0] = packet['Header']['dataSize']
        meta_matrix[index, 1] = packet['Header']['dataTypeSequence']
        meta_matrix[index, 2] = packet['Header']['systemTick']
        meta_matrix[index, 3] = packet['Header']['timestamp']['seconds']
        meta_matrix[index, 7] = index
        #  Each accelerometer packet has samples for 3 axes
        meta_matrix[index, 8] = 3
        meta_matrix[index, 9] = 8  # Number of samples in each channel always 8 for accel data
        meta_matrix[index, 10] = packet['SampleRate']
    
    meta_matrix = correct_meta_matrix_consecutive_sys_tick(
        meta_matrix, frameLength = 500)
    
    return meta_matrix


def code_micro_and_macro_packet_loss(meta_matrix):
    meta_matrix[np.where((np.diff(meta_matrix[:, 1]) % (2 ** 8)) > 1)[0] + 1, 4] = 1  # Top packet of microloss
    meta_matrix[np.where((np.diff(meta_matrix[:, 3]) >= ((2 ** 16) * .0001)))[0] + 1, 5] = 1  # Top packet of macroloss
    meta_matrix[:, 6] = ((meta_matrix[:, 4]).astype(int) & (meta_matrix[:, 5]).astype(
        int))  # Code coincidence of micro and macro loss
    return meta_matrix


def calculate_statistics(meta_array, intersample_tick_count):
    # Calculate total number of actual data points that were received
    num_real_points = meta_array[:, 9].sum()

    # Calculate the total number of large roll-overs (>= 6.5536 seconds)
    num_macro_rollovers = meta_array[:, 5].sum()

    # Calculate the packet number before and after the occurrence of a small packet losse
    micro_loss_stack = np.dstack((np.where(meta_array[:, 4] == 1)[0] - 1, np.where(meta_array[:, 4] == 1)[0]))[0]

    # Remove small packet losses that coincided with large packet losses
    micro_loss_stack = micro_loss_stack[
        np.isin(micro_loss_stack[:, 1], np.where(meta_array[:, 5] == 1)[0], invert=True)]

    # Allocate array for calculating micropacket loss
    loss_array = np.zeros(len(micro_loss_stack))

    # Loop over meta data to extract and calculate number of data points lost to small packet loss.
    for index, packet in enumerate(micro_loss_stack):
        loss_array[index] = (((meta_array[packet[1], 2] - (meta_array[packet[1], 9] * intersample_tick_count)) -
                              meta_array[packet[0], 2]) % (2 ** 16)) / intersample_tick_count

    # Sum the total number of lost data points due to small packet loss.
    loss_as_scalar = np.around(loss_array).sum()

    return num_real_points, num_macro_rollovers, loss_as_scalar


def unpacker_td(meta_array, input_json, intersample_tick_count):
    # First we verify that num of channels and sampling rate does not change
    if np.diff(meta_array[:, 8]).sum():
        raise ValueError('Number of Active Channels Changes Throughout the Recording')
    if np.diff(meta_array[:, 10]).sum():
        raise ValueError('Sampling Rate Changes Throughout the Recording')

    # Initialize array to hold output data
    final_array = np.zeros((meta_array[:, 9].sum().astype(int), 3 + meta_array[0, 8].astype(int)))

    # Initialize variables for looping:

    array_bottom = 0
    master_time = meta_array[0, 3]
    old_system_tick = 0
    running_ms_counter = 0
    for packet_number, i in enumerate(meta_array):
        if i[5]:
            # We just suffered a macro packet loss...
            old_system_tick = 0
            running_ms_counter = 0
            master_time = i[3]  # Resets the master time

        running_ms_counter += ((i[2] - old_system_tick) % (2 ** 16))
        backtrack_time = running_ms_counter - ((i[9] - 1) * intersample_tick_count)

        # Populate master clock time into array
        final_array[int(array_bottom):int(array_bottom + i[9]), 0] = np.array([master_time] * int(i[9]))

        # Linspace microsecond clock and populate into array
        final_array[int(array_bottom):int(array_bottom + i[9]), 1] = np.linspace(backtrack_time, running_ms_counter,
                                                                                 int(i[9]))

        # Put packet number into array for debugging
        final_array[int(array_bottom):int(array_bottom + i[9]), -1] = np.array([packet_number] * int(i[9]))

        # Unpack timedomain data from original packets into array
        for j in range(0, int(i[8])):
            final_array[int(array_bottom):int(array_bottom + i[9]), j + 2] = \
            input_json[0]['TimeDomainData'][int(i[7])]['ChannelSamples'][j]['Value']

        # Update counters for next loop
        old_system_tick = i[2]
        array_bottom += i[9]

    # Convert systemTick into microseconds

    final_array[:, 1] = final_array[:, 1] * 100
    return final_array


def unpacker_accel(meta_array, input_json, intersample_tick_count):
    # First we verify that num of channels and sampling rate does not change
    if np.diff(meta_array[:, 8]).sum():
        raise ValueError('Number of Active Channels Changes Throughout the Recording')
    if np.diff(meta_array[:, 10]).sum():
        raise ValueError('Sampling Rate Changes Throughout the Recording')

    # Initialize array to hold output data
    final_array = np.zeros((meta_array[:, 9].sum().astype(int), 3 + meta_array[0, 8].astype(int)))

    # Initialize variables for looping:

    array_bottom = 0
    master_time = meta_array[0, 3]
    old_system_tick = 0
    running_ms_counter = 0
    for packet_number, i in enumerate(meta_array):
        if i[5]:
            # We just suffered a macro packet loss...
            old_system_tick = 0
            running_ms_counter = 0
            master_time = i[3]  # Resets the master time

        running_ms_counter += ((i[2] - old_system_tick) % (2 ** 16))
        backtrack_time = running_ms_counter - ((i[9] - 1) * intersample_tick_count)

        # Populate master clock time into array
        final_array[int(array_bottom):int(array_bottom + i[9]), 0] = np.array([master_time] * int(i[9]))

        # Linspace microsecond clock and populate into array
        final_array[int(array_bottom):int(array_bottom + i[9]), 1] = np.linspace(backtrack_time, running_ms_counter,
                                                                                 int(i[9]))

        # Put packet number into array for debugging
        final_array[int(array_bottom):int(array_bottom + i[9]), -1] = np.array([packet_number] * int(i[9]))

        # Unpack timedomain data from original packets into array
        for accel_index, accel_channel in enumerate(['XSamples', 'YSamples', 'ZSamples']):
            final_array[int(array_bottom):int(array_bottom + i[9]), accel_index + 2] = \
            input_json[0]['AccelData'][int(i[7])][accel_channel]

        # Update counters for next loop
        old_system_tick = i[2]
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

    column_names = ['time_master', 'microseconds'] + channel_names + ['packetIdx']
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
