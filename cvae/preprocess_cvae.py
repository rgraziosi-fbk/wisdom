# This script takes as input a CSV log splitted in train-val-test and prepares it to be used with the CVAE for process mining (https://github.com/rgraziosi-fbk/cvae-process-mining)
# More in detail, this script:
# 1. Merges the three splits into a single log
# 2. For each activity in the log, creates a START activity with the start:timestamp value + a COMPLETE activity with the time:timestamp value of the original event
# (this step is needed because, as of May 2025, CVAE does not support the generation of start+end timestamps, but just a single timestamp)
# 3. Preprocess timestamps as required by the CVAE architecture
# 4. Splits the log again into the original train-val-test splits
#
# After generating logs with CVAE, you can merge START and COMPLETE events using the postprocess_cvae.py script

import pandas as pd
import os
import time
import matplotlib.pyplot as plt

BASE_PATH = '/your/base/path'
FOLDER_NAME = 'sepsis_imbalance_0.1'
LOG_NAME = 'sepsis'

OUTPUT_PATH = os.path.join(BASE_PATH, FOLDER_NAME)

TRAIN_LOG_PATH = os.path.join(BASE_PATH, FOLDER_NAME, f'{LOG_NAME}_train.csv')
VAL_LOG_PATH = os.path.join(BASE_PATH, FOLDER_NAME, f'{LOG_NAME}_val.csv')
TEST_LOG_PATH = os.path.join(BASE_PATH, FOLDER_NAME, f'{LOG_NAME}_test.csv')

CSV_SEP = ','

# Keys present in original log
# sepsis | BPI17
OG_CASE_ID_KEY = 'trace_id'
OG_ACTIVITY_KEY = 'prefix'
OG_TIMESTAMP_KEY = 'time:timestamp'
OG_RESOURCE_KEY = 'Resource'

# bpic2012
# OG_CASE_ID_KEY = 'Case ID'
# OG_ACTIVITY_KEY = 'concept:name'
# OG_TIMESTAMP_KEY = 'time:timestamp'
# OG_RESOURCE_KEY = 'Resource'

# Keys to convert to
CASE_ID_KEY = 'Case ID'
ACTIVITY_KEY = 'Activity'
TIMESTAMP_KEY = 'time:timestamp'
RESOURCE_KEY = 'Resource'

# Method to read log
def read_log(dataset_path, separator=';', timestamp_key='time:timestamp', verbose=True):
  """Read xes or csv logs"""
  if dataset_path.endswith('.csv'):
    log = pd.read_csv(dataset_path, sep=separator)
    log[timestamp_key] = pd.to_datetime(log[timestamp_key], format='mixed')
  else:
    raise ValueError("Unsupported file extension")
    
  return log


# (1) Merge logs and rename columns

print('(1) Merge logs...')

# Read logs and add respective 'log' column
train_log = read_log(TRAIN_LOG_PATH, separator=CSV_SEP, timestamp_key=OG_TIMESTAMP_KEY)
train_log['log'] = 'train'

val_log = read_log(VAL_LOG_PATH, separator=CSV_SEP, timestamp_key=OG_TIMESTAMP_KEY)
val_log['log'] = 'val'

test_log = read_log(TEST_LOG_PATH, separator=CSV_SEP, timestamp_key=OG_TIMESTAMP_KEY)
test_log['log'] = 'test'

# Count traces in logs
train_cases_og = list(train_log[OG_CASE_ID_KEY].unique())
val_cases_og = list(val_log[OG_CASE_ID_KEY].unique())
test_cases_og = list(test_log[OG_CASE_ID_KEY].unique())
print(f'Train cases = {len(train_cases_og)}, val cases = {len(val_cases_og)}, test cases = {len(test_cases_og)}')

# Merge logs
log = pd.concat([train_log, val_log, test_log], ignore_index=True)

# Change cols name
log = log.rename(columns={
  OG_CASE_ID_KEY: CASE_ID_KEY,
  OG_ACTIVITY_KEY: ACTIVITY_KEY,
  OG_TIMESTAMP_KEY: TIMESTAMP_KEY,
  OG_RESOURCE_KEY: RESOURCE_KEY,
})

# (2) Split each activity in START and COMPLETE activities
print('(2) Splitting in START and COMPLETE activities...')

log_start_events = log.copy(deep=True)
log_complete_events = log.copy(deep=True)

for idx, row in log_start_events.iterrows():
  # append START to activity name
  log_start_events.at[idx, ACTIVITY_KEY] = row[ACTIVITY_KEY] + '_START'

  # convert start:timestamp from Unix time to standard time
  log_start_events.at[idx, 'start:timestamp'] = pd.to_datetime(row['start:timestamp'], unit='s')
  log_start_events.at[idx, TIMESTAMP_KEY] = log_start_events.at[idx, 'start:timestamp']

for idx, row in log_complete_events.iterrows():
  # append COMPLETE to activity name
  log_complete_events.at[idx, ACTIVITY_KEY] = row[ACTIVITY_KEY] + '_COMPLETE'

log_start_events = log_start_events.drop(columns=['start:timestamp'])
log_complete_events = log_complete_events.drop(columns=['start:timestamp'])

log = pd.concat([log_start_events, log_complete_events])
log = log.sort_values(by=[TIMESTAMP_KEY])


# Save log
log.to_csv(os.path.join(OUTPUT_PATH, f'{LOG_NAME}.csv'), sep=';', index=False)

# (3) Transform timestamps in relative_timestamp
TIMESTAMP_FROM_START_KEY = 'relative_timestamp_from_start'
TIMESTAMP_FROM_PREV_KEY = 'relative_timestamp_from_previous_activity'

def add_trace_attr_relative_timestamp_to_first_activity(
    log, trace_key='case:concept:name', timestamp_key='time:timestamp',
    custom_timestamp_key='relative_timestamp_from_start'):
  
  traces = list(log.groupby(trace_key).groups.values())
  log[custom_timestamp_key] = 0.0
  
  # get ts of first activity in the log
  lowest_timestamp = log[timestamp_key].min()

  for t in traces:
    # get ts of first activity in trace
    lowest_timestamp_trace = log.iloc[t][timestamp_key].min()
    # compute diff between first activity in trace and first activity in log
    custom_timestamp = (lowest_timestamp_trace - lowest_timestamp).total_seconds() / 60.0

    log.loc[t, custom_timestamp_key] = custom_timestamp

  return log


def add_relative_timestamp_between_activities(
    log, trace_key='case:concept:name', timestamp_key='time:timestamp',
    custom_timestamp_key='relative_timestamp_from_previous_activity'):
  
  traces = list(log.groupby(trace_key).groups.values())
  log[custom_timestamp_key] = 0.0

  for t in traces:
    for n, a in enumerate(t):
      if n == 0:
        log.loc[a, custom_timestamp_key] = 0.0
        continue

      log.loc[a, custom_timestamp_key] = (log.iloc[t[n]][timestamp_key] - log.iloc[t[n-1]][timestamp_key]).total_seconds() / 60.0

  return log

print('(3) Preprocessing timestamps...')

start_time = time.time()

log = read_log(os.path.join(OUTPUT_PATH, f'{LOG_NAME}.csv'))

log = add_trace_attr_relative_timestamp_to_first_activity(
  log,
  trace_key=CASE_ID_KEY,
  timestamp_key=TIMESTAMP_KEY,
  custom_timestamp_key=TIMESTAMP_FROM_START_KEY,
)
log = add_relative_timestamp_between_activities(
  log,
  trace_key=CASE_ID_KEY,
  timestamp_key=TIMESTAMP_KEY,
  custom_timestamp_key=TIMESTAMP_FROM_PREV_KEY,
)

log.to_csv(os.path.join(OUTPUT_PATH,  f'{LOG_NAME}_final.csv'), sep=';', index=False)

end_time = time.time()

print(f'Execution time: {end_time - start_time} seconds')

# Plot histogram of new timestamp
plt.hist(log[TIMESTAMP_FROM_START_KEY], bins=50)
plt.xlabel(TIMESTAMP_FROM_START_KEY)
plt.ylabel('Frequency')
plt.title(f'Histogram of {TIMESTAMP_FROM_START_KEY}')
plt.show()

plt.hist(log[TIMESTAMP_FROM_PREV_KEY], bins=50)
plt.xlabel(TIMESTAMP_FROM_PREV_KEY)
plt.ylabel('Frequency')
plt.title(f'Histogram of {TIMESTAMP_FROM_PREV_KEY}')
plt.show()

# (4) Split log

print('(4) Split logs...')

log = read_log(os.path.join(OUTPUT_PATH,  f'{LOG_NAME}_final.csv'))

# Split log in train, val, test based on 'log' column
train_log = log[log['log'] == 'train']
val_log = log[log['log'] == 'val']
test_log = log[log['log'] == 'test']

train_cases = list(train_log['Case ID'].unique())
val_cases = list(val_log['Case ID'].unique())
test_cases = list(test_log['Case ID'].unique())

print(f'Train cases = {len(train_cases)}, val cases = {len(val_cases)}, test cases = {len(test_cases)}')

assert len(train_cases) == len(train_cases_og)
assert len(val_cases) == len(val_cases_og)
assert len(test_cases) == len(test_cases_og)

# Drop column 'log'
train_log = train_log.drop(columns=['log'])
val_log = val_log.drop(columns=['log'])
test_log = test_log.drop(columns=['log'])

# Save logs
train_log.to_csv(os.path.join(OUTPUT_PATH, f'{LOG_NAME}_TRAIN_final.csv'), sep=';', index=False)
val_log.to_csv(os.path.join(OUTPUT_PATH, f'{LOG_NAME}_VAL_final.csv'), sep=';', index=False)
test_log.to_csv(os.path.join(OUTPUT_PATH, f'{LOG_NAME}_TEST_final.csv'), sep=';', index=False)


log = read_log(os.path.join(OUTPUT_PATH,  f'{LOG_NAME}_final.csv'))

log = log.drop(columns=['log'])

log_cases = list(log['Case ID'].unique())

assert len(log_cases) == len(train_cases) + len(val_cases) + len(test_cases)

log.to_csv(os.path.join(OUTPUT_PATH, f'{LOG_NAME}_final.csv'), sep=';', index=False)