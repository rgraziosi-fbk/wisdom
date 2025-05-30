# This script can be used as a post-processing step for merging the START and COMPLETE activities (that have been added by the preprocess_cvae.py script) into a single activity
# As a reminder, CVAE supports the generation of just one timestamp, so in the preprocessing each event had been split into START and COMPLETE events; now we merge them back into one event

import pandas as pd

LOG_PATH = '/your/path/to/log.csv'


log = pd.read_csv(LOG_PATH, sep=';')
log['start:timestamp'] = None
new_log = pd.DataFrame(columns=log.columns)

attended_rows = []
num_errors = 0

log_cases = log['case:concept:name'].unique().tolist()
print(f'There are {len(log_cases)} cases.')

for idx, row in log.iterrows():
  print(f'{idx}/{len(log)}', end='\r')

  if idx in attended_rows: continue
  attended_rows.append(idx)

  row_case = row['case:concept:name']
  row_activity = row['concept:name']
  row_activity_name = row_activity.split('_')[0]
  row_activity_type = row_activity.split('_')[-1]

  row_to_add = row.copy(deep=True)
  row_to_add['concept:name'] = row_activity_name
  row_to_add['Activity'] = row_activity_name
  row_to_add['start:timestamp'] = row['time:timestamp']

  if row_activity_type == 'COMPLETE':
    num_errors += 1
    print('Found a COMPLETE before a START')

  elif row_activity_type == 'START':
    complete_found = False

    for inner_idx, inner_row in log.loc[(idx+1):].iterrows():
      if inner_idx in attended_rows: continue
      if inner_row['case:concept:name'] != row['case:concept:name']: continue

      inner_row_activity = inner_row['concept:name']
      inner_row_activity_name = inner_row_activity.split('_')[0]
      inner_row_activity_type = inner_row_activity.split('_')[-1]

      if inner_row_activity_name == row_activity_name and inner_row_activity_type == 'COMPLETE':
        attended_rows.append(inner_idx)
        
        row_to_add['time:timestamp'] = inner_row['time:timestamp'] # use COMPLETE activity timestamp as end_timestamp

        complete_found = True
        break
    
    if complete_found == False:
      num_errors += 1
      print('Found a START without a COMPLETE')
    
  else:
    assert False, f'Unrecognized activity lifecycle {row_activity_type}'

  new_log = pd.concat([new_log, pd.DataFrame([row_to_add])], ignore_index=True)


new_log_cases = new_log['case:concept:name'].unique().tolist()
print(f'There are {len(new_log_cases)} cases in the new log.')

print(f'Num errors = {num_errors}')

new_log.to_csv('new.csv', sep=';', index=False)


assert len(log_cases) == len(new_log_cases), f'Discrepancy in new log cases ({len(new_log_cases)} != {len(log_cases)})'