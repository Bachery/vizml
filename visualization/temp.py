import pandas as pd
from tqdm import tqdm
import linecache
import json
import numpy as np


def clean_chunk(chunk):
    # Filtering
    df_final_rows = []
    errors = 0
    empty_fields = 0
    global charts_without_data
    global chart_loading_errors

    for i, x in chunk.iterrows():
        try:
            chart_data = json.loads(x.chart_data)
            layout = json.loads(x.layout)
            table_data = json.loads(x.table_data)

            # Filter empty fields
            if not (bool(chart_data) and bool(table_data)):
                empty_fields += 1

                charts_without_data += 1
                chart_loading_errors += 1
                continue

            df_final_rows.append({
                'fid': x['fid'],
                'chart_data': chart_data,
                'layout': layout,
                'table_data': table_data
            })

        except Exception as e:
            errors += 1
            print(e)
            continue

    return pd.DataFrame(df_final_rows)


data_file = '../data/plot_data.tsv'
chunk_size = 1000

valid_each_chunk = []

df = pd.read_table(
    data_file,
    error_bad_lines=False,
    chunksize=chunk_size,
    encoding='utf-8',
    # usecols=[0, 3]
)

for i, chunk in tqdm(enumerate(df)):
    df = clean_chunk(chunk)
    valid_each_chunk.append(df.shape[0])
    if i == 2100: break

print('avg: ', np.mean(valid_each_chunk))
print('sum: ', np.sum(valid_each_chunk))


# row = 2101978
# # row = 1

# cache_data = linecache.getline(data_file, row)
# f =  open('row_'+str(row)+'.txt', 'w')
# f.write(cache_data)
# f.close()

# cache_data = linecache.getline(data_file, row+1)
# f =  open('row_'+str(row+1)+'.txt', 'w')
# f.write(cache_data)
# f.close()

# cache_data = linecache.getline(data_file, row+2)
# f =  open('row_'+str(row+2)+'.txt', 'w')
# f.write(cache_data)
# f.close()