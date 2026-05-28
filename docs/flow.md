
# backfill flow logic

- snp 500 list of symbols 
- AV api call to get earnings tables
- iterate on the earnings table to get the trasncripts for each earnings event
- write the earnings table and trancripts to s3 as hive partition.