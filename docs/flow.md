
# backfill flow logic

- snp 500 list of symbols 
- AV api call to get earnings tables
- iterate on the earnings table to get the trasncripts for each earnings event
- write the earnings table and trancripts to s3 as hive partition.


# modeling logic 

- snp500 companies ohclv data, Technical indicators, earnings data, and transcripts data
- each row represents a quarter with data transformed into d-10 to d+10 format where d is the earnings date.
- target variable is the pct return on d+10 from d+2 
- modelling will involve both dnn and ml models to predict the target variable.