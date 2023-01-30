# CorrDetFramework, steps for reproducibility:
 
Data:

Original data needs to be downloaded to the local machine from data directory, unzipped and stored as casovne-vrste.csv in data/given directory. password for download zip files: m;84J,4'$\m)VSRg
Data file should be named 'original_data.csv' and stored in data/data directory

Step 1. Install all the requirements
pip install -r requirements.txt

Step 2. In params.yaml set the parameters values:
1. cutoff: for the total transactions amount per company and 
2. event, period: event represents the exact date and period is number of months before and after the event

All .py scripts are in src directory and all the output files with the data come from executing the scripts

Step 3. Run python src/cutoff.py, outcome: data/data/cutoff_data.csv

Step 4. Follow the STEP 1 and STEP 2 from the Framework pipeline:

![step1](https://user-images.githubusercontent.com/62762528/215510473-7ff91d10-1cea-4f60-a4c6-34ba286f4589.png)
![step2](https://user-images.githubusercontent.com/62762528/215510958-6f39eeec-3ea4-4929-822f-5b246566feb3.png)
