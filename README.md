# CorrDetFramework, steps for reproducibility:
 
Data:

Original data needs to be downloaded to the local machine from data directory, unzipped and stored as casovne-vrste.csv in data/given directory. password for download zip files: m;84J,4'$\m)VSRg
Data file should be named 'original_data.csv' and stored in data/data directory

Step 1. Install all the requirements
pip install -r requirements.txt

Step 2. In params.yaml set the parameters values:
1. for the total transactions amount per company and 
2. for the event and time period

Step 3. Run  python src/cutoff.py, outcome: data/data/cutoff_data.csv

Step 4. Follow the process from the pipeline.png

![pipeline](https://user-images.githubusercontent.com/62762528/214508239-61869718-ef90-4344-850d-408482fd5c4f.png)
