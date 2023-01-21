# CorrDetFramework, steps for reproducibility:
 
Data:

Original data needs to be downloaded to the local machine from data directory, unzipped and stored as casovne-vrste.csv in data/given directory. password for download zip files: m;84J,4'$\m)VSRg


Step 1. Install all the requirements
pip install -r requirements.txt

Step 2. In params.yaml set the parameters values for the total transactions amount per company and for the event and time period

Step 3. Run  python src/cutoff.py, outcome: data/cutoff_data.csv
