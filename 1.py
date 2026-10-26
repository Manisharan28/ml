
import pandas as pd
from pandas import DataFrame

# Importing data from CSV file
nam = ['buying', 'maint', 'doors', 'persons',
       'lug_boot', 'safety', 'class']
data = pd.read_csv(
    r'C:\Users\pende\Desktop\ML LAB\Datasets\Datasets\cardata.csv',
    names=nam
)

print(data)

# Exporting data to Excel file
Cars = {
    'Brand': ['Honda Civic', 'Toyota Corolla',
              'Ford Focus', 'Audi A4'],
    'Price': [32000, 35000, 37000, 45000]
}

df = DataFrame(Cars, columns=['Brand', 'Price'])

df.to_excel('export_dataframe.xlsx',
            index=False, header=True)

print("Data exported successfully")
