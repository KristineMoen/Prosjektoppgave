import pandas as pd
import patsy
import statsmodels.api as sm
import numpy as np


# Eksempeldata
data = {
    'lønn': [500000, 600000, 550000, 700000, 520000, 580000, 620000, 650000],
    'alder': [25, 45, 35, 50, 30, 40, 45, 50],
    'kjønn': ['mann', 'kvinne', 'mann', 'kvinne', 'mann', 'kvinne', 'mann', 'kvinne']
}
df = pd.DataFrame(data)

# Alternativt kan du sette rekkefølgen direkte i DataFrame:
df['kjønn'] = pd.Categorical(df['kjønn'], categories=['mann', 'kvinne'], ordered=False)

# Kvadrert alder
y, X = patsy.dmatrices('lønn ~ alder + I(alder**2) + C(kjønn, Treatment(reference="mann"))', data=df, return_type='dataframe')
model = sm.OLS(y, X).fit()


model = sm.OLS(y, X).fit()
print(model.summary())

