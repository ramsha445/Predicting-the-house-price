# Predicting-the-house-price
 A model is trained through random forest. Afterwards a new data is been added to predict the house price. You can change the feature as well. but when u change the feature u should change the data accordingly.


     import pandas as pd
     import numpy as np

     import seaborn as sns

     import matplotlib.pyplot as plt

     from sklearn.model_selection import train_test_split
     from sklearn.ensemble import RandomForestRegressor
     from sklearn.metrics import mean_squared_error, r2_score
     from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv("AmesHousing.csv")  # Replace with your CSV filename
    print(df.shape)
    print(df.columns)
# Drop rows with missing values for simplicity
    print("Columns in dataset:\n", df.columns.tolist())
    features = ['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Total Bsmt SF', 'Year Built']
    target = 'SalePrice'
# Fill missing values
    for col in features + [target]:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
               df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())


# Define X and y
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("R² Score:", r2_score(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

#Model is trained now. I am giving a new data to predict the price of house 

    new_house = pd.DataFrame([{'Overall Qual': 7, 'Gr Liv Area': 1800, 'Garage Cars': 2, 'Total Bsmt SF': 800, 'Year Built': 2005}])
    predicted_price = model.predict(new_house)

    print(f"Predicted price: ${predicted_price[0]:,.2f}")
