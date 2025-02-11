import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error
from joblib import dump,load


def main():
    # Reading CVS
    DATASET_PATH = 'project/insurance.csv'
    df = pd.read_csv(DATASET_PATH)

    # Aplying One-Hot-Enconding
    smoker_Encoded = pd.get_dummies(df['smoker']) # YES / NO
    sex_Encoded = pd.get_dummies(df['sex']) # Male / Female
    region_Encoded = pd.get_dummies(df['region']) # Northeast / Northwest / Southeast / Southwest

    # Dropping the columns with where we did One-Hot-Enconding

    df = df.drop(["sex", "smoker", "region"], axis=1)

    # Concatenating the One-Hot-Enconding columns with the original df

    df = pd.concat([df, smoker_Encoded,sex_Encoded,region_Encoded],axis=1)

    # Dividing df into inputs and outputs
    X = df.drop('charges',axis=1)
    y = df['charges']

    # Dividing the data into training and testing sets
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=101)
    
    # Applying scalar function 
    scaler = StandardScaler()

    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)

    # ElasticNet 

    base_elastic_model = ElasticNet()
    param_grid = {'alpha':[0.1,1,5,10,50,100],
              'l1_ratio':[.1, .5, .7, .9, .95, .99, 1]}

    # verbose number a personal preference
    grid_model = GridSearchCV(estimator=base_elastic_model,
                          param_grid=param_grid,
                          scoring='neg_mean_squared_error',
                          cv=5,
                          verbose=1)
    

    grid_model.fit(scaled_X_train,y_train)

    #y_pred = grid_model.predict(scaled_X_test)
    #mean_absolute_error(y_test,y_pred)
    #np.sqrt(mean_squared_error(y_test,y_pred))
    #np.mean(df['charges'])

    # Save both the scaler and the model
    dump(scaler, './saved_model/scaler.joblib')
    dump(grid_model,'./saved_model/insurance_model.joblib')

    # Loading the model
    #loaded_model = load('insurance_model.joblib')

    # Doing the same transformation that we did for the training data
    #input_data = scaler.transform(input_data)

    # Euning the model
    #prediction = loaded_model.predict(input_data)

    #return prediction

# Run the app
if __name__ == "__main__":
    main()