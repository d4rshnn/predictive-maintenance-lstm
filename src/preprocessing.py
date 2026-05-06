from sklearn.preprocessing import MinMaxScaler

# calculate remaining useful life
def calculate_rul(df):
    max_cycle = df.groupby('id')['cycle'].transform('max')
    df['RUL'] = max_cycle - df['cycle']

    # cap RUL at 125
    df['RUL'] = df['RUL'].clip(upper=125)

    return df


# preprocess data
def preprocess_data(df, scaler=None):

    # remove cycle before scaling
    df = df.drop(columns=['cycle'])

    # remove flat sensors
    df = df.drop(columns=['op3', 's1', 's5', 's10', 's16', 's18', 's19'])

    X = df.drop(columns=['RUL', 'id'])
    y = df['RUL']

    if scaler is None:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y, scaler