import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Create a dataframe
data = {'name': ['George', 'Paul', 'Ringo', 'John', 'Pete', 'Stuart', 'Brian', 'Billy', 'Tony', 'Tommy'],
        'instrument': ['guitar', 'bass', 'drums', 'guitar', 'drums', 'bass', 'guitar', 'bass', 'drums', 'guitar']}
df = pd.DataFrame(data)

# Create an instance of the OneHotEncoder
encoder = OneHotEncoder()

# Fit the encoder to the data
encoder.fit(df[['instrument']])

# Transform the data
encoded = encoder.transform(df[['instrument']]).toarray()

# Rewrite the dataframe with the encoded data
df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['instrument']))
df_encoded = pd.concat([df, df_encoded], axis=1)

# Split the data, keeping the original instrument column for reference
X_train, X_test = train_test_split(df_encoded, test_size=0.2, random_state=42)
print(X_test)
# Inverse transform the one-hot encoded columns to get the original instrument values
df_inverse = encoder.inverse_transform(X_test[encoder.get_feature_names_out(['instrument'])])
df_inverse = pd.DataFrame(df_inverse, columns=['instrument'])

# Restore the original instrument column in X_test
X_test = X_test.drop(columns=encoder.get_feature_names_out(['instrument']))
X_test = pd.concat([X_test.reset_index(drop=True), df_inverse.reset_index(drop=True)], axis=1)

print(X_test)


# #Same for label encoding
# from sklearn.preprocessing import LabelEncoder

# # Create an instance of the LabelEncoder
# encoder = LabelEncoder()

# # Fit the encoder to the data

# encoder.fit(df['instrument'])

# # Transform the data
# encoded = encoder.transform(df['instrument'])

# # rewrite the dataframe with the encoded data

# df['instrument'] = encoded
# print(df)

# #Display the encoder categories

# print(encoder.classes_)

# #inverse transform and display the original dataframe

# df_inverse = encoder.inverse_transform(encoded)
# df = df.drop(columns=['instrument'])
# df = pd.concat([df, pd.DataFrame(df_inverse, columns=['instrument'])], axis=1)

# print(df)