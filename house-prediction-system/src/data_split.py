import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv("data/raw/AmesHousing.csv")

# -------------------------
# Separate Features & Target
# -------------------------

X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

# -------------------------
# Train-Test Split
# -------------------------

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# -------------------------
# Display Shapes
# -------------------------

print("Training Features :", X_train.shape)
print("Testing Features  :", X_test.shape)
print("Training Target   :", y_train.shape)
print("Testing Target    :", y_test.shape)