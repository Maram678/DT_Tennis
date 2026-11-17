import joblib
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

tennis = pd.read_csv("tennis.csv")
X, y = tennis.drop(columns="Play"), tennis["Play"]

pipe = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"),
    DecisionTreeClassifier(criterion="entropy", random_state=0),
)
pipe.fit(X, y)
joblib.dump(pipe, "tennis_model.joblib")
print("Saved tennis_model.joblib")
