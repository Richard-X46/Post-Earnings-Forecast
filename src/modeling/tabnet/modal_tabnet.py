
from sklearn.datasets import fetch_covtype
from dotenv import load_dotenv
import os 
import modal
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import pandas as pd
import torch
load_dotenv() 



image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "pytorch_tabnet",
    "scikit-learn",
    "torch",
    "xgboost",
    "pandas",
    "dotenv",
)

# handling imports for modal function

with image.imports():
    import pandas as pd
    from sklearn.datasets import fetch_covtype
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
    from sklearn.metrics import accuracy_score

app = modal.App('tabnet-covertype', image = image)



def load_covertype_data():
    data = fetch_covtype()  # Download the dataset if not already present
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target  # Add the target variable to the DataFrame

    X = df.drop(columns=['target'])
    y = df['target']
    # reducing y to start with 0 
    y = y - 1  # Adjust target to start from 0
    y.value_counts()  # Check the distribution of the adjusted target variable

    # convert to float32 for TabNet
    X = X.astype('float32')
    
    return X, y


def split_data(X, y, train_size, test_size, random_state=11):
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test



@app.function(gpu="t4",timeout=60 * 10)
def run_tabnet_modal():
    X, y = load_covertype_data()
    X_train, X_test, y_train, y_test = split_data(X, y,train_size = 50000, test_size=10000)

    # init tabnet classifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = TabNetClassifier(device_name=device, verbose=1)

    clf.fit(
        X_train.values, y_train.values,
        eval_set=[(X_test.values, y_test.values)],
        eval_name=["val"],
        eval_metric=["accuracy"],
        max_epochs=30,
        patience=6,
        batch_size=8192,
        weights=1
    )
    return {
        "accuracy": accuracy_score(y_test, clf.predict(X_test.values)),
        "best_epoch": clf.best_epoch,
    }


# def run_local_xgbc():
#     # Initialize the XGBClassifier
#     model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

#     # Fit the model on the training data
#     model.fit(X_train, y_train)

#     # Evaluate the model on the test data
#     accuracy = model.score(X_test, y_test)
#     print(f"Test Accuracy: {accuracy:.4f}")
#     return model

# model =     run_local_xgbc()


# # poking at the model
# model.score(X_test, y_test)  # Evaluate the model on the test data



# # setting device to mps 
# device = "mps" if torch.backends.mps.is_available() else "cpu"

# clf = TabNetClassifier(device_name=device,verbose = 1)  # local

# clf.fit(X_train.values, y_train.values, eval_set=[(X_test.values, y_test.values)],
#         eval_name=["val"], eval_metric=["accuracy"],
#         max_epochs=30, patience=8, batch_size=4096, weights=1
#      )
       



@app.local_entrypoint()
def main():
    # Run the TabNet model on Modal
    result = run_tabnet_modal.remote()
    print(f"Accuracy: {result['accuracy']:.4f}, best epoch: {result['best_epoch']}")

if __name__ == "__main__":
    main()






# --- modal costing ---
cost_t4_second = 0.000164 
cost_t4_hour = cost_t4_second * 3600
# number of hours i can run given 30 dollars
hours_i_can_run = 30 / cost_t4_hour

