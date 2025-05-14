# Import necessary libraries for data manipulation, file handling, visualization, and machine learning.
import pandas as pd # For data manipulation and analysis, especially DataFrames.
import numpy as np # For numerical operations, handling arrays, and NaN/infinity values.
import os # For interacting with the operating system, like checking directory paths.
import glob # For finding files matching a specific pattern (e.g., all CSV files in a directory).
import matplotlib.pyplot as plt # For creating static, animated, and interactive visualizations.
import seaborn as sns # For making statistical graphics more attractive and informative.
from sklearn.model_selection import train_test_split # For splitting data into training and testing sets.
from sklearn.preprocessing import StandardScaler # For standardizing features (mean 0, variance 1).
from sklearn.impute import SimpleImputer # For handling missing values in the data.
from sklearn.metrics import roc_curve # For generating ROC curve data.
from imblearn.over_sampling import SMOTE # For handling class imbalance using Synthetic Minority Over-sampling Technique.
from sklearn.linear_model import LogisticRegression # Logistic Regression classifier.
from sklearn.tree import DecisionTreeClassifier # Decision Tree classifier.
from sklearn.ensemble import RandomForestClassifier # Random Forest classifier.
from sklearn.neighbors import KNeighborsClassifier # K-Nearest Neighbors classifier.
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score # For model evaluation.
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV # For hyperparameter tuning.

# --- 1. Configuration & Data Loading ---
# I defined the path to the directory containing my CICIDS2017 CSV files.
# This path needs to be adjusted if the data is located elsewhere.
csv_directory_path = 'C:/Users/aaara/Documents/Deakin/Deakin cybersec T1 2025/SIT 326/HD Task/CICIDS2017' 
print(f"Looking for CSV files in: {csv_directory_path}")

# I performed a check to ensure the specified directory exists.
if not os.path.isdir(csv_directory_path):
    raise FileNotFoundError(f"Directory not found: {csv_directory_path}")

# I used glob to get a list of all CSV file paths in the directory.
csv_files = glob.glob(os.path.join(csv_directory_path, "*.csv")) 
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {csv_directory_path}") 

print(f"Found {len(csv_files)} CSV files. Loading...") 
# I initialized an empty list to store individual DataFrames.
list_of_dataframes = []
# I iterated through each found CSV file path.
for file_path in csv_files:
    try:
        # I read each CSV into a temporary DataFrame.
        # low_memory=False was used to handle potentially mixed data types more effectively.
        # on_bad_lines='warn' will print a warning for lines that can't be parsed correctly.
        # encoding='latin1' was chosen as it often works for datasets with special characters.
        df_temp = pd.read_csv(file_path, low_memory=False, on_bad_lines='warn', encoding='latin1')
        list_of_dataframes.append(df_temp)
    except Exception as e:
        # If an error occurred while loading a file, I printed an error message and skipped the file.
        print(f"Error loading {file_path}: {e}. Skipping.")

# I checked if any DataFrames were successfully loaded.
if not list_of_dataframes:
    raise ValueError("No dataframes were loaded. Check CSV files.")

# I concatenated all the loaded DataFrames into a single DataFrame.
# ignore_index=True resets the index of the combined DataFrame.
combined_df = pd.concat(list_of_dataframes, ignore_index=True)
print(f"Successfully loaded and combined data. Shape: {combined_df.shape}")


# --- 2. Initial Cleaning & Preprocessing ---
# I stripped leading/trailing whitespace from all column names for consistency.
combined_df.columns = combined_df.columns.str.strip() 
print("Column names stripped.")

# I handled specific string artifacts (like 'ï¿½') in the 'Label' column if it exists, replacing them with a hyphen.
# This was based on an observation of potential encoding issues in the dataset.
if 'Label' in combined_df.columns:
    combined_df['Label'] = combined_df['Label'].str.replace('ï¿½', '-', regex=False)

# I replaced infinity and negative infinity values with NaN (Not a Number) to allow for proper numerical processing.
combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
print(f"Rows before dropping NaNs after infinity replacement: {len(combined_df)}")
# For simplicity in this stage, I dropped all rows that contained any NaN values.
# In a more robust pipeline, I might consider targeted imputation for features (X) before splitting,
# but for this initial pass, dropping NaNs ensures a clean dataset for subsequent steps.
combined_df.dropna(inplace=True) 
print(f"Rows after dropping all NaNs: {len(combined_df)}")


# I removed duplicate rows to ensure each data point is unique.
initial_rows = len(combined_df)
combined_df.drop_duplicates(inplace=True)
print(f"Dropped {initial_rows - len(combined_df)} duplicate rows. Shape after duplicates removal: {combined_df.shape}")

# --- 3. Feature Engineering: Binary Label ---
# I created a binary target label ('Binary_Label') for the classification task.
# 'BENIGN' traffic was mapped to 0, and all other labels (attacks) were mapped to 1.
if 'Label' not in combined_df.columns:
    # I added a check to ensure the 'Label' column exists before trying to create the binary target.
    raise ValueError("'Label' column not found. Cannot create binary target.")
combined_df['Binary_Label'] = combined_df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
print("\nBinary Label Distribution:")
# I printed the normalized distribution of the binary label to check for class imbalance.
print(combined_df['Binary_Label'].value_counts(normalize=True))

# --- 4. Prepare Features (X) and Target (y) ---
# I defined a list of columns to drop when creating the feature set (X).
# These include the original multi-class label, the new binary label, and identifier columns.
columns_to_drop_for_X = ['Label', 'Binary_Label', 'Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
# I filtered this list to only include columns that actually exist in my DataFrame, preventing errors if some are missing.
columns_to_drop_for_X_existing = [col for col in columns_to_drop_for_X if col in combined_df.columns]

# I created the feature matrix X by dropping the specified columns.
X = combined_df.drop(columns=columns_to_drop_for_X_existing)
# I assigned the 'Binary_Label' column as the target variable y.
y = combined_df['Binary_Label']

# I ensured that X contains only numeric features.
# If object-type columns (typically strings) were found in X, I printed a warning.
if X.select_dtypes(include=['object']).shape[1] > 0:
    print("Warning: Object type columns found in X. These need to be handled (e.g., encoded or dropped).")
    print(X.select_dtypes(include=['object']).columns)
    # For this script, I attempted to convert any object columns to numeric, coercing errors to NaN.
    # This is a simplifying assumption; more sophisticated encoding might be needed for truly categorical features.
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    # If NaNs were introduced by pd.to_numeric, they would be handled by the imputer below.
    
# I imputed any remaining NaNs in the feature set X using the median strategy.
# This handles NaNs that might have resulted from the pd.to_numeric coercion or if the earlier dropna wasn't aggressive enough for X.
if X.isnull().values.any():
    print("Imputing remaining NaNs in features (X)...")
    imputer_X = SimpleImputer(strategy='median') # Using median as it's robust to outliers.
    X_imputed = imputer_X.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns) # Reconstructing DataFrame to keep column names.

print(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")

# --- 5. Train-Test Split ---
# I split the data into training (70%) and testing (30%) sets.
# random_state=42 ensures reproducibility of the split.
# stratify=y ensures that the class proportions are maintained in both train and test sets, which is important for imbalanced datasets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) 
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# --- 6. Feature Scaling ---
# I applied StandardScaler to standardize the features (mean 0, variance 1).
# This is important for algorithms sensitive to feature magnitudes (e.g., KNN, Logistic Regression with regularization).
scaler = StandardScaler()
# I fitted the scaler ONLY on the training data to prevent data leakage from the test set.
X_train_scaled = scaler.fit_transform(X_train)
# I then transformed both the training and test data using the fitted scaler.
X_test_scaled = scaler.transform(X_test) 

# I converted the scaled NumPy arrays back to pandas DataFrames to retain column names for easier interpretation.
X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X.columns)
print("Features scaled.")

# --- 7. Handle Class Imbalance on Training Data (SMOTE) ---
# I checked the class distribution in the training set before applying SMOTE.
print(f"\nClass distribution in y_train before SMOTE: \n{y_train.value_counts(normalize=True)}")
# I calculated the number of samples in the minority class to set k_neighbors for SMOTE.
# SMOTE's k_neighbors must be less than the number of samples in the minority class.
minority_class_count = y_train.value_counts().min()
# I set k_neighbors to min(5, count-1) to avoid errors if the minority class is very small.
smote_k_neighbors = min(5, minority_class_count - 1) if minority_class_count > 1 else 1

# I defaulted the resampled data to the original training data in case SMOTE fails or is not applicable.
X_train_resampled, y_train_resampled = X_train, y_train 
if smote_k_neighbors >= 1: # SMOTE can only be applied if k_neighbors is at least 1.
    print(f"Applying SMOTE with k_neighbors={smote_k_neighbors}...")
    smote = SMOTE(random_state=42, k_neighbors=smote_k_neighbors)
    try:
        # I applied SMOTE to the training data (X_train, y_train) to oversample the minority class.
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"Class distribution after SMOTE: \n{y_train_resampled.value_counts(normalize=True)}")
    except Exception as e:
        # If SMOTE failed, I printed an error and used the original (imbalanced) training data.
        print(f"Error during SMOTE: {e}. Using original training data.")
else:
    # If there weren't enough samples in the minority class for SMOTE, I used the original training data.
    print("Not enough samples in minority class for SMOTE (k_neighbors < 1). Using original training data.")


# --- 8. Model Training and Evaluation Function ---
# I defined a helper function to train each model and print its evaluation metrics.
# This promotes code reusability and consistency in evaluation.
def train_and_evaluate_model(model, X_train_data, y_train_data, X_test_data, y_test_data, model_name_str):
    """
    Trains a given model and evaluates its performance.
    My function takes the model instance, training data, testing data, and model name as input.
    It prints classification reports, confusion matrices, and key performance scores.
    """
    print(f"\n--- Training {model_name_str} ---")
    # I fitted the model on the provided training data.
    model.fit(X_train_data, y_train_data)
    
    print(f"\n--- Evaluating {model_name_str} ---")
    # I made predictions on the test data.
    y_pred_values = model.predict(X_test_data)
    # I obtained probability predictions for ROC AUC calculation, if the model supports predict_proba.
    y_pred_proba_values = model.predict_proba(X_test_data)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"{model_name_str} Classification Report:")
    # I printed the classification report, including precision, recall, and F1-score for each class.
    # zero_division=0 prevents warnings if a class has no predicted samples (though unlikely here).
    print(classification_report(y_test_data, y_pred_values, zero_division=0)) 
    
    print(f"{model_name_str} Confusion Matrix:")
    # I printed the confusion matrix to show true positives, true negatives, false positives, and false negatives.
    print(confusion_matrix(y_test_data, y_pred_values))
    
    if y_pred_proba_values is not None:
        # If probability scores are available, I calculated and printed the ROC AUC score.
        print(f"{model_name_str} ROC AUC Score: {roc_auc_score(y_test_data, y_pred_proba_values):.4f}")
    # I calculated and printed precision, recall, and F1-score specifically for the anomalous class (pos_label=1).
    print(f"{model_name_str} Precision (Anomalous): {precision_score(y_test_data, y_pred_values, pos_label=1, zero_division=0):.4f}")
    print(f"{model_name_str} Recall (Anomalous): {recall_score(y_test_data, y_pred_values, pos_label=1, zero_division=0):.4f}")
    print(f"{model_name_str} F1-Score (Anomalous): {f1_score(y_test_data, y_pred_values, pos_label=1, zero_division=0):.4f}")
    
    return model # I returned the trained model instance.

# --- Define Models ---
# I defined a dictionary of models I wanted to train, with their respective configurations.
# These configurations were chosen based on common practices or initial exploratory analysis.
models_to_train = {
    "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1), # n_jobs=-1 uses all available CPU cores.
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1) # Default n_neighbors=5 is a common starting point.
}

# I initialized an empty dictionary to store the trained models.
trained_models = {}
# I iterated through my defined models, training and evaluating each one using the helper function.
# Models were trained on the SMOTE-resampled training data and evaluated on the original (scaled) test data.
for model_name, model_instance in models_to_train.items():
    trained_models[model_name] = train_and_evaluate_model(model_instance, 
                                                          X_train_resampled, 
                                                          y_train_resampled, 
                                                          X_test, # Evaluate on the original, scaled X_test
                                                          y_test, 
                                                          model_name)

# --- 9. Further Steps (Implementation as per my analysis plan) ---

# --- 9.1 In-depth Comparison of Model Metrics ---
# To systematically compare models, I decided to store their key performance metrics in a DataFrame.
performance_summary_list = []
for model_name, model_instance in trained_models.items():
    y_pred_values = model_instance.predict(X_test) # X_test is already scaled from earlier steps.
    # I obtained probability predictions. For KNN, if predict_proba isn't well-calibrated or standard, I used a placeholder.
    # A more robust approach for KNN might involve CalibratedClassifierCV.
    y_pred_proba_values = model_instance.predict_proba(X_test)[:, 1] if hasattr(model_instance, "predict_proba") else [0.5] * len(y_test) 

    # I calculated key metrics for the anomalous class.
    precision_anomalous = precision_score(y_test, y_pred_values, pos_label=1, zero_division=0)
    recall_anomalous = recall_score(y_test, y_pred_values, pos_label=1, zero_division=0)
    f1_anomalous = f1_score(y_test, y_pred_values, pos_label=1, zero_division=0)
    # I calculated ROC AUC, handling cases where predict_proba might not be standard (like uncalibrated KNN).
    roc_auc = roc_auc_score(y_test, y_pred_proba_values) if hasattr(model_instance, "predict_proba") and not (model_name == "K-Nearest Neighbors" and y_pred_proba_values[0] == 0.5) else np.nan
    
    performance_summary_list.append({
        'Model': model_name,
        'Precision (Anomalous)': precision_anomalous,
        'Recall (Anomalous)': recall_anomalous,
        'F1-Score (Anomalous)': f1_anomalous,
        'ROC AUC': roc_auc
    })

# I converted the list of dictionaries into a pandas DataFrame for easy viewing and comparison.
performance_df = pd.DataFrame(performance_summary_list)
print("\n--- Performance Summary of Initial Models ---")
print(performance_df.to_string()) # .to_string() ensures the full DataFrame is printed.


# --- 9.2 Feature Importance Analysis (Random Forest) ---
# I focused on Random Forest for feature importance as it provides this information readily.
if "Random Forest" in trained_models:
    rf_model = trained_models["Random Forest"]
    importances = rf_model.feature_importances_
    # I created a DataFrame to display feature names and their importance scores.
    feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
    # I sorted the features by importance in descending order.
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    
    print("\n--- Top 15 Feature Importances (Random Forest) ---")
    print(feature_importance_df.head(15).to_string()) # Displaying the top 15 features.
    
    # I generated a bar plot to visualize the top 15 feature importances.
    plt.figure(figsize=(10, 8)) # Setting the figure size for better readability.
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15), palette='viridis')
    plt.title('Top 15 Feature Importances from Random Forest')
    plt.tight_layout() # Adjusts plot to ensure everything fits without overlapping.
    plt.show() # Displays the plot.


# --- 9.3 Visualization of Results (ROC Curves) ---
# I generated ROC curves for all trained models to visualize their performance in distinguishing classes.
plt.figure(figsize=(10, 8)) # Setting figure size.
for model_name, model_instance in trained_models.items():
    if hasattr(model_instance, "predict_proba"):
        y_pred_proba_values = model_instance.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_values) # Calculating False Positive Rate and True Positive Rate.
        auc_score = roc_auc_score(y_test, y_pred_proba_values) # Calculating Area Under the Curve.
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})') # Plotting the ROC curve.
    elif model_name == "K-Nearest Neighbors":
        # For KNN, predict_proba might not be well-calibrated by default.
        # I decided to skip plotting its ROC curve if standard predict_proba was not used or deemed unreliable.
        print(f"Skipping ROC curve for {model_name} as predict_proba might not be well-calibrated by default or was handled as placeholder.")

plt.plot([0, 1], [0, 1], 'k--') # Plotting the dashed diagonal line (random chance).
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Trained Models')
plt.legend(loc='lower right') # Adding a legend.
plt.grid(True) # Adding a grid for better readability.
plt.show() # Displaying the plot.


# --- 9.4 Hyperparameter Tuning (Example: Random Forest & Logistic Regression) ---
# I planned to perform hyperparameter tuning, but noted this can be very time-consuming.
# For this script, I used the full resampled training data for tuning demonstration.
# In a real scenario with extremely large data, I might use a subset or a dedicated validation set.

# I assigned the resampled training data to X_tune and y_tune for clarity in the tuning phase.
X_tune, y_tune = X_train_resampled, y_train_resampled
print(f"\nShape of data used for hyperparameter tuning demo: X_tune={X_tune.shape}, y_tune={y_tune.shape}")

# --- Hyperparameter Tuning for Random Forest ---
print("\n--- Hyperparameter Tuning for Random Forest (using GridSearchCV) ---")
# I defined a parameter grid for Random Forest. This grid specifies the hyperparameters I wanted to test.
# WARNING: A larger grid or more CV folds can make this VERY time-consuming.
rf_param_grid = {
    'n_estimators': [100, 150],       # Number of trees in the forest.
    'max_depth': [10, 20, None],    # Maximum depth of the trees.
    'min_samples_split': [2, 5],    # Minimum number of samples required to split an internal node.
    'min_samples_leaf': [1, 2]      # Minimum number of samples required to be at a leaf node.
}

# I set up GridSearchCV.
# cv=2 (2-fold cross-validation) was used for speed in this example; for robust results, cv=3 or cv=5 is more common.
# scoring='f1_weighted' was chosen as the metric to optimize, considering potential class imbalance (though SMOTE was applied).
# n_jobs=-1 uses all available CPU cores for the grid search.
# verbose=1 provides updates during the fitting process.
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), # Using n_jobs in the estimator too for tree building.
                              rf_param_grid,
                              cv=2, 
                              scoring='f1_weighted', 
                              verbose=1,
                              n_jobs=-1) 
try:
    # I fitted GridSearchCV on the tuning data.
    rf_grid_search.fit(X_tune, y_tune)
    print("Best Random Forest Parameters found:")
    print(rf_grid_search.best_params_) # Printing the best hyperparameter combination found.
    best_rf_model = rf_grid_search.best_estimator_ # Getting the best model instance.
    trained_models["Random Forest Tuned"] = best_rf_model # Adding the tuned model to my dictionary.
    # I re-evaluated the tuned model using my helper function.
    train_and_evaluate_model(best_rf_model, X_train_resampled, y_train_resampled, X_test, y_test, "Random Forest Tuned")
except Exception as e:
    # If GridSearchCV failed (e.g., due to memory errors as seen in the output), I printed an error and skipped this tuning.
    print(f"Error during Random Forest GridSearchCV: {e}. Skipping RF tuning.")


# --- Hyperparameter Tuning for Logistic Regression ---
print("\n--- Hyperparameter Tuning for Logistic Regression (using GridSearchCV) ---")
# I defined a parameter grid for Logistic Regression.
lr_param_grid = {
    'C': [0.01, 0.1, 1, 10], # Inverse of regularization strength.
    'penalty': ['l1', 'l2'],   # Regularization penalty type.
    'solver': ['liblinear']    # Solver that supports both l1 and l2 penalties.
}

# I set up GridSearchCV for Logistic Regression, similar to Random Forest.
lr_grid_search = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000), # Increased max_iter for convergence.
                              lr_param_grid,
                              cv=2, 
                              scoring='f1_weighted',
                              verbose=1,
                              n_jobs=-1)
try:
    # I fitted GridSearchCV on the tuning data.
    lr_grid_search.fit(X_tune, y_tune) 
    print("Best Logistic Regression Parameters found:")
    print(lr_grid_search.best_params_)
    best_lr_model = lr_grid_search.best_estimator_
    trained_models["Logistic Regression Tuned"] = best_lr_model
    # I re-evaluated the tuned model.
    train_and_evaluate_model(best_lr_model, X_train_resampled, y_train_resampled, X_test, y_test, "Logistic Regression Tuned")
except Exception as e:
    # If GridSearchCV failed, I printed an error and skipped this tuning.
    print(f"Error during Logistic Regression GridSearchCV: {e}. Skipping LR tuning.")

# --- Re-generate Performance Summary with Tuned Models (if any) ---
# I prepared to regenerate the performance summary to include any successfully tuned models.
performance_summary_tuned_list = []
# I iterated through all models in the trained_models dictionary (which now might include tuned versions).
for model_name, model_instance in trained_models.items(): 
    y_pred_values = model_instance.predict(X_test)
    y_pred_proba_values = model_instance.predict_proba(X_test)[:, 1] if hasattr(model_instance, "predict_proba") else [0.5] * len(y_test)

    precision_anomalous = precision_score(y_test, y_pred_values, pos_label=1, zero_division=0)
    recall_anomalous = recall_score(y_test, y_pred_values, pos_label=1, zero_division=0)
    f1_anomalous = f1_score(y_test, y_pred_values, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba_values) if hasattr(model_instance, "predict_proba") and not (model_name == "K-Nearest Neighbors" and y_pred_proba_values[0] == 0.5 and "Tuned" not in model_name) else np.nan # Adjusted KNN condition
    
    performance_summary_tuned_list.append({
        'Model': model_name,
        'Precision (Anomalous)': precision_anomalous,
        'Recall (Anomalous)': recall_anomalous,
        'F1-Score (Anomalous)': f1_anomalous,
        'ROC AUC': roc_auc
    })

# I created a DataFrame from the new summary list.
performance_df_tuned = pd.DataFrame(performance_summary_tuned_list)
print("\n--- Performance Summary Including Tuned Models ---")
# I printed the tuned performance summary, sorted by F1-Score for the anomalous class in descending order.
print(performance_df_tuned.sort_values(by='F1-Score (Anomalous)', ascending=False).to_string())


# --- 9.5 Final Model Selection (Qualitative Discussion based on results) ---
# This section is for my qualitative discussion on selecting the final model.
print("\n--- Final Model Selection Considerations ---")
print("Based on the performance summary (especially F1-score and Recall for the anomalous class, and ROC AUC):")
# Here, I would typically add my detailed reasoning for model selection.
# For example:
# best_model_name_candidate = performance_df_tuned.sort_values(by='F1-Score (Anomalous)', ascending=False).iloc[0]['Model']
# print(f"The model performing best on F1-Score (Anomalous) is tentatively: {best_model_name_candidate}")
print("I need to consider several factors for final selection:")
print("1. F1-Score (Anomalous): This balances precision and recall, crucial for the attack class.")
print("2. Recall (Anomalous): How many actual attacks did my model find? Minimizing False Negatives is key.")
print("3. Precision (Anomalous): Of those flagged as attacks, how many were truly attacks? Minimizing False Positives is also important.")
print("4. ROC AUC: This shows overall model distinguishability.")
print("5. Training Time / Inference Time: These were not explicitly measured in this script but are vital for real-world deployment.")
print("6. Interpretability: Models like Decision Trees and Logistic Regression (with feature importance) are easier to interpret than more complex Random Forests or KNN.")
print("My initial proposal mentioned selecting based on 'accuracy, interpretability, and computational efficiency'.")

# I added an example of how I might programmatically identify the best model based on a primary metric.
if not performance_df_tuned.empty:
    # I decided to sort by F1-Score (Anomalous) first, then ROC AUC, then Recall (Anomalous) as tie-breakers.
    best_performing_models_df = performance_df_tuned.sort_values(
        by=['F1-Score (Anomalous)', 'ROC AUC', 'Recall (Anomalous)'],
        ascending=[False, False, False] # All descending for "best"
    )
    print("\nModels ranked by F1-Score (Anomalous), then ROC AUC, then Recall (Anomalous):")
    print(best_performing_models_df.to_string())
    if not best_performing_models_df.empty:
        best_model_name_overall = best_performing_models_df.iloc[0]['Model']
        print(f"\nTentatively, '{best_model_name_overall}' shows strong overall performance based on these prioritized metrics.")
    else:
        print("Ranking could not be performed as the performance DataFrame is empty after sorting.")
else:
    print("Performance DataFrame (tuned) is empty, cannot determine the best model.")

print("\n--- My Intrusion Detection Script Finished ---")
