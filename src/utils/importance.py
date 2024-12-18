import os

import numpy as np
import pickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import shap

matplotlib.use('Agg') # for headless environment

# Load the model
def load_predictions_and_model():
    model_path = os.path.join('..', 'tab_data', 'saved_predictions_and_model.pkl')
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['trained_model'], data['X_test'], data['y_test'], data['X_train']

def preprocess_data_for_shap(X_train, X_test):
    """ Ensure X_train and X_test are numeric, handle booleans, and handle missing values """
    # Convert boolean columns to integers (True -> 1, False -> 0)
    # pdb.set_trace()
    # if column's dtype is object, cast it to category
    X_train = X_train.apply(lambda x: x.astype(float) if x.dtype == 'object' else x)
    X_test = X_test.apply(lambda x: x.astype(float) if x.dtype == 'object' else x)
    X_train = X_train.map(lambda x: int(x) if isinstance(x, bool) else x)
    X_test = X_test.map(lambda x: int(x) if isinstance(x, bool) else x)

    # Convert non-numeric columns to numeric (coerce invalid values to NaN)
    # X_train = X_train.apply(pd.to_numeric, errors='coerce')
    # X_test = X_test.apply(pd.to_numeric, errors='coerce')

    # Handle missing values
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())
    
    return X_train, X_test

def calculate_shap_importance(trained_model, X_train, X_test, random_state):
    """ Calculate SHAP-based feature importance """
    # Preprocess the data 
    # X_train, X_test = preprocess_data_for_shap(X_train, X_test)

    # Initialize SHAP explainer and compute SHAP valuesn
    # when feature_pertubation is set to 'tree_path_dependent', no background data is needed
    # setting feature_pertubation to 'interventional' will fail additivity check but not sure why
    explainer = shap.TreeExplainer(trained_model, feature_perturbation='tree_path_dependent')
    # NOTE: check_additivity=False is used to avoid the warning about the expected sum of SHAP values not matching the output
    # TODO: is there a way to run with check_additivity=True?
    shap_values = explainer(X_test, check_additivity=True)

    # Calculate mean SHAP values per feature (absolute value to reflect importance)
    mean_shap_values = np.abs(shap_values.values).mean(axis=0)
    return mean_shap_values, shap_values

def calculate_permutation_importance(trained_model, X_test, y_test, random_state):
    """ Calculate Permutation Importance """
    # Ensure y_test does not contain NaN values
    # y_test = y_test.fillna(y_test.median())

    # Ensure X_test is fully clean and contains no NaN values
    # X_test = X_test.fillna(X_test.mean())

    # Calculate permutation importance
    perm_importance = permutation_importance(trained_model, X_test, y_test, n_jobs=4, n_repeats=5, random_state=random_state)
    return perm_importance.importances_mean

def save_importance_results(feature_names, shap_values, perm_importance, filename):
    """ Combine SHAP and permutation importance results and save as CSV """
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean SHAP Value': shap_values,
        'Permutation Importance': perm_importance
    })
    
    # Save the importance results to CSV with target name in the filename
    importance_df.to_csv(filename, index=False)
    return importance_df

def plot_top_features(df, importance_column, top_n=20, title='', filename='top_features.jpg'):
    """ Plot the top N features based on the specified importance column """
    # Sort by the selected importance column and take the top N features
    top_features_df = df.nlargest(top_n, importance_column)
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.barh(top_features_df['Feature'], top_features_df[importance_column], color='blue', alpha=0.7)
    plt.xlabel(importance_column)
    plt.title(title)
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important features at the top
    plt.tight_layout()
    # Save the plot as a JPG image
    plt.savefig(filename)
    plt.close()

def save_shap_summary_plot(shap_values, X_test, filename=f'shap_summary_plot.jpg'):
    """ Save SHAP summary plot to a file with the target name in the filename """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def run_importance(trained_model, X_train, X_test, y_test, target_name, save_dir, random_state):
    # Calculate SHAP values for feature importance
    shap_values, shap_explanation = calculate_shap_importance(trained_model, X_train, X_test, random_state)
    # save shap_values as pkl file:
    shap_values_path = os.path.join(save_dir, f'shap_values_target={target_name}.pkl')
    with open(shap_values_path, 'wb') as f:
        pickle.dump(shap_values, f)
    # Calculate permutation importance
    perm_importance = calculate_permutation_importance(trained_model, X_test, y_test, random_state)
    # Save and display feature importance results
    feature_names = X_test.columns
    filename_importance_results = os.path.join(save_dir, f'feature_importance_results_{target_name}.csv')
    importance_df = save_importance_results(feature_names, shap_values, perm_importance, filename_importance_results)
    # Plot the top 20 features based on SHAP values
    file_path_top20 = os.path.join(save_dir, f'top_shap_features_{target_name}.jpg')
    plot_top_features(
        importance_df, 
        'Mean SHAP Value', 
        top_n=20, 
        title=f'Top 20 SHAP Values ({target_name})', 
        filename=file_path_top20
    )
    # Plot the top 20 features based on Permutation Importance
    file_path_top_features = os.path.join(save_dir, f'top_permutation_features_{target_name}.jpg')
    plot_top_features(
        importance_df, 
        'Permutation Importance', 
        top_n=20, 
        title=f'Top 20 Permutation Importance ({target_name})', 
        filename=file_path_top_features
    )

    # Save SHAP summary plot
    file_path_summary_plot = os.path.join(save_dir, f'shap_summary_plot_{target_name}.jpg')
    save_shap_summary_plot(shap_explanation, X_test, filename=file_path_summary_plot)
