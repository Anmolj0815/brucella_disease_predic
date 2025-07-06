import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
import os # Import os for path joining

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Brucellosis Prediction App", layout="wide")

# Define the path to your artifacts directory
ARTIFACTS_DIR = 'model_artifacts'

# Function to load artifacts
@st.cache_resource
def load_artifacts():
    try:
        # Construct full paths to files inside the artifacts directory
        df_clean_path = os.path.join(ARTIFACTS_DIR, 'df_clean.csv')
        le_dict_path = os.path.join(ARTIFACTS_DIR, 'le_dict.pkl')
        le_target_path = os.path.join(ARTIFACTS_DIR, 'le_target.pkl')
        scaler_path = os.path.join(ARTIFACTS_DIR, 'scaler.pkl')
        model_results_path = os.path.join(ARTIFACTS_DIR, 'model_results.pkl')
        feature_names_path = os.path.join(ARTIFACTS_DIR, 'feature_names.pkl')

        df_clean = pd.read_csv(df_clean_path)
        with open(le_dict_path, 'rb') as f:
            le_dict = pickle.load(f)
        with open(le_target_path, 'rb') as f:
            le_target = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(model_results_path, 'rb') as f:
            model_results = pickle.load(f)
        with open(feature_names_path, 'rb') as f:
            feature_names = pickle.load(f)

        # Retrieve y_test for CM plotting from one of the model results (assuming it's consistent)
        # Or ideally, save y_test as a separate artifact during training
        # For robustness, we'll pick y_test from the best model's results for plotting
        best_model_name_for_y_test = max(model_results.keys(), key=lambda x: model_results[x]['accuracy'])
        y_test_for_streamlit_cm = model_results[best_model_name_for_y_test]['y_test']

        return df_clean, le_dict, le_target, scaler, model_results, feature_names, y_test_for_streamlit_cm
    except FileNotFoundError as e:
        st.error(f"Error loading required files. Make sure the '{ARTIFACTS_DIR}' directory and all necessary files are present. Missing file: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading artifacts: {e}")
        st.stop()


# Load artifacts at the start of the app
df_clean, le_dict, le_target, scaler, model_results, feature_names, y_test_from_artifacts = load_artifacts()

# --- Prediction Function (from original script, adapted for Streamlit) ---
def predict_single_case(input_dict, model_results, le_dict, le_target, scaler, feature_names, model_name=None):
    if model_name is None:
        # Determine the best model based on accuracy
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['accuracy'])
        model_name = best_model_name

    if model_name not in model_results:
        st.error(f"Model {model_name} not found!")
        return None

    model = model_results[model_name]['model']

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Clean column names (strip spaces)
    input_df.columns = input_df.columns.str.strip()

    # Encode categorical features
    for col in input_df.columns:
        if col in le_dict and input_df[col].dtype == 'object':
            try:
                # Handle unseen labels: If a category is not in the encoder's classes, it will cause an error.
                # For a robust app, you might want to map it to a 'default' or raise a more user-friendly error.
                # For now, we'll check if the value is known.
                if input_df[col].iloc[0] not in le_dict[col].classes_:
                    st.warning(f"Input value '{input_df[col].iloc[0]}' for '{col}' was not seen during model training. This might affect prediction accuracy.")
                    # Option: Replace with mode or a default known value
                    # input_df[col] = le_dict[col].transform([le_dict[col].classes_[0]]) # Example: map to first class
                    # For this example, we proceed with transform, which will raise ValueError if not in classes
                input_df[col] = le_dict[col].transform(input_df[col])
            except ValueError as e:
                st.error(f"Error encoding column '{col}': {e}. Please ensure input values are valid categories.")
                return None

    # Ensure all features are present and in the correct order as per training data
    # Create a DataFrame with all expected features, initialized to 0 or a sensible default
    processed_input_df = pd.DataFrame(columns=feature_names)
    processed_input_df.loc[0] = 0 # Initialize with zeros or mean/mode

    # Fill in the values from the user's input
    for col in input_df.columns:
        if col in feature_names: # Only copy if the column is an expected feature
            processed_input_df[col] = input_df[col].iloc[0]

    # Scale if needed
    if model_name in ["MLP", "SVM", "Logistic Regression", "KNN"]:
        processed_input_df_scaled = scaler.transform(processed_input_df)
        pred_class = model.predict(processed_input_df_scaled)[0]
        pred_prob = model.predict_proba(processed_input_df_scaled)[0]
    else:
        pred_class = model.predict(processed_input_df)[0]
        pred_prob = model.predict_proba(processed_input_df)[0]

    # Convert back to original labels
    predicted_result = le_target.inverse_transform([pred_class])[0]
    confidence = pred_prob[pred_class]

    return {
        'predicted_class': predicted_result,
        'confidence': confidence,
        'probabilities': dict(zip(le_target.classes_, pred_prob)),
        'model_used': model_name
    }

# --- Streamlit UI ---
st.title("🐄 Brucellosis Prediction App for Mathura")
st.markdown("""
This application predicts the likelihood of Brucellosis in animals based on various parameters.
""")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Model Performance", "Make a Prediction"])

# --- Model Performance Section ---
if page == "Model Performance":
    st.header("📊 Model Performance Overview")

    best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['accuracy'])
    best_accuracy = model_results[best_model_name]['accuracy']

    st.subheader("Summary of Model Accuracies and AUC Scores")
    performance_data = []
    for name, results in model_results.items():
        performance_data.append({"Model": name, "Accuracy": results['accuracy'], "AUC": results['auc']})
    performance_df = pd.DataFrame(performance_data).sort_values(by="Accuracy", ascending=False)
    st.dataframe(performance_df.style.highlight_max(axis=0, subset=['Accuracy', 'AUC']), use_container_width=True)

    st.markdown(f"**🏆 Best Performing Model: {best_model_name}** with an Accuracy of **{best_accuracy:.4f}**.")

    st.subheader("Model Comparison Plots")
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Accuracy Comparison")
        model_names = list(model_results.keys())
        accuracies = [model_results[name]['accuracy'] for name in model_names]
        fig_acc, ax_acc = plt.subplots(figsize=(8, 6))
        ax_acc.barh(model_names, accuracies, color='skyblue')
        ax_acc.set_xlabel('Accuracy')
        ax_acc.set_title('Model Accuracy Comparison')
        ax_acc.set_xlim(0, 1) # Set x-axis limit from 0 to 1
        for i, v in enumerate(accuracies):
            ax_acc.text(v + 0.01, i, f'{v:.3f}', va='center')
        st.pyplot(fig_acc)

    with col2:
        st.write("### AUC Score Comparison")
        aucs = [model_results[name]['auc'] for name in model_names]
        fig_auc, ax_auc = plt.subplots(figsize=(8, 6))
        ax_auc.barh(model_names, aucs, color='lightcoral')
        ax_auc.set_xlabel('AUC Score')
        ax_auc.set_title('Model AUC Comparison')
        ax_auc.set_xlim(0, 1) # Set x-axis limit from 0 to 1
        for i, v in enumerate(aucs):
            ax_auc.text(v + 0.01, i, f'{v:.3f}', va='center')
        st.pyplot(fig_auc)

    st.subheader(f"Confusion Matrix for {best_model_name}")
    # Use y_test_from_artifacts which was passed from load_artifacts
    y_pred_best = model_results[best_model_name]['predictions']
    cm = confusion_matrix(y_test_from_artifacts, y_pred_best)

    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=le_target.classes_,
               yticklabels=le_target.classes_, ax=ax_cm)
    ax_cm.set_title(f'Confusion Matrix - {best_model_name}')
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    st.pyplot(fig_cm)


    st.subheader(f"Feature Importance for {best_model_name} (if applicable)")
    model = model_results[best_model_name]['model']
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.Series(model.feature_importances_, index=feature_names)
        top_features = feature_importance.sort_values(ascending=False).head(10)

        fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
        top_features.plot(kind='barh', color='lightgreen', ax=ax_fi)
        ax_fi.set_title(f'Top 10 Feature Importances - {best_model_name}')
        ax_fi.set_xlabel('Importance Score')
        ax_fi.grid(axis='x', alpha=0.3)
        st.pyplot(fig_fi)
    else:
        st.info(f"{best_model_name} does not directly support feature importance visualization.")


# --- Make a Prediction Section ---
elif page == "Make a Prediction":
    st.header("🔮 Make a Brucellosis Prediction")
    st.markdown("Enter the animal's details below to get a prediction.")

    # Collect user inputs
    with st.form("prediction_form"):
        st.subheader("Animal Information")

        # Get unique values for dropdowns from df_clean
        # Ensure column names match the original data after preprocessing (stripped spaces)

        # Age: Numerical input
        age = st.number_input("Age (Years)", min_value=0, max_value=20, value=3)

        # Breed species: Categorical (Dropdown)
        # Using sorted list for consistent display
        breed_species_options = sorted(df_clean['Breed species'].unique().tolist())
        breed_species = st.selectbox("Breed species", options=breed_species_options)

        # Sex: Categorical (Dropdown)
        sex_options = sorted(df_clean['Sex'].unique().tolist())
        sex = st.selectbox("Sex", options=sex_options)

        # Calvings: Numerical input
        calvings = st.number_input("Number of Calvings", min_value=0, max_value=15, value=1)

        # Abortion History: Categorical (Dropdown)
        abortion_history_options = sorted(df_clean['Abortion History (Yes No)'].unique().tolist())
        abortion_history = st.selectbox("Abortion History", options=abortion_history_options)

        # Infertility Repeat breeder: Categorical (Dropdown)
        infertility_options = sorted(df_clean['Infertility Repeat breeder(Yes No)'].unique().tolist())
        infertility = st.selectbox("Infertility / Repeat breeder", options=infertility_options)

        # Brucella vaccination status: Categorical (Dropdown)
        vaccination_status_options = sorted(df_clean['Brucella vaccination status (Yes No)'].unique().tolist())
        vaccination_status = st.selectbox("Brucella vaccination status", options=vaccination_status_options)

        # Sample Type: Categorical (Dropdown)
        sample_type_options = sorted(df_clean['Sample Type(Serum Milk)'].unique().tolist())
        sample_type = st.selectbox("Sample Type", options=sample_type_options)

        # Test Type: Categorical (Dropdown)
        test_type_options = sorted(df_clean['Test Type (RBPT ELISA MRT)'].unique().tolist())
        test_type = st.selectbox("Test Type", options=test_type_options)

        # Retained Placenta Stillbirth: Categorical (Dropdown)
        retained_placenta_options = sorted(df_clean['Retained Placenta Stillbirth(Yes No No Data)'].unique().tolist())
        retained_placenta = st.selectbox("Retained Placenta / Stillbirth", options=retained_placenta_options)

        # Proper Disposal of Aborted Fetuses: Categorical (Dropdown)
        disposal_options = sorted(df_clean['Proper Disposal of Aborted Fetuses (Yes No)'].unique().tolist())
        disposal = st.selectbox("Proper Disposal of Aborted Fetuses", options=disposal_options)

        submitted = st.form_submit_button("Get Prediction")

        if submitted:
            input_data = {
                'Age ': age, # Ensure column names exactly match those used in training, including spaces
                'Breed species': breed_species,
                ' Sex ': sex,
                'Calvings': calvings,
                'Abortion History (Yes No)': abortion_history,
                'Infertility Repeat breeder(Yes No)': infertility,
                'Brucella vaccination status (Yes No)': vaccination_status,
                'Sample Type(Serum Milk)': sample_type,
                'Test Type (RBPT ELISA MRT)': test_type,
                'Retained Placenta Stillbirth(Yes No No Data)': retained_placenta,
                'Proper Disposal of Aborted Fetuses (Yes No)': disposal
            }

            st.subheader("Prediction Results")
            prediction = predict_single_case(input_data, model_results, le_dict, le_target, scaler, feature_names)

            if prediction:
                st.markdown(f"The animal is predicted to be **{prediction['predicted_class']}** with **{prediction['confidence']:.2%}** confidence using the **{prediction['model_used']}** model.")

                st.write("---")
                st.subheader("Detailed Probabilities:")
                prob_df = pd.DataFrame({
                    'Class': list(prediction['probabilities'].keys()),
                    'Probability': [f"{v:.2%}" for v in prediction['probabilities'].values()]
                })
                st.dataframe(prob_df, hide_index=True)

                st.info("Disclaimer: This prediction is based on the trained model and provided inputs. Consult a veterinary professional for diagnosis and treatment.")
            else:
                st.error("Could not generate a prediction. Please check your inputs and try again.")
