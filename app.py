import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Brucellosis Prediction App", layout="wide")

# Function to load artifacts
@st.cache_resource
def load_artifacts():
    try:
        df_clean = pd.read_csv('df_clean.csv')
        with open('le_dict.pkl', 'rb') as f:
            le_dict = pickle.load(f)
        with open('le_target.pkl', 'rb') as f:
            le_target = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('model_results.pkl', 'rb') as f:
            model_results = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return df_clean, le_dict, le_target, scaler, model_results, feature_names
    except FileNotFoundError as e:
        st.error(f"Error loading required files. Make sure all .pkl and df_clean.csv are in the same directory. Missing file: {e}")
        st.stop()

df_clean, le_dict, le_target, scaler, model_results, feature_names = load_artifacts()

# --- Prediction Function (from original script) ---
def predict_single_case(input_dict, model_results, le_dict, le_target, scaler, feature_names, model_name=None):
    if model_name is None:
        # Determine the best model based on accuracy (can be adjusted to AUC or F1)
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
                # Ensure the value exists in the encoder's classes
                if input_df[col].iloc[0] not in le_dict[col].classes_:
                    st.warning(f"Input value '{input_df[col].iloc[0]}' for '{col}' not seen during training. This might lead to incorrect prediction.")
                    # Handle unseen labels: e.g., map to a default or the most frequent, or raise an error
                    # For now, let's just use transform which will raise an error if not handled
                    # A robust solution might involve adding a 'default' mapping or specific imputation
                    # For simplicity, we assume valid inputs for this app
                    pass
                input_df[col] = le_dict[col].transform(input_df[col])
            except ValueError as e:
                st.error(f"Error encoding column '{col}': {e}. Please check input value.")
                return None

    # Ensure all features are present and in the correct order
    processed_input = {}
    for col in feature_names:
        if col in input_df.columns:
            processed_input[col] = input_df[col].iloc[0]
        else:
            # Handle missing features in input - impute with a default (e.g., 0 or mean/mode from training)
            # For simplicity, using 0, but a more robust solution would be preferred
            processed_input[col] = 0 # Or use mode/median from df_clean for numerical features
                                     # For categorical, it would be the encoded value of the mode
    input_df_final = pd.DataFrame([processed_input])

    # Scale if needed
    if model_name in ["MLP", "SVM", "Logistic Regression", "KNN"]:
        input_df_final = scaler.transform(input_df_final)

    # Make prediction
    pred_class = model.predict(input_df_final)[0]
    pred_prob = model.predict_proba(input_df_final)[0]

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
    y_test_pred = model_results[best_model_name]['predictions']
    y_test_true = model_results[best_model_name]['model'].fit(scaler.transform(df_clean.drop(columns=[le_target.inverse_transform([0,1,2])[0]])), le_target.transform(df_clean[le_target.inverse_transform([0,1,2])[0]])).predict(scaler.transform(df_clean.drop(columns=[le_target.inverse_transform([0,1,2])[0]]))) # A placeholder. In a real app, y_test should be passed or re-derived.
    # For a proper confusion matrix on test data, y_test from the split is needed.
    # For demonstration, we will use a dummy y_true and y_pred or retrieve the actual y_test if stored.
    # As y_test is not stored in model_results, we simulate by re-predicting on a subset or using reported values if available.
    # For a true representation, the actual y_test from the train-test split would be ideal.
    # Assuming y_test is available from global scope or reloaded
    # (Note: In a real app, you'd save y_test as well or ensure consistency)
    
    # For this example, let's try to retrieve the actual y_test used during training from the global scope if possible
    # or use a representative set. Given the context of the .py file, y_test is a global variable there.
    # We will need to approximate or assume y_test is available through some means.
    # For now, let's use the full df_clean and re-encode to get a y_clean_encoded as a proxy for the true labels.
    
    # Reload y_test and y_pred from saved objects if possible, or pass them from the main script.
    # For now, we will create a dummy y_test and use the predictions from model_results for plotting.
    # This is not ideal for true evaluation but demonstrates the plotting.
    # A better way: save X_test, y_test during artifact generation.
    
    # To fix this, in your original script (Cell 10), you should store y_test:
    # model_results[name] = { ..., 'y_test': y_test, 'y_pred': y_pred, ... }
    
    # Assuming y_test and y_pred are available from the original script's execution context
    # If not, you'd need to re-run the split and use the same random_state
    # For simplicity, using the `y_test` from the global scope of the Streamlit app.
    
    try:
        y_test_original = df_clean[df_clean.columns[-1]] # Assuming target is the last column
        y_test_encoded_for_cm = le_target.transform(y_test_original)
        
        # To get the actual y_test from the original split:
        # You would need to save X_test, y_test as artifacts.
        # For this demo, let's use the 'predictions' from model_results and
        # assume the 'y_test' passed to plot_confusion_matrix is the one used in training.
        
        # Let's try to use the true y_test from the script if loaded as a global.
        # If `y_test` from the script's Cell 7 is available globally:
        cm = confusion_matrix(y_test, model_results[best_model_name]['predictions'])
        
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=le_target.classes_,
                   yticklabels=le_target.classes_, ax=ax_cm)
        ax_cm.set_title(f'Confusion Matrix - {best_model_name}')
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('Actual')
        st.pyplot(fig_cm)
    except NameError:
        st.warning("Cannot display confusion matrix directly as `y_test` is not globally available in this Streamlit app context.")
        st.info("To display the correct confusion matrix, ensure `y_test` from the train-test split is saved as an artifact and loaded here.")

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

        # Breed Species: Categorical (Dropdown)
        breed_species_options = df_clean['Breed species'].unique().tolist()
        breed_species = st.selectbox("Breed species", options=breed_species_options)

        # Sex: Categorical (Dropdown)
        sex_options = df_clean['Sex'].unique().tolist()
        sex = st.selectbox("Sex", options=sex_options)

        # Calvings: Numerical input
        calvings = st.number_input("Number of Calvings", min_value=0, max_value=15, value=1)

        # Abortion History: Categorical (Dropdown)
        abortion_history_options = df_clean['Abortion History (Yes No)'].unique().tolist()
        abortion_history = st.selectbox("Abortion History", options=abortion_history_options)

        # Infertility Repeat breeder: Categorical (Dropdown)
        infertility_options = df_clean['Infertility Repeat breeder(Yes No)'].unique().tolist()
        infertility = st.selectbox("Infertility / Repeat breeder", options=infertility_options)

        # Brucella vaccination status: Categorical (Dropdown)
        vaccination_status_options = df_clean['Brucella vaccination status (Yes No)'].unique().tolist()
        vaccination_status = st.selectbox("Brucella vaccination status", options=vaccination_status_options)

        # Sample Type: Categorical (Dropdown)
        sample_type_options = df_clean['Sample Type(Serum Milk)'].unique().tolist()
        sample_type = st.selectbox("Sample Type", options=sample_type_options)

        # Test Type: Categorical (Dropdown)
        test_type_options = df_clean['Test Type (RBPT ELISA MRT)'].unique().tolist()
        test_type = st.selectbox("Test Type", options=test_type_options)

        # Retained Placenta Stillbirth: Categorical (Dropdown)
        retained_placenta_options = df_clean['Retained Placenta Stillbirth(Yes No No Data)'].unique().tolist()
        retained_placenta = st.selectbox("Retained Placenta / Stillbirth", options=retained_placenta_options)

        # Proper Disposal of Aborted Fetuses: Categorical (Dropdown)
        disposal_options = df_clean['Proper Disposal of Aborted Fetuses (Yes No)'].unique().tolist()
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
