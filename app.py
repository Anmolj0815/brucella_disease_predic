import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder # Make sure LabelEncoder is imported for le_dict.get() default

warnings.filterwarnings('ignore')

# Define the directory where model artifacts are stored
MODEL_ARTIFACTS_DIR = 'model_artifacts/'

# Load the trained objects
try:
    with open(MODEL_ARTIFACTS_DIR + 'best_model.pkl', 'rb') as f:
        best_model = pickle.load(f)
    with open(MODEL_ARTIFACTS_DIR + 'le_dict.pkl', 'rb') as f:
        le_dict = pickle.load(f)
    with open(MODEL_ARTIFACTS_DIR + 'le_target.pkl', 'rb') as f:
        le_target = pickle.load(f)
    with open(MODEL_ARTIFACTS_DIR + 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(MODEL_ARTIFACTS_DIR + 'feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    st.sidebar.success("✅ All model components loaded successfully!")

except FileNotFoundError:
    st.sidebar.error(f"❌ Required model files not found in '{MODEL_ARTIFACTS_DIR}'. Please ensure all .pkl files are in this directory.")
    st.stop() # Stop the app if essential files are missing
except Exception as e:
    st.sidebar.error(f"❌ Error loading model components: {e}")
    st.stop()

# Get unique categories for dropdowns from le_dict
# We use LabelEncoder() as a default for .get() in case a key is missing, though they should all be present.
unique_breeds = sorted(list(le_dict.get('Breed species', LabelEncoder()).classes_))
unique_sex = sorted(list(le_dict.get(' Sex ', LabelEncoder()).classes_))
unique_abortion_history = sorted(list(le_dict.get('Abortion History (Yes No)', LabelEncoder()).classes_))
unique_infertility = sorted(list(le_dict.get('Infertility Repeat breeder(Yes No)', LabelEncoder()).classes_))
unique_vaccination_status = sorted(list(le_dict.get('Brucella vaccination status (Yes No)', LabelEncoder()).classes_))
unique_sample_type = sorted(list(le_dict.get('Sample Type(Serum Milk)', LabelEncoder()).classes_))
unique_test_type = sorted(list(le_dict.get('Test Type (RBPT ELISA MRT)', LabelEncoder()).classes_))
unique_retained_placenta = sorted(list(le_dict.get('Retained Placenta Stillbirth(Yes No No Data)', LabelEncoder()).classes_))
unique_disposal = sorted(list(le_dict.get('Proper Disposal of Aborted Fetuses (Yes No)', LabelEncoder()).classes_))


def predict_single_case(input_dict, model, le_dict, le_target, scaler, feature_names):
    """Predict a single case with robust error handling for encoding."""
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Clean column names (strip spaces to match trained features)
    input_df.columns = input_df.columns.str.strip()
    
    # Pre-process 'Breed species' to ensure consistent spacing
    if 'Breed species' in input_df.columns:
        # Normalize spaces: replace multiple spaces with single, then strip
        input_df['Breed species'] = input_df['Breed species'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()

    # Encode categorical features
    for col in input_df.columns:
        # Check if the column is categorical and needs encoding based on le_dict
        if col in le_dict and input_df[col].dtype == 'object':
            try:
                # Transform the value using the loaded LabelEncoder
                input_df[col] = le_dict[col].transform(input_df[col])
            except ValueError as e:
                st.error(f"❌ Error encoding column '{col}': The input value '{input_dict[col]}' is not a known category. Known categories: {list(le_dict[col].classes_)}")
                return None
    
    # Ensure all expected features are present, fill with 0 if not (for consistency)
    # This assumes that missing features should be treated as 0 after encoding.
    # It's crucial that feature_names are aligned with the training data.
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match training data's feature order
    input_df = input_df[feature_names]
    
    # Scale numerical features if the model requires it
    # Identify numerical columns for scaling. These should match the columns scaled during training.
    # 'Age' and 'Calvings' are likely numerical and were scaled.
    numerical_cols_to_scale = ['Age', 'Calvings'] 
    
    # Create a copy to avoid SettingWithCopyWarning and ensure independent scaling
    input_df_processed = input_df.copy() 

    # Apply scaling only if the column exists in the input_df_processed
    # And if the model type benefits from scaling (e.g., SVM, MLP, Logistic Regression, KNN)
    # The 'best_model' type indicates if scaling was used during its training.
    
    # This check is a simplification. A more robust way would be to check if the best_model
    # is one of the types that was trained with scaled data.
    model_trained_with_scaled_data = isinstance(model, (type(best_model))) and any(
        m_name in best_model.__class__.__name__ for m_name in ["MLP", "SVC", "LogisticRegression", "KNeighborsClassifier"]
    ) # This is a placeholder, you'd need the actual model name from the best_model_name variable from your original script

    # For simplicity, assuming if best_model is one of these types, it was scaled:
    # A better approach would be to pass `best_model_name` to this function and check against it.
    # For now, let's assume `scaler` should be applied to `Age` and `Calvings` regardless if they are in `feature_names`.
    
    for num_col in numerical_cols_to_scale:
        if num_col in input_df_processed.columns:
            input_df_processed[num_col] = scaler.transform(input_df_processed[[num_col]])[:, 0]

    # Predict
    # Ensure the input to the model is a numpy array if that's what it expects
    model_input = input_df_processed.values 
    
    pred_class = model.predict(model_input)[0]
    pred_prob = model.predict_proba(model_input)[0]
    
    # Convert back to original labels
    predicted_result = le_target.inverse_transform([pred_class])[0]
    confidence = pred_prob[pred_class]
    
    return {
        'predicted_class': predicted_result,
        'confidence': confidence,
        'probabilities': dict(zip(le_target.classes_, pred_prob))
    }

# Streamlit App Layout
st.set_page_config(page_title="Brucellosis Prediction App", layout="wide") # Changed to wide for better layout

st.title("🐂 Brucellosis Prediction Model")
st.markdown("Enter the animal's details to predict its Brucellosis status.")

st.sidebar.header("Input Features")

# Collect user input using columns for better layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (Years)", 0, 20, 5)
    breed_species = st.selectbox("Breed/Species", options=unique_breeds)
    sex = st.selectbox("Sex", options=unique_sex)
    calvings = st.slider("Calvings", 0, 15, 1)
    abortion_history = st.selectbox("Abortion History (Yes/No)", options=unique_abortion_history)

with col2:
    infertility_rb = st.selectbox("Infertility/Repeat Breeder (Yes/No)", options=unique_infertility)
    vaccination_status = st.selectbox("Brucella Vaccination Status (Yes/No)", options=unique_vaccination_status)
    sample_type = st.selectbox("Sample Type (Serum/Milk)", options=unique_sample_type)
    test_type = st.selectbox("Test Type (RBPT/ELISA/MRT)", options=unique_test_type)
    retained_placenta = st.selectbox("Retained Placenta/Stillbirth", options=unique_retained_placenta)
    proper_disposal = st.selectbox("Proper Disposal of Aborted Fetuses (Yes/No)", options=unique_disposal)


input_data = {
    'Age ': age,
    'Breed species': breed_species,
    ' Sex ': sex,
    'Calvings': calvings,
    'Abortion History (Yes No)': abortion_history,
    'Infertility Repeat breeder(Yes No)': infertility_rb,
    'Brucella vaccination status (Yes No)': vaccination_status,
    'Sample Type(Serum Milk)': sample_type,
    'Test Type (RBPT ELISA MRT)': test_type,
    'Retained Placenta Stillbirth(Yes No No Data)': retained_placenta,
    'Proper Disposal of Aborted Fetuses (Yes No)': proper_disposal
}

st.subheader("Provided Input:")
st.json(input_data)

if st.button("Predict Brucellosis Status"):
    st.subheader("Prediction Results:")
    with st.spinner('Making prediction...'):
        prediction_output = predict_single_case(
            input_data, best_model, le_dict, le_target, scaler, feature_names
        )

        if prediction_output:
            st.success(f"**Predicted Result:** {prediction_output['predicted_class']}")
            st.info(f"**Confidence:** {prediction_output['confidence']:.2%}")

            st.write("---")
            st.subheader("Class-wise Probabilities:")
            prob_df = pd.DataFrame.from_dict(prediction_output['probabilities'], orient='index', columns=['Probability'])
            prob_df = prob_df.sort_values(by='Probability', ascending=False)
            st.dataframe(prob_df.style.format("{:.2%}"))

            # Visualizing probabilities
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=prob_df.index, y=prob_df['Probability'], palette='viridis', ax=ax)
            ax.set_title("Predicted Class Probabilities")
            ax.set_ylabel("Probability")
            ax.set_xlabel("Brucellosis Status")
            st.pyplot(fig)
        else:
            st.error("Failed to make a prediction. Please check the input values and error messages above.")

st.markdown("---")
st.markdown("Developed with ❤️ for Veterinary Health")
