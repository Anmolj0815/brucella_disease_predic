import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder # Make sure LabelEncoder is imported

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

# --- IMPORTANT CHANGE HERE ---
# Get unique categories for dropdowns. Use the clean, stripped column names as keys.
# The `df_processed.columns.str.strip()` in your training script means keys in le_dict
# will likely NOT have leading/trailing spaces.
unique_breeds = sorted(list(le_dict.get('Breed species', LabelEncoder()).classes_))
# Assuming ' Sex ' was stripped to 'Sex' during training before le_dict was created
unique_sex = sorted(list(le_dict.get('Sex', LabelEncoder()).classes_)) # Changed from ' Sex ' to 'Sex'
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
    
    # --- CRITICAL: Strip spaces from input_df columns to match trained features ---
    input_df.columns = input_df.columns.str.strip()
    
    # Pre-process 'Breed species' content to ensure consistent spacing
    if 'Breed species' in input_df.columns:
        input_df['Breed species'] = input_df['Breed species'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()

    # Encode categorical features
    for col in input_df.columns:
        # Check if the column is in le_dict (which will now have stripped keys) and is an object type
        if col in le_dict and input_df[col].dtype == 'object':
            try:
                input_df[col] = le_dict[col].transform(input_df[col])
            except ValueError as e:
                st.error(f"❌ Error encoding column '{col}': The input value '{input_dict[col]}' is not a known category. Known categories: {list(le_dict[col].classes_)}")
                return None
    
    # Ensure all expected features are present, fill with 0 if not
    for col in feature_names: # feature_names should also have stripped column names
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match training data's feature order
    input_df = input_df[feature_names]
    
    # Decide if scaling is needed based on the type of the 'best_model'
    model_requires_scaling = isinstance(model, (type(best_model))) and any(
        m_name in best_model.__class__.__name__ for m_name in ["MLPClassifier", "SVC", "LogisticRegression", "KNeighborsClassifier"]
    )

    if model_requires_scaling:
        # Scale ALL features. The scaler was fitted on the entire X_train.
        input_df_scaled = scaler.transform(input_df)
    else:
        input_df_scaled = input_df.values # Convert to numpy array directly if not scaling

    # Predict
    pred_class = model.predict(input_df_scaled)[0]
    pred_prob = model.predict_proba(input_df_scaled)[0]
    
    # Convert back to original labels
    predicted_result = le_target.inverse_transform([pred_class])[0]
    confidence = pred_prob[pred_class]
    
    return {
        'predicted_class': predicted_result,
        'confidence': confidence,
        'probabilities': dict(zip(le_target.classes_, pred_prob))
    }

# Streamlit App Layout
st.set_page_config(page_title="Brucellosis Prediction App", layout="wide")

st.title("🐂 Brucellosis Prediction Model")
st.markdown("Enter the animal's details to predict its Brucellosis status.")

st.sidebar.header("Input Features")

# Collect user input using columns for better layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (Years)", 0, 20, 5)
    breed_species = st.selectbox("Breed/Species", options=unique_breeds)
    # --- IMPORTANT: Pass the stripped version of column names to the input_data dict keys ---
    # This means the dictionary keys should match how they appear after df.columns.str.strip()
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
    'Age': age, # Changed from 'Age ' to 'Age'
    'Breed species': breed_species,
    'Sex': sex, # Changed from ' Sex ' to 'Sex'
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
