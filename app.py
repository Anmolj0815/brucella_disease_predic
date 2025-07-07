import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

# Load the trained objects
try:
    with open('best_model.pkl', 'rb') as f:
        best_model = pickle.load(f)
    with open('le_dict.pkl', 'rb') as f:
        le_dict = pickle.load(f)
    with open('le_target.pkl', 'rb') as f:
        le_target = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    st.sidebar.success("✅ All model components loaded successfully!")

except FileNotFoundError:
    st.sidebar.error("❌ Required model files not found. Please ensure 'best_model.pkl', 'le_dict.pkl', 'le_target.pkl', 'scaler.pkl', and 'feature_names.pkl' are in the same directory.")
    st.stop() # Stop the app if essential files are missing
except Exception as e:
    st.sidebar.error(f"❌ Error loading model components: {e}")
    st.stop()

# Get unique categories for dropdowns from le_dict
# We'll use a placeholder for now, but in a real scenario, you'd extract these from le_dict
# For demonstration, let's assume we can derive these from the loaded le_dict
unique_breeds = list(le_dict.get('Breed species', LabelEncoder()).classes_)
unique_sex = list(le_dict.get(' Sex ', LabelEncoder()).classes_)
unique_abortion_history = list(le_dict.get('Abortion History (Yes No)', LabelEncoder()).classes_)
unique_infertility = list(le_dict.get('Infertility Repeat breeder(Yes No)', LabelEncoder()).classes_)
unique_vaccination_status = list(le_dict.get('Brucella vaccination status (Yes No)', LabelEncoder()).classes_)
unique_sample_type = list(le_dict.get('Sample Type(Serum Milk)', LabelEncoder()).classes_)
unique_test_type = list(le_dict.get('Test Type (RBPT ELISA MRT)', LabelEncoder()).classes_)
unique_retained_placenta = list(le_dict.get('Retained Placenta Stillbirth(Yes No No Data)', LabelEncoder()).classes_)
unique_disposal = list(le_dict.get('Proper Disposal of Aborted Fetuses (Yes No)', LabelEncoder()).classes_)


def predict_single_case(input_dict, model, le_dict, le_target, scaler, feature_names):
    """Predict a single case with robust error handling for encoding."""
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Clean column names (strip spaces to match trained features)
    input_df.columns = input_df.columns.str.strip()
    
    # Pre-process 'Breed species' to ensure consistent spacing
    if 'Breed species' in input_df.columns:
        input_df['Breed species'] = input_df['Breed species'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()


    # Encode categorical features
    for col in input_df.columns:
        # Check if the column is categorical and needs encoding based on le_dict
        if col in le_dict and input_df[col].dtype == 'object':
            try:
                # Ensure the input value is in the classes known by the encoder
                if input_df[col].values[0] not in le_dict[col].classes_:
                    st.warning(f"⚠️ Input value '{input_df[col].values[0]}' for column '{col}' was not seen during training. Attempting to proceed with the closest match or default if available.")
                    # Fallback: if not found, you might want to map to a default or handle.
                    # For now, let it raise ValueError if not found.
                input_df[col] = le_dict[col].transform(input_df[col])
            except ValueError as e:
                st.error(f"❌ Error encoding column '{col}': {input_df[col].values[0]} is not a known category. Known categories: {list(le_dict[col].classes_)}")
                return None
    
    # Ensure all expected features are present, fill with 0 if not (for consistency)
    # This assumes that missing features should be treated as 0 after encoding
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match training data
    input_df = input_df[feature_names]
    
    # Scale numerical features if the model requires it
    # We apply scaling only to the relevant columns, assuming scaler was fit on X_train
    # It's safer to apply scaler only to numerical columns
    numerical_cols_to_scale = [col for col in feature_names if col in ['Age', 'Calvings']] # Adjust based on your actual numerical columns
    
    # Create a copy to avoid SettingWithCopyWarning
    input_df_scaled = input_df.copy() 

    if 'Age' in input_df_scaled.columns:
        input_df_scaled['Age'] = scaler.transform(input_df_scaled[['Age']])[:, 0]
    if 'Calvings' in input_df_scaled.columns:
        input_df_scaled['Calvings'] = scaler.transform(input_df_scaled[['Calvings']])[:, 0]

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
st.set_page_config(page_title="Brucellosis Prediction App", layout="centered")

st.title("🐂 Brucellosis Prediction Model")
st.markdown("Enter the animal's details to predict its Brucellosis status.")

st.sidebar.header("Input Features")

# Collect user input
age = st.sidebar.slider("Age (Years)", 0, 20, 5)
breed_species = st.sidebar.selectbox("Breed/Species", options=unique_breeds)
sex = st.sidebar.selectbox("Sex", options=unique_sex)
calvings = st.sidebar.slider("Calvings", 0, 15, 1)
abortion_history = st.sidebar.selectbox("Abortion History (Yes/No)", options=unique_abortion_history)
infertility_rb = st.sidebar.selectbox("Infertility/Repeat Breeder (Yes/No)", options=unique_infertility)
vaccination_status = st.sidebar.selectbox("Brucella Vaccination Status (Yes/No)", options=unique_vaccination_status)
sample_type = st.sidebar.selectbox("Sample Type (Serum/Milk)", options=unique_sample_type)
test_type = st.sidebar.selectbox("Test Type (RBPT/ELISA/MRT)", options=unique_test_type)
retained_placenta = st.sidebar.selectbox("Retained Placenta/Stillbirth", options=unique_retained_placenta)
proper_disposal = st.sidebar.selectbox("Proper Disposal of Aborted Fetuses (Yes/No)", options=unique_disposal)

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
