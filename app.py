import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
# Import necessary classifiers for type checking in predict_single_case
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


warnings.filterwarnings('ignore')

# Define the directory where model artifacts are stored
MODEL_ARTIFACTS_DIR = 'model_artifacts/'
IMAGE_PATH = MODEL_ARTIFACTS_DIR + 'veterinary.jpg' # Your image path and filename

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

except FileNotFoundError as e:
    st.sidebar.error(f"❌ Required model file not found: {e}. Please ensure all .pkl files and the image are in the '{MODEL_ARTIFACTS_DIR}' directory (or update the path).")
    st.stop() # Stop the app if essential files are missing
except Exception as e:
    st.sidebar.error(f"❌ Error loading model components: {e}")
    st.stop()

# Streamlit App Layout
st.set_page_config(page_title="Brucellosis Prediction App", layout="wide")

# Display the veterinary image at the very top
try:
    st.image(IMAGE_PATH, caption="Caring for Animal Health", use_column_width=True)
except FileNotFoundError:
    st.warning(f"⚠️ Veterinary image not found at '{IMAGE_PATH}'. Please ensure 'veterinary.jpg' is in the '{MODEL_ARTIFACTS_DIR}' folder.")
except Exception as e:
    st.error(f"⚠️ Error loading veterinary image: {e}")

st.title("🐂 Brucellosis Prediction Model")
st.markdown("Enter the animal's details to predict its Brucellosis status.")

# Get unique categories for dropdowns
unique_breeds = sorted(list(le_dict.get('Breed species', LabelEncoder()).classes_))
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

    # Strip spaces from input_df columns
    input_df.columns = input_df.columns.str.strip()

    # Pre-process 'Breed species' content
    if 'Breed species' in input_df.columns:
        input_df['Breed species'] = input_df['Breed species'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()

    # Encode categorical features
    for col in input_df.columns:
        if col in le_dict and input_df.dtypes.get(col) == 'object':
            try:
                input_df.loc[:, col] = le_dict.get(col).transform(input_df.loc[:, col])
            except ValueError as e:
                st.error(f"❌ Error encoding column '{col}': The input value '{input_dict.get(col)}' is not a known category. Known categories: {list(le_dict.get(col).classes_)}")
                return None

    # Ensure all expected features are present, fill with 0 if not
    for col in feature_names:
        if col not in input_df.columns:
            input_df.loc[:, col] = 0

    # Reorder columns to match training data's feature order
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Decide if scaling is needed based on the type of the 'best_model'
    # Import relevant classes at the top of the file to avoid NameError
    model_requires_scaling = isinstance(model, (MLPClassifier, SVC, LogisticRegression, KNeighborsClassifier))

    if model_requires_scaling:
        input_df_scaled = scaler.transform(input_df)
    else:
        input_df_scaled = input_df.values

    # Predict
    pred_class = model.predict(input_df_scaled)[0]
    pred_prob = model.predict_proba(input_df_scaled)[0]

    # Convert back to original labels
    predicted_result = le_target.inverse_transform([pred_class])[0]
    confidence = pred_prob.max() # Using max probability as confidence

    return {
        'predicted_class': predicted_result,
        'confidence': confidence,
        'probabilities': dict(zip(le_target.classes_, pred_prob))
    }

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
    'Age': age,
    'Breed species': breed_species,
    'Sex': sex,
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
