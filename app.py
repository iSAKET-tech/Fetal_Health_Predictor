import streamlit as st
import pickle
import pandas as pd

pipe = pickle.load(open('pipe.pkl', 'rb'))




# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Fetal Health Predictor",
    page_icon="🩺",
    layout="wide"
)






# -----------------------------
# Header
# -----------------------------
st.title("Fetal Health Predictor")
st.write(
    "Machine learning-based fetal health classification using "
    "cardiotocography (CTG) features."
)

st.info(
    "This application is an educational/research project. "
    "The prediction is based on the trained dataset model and "
    "is not a clinical diagnosis."
)


# -----------------------------
# Input Form
# -----------------------------
with st.form("prediction_form"):

    st.subheader("CTG Measurements")

    col1, col2, col3 = st.columns(3)

    with col1:
        baseline_value = st.number_input(
            "Baseline Value",
            value=120.0,
            step=1.0
        )

        accelerations = st.number_input(
            "Accelerations",
            value=0.0,
            step=0.001,
            format="%.3f"
        )

        fetal_movement = st.number_input(
            "Fetal Movement",
            value=0.0,
            step=0.001,
            format="%.3f"
        )

    with col2:
        uterine_contractions = st.number_input(
            "Uterine Contractions",
            value=0.0,
            step=0.001,
            format="%.3f"
        )

        light_decelerations = st.number_input(
            "Light Decelerations",
            value=0.0,
            step=0.001,
            format="%.3f"
        )

        severe_decelerations = st.number_input(
            "Severe Decelerations",
            value=0.0,
            step=0.001,
            format="%.3f"
        )

    with col3:
        prolonged_decelerations = st.number_input(
            "Prolonged Decelerations",
            value=0.0,
            step=0.001,
            format="%.3f"
        )

        abnormal_short_term_variability = st.number_input(
            "Abnormal Short-Term Variability",
            value=0.0,
            step=1.0
        )

        mean_value_of_short_term_variability = st.number_input(
            "Mean Short-Term Variability",
            value=0.0,
            step=0.1
        )


    # -----------------------------
    # Variability
    # -----------------------------
    st.subheader("Variability Features")

    col1, col2 = st.columns(2)

    with col1:
        percentage_abnormal_long_term_variability = st.number_input(
            "Percentage of Abnormal Long-Term Variability",
            value=0.0,
            step=1.0
        )

        mean_value_of_long_term_variability = st.number_input(
            "Mean Long-Term Variability",
            value=0.0,
            step=0.1
        )

    with col2:
        histogram_width = st.number_input(
            "Histogram Width",
            value=0.0,
            step=1.0
        )

        histogram_min = st.number_input(
            "Histogram Minimum",
            value=0.0,
            step=1.0
        )

        histogram_max = st.number_input(
            "Histogram Maximum",
            value=0.0,
            step=1.0
        )


    # -----------------------------
    # Histogram Features
    # -----------------------------
    st.subheader("Histogram Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        histogram_number_of_peaks = st.number_input(
            "Number of Histogram Peaks",
            value=0.0,
            step=1.0
        )

        histogram_number_of_zeroes = st.number_input(
            "Number of Histogram Zeroes",
            value=0.0,
            step=1.0
        )

        histogram_mode = st.number_input(
            "Histogram Mode",
            value=0.0,
            step=1.0
        )

    with col2:
        histogram_mean = st.number_input(
            "Histogram Mean",
            value=0.0,
            step=1.0
        )

        histogram_median = st.number_input(
            "Histogram Median",
            value=0.0,
            step=1.0
        )

        histogram_variance = st.number_input(
            "Histogram Variance",
            value=0.0,
            step=1.0
        )

    with col3:
        histogram_tendency = st.number_input(
            "Histogram Tendency",
            value=0.0,
            step=1.0
        )


    # -----------------------------
    # Prediction Button
    # -----------------------------
    submitted = st.form_submit_button(
        "Predict Fetal Health",
        use_container_width=True
    )


# -----------------------------
# Prediction
# -----------------------------
if submitted:

    input_data = pd.DataFrame([{
        "baseline value": baseline_value,
        "accelerations": accelerations,
        "fetal_movement": fetal_movement,
        "uterine_contractions": uterine_contractions,
        "light_decelerations": light_decelerations,
        "severe_decelerations": severe_decelerations,
        "prolongued_decelerations": prolonged_decelerations,
        "abnormal_short_term_variability": abnormal_short_term_variability,
        "mean_value_of_short_term_variability": mean_value_of_short_term_variability,
        "percentage_of_time_with_abnormal_long_term_variability":
            percentage_abnormal_long_term_variability,
        "mean_value_of_long_term_variability":
            mean_value_of_long_term_variability,
        "histogram_width": histogram_width,
        "histogram_min": histogram_min,
        "histogram_max": histogram_max,
        "histogram_number_of_peaks": histogram_number_of_peaks,
        "histogram_number_of_zeroes": histogram_number_of_zeroes,
        "histogram_mode": histogram_mode,
        "histogram_mean": histogram_mean,
        "histogram_median": histogram_median,
        "histogram_variance": histogram_variance,
        "histogram_tendency": histogram_tendency
    }])


    # Prediction
    prediction = pipe.predict(input_data)[0]

    # Model probabilities
    probabilities = pipe.predict_proba(input_data)[0]


    # Class mapping
    class_names = {
        1.0: "Normal",
        2.0: "Suspect",
        3.0: "Pathological"
    }

    result = class_names.get(
        float(prediction),
        str(prediction)
    )

    st.table(input_data)

    # -----------------------------
    # Result
    # -----------------------------
    st.divider()

    st.subheader("Prediction Result")

    if result == "Normal":
        st.success(f"Predicted Fetal Health: **{result}**")

    elif result == "Suspect":
        st.warning(f"Predicted Fetal Health: **{result}**")

    else:
        st.error(f"Predicted Fetal Health: **{result}**")




    # -----------------------------
    # Model Output Probabilities
    # -----------------------------
    st.subheader("Model Output")

    probability_df = pd.DataFrame({
        "Class": [
            "Normal",
            "Suspect",
            "Pathological"
        ],
        "Model Probability": probabilities
    })

    probability_df["Model Probability"] =((
        probability_df["Model Probability"] * 100
    ).round(2).astype(str) + "%")

    st.dataframe(
        probability_df,
        use_container_width=True,
        hide_index=True
    )
