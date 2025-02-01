import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd

def create_model():
    """
    Create a Random Forest model with the same parameters as the original code
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

def load_models(behavior_pairs):
    """
    Load or create models for each behavior pair
    """
    models = {}
    for pair in behavior_pairs:
        try:
            with open(f'model_{pair[0]}_{pair[1]}.pkl', 'rb') as f:
                models[f'{pair[0]}_vs_{pair[1]}'] = pickle.load(f)
        except FileNotFoundError:
            # If model doesn't exist, create a new one
            models[f'{pair[0]}_vs_{pair[1]}'] = create_model()
    return models

def main():
    st.title("Behavior Prediction System")
    st.write("""
    Think of an electronic device that you own and currently use. This could be a smartphone, tablet, AR/VR or gaming device, streaming device, smart TV, kitchen appliance, connected home device. Answer the following questions with that device in mind.
    To what extent do you agree or disagree with the fllowing? You can move the sliders according to the below guide: 
    (-2: Strongly Disagree, 2: Strongly Agree)
    """)

    # Define attitude questions - these should match your attitude_columns
    attitude_questions = {
        "resell_value": "I know the resale value of the device",
        "resell_convenience": "It is convenient to resell the device",
        "resell_worthwhile": "The money I can get makes reselling the device worthwhile",
        "resell_investment": "I considered the resale value when buying this device",
        "resell_online": "I know how to resell the device online",
        "resell_offline":"I know how to resell the device at a local store",
        "resell_data":"I am hesitant about reselling because of data left on the device",
        "resell_environment":"Reselling protects envrionment"
    }

    # Create sliders for each attitude question
    user_responses = {}
    for key, question in attitude_questions.items():
        user_responses[key] = st.slider(
            question,
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=1.0,
            help="Slide to indicate your level of agreement"
        )

    # Define behavior pairs
    behavior_pairs = [
        ("Recycle it", "Resell it"),
        #("recycle", "donate"),
        ("Resell it", "Throw it away with household trash")
    ]

    # Load models
    models = load_models(behavior_pairs)

    if st.button("Predict My Behaviors"):
        # Prepare input data
        input_data = np.array([[v for v in user_responses.values()]])

        st.subheader("Your Behavior Predictions:")

        # Make predictions for each pair
        for pair in behavior_pairs:
            behavior1, behavior2 = pair
            model_key = f'{behavior1}_vs_{behavior2}'
            
            if model_key in models:
                # Get prediction probability
                probs = models[model_key].predict_proba(input_data)[0]
                
                # Create result message
                likely_behavior = behavior2 if probs[1] > 0.5 else behavior1
                unlikely_behavior = behavior1 if probs[1] > 0.5 else behavior2
                higher_prob = max(probs[1], 1 - probs[1]) * 100
                
                # Display result with probability
                st.write(f"""
                🔮 When choosing between **{behavior1}** and **{behavior2}**:
                - You are more likely to **{likely_behavior}** than **{unlikely_behavior}**
                - Probability: {higher_prob:.1f}%
                """)
                
                # Create a progress bar for visualization
                st.progress(probs[1])

        st.write("""
        ---
        Note: These predictions are based on machine learning models trained on historical data.
        Your actual behavior may vary depending on specific circumstances.
        """)

if __name__ == "__main__":
    main()