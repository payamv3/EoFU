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
    
    To what extent do you agree or disagree with the following? You can move the sliders from Strongly Disagree to Strongly Agree
    """)

    # Define attitude questions - these should match your attitude_columns
    attitude_questions = {
        "resell_value": "I know the resale value of the device",
        "resell_convenience": "It is convenient to resell the device",
        "resell_worthwhile": "The money I can get makes reselling the device worthwhile",
    }

    # Create sliders for each attitude question with text labels
    likert_labels = {
    "Strongly Disagree": -2,
    "Somewhat Disagree": -1,
    "I do not know / No opinion": 0,
    "Somewhat Agree": 1,
    "Strongly Agree": 2
    }

    user_responses = {}

    for key, question in attitude_questions.items():
        response = st.select_slider(
            question,
            options=list(likert_labels.keys()),  # Display labels
            value="I do not know / No opinion",  # Default selection
            help="Slide to indicate your level of agreement"
        )
    
    # Store the corresponding numerical value
        user_responses[key] = likert_labels[response]

    # Define behavior pairs
    behavior_pairs = [
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

                # Display messages based on user responses
                if user_responses["resell_convenience"] in ['Strongly Disagree', 'Somewhat Disagree']:

                    st.write("You disagree that reselling is convenient. You can easily sell your consumer product on e-bay")
                
                if user_responses["resell_worthwhile"] in ['Strongly Disagree', 'Somewhat Disagree']:
                    st.write("You disagree that the money you can get from reselling makes it worthwhile, have you tried looking up the value of your device on e-bay?")

        st.write("""
        ---
        Note: These predictions are based on machine learning models trained on historical data.
        Your actual behavior may vary depending on specific circumstances.
        """)

if __name__ == "__main__":
    main()
