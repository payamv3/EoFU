import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def create_model():
    """
    Create a Random Forest model with optimal parameters
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

def load_model(model_path='model.pkl'):
    """Load the trained model and label encoder"""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['label_encoder']

def create_attitude_questions():
    """Define the attitude questions exactly as in training data"""
    return {
        'resell_value': 'The device has significant resale value',
        'resell_convenience': 'It is convenient for me to resell my device',
        'resell_worthwhile': 'The money I could get from reselling my device is worth the effort',
        'resell_investment': 'Electronic devices are an investment and I should recover some of the cost by reselling',
        'recycle_know_where': 'I know where to recycle my device',
        'recycle_convenience': 'It is convenient for me to recycle my device'
    }

def main():
    st.title("Consumer Electronics End of First Use Behavior Prediction")
    st.write("""
    Think of an electronic device that you own and currently use. This could be a smartphone, tablet, AR/VR or gaming
    device, streaming device, smart TV, kitchen appliance, connected home device.
    """)
    
    try:
        # Load the trained model and label encoder
        model, label_encoder = load_model()
        
        # Get attitude questions
        attitude_questions = create_attitude_questions()
        
        # Display instructions
        st.write("Please indicate your level of agreement or disagreement with each statement.")
        
        # Add legend for slider values
        #st.write("Scale:")
        #col1, col2, col3, col4, col5 = st.columns(5)
        #with col1:
        #    st.write("**-2**: Strongly Disagree")
        #with col2:
        #    st.write("**-1**: Somewhat Disagree")
        #with col3:
        #    st.write("**0**: No Opinion")
        #with col4:
        #    st.write("**1**: Somewhat Agree")
        #with col5:
        #    st.write("**2**: Strongly Agree")
        
        # Create sliders for each attitude question
        st.subheader("Attitude Questions")
        responses = {}
        
        # Define labels for steps
        labels = {
            -2: "Strongly Disagree",
            -1: "Somewhat Disagree",
            0: "No Opinion",
            1: "Somewhat Agree",
            2: "Strongly Agree"
        }
        
        for attitude, question in attitude_questions.items():
            # Create two columns: one for the question, one for the current value label
            col1, col2 = st.columns([3, 1])
            
            with col1:
                value = st.slider(
                    question,
                    min_value=-2,
                    max_value=2,
                    value=0,
                    step=1,
                    key=attitude
                )
            
            with col2:
                st.write(f"**{labels[value]}**")
                
            responses[attitude] = value
        
        # Create a submit button
        if st.button("Predict My Behavior"):
            # Prepare the input data
            input_data = pd.DataFrame([responses])
            
            # Ensure columns are in the correct order
            attitude_columns = ['resell_value', 'resell_convenience', 'resell_worthwhile', 
                              'resell_investment', 'recycle_know_where', 'recycle_convenience']
            input_data = input_data[attitude_columns]
            
            # Convert input data to numpy array to match training format
            X = np.asarray(input_data)
            
            # Get behavior probabilities
            probas = model.predict_proba(X)
            
            # Get the original behavior labels using the label encoder
            behaviors = label_encoder.inverse_transform(range(len(label_encoder.classes_)))
            
            # Create DataFrame with behaviors and their probabilities
            proba_df = pd.DataFrame({
                'Behavior': behaviors,
                'Probability': probas[0]
            })
            
            # Sort by probability in descending order
            proba_df = proba_df.sort_values('Probability', ascending=False)
            
            # Display most likely behavior
            most_likely_behavior = proba_df.iloc[0]['Behavior']
            st.header(f"You are most likely to {most_likely_behavior}")
            
            # Create bar chart using plotly
            fig = px.bar(
                proba_df,
                x='Probability',
                y='Behavior',
                orientation='h',
                title='Probability of Each Disposal Behavior'
            )
            
            # Customize the chart
            fig.update_layout(
                yaxis={'categoryorder': 'max ascending'},
                xaxis_title="Probability",
                yaxis_title="Behavior",
                showlegend=False,
                xaxis={'range': [0, 1]},
                xaxis_tickformat='.0%'
            )
            
            # Update bar colors
            fig.update_traces(marker_color='rgb(49, 130, 189)')
            
            # Display the chart
            st.plotly_chart(fig)
                
    except FileNotFoundError:
        st.error("Error: Model file not found. Please ensure the model has been trained and saved.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()