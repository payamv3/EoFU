import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_responses" not in st.session_state:
    st.session_state.current_responses = {}

def create_model():
    """Create a Random Forest model with optimal parameters"""
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

def get_question_from_responses(responses, attitude_questions):
    """Get the next unanswered question"""
    for attitude, question in attitude_questions.items():
        if attitude not in responses:
            return attitude, question
    return None, None

def format_prediction(proba_df, most_likely_behavior):
    """Format the prediction results as a chat message"""
    message = f"📊 Based on your responses, you are most likely to: **{most_likely_behavior}**\n\n"
    message += "Other possible behaviors and their probabilities:\n"
    for _, row in proba_df.iterrows():
        if row['Behavior'] != most_likely_behavior:
            message += f"- {row['Behavior']}: {row['Probability']:.1%}\n"
    return message

def main():
    st.title("Consumer Electronics Behavior Chatbot")
    
    try:
        # Load the trained model and label encoder
        model, label_encoder = load_model()
        attitude_questions = create_attitude_questions()
        
        # Display chat interface
        st.write("""
        Hi! I'm here to help predict how you might handle your electronic device at the end of its use. 
        Let's have a conversation about your attitudes towards device resale and recycling.
        """)
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Define labels for responses
        labels = {
            -2: "Strongly Disagree",
            -1: "Somewhat Disagree",
            0: "No Opinion",
            1: "Somewhat Agree",
            2: "Strongly Agree"
        }
        
        # Get next question if needed
        attitude, question = get_question_from_responses(
            st.session_state.current_responses, 
            attitude_questions
        )
        
        # If there are still questions to ask
        if attitude:
            # Display current question
            with st.chat_message("assistant"):
                st.write(f"Please indicate your level of agreement with this statement:\n\n**{question}**")
                
                # Create a column layout for the response options
                cols = st.columns(5)
                for i, (value, label) in enumerate(labels.items()):
                    if cols[i].button(label, key=f"{attitude}_{value}"):
                        # Store response
                        st.session_state.current_responses[attitude] = value
                        
                        # Add to chat history
                        st.session_state.messages.append({
                            "role": "user",
                            "content": f"*{label}* to: {question}"
                        })
                        
                        # Rerun to update chat and possibly make prediction
                        st.rerun()
        
        # If all questions are answered, make prediction
        elif len(st.session_state.current_responses) == len(attitude_questions):
            # Prepare the input data
            input_data = pd.DataFrame([st.session_state.current_responses])
            
            # Ensure columns are in the correct order
            attitude_columns = list(attitude_questions.keys())
            input_data = input_data[attitude_columns]
            
            # Convert input data to numpy array
            X = np.asarray(input_data)
            
            # Get behavior probabilities
            probas = model.predict_proba(X)
            
            # Get the original behavior labels
            behaviors = label_encoder.inverse_transform(range(len(label_encoder.classes_)))
            
            # Create DataFrame with behaviors and probabilities
            proba_df = pd.DataFrame({
                'Behavior': behaviors,
                'Probability': probas[0]
            }).sort_values('Probability', ascending=False)
            
            # Get most likely behavior
            most_likely_behavior = proba_df.iloc[0]['Behavior']
            
            # Format prediction message
            prediction_message = format_prediction(proba_df, most_likely_behavior)
            
            # Create visualization and show prediction
            fig = px.bar(
                proba_df,
                x='Probability',
                y='Behavior',
                orientation='h',
                title='Probability of Each Disposal Behavior'
            )
            
            fig.update_layout(
                yaxis={'categoryorder': 'max ascending'},
                xaxis_title="Probability",
                yaxis_title="Behavior",
                showlegend=False,
                xaxis={'range': [0, 1]},
                xaxis_tickformat='.0%'
            )
            
            fig.update_traces(marker_color='rgb(49, 130, 189)')
            
            # Show prediction and visualization together
            with st.chat_message("assistant"):
                st.write(prediction_message)
                st.plotly_chart(fig)
                if st.button("Start Over"):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
                
    except FileNotFoundError:
        st.error("Error: Model file not found. Please ensure the model has been trained and saved.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()