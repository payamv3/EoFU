import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_responses" not in st.session_state:
    st.session_state.current_responses = {}
if "stage" not in st.session_state:
    st.session_state.stage = "intro"  # Possible values: "intro", "stage1", "stage2", "store_result", "final_result"

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
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['label_encoder']
    except Exception as e:
        st.error(f"Error loading main model: {str(e)}")
        raise

def load_store_model(model_path='model_store.pkl'):
    """Load the trained storage decision model."""
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        return data['model']
    except Exception as e:
        st.error(f"Error loading store model: {str(e)}")
        raise

def create_attitude_questions():
    """Define the attitude questions exactly as in training data,
    with the added store_sentiment question."""
    return {
        'resell_value': 'I know the resell value of my device',
        'resell_convenience': 'It is convenient for me to resell my device',
        'resell_worthwhile': 'The money I could get from reselling my device is worth the effort',
        'resell_investment': 'I considered the resale value of my device when I bought it',
        'recycle_know_where': 'I know where to recycle my device',
        'recycle_convenience': 'It is convenient for me to recycle my device',
        'store_sentiment': 'I store because the device has sentimental value'
    }

def get_next_question(responses, question_keys, attitude_questions):
    """Return the next unanswered question from the provided keys."""
    for key in question_keys:
        if key not in responses:
            return key, attitude_questions[key]
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
        # Load the models
        model, label_encoder = load_model()
        store_model = load_store_model()
        attitude_questions = create_attitude_questions()
        
        # Define response labels (for buttons)
        labels = {
            -2: "Strongly Disagree",
            -1: "Somewhat Disagree",
             0: "No Opinion",
             1: "Somewhat Agree",
             2: "Strongly Agree"
        }
        
        # Define stage keys
        stage1_keys = ['resell_investment', 'recycle_know_where', 'store_sentiment']
        stage2_keys = ['resell_value', 'resell_convenience', 'resell_worthwhile', 'recycle_convenience']
        final_prediction_keys = ['resell_investment', 'recycle_know_where', 'resell_value', 
                               'resell_convenience', 'resell_worthwhile', 'recycle_convenience']
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Intro stage
        if st.session_state.stage == "intro":
            st.write("""
            Hi! I'm here to help predict how you might handle your electronic device at the end of its use.
            We'll start with a few questions about your attitudes. Please indicate your level of agreement 
            with each statement by clicking one of the buttons.
            """)
            st.session_state.stage = "stage1"
            st.rerun()
        
        # Stage 1: Storage Decision
        elif st.session_state.stage == "stage1":
            st.write("### Stage 1: Storage Decision")
            st.write("Please answer the following questions:")
            
            attitude, question = get_next_question(st.session_state.current_responses, stage1_keys, attitude_questions)
            if attitude:
                with st.chat_message("assistant"):
                    st.write(f"Please indicate your level of agreement with this statement:\n\n**{question}**")
                    cols = st.columns(5)
                    for i, (value, label_text) in enumerate(labels.items()):
                        if cols[i].button(label_text, key=f"{attitude}_{value}"):
                            st.session_state.current_responses[attitude] = value
                            st.session_state.messages.append({
                                "role": "user",
                                "content": f"*{label_text}* to: {question}"
                            })
                            st.rerun()
            else:
                # All Stage 1 questions answered
                stage1_input = np.array([[st.session_state.current_responses[k] for k in stage1_keys]])
                
                # Get both prediction and probability
                store_pred = store_model.predict(stage1_input)[0]
                store_proba = store_model.predict_proba(stage1_input)[0]
                
                # Create data for visualization
                store_proba_df = pd.DataFrame({
                    'Decision': ['Not Store', 'Store'],
                    'Probability': [store_proba[0], store_proba[1]]
                })
                
                # Create visualization for storage decision probabilities
                fig = px.bar(
                    store_proba_df,
                    x='Probability',
                    y='Decision',
                    orientation='h',
                    title='Probability of Storage Decision'
                )
                fig.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    xaxis_title="Probability",
                    yaxis_title="Decision",
                    showlegend=False,
                    xaxis={'range': [0, 1]},
                    xaxis_tickformat='.0%'
                )
                fig.update_traces(marker_color='rgb(49, 130, 189)')
                
                # Add message to chat history based on prediction
                if store_pred == 0:
                    store_message = "Based on your responses, you are more likely not to store your device."
                    # Move to stage 2 if not storing
                    st.session_state.stage = "stage2"
                else:
                    url = 'https://ebay.us/RW8rZZ'
                    url1 = "https://chatgpt.com/g/g-935VQ6CmW-recycle-pro"
                    store_message = ("Based on your responses, you are likely to keep/store your device.\n"
                                    "But you can also consider reselling using one of the following channels:\n"
                                    f"1. [eBay]({url}) - The largest online marketplace for used electronics\n"
                                    f"2. Back Market - Specialized in refurbished electronics\n"
                                    f"3. Swappa - For direct sales of used electronics\n"
                                    f"4. Facebook Marketplace - For local sales\n"
                                    f"5. Decluttr - Simple selling process with instant valuation\n\n"
                                    f"If you think your device does not fetch a high price but is still functional, why not consider donating? If that is not your thing, you might choose to recycle it instead by visiting this [link]({url1})"
                                    )
                    

#However, you could also consider:

#**Reselling your device to get some money back.** You can check:

#1. eBay - The largest online marketplace for used electronics
#2. Back Market - Specialized in refurbished electronics
#3. Swappa - For direct sales of used electronics
#4. Facebook Marketplace - For local sales
#5. Decluttr - Simple selling process with instant valuation

#If the resell value is not too high but the device is still usable, you can consider donating it, or you can recycle it using the #app in this link:

#https://chatgpt.com/g/g-935VQ6CmW-recycle-pro"""
                    # Move to store result stage if storing
                    st.session_state.stage = "store_result"
                
                # Add the message to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": store_message
                })
                
                st.rerun()
        
        # Stage 2: Final Outcome Decision
        elif st.session_state.stage == "stage2":
            # Get next unanswered question
            attitude, question = get_next_question(st.session_state.current_responses, stage2_keys, attitude_questions)
            
            if attitude:
                st.write("### Stage 2: Final Outcome Decision")
                st.write("Since you're likely not to store your device, please answer these questions about what you might do with it:")
                
                with st.chat_message("assistant"):
                    st.write(f"Please indicate your level of agreement with this statement:\n\n**{question}**")
                    cols = st.columns(5)
                    for i, (value, label_text) in enumerate(labels.items()):
                        if cols[i].button(label_text, key=f"{attitude}_{value}"):
                            st.session_state.current_responses[attitude] = value
                            st.session_state.messages.append({
                                "role": "user",
                                "content": f"*{label_text}* to: {question}"
                            })
                            st.rerun()
            else:
                # All Stage 2 questions answered, move to final result
                st.session_state.stage = "final_result"
                st.rerun()
        
        # Final result for non-storage path
        elif st.session_state.stage == "final_result":
            # Prepare input data for the final prediction model
            input_values = [st.session_state.current_responses[k] for k in final_prediction_keys]
            X = np.array([input_values])
            
            # Get behavior probabilities from the final model
            probas = model.predict_proba(X)[0]
            behaviors = label_encoder.classes_
            
            # Create DataFrame for visualization
            proba_df = pd.DataFrame({
                'Behavior': behaviors,
                'Probability': probas
            }).sort_values('Probability', ascending=False)
            
            most_likely_behavior = proba_df.iloc[0]['Behavior']
            prediction_message = format_prediction(proba_df, most_likely_behavior)
            
            # Append extra messaging based on the outcome
            if "resell" in most_likely_behavior.lower():
                url = 'https://ebay.us/RW8rZZ'
                extra_message = (
                    "It looks like you might be planning to resell your device.\n"
                    "You should check the following resources!\n"
                    f"1. [eBay]({url}) - The largest online marketplace for used electronics\n"
                    f"2. Back Market - Specialized in refurbished electronics\n"
                    f"3. Swappa - For direct sales of used electronics\n"
                    f"4. Facebook Marketplace - For local sales\n"
                    f"5. Decluttr - Simple selling process with instant valuation\n\n"
                )
            elif "recycle" in most_likely_behavior.lower():
                url = 'https://ebay.us/RW8rZZ'
                url1 = "https://chatgpt.com/g/g-935VQ6CmW-recycle-pro"
                extra_message = (
                    "It appears you're considering recycling your device.\n"
                    f"You can use this [bot]({url1}) for recycling guidance:\n"
                    "However, you could also consider reselling your device to get some money back. You can check:\n"
                    f"1. [eBay]({url}) - The largest online marketplace for used electronics\n"
                    f"2. Back Market - Specialized in refurbished electronics\n"
                    f"3. Swappa - For direct sales of used electronics\n"
                    f"4. Facebook Marketplace - For local sales\n"
                    f"5. Decluttr - Simple selling process with instant valuation\n\n"
                )
            elif "throw" in most_likely_behavior.lower() or "trash" in most_likely_behavior.lower():
                url = 'https://ebay.us/RW8rZZ'
                url1 = "https://chatgpt.com/g/g-935VQ6CmW-recycle-pro"
                extra_message = (
                    "It seems you're thinking about throwing your device away.\n"
                    "However, you could also consider reselling your device to get some money back. You can check:\n"
                    f"1. [eBay]({url}) - The largest online marketplace for used electronics\n"
                    f"2. Back Market - Specialized in refurbished electronics\n"
                    f"3. Swappa - For direct sales of used electronics\n"
                    f"4. Facebook Marketplace - For local sales\n"
                    f"5. Decluttr - Simple selling process with instant valuation\n\n"
                    f"If that option isn't available, you might choose to recycle it instead by visiting this [link]({url1})"
                )
            else:
                extra_message = ""
            
            final_message = prediction_message + "\n\n" + extra_message
            
            # Create visualization for the outcome probabilities
            fig = px.bar(
                proba_df,
                x='Probability',
                y='Behavior',
                orientation='h',
                title='Probability of Each Disposal Behavior'
            )
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title="Probability",
                yaxis_title="Behavior",
                showlegend=False,
                xaxis={'range': [0, 1]},
                xaxis_tickformat='.0%'
            )
            fig.update_traces(marker_color='rgb(49, 130, 189)')
            
            # Add the prediction and visualization to chat
            with st.chat_message("assistant"):
                st.write(final_message)
                st.plotly_chart(fig)
            
            # Show start over button
            if st.button("Start Over"):
                st.session_state.messages = []
                st.session_state.current_responses = {}
                st.session_state.stage = "intro"
                st.rerun()
        
        # Result for storage path
        elif st.session_state.stage == "store_result":
            # Show start over button for the storage outcome
            if st.button("Start Over"):
                st.session_state.messages = []
                st.session_state.current_responses = {}
                st.session_state.stage = "intro"
                st.rerun()
        
    except FileNotFoundError as e:
        st.error(f"Error: Model file not found. Please ensure the model has been trained and saved. Details: {str(e)}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)  # This will display a more detailed error message

if __name__ == "__main__":
    main()