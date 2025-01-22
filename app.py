import nest_asyncio
import streamlit as st
import json
from typing import List
from pathlib import Path
from fit_fusion import _fit_fusion_instance
import os

# Optional: set proxy if needed
os.environ["HTTP_PROXY"] = "socks5://127.0.0.1:8087"
os.environ["HTTPS_PROXY"] = "socks5://127.0.0.1:8087"

# Allow Streamlit + async calls in a notebook environment
nest_asyncio.apply()

# Streamlit page settings
st.set_page_config(
    page_title="FitFusion",
    page_icon=":apple:",
    layout="centered",
)

st.title("FitFusion")
st.markdown("##### A diet and workout planning coach")

# --- Load default user data
def load_default_data() -> dict:
    """Load the default user data from a JSON file."""
    default_file = Path(__file__).parent / "default_user_data.json"
    with open(default_file, "r") as f:
        return json.load(f)

# --- Utility function to reset chat
def restart_agent():
    """Clears the chat history and restarts the session state for the agent."""
    for key in ["fit_fusion_agent", "messages", "user_data"]:
        if key in st.session_state:
            del st.session_state[key]
    # We don’t call st.experimental_rerun() on user input,
    # only when user explicitly clicks "New Chat Session"
    st.experimental_rerun()

# --- Main function
def main() -> None:
    # Sidebar controls
    st.sidebar.header("Controls")
    if st.sidebar.button("New Chat Session"):
        restart_agent()

    # 1. Retrieve or initialize the FitFusion Agent
    if "fit_fusion_agent" not in st.session_state:
        st.session_state["fit_fusion_agent"] = _fit_fusion_instance.agent

    fit_fusion_agent = st.session_state["fit_fusion_agent"]

    # 2. Initialize chat messages in session state
    if "messages" not in st.session_state:
        # Start with a greeting from the assistant
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I’m your **FitFusion** coach. I can help plan your diet "
                    "and workout routines. Ask me anything, or share your fitness goals!"
                ),
            }
        ]

    # 3. Initialize or load user_data from JSON file
    if "user_data" not in st.session_state:
        st.session_state["user_data"] = load_default_data()

    # 4. Create a form in the sidebar for user data
    with st.sidebar.form("user_data_form", clear_on_submit=False):
        st.write("### Personal Information")
        st.session_state["user_data"]["Age"] = st.number_input(
            "Age", value=st.session_state["user_data"]["Age"]
        )
        st.session_state["user_data"]["Gender"] = st.selectbox(
            "Gender", ["Male", "Female", "Other"],
            index=["Male", "Female", "Other"].index(
                st.session_state["user_data"]["Gender"]
            )
        )
        st.session_state["user_data"]["Height"] = st.text_input(
            "Height", st.session_state["user_data"]["Height"]
        )
        st.session_state["user_data"]["Weight"] = st.text_input(
            "Weight", st.session_state["user_data"]["Weight"]
        )

        st.write("### Goals")
        for field in ["Primary Goal", "Target Weight", "Timeframe"]:
            st.session_state["user_data"][field] = st.text_input(
                field, st.session_state["user_data"][field]
            )

        st.write("### Activity Levels")
        st.session_state["user_data"]["Current Physical Activity"] = st.text_input(
            "Current Physical Activity",
            st.session_state["user_data"]["Current Physical Activity"],
        )

        st.write("### Medical and Health Information")
        for field in ["Existing Medical Conditions", "Food Allergies"]:
            st.session_state["user_data"][field] = st.text_input(
                field, st.session_state["user_data"][field]
            )

        st.write("### Dietary Preferences")
        for field in ["Diet Type", "Meal Frequency Preferences"]:
            st.session_state["user_data"][field] = st.text_input(
                field, st.session_state["user_data"][field]
            )

        st.write("### Workout Preferences")
        for field in [
            "Preferred Workout Types",
            "Current Fitness Level",
            "Workout Frequency",
            "Workout Duration",
        ]:
            st.session_state["user_data"][field] = st.text_input(
                field, st.session_state["user_data"][field]
            )

        st.write("### Lifestyle and Habits")
        for field in ["Sleep Patterns", "Stress Levels", "Hydration Habits"]:
            st.session_state["user_data"][field] = st.text_input(
                field, st.session_state["user_data"][field]
            )

        st.write("### Behavioral Insights")
        for field in ["Motivators", "Barriers"]:
            st.session_state["user_data"][field] = st.text_input(
                field, st.session_state["user_data"][field]
            )

        st.write("### Feedback and Customization")
        for field in ["Adjustability", "Feedback Loop"]:
            st.session_state["user_data"][field] = st.text_input(
                field, st.session_state["user_data"][field]
            )

        submitted = st.form_submit_button("Update Data")
        if submitted:
            st.success("Data updated!")

    # 5. Button to generate a full 7-day plan with current user data
    if st.sidebar.button("Generate Diet & Workout Plan"):
        data_dict = st.session_state["user_data"]
        user_info_text = f"""
Generate a comprehensive 7-day diet plan and workout schedule based on the following details:

### Personal Information
Age: {data_dict["Age"]}
Gender: {data_dict["Gender"]}
Height: {data_dict["Height"]}
Weight: {data_dict["Weight"]}

### Goals
Primary Goal: {data_dict["Primary Goal"]}
Target Weight: {data_dict["Target Weight"]}
Timeframe: {data_dict["Timeframe"]}

### Activity Levels
Current Physical Activity: {data_dict["Current Physical Activity"]}

### Medical and Health Information
Existing Medical Conditions: {data_dict["Existing Medical Conditions"]}
Food Allergies: {data_dict["Food Allergies"]}

### Dietary Preferences
Diet Type: {data_dict["Diet Type"]}
Meal Frequency Preferences: {data_dict["Meal Frequency Preferences"]}

### Workout Preferences
Preferred Workout Types: {data_dict["Preferred Workout Types"]}
Current Fitness Level: {data_dict["Current Fitness Level"]}
Workout Frequency: {data_dict["Workout Frequency"]}
Workout Duration: {data_dict["Workout Duration"]}

### Lifestyle and Habits
Sleep Patterns: {data_dict["Sleep Patterns"]}
Stress Levels: {data_dict["Stress Levels"]}
Hydration Habits: {data_dict["Hydration Habits"]}

### Behavioral Insights
Motivators: {data_dict["Motivators"]}
Barriers: {data_dict["Barriers"]}

### Feedback and Customization
Adjustability: {data_dict["Adjustability"]}
Feedback Loop: {data_dict["Feedback Loop"]}
""".strip()

        # Treat this like a user message in the conversation
        st.session_state["messages"].append({"role": "user", "content": user_info_text})
        response = fit_fusion_agent.run(user_info_text)
        plan_content = response.content

        # Assistant's response
        st.session_state["messages"].append({"role": "assistant", "content": plan_content})

    # --- Always display the existing conversation
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Chat input for follow-up or new questions (always visible)
    user_input = st.chat_input(placeholder="Ask more questions or refine your plan...")
    if user_input:
        # Append user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Get assistant response
        with st.chat_message("assistant"):
            resp_container = st.empty()
            # If you want to stream token-by-token:
            # fit_fusion_agent.print_response(user_input, stream=True)
            # Then read the final content:
            response = fit_fusion_agent.run(user_input)
            agent_full_response = response.content
            resp_container.markdown(agent_full_response)

        # Finally, append assistant’s response to chat history
        st.session_state["messages"].append(
            {"role": "assistant", "content": agent_full_response}
        )


if __name__ == "__main__":
    main()
