import nest_asyncio
import streamlit as st
from typing import List
from fit_fusion import _fit_fusion_instance

# Allow Streamlit + async calls in a notebook environment
nest_asyncio.apply()

st.set_page_config(
    page_title="FitFusion",
    page_icon=":apple:",
    layout="centered",
)

st.title("FitFusion")
st.markdown("##### A diet and workout planning coach")

def restart_agent():
    """
    Clears the chat history and restarts the session state for the agent.
    """
    if "fit_fusion_agent" in st.session_state:
        del st.session_state["fit_fusion_agent"]
    if "messages" in st.session_state:
        del st.session_state["messages"]
    if "user_data" in st.session_state:
        del st.session_state["user_data"]
    st.experimental_rerun()

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
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I’m your **FitFusion** coach. I can help plan your diet "
                    "and workout routines. Ask me anything, or share your fitness goals!"
                ),
            }
        ]

    # 3. Initialize or update user_data in session state
    if "user_data" not in st.session_state:
        st.session_state["user_data"] = {
            "Age": 25,
            "Gender": "Male",
            "Height": "180 cm",
            "Weight": "70 kg",
            "Primary Goal": "Muscle Gain",
            "Target Weight": "78 kg",
            "Timeframe": "3 months",
            "Current Physical Activity": "Moderately active (fitness enthusiast)",
            "Existing Medical Conditions": "None",
            "Food Allergies": "None",
            "Diet Type": "Omnivore",
            "Meal Frequency Preferences": "3 meals + 2 snacks",
            "Preferred Workout Types": "Cardio, Strength training",
            "Current Fitness Level": "Beginner",
            "Workout Frequency": "3 days/week",
            "Workout Duration": "~45 min",
            "Sleep Patterns": "~8 hours",
            "Stress Levels": "low",
            "Hydration Habits": "~3L water/day",
            "Motivators": "General Health, Appearance",
            "Barriers": "Time constraints (office job)",
            "Adjustability": "Willing to adjust plan each month",
            "Feedback Loop": "Weekly weigh-ins and monthly measurements",
        }

    # 4. Create a form in the sidebar for user data
    with st.sidebar.form("user_data_form", clear_on_submit=False):
        st.write("### Personal Information")
        st.session_state["user_data"]["Age"] = st.number_input(
            "Age", value=st.session_state["user_data"]["Age"]
        )
        st.session_state["user_data"]["Gender"] = st.selectbox(
            "Gender", ["Male", "Female", "Other"],
            index=["Male", "Female", "Other"].index(st.session_state["user_data"]["Gender"])
            if st.session_state["user_data"]["Gender"] in ["Male", "Female", "Other"] else 0
        )
        st.session_state["user_data"]["Height"] = st.text_input(
            "Height", st.session_state["user_data"]["Height"]
        )
        st.session_state["user_data"]["Weight"] = st.text_input(
            "Weight", st.session_state["user_data"]["Weight"]
        )

        st.write("### Goals")
        st.session_state["user_data"]["Primary Goal"] = st.text_input(
            "Primary Goal", st.session_state["user_data"]["Primary Goal"]
        )
        st.session_state["user_data"]["Target Weight"] = st.text_input(
            "Target Weight", st.session_state["user_data"]["Target Weight"]
        )
        st.session_state["user_data"]["Timeframe"] = st.text_input(
            "Timeframe", st.session_state["user_data"]["Timeframe"]
        )

        st.write("### Activity Levels")
        st.session_state["user_data"]["Current Physical Activity"] = st.text_input(
            "Current Physical Activity", st.session_state["user_data"]["Current Physical Activity"]
        )

        st.write("### Medical and Health Information")
        st.session_state["user_data"]["Existing Medical Conditions"] = st.text_input(
            "Existing Medical Conditions", st.session_state["user_data"]["Existing Medical Conditions"]
        )
        st.session_state["user_data"]["Food Allergies"] = st.text_input(
            "Food Allergies", st.session_state["user_data"]["Food Allergies"]
        )

        st.write("### Dietary Preferences")
        st.session_state["user_data"]["Diet Type"] = st.text_input(
            "Diet Type", st.session_state["user_data"]["Diet Type"]
        )
        st.session_state["user_data"]["Meal Frequency Preferences"] = st.text_input(
            "Meal Frequency Preferences", st.session_state["user_data"]["Meal Frequency Preferences"]
        )

        st.write("### Workout Preferences")
        st.session_state["user_data"]["Preferred Workout Types"] = st.text_input(
            "Preferred Workout Types", st.session_state["user_data"]["Preferred Workout Types"]
        )
        st.session_state["user_data"]["Current Fitness Level"] = st.text_input(
            "Current Fitness Level", st.session_state["user_data"]["Current Fitness Level"]
        )
        st.session_state["user_data"]["Workout Frequency"] = st.text_input(
            "Workout Frequency", st.session_state["user_data"]["Workout Frequency"]
        )
        st.session_state["user_data"]["Workout Duration"] = st.text_input(
            "Workout Duration", st.session_state["user_data"]["Workout Duration"]
        )

        st.write("### Lifestyle and Habits")
        st.session_state["user_data"]["Sleep Patterns"] = st.text_input(
            "Sleep Patterns", st.session_state["user_data"]["Sleep Patterns"]
        )
        st.session_state["user_data"]["Stress Levels"] = st.text_input(
            "Stress Levels", st.session_state["user_data"]["Stress Levels"]
        )
        st.session_state["user_data"]["Hydration Habits"] = st.text_input(
            "Hydration Habits", st.session_state["user_data"]["Hydration Habits"]
        )

        st.write("### Behavioral Insights")
        st.session_state["user_data"]["Motivators"] = st.text_input(
            "Motivators", st.session_state["user_data"]["Motivators"]
        )
        st.session_state["user_data"]["Barriers"] = st.text_input(
            "Barriers", st.session_state["user_data"]["Barriers"]
        )

        st.write("### Feedback and Customization")
        st.session_state["user_data"]["Adjustability"] = st.text_input(
            "Adjustability", st.session_state["user_data"]["Adjustability"]
        )
        st.session_state["user_data"]["Feedback Loop"] = st.text_input(
            "Feedback Loop", st.session_state["user_data"]["Feedback Loop"]
        )

        submitted = st.form_submit_button("Update Data")
        if submitted:
            st.success("Data updated!")

    # 5. Button to generate a full 7-day plan with current user data
    if st.sidebar.button("Generate Diet & Workout Plan"):
        # Build a prompt from the user data
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

        # 5a. Append user query to the chat history
        st.session_state["messages"].append({"role": "user", "content": user_info_text})

        # 5b. Have the agent generate the plan and extract content
        plan_response = fit_fusion_agent.run(user_info_text)
        plan_content = plan_response.content  # Extract content from response object

        # 5c. Append the content as assistant's message
        st.session_state["messages"].append({"role": "assistant", "content": plan_content})

        # 6. Display current chat messages using markdown
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])  # Changed from st.write() to st.markdown()

        # 7. Chat input for follow-up questions
        user_input = st.chat_input(placeholder="Ask more questions or refine your plan...")
        if user_input:
            # Display user query
            st.session_state["messages"].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            # Generate and display agent response
            with st.chat_message("assistant"):
                resp_container = st.empty()
                fit_fusion_agent.print_response(user_input, stream=True)
                response = fit_fusion_agent.run(user_input)
                agent_full_response = response.content  # Extract content from response
                resp_container.markdown(agent_full_response)  # Render as markdown

            # Save agent response to chat history
            st.session_state["messages"].append({"role": "assistant", "content": agent_full_response})
# Run the app
if __name__ == "__main__":
    main()