import uuid
import gradio as gr
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from openai import OpenAI
import os

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class State(TypedDict):
    recipe_request: str
    recipe: str
    allergenes: str
    allergenes_detected: bool
    human_feedback: str
    final_recipe: str

# Initialize graph builder
graph_builder = StateGraph(State)

def recipe_generator(state: State) -> State:
    """Generate initial recipe based on request"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"Generate a detailed recipe for: {state['recipe_request']}"
        }]
    )
    state["recipe"] = response.choices[0].message.content
    
    # Check for common allergenes
    allergenes_check = client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user", 
            "content": f"Analyze this recipe for common allergenes (nuts, dairy, gluten, shellfish, eggs). Only respond with 'ALLERGENES FOUND' plus the allergenes detected or 'NO ALLERGENES': {state['recipe']}"
        }]
    )
    state["allergenes"] = allergenes_check.choices[0].message.content.split("ALLERGENES FOUND")[1].strip()
    state["allergenes_detected"] = "ALLERGENES FOUND" in allergenes_check.choices[0].message.content
    return state

def human_feedback_handler(state: State) -> State:
    """Handle human feedback for allergenes"""
    state["allergenes_detected"] = False
    return state

def recipe_finalizer(state: State) -> State:
    """Finalize recipe based on feedback if any"""
    if "human_feedback" in state and state["human_feedback"]:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": f"Modify this recipe according to dietary restrictions: {state['recipe']}\nRestrictions: {state['human_feedback']}"
            }]
        )
        state["final_recipe"] = response.choices[0].message.content
    else:
        state["final_recipe"] = state["recipe"]
    return state

def allergenes_check_condition(state: State) -> str:
    """Determine next step based on allergenes detection"""
    if state["allergenes_detected"]:
        return "human_feedback_handler"
    else:
        return "recipe_finalizer"

# Add nodes
graph_builder.add_node("recipe_generator", recipe_generator)
graph_builder.add_node("human_feedback_handler", human_feedback_handler)
graph_builder.add_node("recipe_finalizer", recipe_finalizer)

# Add edges
graph_builder.add_edge(START, "recipe_generator")
graph_builder.add_conditional_edges(
    "recipe_generator",
    allergenes_check_condition,
    {
        "human_feedback_handler": "human_feedback_handler",
        "recipe_finalizer": "recipe_finalizer"
    }
)
graph_builder.add_edge("human_feedback_handler", "recipe_finalizer")
graph_builder.add_edge("recipe_finalizer", END)

# Create Gradio interface
CSS = """
.contain { display: flex; flex-direction: column; }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; overflow: auto;}
"""
with gr.Blocks(css=CSS) as demo:
    memory = MemorySaver()
    graph = graph_builder.compile(
        checkpointer=memory,
        interrupt_before=["human_feedback_handler"]
    )

    gr.Markdown("""
    # 🎂 SafePlates
    
    SafePlates: A smart recipe assistant that creates personalized meals while checking for allergens to ensure safe and delicious dining.
                
    Start by entering your recipe request in the text box below. The assistant will:
                
    1. Generate a detailed recipe based on your request
    2. Check for common allergens (nuts, dairy, gluten, shellfish, eggs)
    3. If allergens are detected, ask for your dietary restrictions
    4. Modify the recipe according to your needs
    5. Provide you with a safe and delicious final recipe""")
    
    chatbot = gr.Chatbot(
        show_copy_button=False,
        show_share_button=False,
        label="Recipe Assistant",
        elem_id="chatbot",
        type="tuples"
    )
    msg = gr.Textbox(
        placeholder="Enter your recipe request...",
        container=False,
        scale=7
    )
    user_id = gr.State(None)
    state = gr.State(None)

    def process_request(message, history, state, user_id):
        if not user_id:
            user_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": user_id}}

        # Handle initial request or human feedback
        if not state:
            state = {"recipe_request": message}
            history = [(message, "Generating recipe...")]
        elif "allergenes_detected" in state and state["allergenes_detected"]:
            state["human_feedback"] = message
            graph.update_state(config=config, values=state)
            history.append((message, "Updating recipe..."))
        yield history, state, user_id

        # Process through graph
        for event in graph.stream(None if "human_feedback" in state else state, config=config):
            state = graph.get_state(config=config).values
            if isinstance(event, dict):
                if "recipe_generator" in event:
                    if event["recipe_generator"]["allergenes_detected"]:
                        history.append((None, f"Are you allergic to any of the following ingredients:\n{event['recipe_generator']['allergenes'][2:]}\nIf so, please specify which ones."))
                    else:
                        history.append((None, event["recipe_generator"]["recipe"]))
                elif "recipe_finalizer" in event:
                    history.append((None, f"Here's your final recipe:\n\n{event['recipe_finalizer']['final_recipe']}"))
            yield history, state, user_id

    msg.submit(
        process_request,
        inputs=[msg, chatbot, state, user_id],
        outputs=[chatbot, state, user_id]
    ).then(lambda: None, inputs=[], outputs=[msg])

    with gr.Row():
        example_button1 = gr.Button("Example 1: Lemon Cake")
        example_button2 = gr.Button("Example 2: Chocolate Cookies") 
        example_button3 = gr.Button("Example 3: Banana Bread")

    def load_example(example_num):
        if example_num == 1:
            return "I want a recipe for a lemon cake"
        elif example_num == 2:
            return "I want a recipe for chocolate cookies"
        elif example_num == 3:
            return "I want a recipe for banana bread"
        else:
            return None

    example_button1.click(fn=load_example, inputs=[gr.Number(value=1, visible=False)], outputs=[msg]).then(
        process_request,
        inputs=[msg, chatbot, state, user_id],
        outputs=[chatbot, state, user_id]
    ).then(lambda: None, inputs=[], outputs=[msg])
    example_button2.click(fn=load_example, inputs=[gr.Number(value=2, visible=False)], outputs=[msg]).then(
        process_request,
        inputs=[msg, chatbot, state, user_id],
        outputs=[chatbot, state, user_id]
    ).then(lambda: None, inputs=[], outputs=[msg])
    example_button3.click(fn=load_example, inputs=[gr.Number(value=3, visible=False)], outputs=[msg]).then(
        process_request,
        inputs=[msg, chatbot, state, user_id],
        outputs=[chatbot, state, user_id]
    ).then(lambda: None, inputs=[], outputs=[msg])

    refresh_button = gr.Button("🔄 Refresh")
    refresh_button.click(fn=lambda: None, inputs=[], outputs=[chatbot, state, user_id], js="() => {window.location.reload()}")

    gr.Markdown(
        """
        By using this application, you agree to OpenAI's [terms of service](https://openai.com/policies/terms-of-service) and [privacy policy](https://openai.com/policies/privacy-policy).
        """
    )
demo.launch()
