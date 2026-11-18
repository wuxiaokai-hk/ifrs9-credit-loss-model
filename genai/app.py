"""Part 3 of the GenAI app: the USER INTERFACE and the PROMPT.

Run it with:   streamlit run genai/app.py

Every time you send a question, the app:
  1. builds the system prompt = instructions + the context (facts from the model run),
  2. adds the conversation so far and your new question,
  3. sends everything to the language model (llm.py),
  4. shows the answer as it streams in.
The model remembers nothing between questions: step 2 is how it "remembers" the chat.
"""
import itertools
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context import build_context  # noqa: E402
from llm import MODEL, chat, is_available  # noqa: E402

INSTRUCTIONS = """\
You are a credit-risk analyst. Answer questions about the IFRS 9 model results below.
Rules:
- Use only the numbers in the results. If the answer is not there, say you don't have it.
- Quote amounts in C$ millions and name the table you used.
- For totals, use the "TOTAL, all segments" rows. Never add or subtract numbers yourself.
- Be brief: at most about 150 words, plain English.

RESULTS
"""

st.set_page_config(page_title="Ask the IFRS 9 model", page_icon="📊")
st.title("Ask the IFRS 9 model")
st.caption(f"A GenAI chat over the shipped EQB results · local model: {MODEL} via Ollama · "
           "answers can be wrong, so check them against the tables")


@st.cache_data
def system_prompt() -> str:
    return INSTRUCTIONS + build_context()


with st.sidebar:
    st.subheader("Under the hood")
    st.write("This is the exact text sent to the model before your question:")
    with st.expander("System prompt (instructions + context)"):
        st.code(system_prompt(), language="markdown")
    if st.button("Clear chat"):
        st.session_state.messages = []

if not is_available():
    st.error(f"Ollama is not running or `{MODEL}` is not installed. Start the Ollama app, "
             f"or run `ollama pull {MODEL}` in Terminal.")
    st.stop()

# The conversation is kept in session_state, which survives between questions.
if "messages" not in st.session_state:
    st.session_state.messages = []

def no_math(text: str) -> str:
    """Streamlit reads text between two $ signs as a maths formula: escape them."""
    return text.replace("\\$", "$").replace("$", "\\$")


for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(no_math(m["content"]))

if not st.session_state.messages:
    st.info("Try: *What is the total allowance and which segment is largest?* · "
            "*How much worse are losses in the severe scenario than baseline?* · "
            "*Why are uninsured mortgages so sensitive to the severe scenario?*")

if question := st.chat_input("Ask about the results"):
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").markdown(no_math(question))

    messages = [{"role": "system", "content": system_prompt()}, *st.session_state.messages]
    with st.chat_message("assistant"):
        with st.spinner("Reading the results… (the first answer takes about a minute on this Mac)"):
            stream = chat(messages)
            first = next(stream, "")  # wait for the first words before streaming
        pieces = []  # the raw answer, kept for the chat history

        def show(parts):
            for piece in parts:
                pieces.append(piece)
                yield no_math(piece)

        st.write_stream(show(itertools.chain([first], stream)))
        answer = "".join(pieces)
    st.session_state.messages.append({"role": "assistant", "content": answer})
