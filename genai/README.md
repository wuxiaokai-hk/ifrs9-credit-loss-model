# Your first GenAI app: "Ask the IFRS 9 model"

A chat app where you ask questions about the model's results in plain English and a language
model answers from the actual numbers. It runs fully on your Mac: the model is `gemma3:4b`
through Ollama. It's free and private, and nothing leaves the computer.

## Run it

```bash
cd ~/Downloads/ifrs9-credit-loss-model
streamlit run genai/app.py
```

It opens in your browser. The Ollama app must be running. On this Intel Mac the first answer
takes about 30–60 seconds; later ones are a little quicker.

## What a GenAI app is made of

Almost every GenAI app, including ChatGPT-style products, is the same four parts:

```
 ┌────────────┐   question   ┌──────────────────────────────┐   messages   ┌──────────────┐
 │ 1. UI      │ ───────────► │ 2. PROMPT                    │ ───────────► │ 4. MODEL     │
 │  app.py    │              │  instructions                │              │  llm.py      │
 │ (Streamlit)│ ◄─────────── │  + 3. CONTEXT (context.py)   │ ◄─────────── │ (Ollama,     │
 └────────────┘ answer,      │  + chat history + question   │  answer      │  gemma3:4b)  │
                streamed     └──────────────────────────────┘              └──────────────┘
```

| Part | File | What it does |
|---|---|---|
| **1. User interface** | `app.py` | The chat window. Keeps the conversation in memory and shows answers as they stream in. |
| **2. Prompt** | `app.py` (`INSTRUCTIONS`) | The rules the model must follow: role, "use only these numbers", answer length. This is where most of the quality comes from. |
| **3. Context** | `context.py` | Your data, turned into text. The model has never seen your results; we paste the key tables into every request. This is called **grounding**. When the right pieces are *searched for* first, it's called **RAG**. |
| **4. Model** | `llm.py` | Sends the messages to the language model and returns the reply. The only file that knows which model is used. |

Open the **"System prompt"** panel in the app's sidebar to see exactly what the model receives.
Run `python genai/context.py` to print just the context.

### Three facts that explain most of GenAI

1. **The model has no memory.** Every question sends the whole conversation again. That's
   why longer chats get slower and cost more tokens.
2. **The model only knows what's in the prompt** (plus general knowledge from training). If a
   number isn't in the context, it will either say so or, worse, make one up.
3. **The model predicts plausible text; it doesn't calculate.** Do sums in code and give it the result.

## Lessons from building it (real mistakes we saw)

- **Arithmetic.** Asked for total losses, the model used the credit-card row as the total and
  invented a subtraction. **Fix:** `context.py` now adds a `TOTAL, all segments` row in code,
  and the instructions say *never add or subtract numbers yourself*. After that, it answered correctly.
- **Mislabelling.** Asked why uninsured mortgages are sensitive, it called C$52.2m the
  "baseline" allowance (that's the probability-weighted figure; baseline is C$20.4m) and
  presented the 3.14% Stage 3 coverage as "severe" coverage. The numbers were real, but the
  labels were wrong. Small models do this often; bigger models much less. **Always check
  answers against the tables**, which is why the app says so under the title.

## Agent vs GenAI app

| | This GenAI app | The `ifrs9-analyst` agent |
|---|---|---|
| What it does | Answers from numbers we give it | Decides its own steps and runs tools |
| Can it run new scenarios? | No, only reads the shipped results | Yes, it writes configs and runs the model |
| Calls to the model per question | One | Many (a loop) |
| Who controls the flow | Your code | The model |

A GenAI app becomes an agent when you let the model **call tools** and loop. That's the
natural next step: give this app a `run_whatif()` tool.

## Exercises

1. **Change the personality.** In `app.py`, edit `INSTRUCTIONS` to *"explain like I'm a new
   board member"*. Ask the same question and compare.
2. **Turn the temperature up.** In `llm.py`, set `temperature=1.0`, ask the same question three
   times and see how the answers vary. Then set it back to `0.2`.
3. **Add context.** In `context.py`, add the `attribution` sheet (why severe losses are higher:
   PD vs collateral). Ask *"what drives the severe loss for uninsured mortgages?"* before and after.
4. **Find the limit.** Ask something that isn't in the context, such as *"what is the Stage 2
   allowance for HELOCs?"* Does it admit it doesn't know, or make something up?
5. **Swap the model.** Install a different model (`ollama pull qwen3:4b`, a 2–3 GB download)
   and change `MODEL` in `llm.py`. Only one line changes, because the parts are separate.
