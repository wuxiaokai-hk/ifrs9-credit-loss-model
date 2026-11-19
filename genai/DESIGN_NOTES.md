# Design notes: the GenAI chat app and the agent

Why the chat app and the agent are built the way they are, and what to watch for if you adapt
them to your own model results. Each section starts with a short answer, then the details.

---

## 1. Grounding: why the model gets the results in the prompt

**Short answer:**
A language model only knows what it learned in training; it has never seen this model's results.
So before every question the app reads the key tables from the results workbook, turns them into
text and puts them in the prompt. The model answers from those numbers instead of from memory.

**Details:**
- `context.py` builds about 400 words of context: the allowance by stage, the allowance under
  each scenario, cumulative losses, coverage ratios and a few notes.
- **Grounding** means giving the model the facts in the prompt. **RAG** (retrieval-augmented
  generation) is grounding where you first *search* a larger collection, such as documents or a
  database, for the relevant pieces. The results here are small enough to send in full, so there
  is no search step.
- Without grounding, the model would either refuse or invent plausible-looking numbers.
- The instructions also say "use only these numbers; if it's not there, say so", to stop it
  filling gaps from general knowledge.

---

## 2. Why the chat gets slower as the conversation grows

**Short answer:**
The model has no memory between calls. To look like a conversation, the app sends the whole
history (instructions, context, every earlier question and answer) with each new question, so
each question means more text to read: slower, and with a paid model, more tokens.

**Details:**
- Each request contains the system prompt (instructions + context), then all previous turns,
  then the new question.
- On a laptop CPU a small local model reads roughly 40 tokens per second, so every extra 400
  words of history adds about 10 seconds.
- Common fixes: limit the history to the last few turns, summarise older turns, or keep the
  context short (the reason the app sends compact tables, not the whole workbook).
- Models also have a **context window**, a maximum amount of text per request. Long chats
  eventually hit it.

---

## 3. Totals and arithmetic: do them in code

**Short answer:**
Language models predict likely text; they don't calculate. In testing, a small model asked for
total losses took the credit-card row as the total and made up a subtraction. The fix: totals are
computed in code and every table has a "TOTAL" row, and the model is told never to do arithmetic
itself. After that it answered correctly (C$1,456m severe versus C$602m baseline).

**Details:**
- General principle: **do deterministic work in code, use the LLM for language.** Sums, ratios
  and lookups belong in Python.
- A second kind of error: the numbers are real but the **labels** are wrong (for example calling
  the C$52.2m weighted allowance the "baseline" figure). Clear table titles and instructions
  reduce it, and the app tells users to check answers against the tables.
- Larger models make these mistakes much less often, but outputs still need checking. In a bank,
  model outputs need validation, and LLM outputs are no different.

---

## 4. The chat app versus the agent

**Short answer:**
In the chat app, the code controls every step: it gathers the data, calls the model once and
shows the answer. In the agent, the model controls the steps: it decides which tools to use, runs
them, looks at the results and repeats until done. The app can only explain existing results; the
agent can run new scenarios.

| | Chat app | Agent |
|---|---|---|
| Who decides the steps | The code | The model |
| Model calls per question | One | Many (a loop) |
| Tools | None | Shell commands, read and write files |
| What it can do | Explain the shipped results | Write a config, run the model, compare results |
| Example | "What's the total allowance?" | "What if severe gets 40% weight?" → ran it: C$399m → C$416m |

- An agent has three parts: **instructions, tools and a loop**. The agent here is defined in one
  Markdown file (`.claude/agents/ifrs9-analyst.md`); Claude Code provides the loop and the tools.
  See [`docs/AGENT_GUIDE.md`](../docs/AGENT_GUIDE.md).
- The instructions include **guardrails** (never edit the shipped config or results; don't run the
  slow backtest unless asked) and **domain knowledge** (credit cards are calibrated to the
  reported allowance, so their weighted ECL doesn't move when weights change).
- The example run surfaced a real model behaviour: changing the weights also re-calibrates some
  parameters, so the severe scenario itself became milder. A pure re-weighting would have
  overstated the impact.

---

## 5. Why the agent uses a hosted model and the chat app a local one

**Short answer:**
Speed and reliability. An agent calls the model many times per question, and each call re-reads a
growing conversation; on a laptop CPU that turns a 40-second answer into many minutes. Small models
are also unreliable at what agents need most: calling tools with correct arguments, reading tool
output and recovering from errors. So the agent uses a strong hosted model and the single-call
chat uses a local one.

**Details:**
- The example agent run took 6 tool steps. At ~40 tokens per second, a local 4B model would need
  roughly 5–15 minutes.
- Typical small-model failures: malformed tool calls, wrong arguments (such as weights that don't
  add up to 1), stopping early or looping, and misreading results.
- Local models still make sense for **privacy** (data never leaves the machine), **cost** (no
  per-token fees) and **simple tasks**, and are much more capable on Apple Silicon or a GPU.
- In one line: **pick the model by the task.** Use a strong model where errors compound over many
  steps; use a small local one where a single answer can be checked.

---

## Adapting it

- **Production:** replace Ollama with a hosted model API (only `llm.py` changes). Keep the API key
  in a secret store, add usage limits, log questions and answers for review, and test answers
  against a set of known questions before each release.
- **Testing an LLM app:** keep a fixed set of questions with known correct answers taken from the
  workbook. Run them after every change to the prompt or model and track how many are right.
- **Temperature:** how random the model's word choice is. The app uses 0.2 for factual answers;
  higher values give more varied text.
- **Next step:** give the chat app a `run_whatif()` tool so it can run scenarios, which turns it
  into an agent, or build the agent loop directly with the Claude Agent SDK.
