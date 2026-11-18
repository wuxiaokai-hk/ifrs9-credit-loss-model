# Interview notes: GenAI app and agent

Each answer has a **short version** (say it first, ~20 seconds) and **details** (use them if
the interviewer asks more). Use your own words; these are notes, not a script.

---

## 1. What is grounding (RAG), and why does the model need your results in the prompt?

**Short version:**
A language model only knows what it learned in training. It has never seen my model's results.
So before every question, my app reads the key tables from the results workbook, turns them
into text, and puts them in the prompt. The model answers from those numbers instead of from
memory. That's called grounding.

**Details:**
- In my app, `context.py` builds about 400 words of context: the allowance by stage, the
  allowance under each scenario, cumulative losses, coverage ratios and a few notes.
- **Grounding** means giving the model the facts in the prompt. **RAG** (retrieval-augmented
  generation) is grounding where you first *search* a larger collection, such as documents or
  a database, for the relevant pieces. My results were small enough to send all of them, so I
  didn't need the search step.
- Without grounding, the model would either refuse or, worse, invent plausible-looking numbers.
- The instructions also say "use only these numbers; if it's not there, say so", to stop it
  filling gaps from general knowledge.

---

## 2. Why does the chat get slower as the conversation grows?

**Short version:**
The model has no memory between calls. To make it look like a conversation, the app sends the
whole history (instructions, context, every earlier question and answer) with each new
question. So each question means more text to read, which takes longer and, with a paid model,
costs more tokens.

**Details:**
- Each request contains the system prompt (instructions + context), then all previous turns,
  then the new question.
- On my Intel Mac the local model reads about 40 tokens per second, so every extra 400 words
  of history adds roughly 10 seconds.
- Common fixes: limit the history to the last few turns, summarise older turns, or keep the
  context as short as possible (which is why I sent compact tables, not the whole workbook).
- Models also have a **context window**, a maximum amount of text per request. Long chats
  eventually hit it.

---

## 3. Why did the model get totals wrong, and how did you fix it?

**Short version:**
Language models predict likely text; they don't calculate. When I asked for total losses, the
small model took the credit-card row as the total and made up a subtraction. I fixed it by
computing totals in code and adding a "TOTAL" row to every table, and by telling the model
never to do arithmetic itself. After that it answered correctly: C$1,456m severe versus
C$602m baseline.

**Details:**
- General principle: **do deterministic work in code, use the LLM for language.** Sums,
  ratios and lookups belong in Python.
- I also saw a second kind of error: the numbers were real but the **labels** were wrong (it
  called the C$52.2m weighted allowance the "baseline" figure). That's harder to fix with
  code. I reduced it with clearer table titles and instructions, and the app tells users to
  check answers against the tables.
- Larger models make these mistakes much less often, but you still verify them. In a bank,
  model outputs need validation, and LLM outputs are no different.

---

## 4. What's the difference between your GenAI app and your agent?

**Short version:**
In the GenAI app, my code controls every step: it gathers the data, calls the model once and
shows the answer. In the agent, the model controls the steps: it decides which tools to use,
runs them, looks at the results and repeats until the job is done. The app can only explain
existing results; the agent can run new scenarios.

**Details:**

| | GenAI app | Agent |
|---|---|---|
| Who decides the steps | My code | The model |
| Model calls per question | One | Many (a loop) |
| Tools | None | Shell commands, read and write files |
| What it can do | Explain the shipped results | Write a config, run the model, compare results |
| Example | "What's the total allowance?" | "What if severe gets 40% weight?" → ran it: C$399m → C$416m |

- An agent has three parts: **instructions, tools and a loop**. My agent is defined in one
  Markdown file (`.claude/agents/ifrs9-analyst.md`); Claude Code provides the loop and the tools.
- The instructions include **guardrails** (never edit the shipped config or results; don't run
  the slow backtest unless asked) and **domain knowledge** (credit cards are calibrated to
  the reported allowance, so their weighted ECL doesn't move when weights change).
- Its answer surfaced a real model behaviour: changing the weights also re-calibrates some
  parameters, so the severe scenario itself became milder. A pure re-weighting would have
  overstated the impact.

---

## 5. Why not use the local model for the agent?

**Short version:**
Two reasons: speed and reliability. An agent calls the model many times per question, and each
call re-reads a growing conversation. On my laptop that would turn a 40-second answer into many
minutes. And small models are unreliable at what agents need most: calling tools with correct
arguments, reading tool output, and recovering from errors. So I use a strong hosted model
(Claude) for the agent and a local model for the simple single-call chat.

**Details:**
- My agent test took 6 tool steps. At ~40 tokens per second for reading, a local 4B model
  would need roughly 5–15 minutes.
- Typical small-model failures: malformed tool calls, wrong arguments (such as weights that
  don't add up to 1), stopping early or looping, and misreading results. My local chat app
  already mislabelled numbers in a single step.
- Local models still make sense for **privacy** (data never leaves the machine), **cost**
  (no per-token fees) and **simple tasks**. They're much more capable on better hardware
  (Apple Silicon or a GPU).
- The trade-off in one line: **pick the model by the task.** Use a strong model where errors
  compound over many steps; use a small local one where a single answer can be checked.

---

## Likely follow-up questions

- **"How would you put the chat app in production?"**
  Replace Ollama with a hosted model API (only `llm.py` changes). Keep the API key in a
  secret store, add usage limits, log questions and answers for review, and test answers
  against a set of known questions before each release.
- **"How would you test an LLM app?"**
  Use a fixed set of questions with known correct answers taken from the workbook. Run them
  after every change to the prompt or model, and track how many are right. I did this by
  hand: the totals question failed, I fixed the context, and it passed.
- **"What's temperature?"**
  How random the model's word choice is. I use 0.2 for factual answers; higher values give
  more varied, creative text.
- **"What would you do next?"**
  Give the chat app a `run_whatif()` tool so it can run scenarios, which turns it into an
  agent. Or build the agent loop myself with the Claude Agent SDK.
