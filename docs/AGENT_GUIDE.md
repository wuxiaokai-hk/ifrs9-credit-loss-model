# Your first agent: `ifrs9-analyst`

This guide explains the agent we built for this project: what an agent is, how this one works,
how to use it, and how to change it. Read it top to bottom once; later, use it as a reference.

---

## 1. What an agent is

An ordinary script does exactly what you wrote, step by step. An **agent** is an AI model that
is given:

1. **A goal** (your question),
2. **Instructions** (how to behave, what to avoid, what a good answer looks like),
3. **Tools** (here: run shell commands, read and write files),

and then **runs in a loop**: think → use a tool → look at the result → think again → … until the
goal is met. It decides the steps itself. For example, if a run fails, it can read the error
and try again.

```
 your question ──► agent thinks ──► uses a tool (e.g. `creditloss run`) ──► reads the output ─┐
                        ▲                                                                       │
                        └───────────────────────── not done yet ◄──────────────────────────────┘
                                                        │ done
                                                        ▼
                                                  written answer
```

---

## 2. What this agent does

`ifrs9-analyst` answers **what-if questions** about the model by actually running it:

> "What happens to the allowance if we put 40% weight on the severe scenario?"

It copies the shipped config, changes only what the question asks, runs the model (~15 s),
compares the result with the shipped numbers, and writes a short answer with a table,
an explanation and the caveats.

---

## 3. Where the agent lives

The whole agent is **one Markdown file**:

```
.claude/agents/ifrs9-analyst.md
```

Claude Code reads every file in `.claude/agents/` when it starts in this folder. The file has two parts.

**The front matter** (between the `---` lines) is the agent's settings:

| Field | Meaning |
|---|---|
| `name` | How you call it: "use the ifrs9-analyst agent …" |
| `description` | When Claude should hand work to it. Claude reads this to decide on its own, so write it like a job description. |
| `tools` | What it may do. It has `Bash, Read, Write, Glob, Grep`, but no web access, because it doesn't need any. Fewer tools mean a safer, more focused agent. |
| `model` | Which Claude model runs it. `sonnet` is fast and cheap enough for this job. |

**The body** is the agent's instructions (its "system prompt"). It's written like a brief to a
new junior analyst:

- **Ground rules:** what it must never do (edit the shipped config, data or results) and what
  it shouldn't do without being asked (the slow backtest).
- **A recipe:** the five steps from question to answer, including tricky details we found
  while testing (the config must sit directly in `agent_runs/`).
- **Domain knowledge:** for example, that credit cards are calibrated to the reported allowance,
  so their weighted ECL barely moves. Without this, the agent might "explain" a result wrongly.
- **An answer format:** so every answer has the same shape.

> **Key idea:** the model is already smart. The instructions add what it can't know on its own:
> your project's rules, its pitfalls, and what a good answer looks like.

---

## 4. How to use it

1. Open this folder (`~/Downloads/ifrs9-credit-loss-model`) in Claude Code: in the desktop app,
   start a new session and choose this folder.
2. Ask, for example:
   - *"Use the ifrs9-analyst agent: what if unemployment in the severe scenario peaks 1 point higher?"*
   - *"Use the ifrs9-analyst agent: what is the allowance with 100% baseline weight?"*
   - *"Ask ifrs9-analyst why the uninsured mortgage ECL is so sensitive to the severe scenario."*
3. Claude hands the question to the agent and it works on its own. You see its answer, and the
   files it made are in `agent_runs/` (ignored by git).

You can also just ask a what-if question without naming it. Because of the `description`,
Claude will often pick the agent by itself.

---

## 5. Exercises: learn by changing it

Do these in order. After each change, start a new session so the file is re-read, then ask a question.

1. **Change the answer format.** In the body, add a line to step 5: *"End with one sentence a
   non-specialist would understand."* Ask a question and see the difference.
2. **Add a rule.** Add to the ground rules: *"If the user asks for weights that don't add up
   to 1, ask before running."* Then try asking for 50% severe and 60% baseline.
3. **Give it a new skill.** Add a section telling it how to answer *"show me the chart"*: the
   charts are the `fig*.png` files in the output folder, and it can open them with `Read`.
4. **Remove a tool.** Delete `Write` from `tools:` and ask a what-if question. Watch how it
   struggles or finds another way, then put `Write` back. This shows why tools matter.
5. **Make a second agent.** Copy the file to `ifrs9-reviewer.md`, change the name and
   description, and write instructions for a *model-risk reviewer* that reads
   `docs/MODEL_CARD.md` and the backtest sheets and lists the model's weaknesses. Give it only
   `Read, Glob, Grep` so it can't run or change anything.

---

## 6. Where to go next

| Level | What | What you learn |
|---|---|---|
| Now | **Subagents** (this file) | Instructions, tools, delegation |
| Next | **Skills**: a folder with instructions plus scripts that any session can load (`.claude/skills/`) | Reusable know-how with code included |
| Then | **The Claude Agent SDK** in Python: the same loop, but you write the tools as Python functions and run the agent from your own script (needs an Anthropic API key) | How agents work under the hood; building agents into your own apps, for example a "Ask the model" box in the Streamlit app |
