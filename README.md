# AI Planning Agent with LangGraph

This project implements a multi-agent orchestration system using **LangGraph**, **LangChain**, and the **Groq API**. It features a collaborative workflow where multiple agents (Planner, Executor, Verifier) work together to achieve complex goals.

## Overview

The system uses a state-driven workflow where a goal is passed through a graph of nodes. Agents iterate on the tasks until the quality meets the required standard.

### Key Technologies
- **LangGraph**: Orchestrates the agentic workflow as a state machine.
- **LangChain**: Provides the building blocks for LLM interaction and tool integration.
- **Groq API**: Powers the reasoning engine using high-performance Llama models (Llama 3.3).
- **DuckDuckGo Search**: Provides real-time web context to the Executor agent.

## Visual Workflow

[![Workflow Diagram](https://mermaid.ink/img/pako:eNptjUFvgkAQhf_KZk6QoBFlBffQhBQPvZimNT1Ueti4I5DCLlkXa6v8964IbU06h5k3ed-8OcFWCQQGu1J9bHOuDVknqSS2ntfx09pxuuG6ZDS6I3XJpUS9ebxOEmcozdsV772OwyNuG6P0ZtmLG3JwO_SAutgVNvOlFzfo4F7Q8wpR7MlDVWt1wMpC55-of-i47jhxJstV4ji2uS54kOlCADO6QQ8q1BW_rHC6BKRgcpubArNScP2eQipbe1Nz-apUNZxp1WQ5sB0v93ZrasENJgXPNP9FUArU96qRBhjtEoCd4Ahs6k_H80UUzmY0oDSMZnMPPoEF0diPwiCkfkgXizkNWg--up-TcRTSyZ_y229QPIpP?type=png)](https://mermaid.live/edit#pako:eNptjUFvgkAQhf_KZk6QoBFlBffQhBQPvZimNT1Ueti4I5DCLlkXa6v8964IbU06h5k3ed-8OcFWCQQGu1J9bHOuDVknqSS2ntfx09pxuuG6ZDS6I3XJpUS9ebxOEmcozdsV772OwyNuG6P0ZtmLG3JwO_SAutgVNvOlFzfo4F7Q8wpR7MlDVWt1wMpC55-of-i47jhxJstV4ji2uS54kOlCADO6QQ8q1BW_rHC6BKRgcpubArNScP2eQipbe1Nz-apUNZxp1WQ5sB0v93ZrasENJgXPNP9FUArU96qRBhjtEoCd4Ahs6k_H80UUzmY0oDSMZnMPPoEF0diPwiCkfkgXizkNWg--up-TcRTSyZ_y229QPIpP)

---

## Setup Instructions

### 1. Prerequisites
- Python 3.9+
- A [Groq API Key](https://console.groq.com/)

### 2. Installation
Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory (based on `.env.example`):

```bash
cp .env.example .env
```

Add your Groq API key to the `.env` file:
```env
GROQ_API_KEY=your_actual_api_key_here
```

---

## Workflow in `main.py`

The execution flow follows a multi-agent loop designed for high-quality output:

1.  **Planner Node**: Breaks the user's goal into up to 5 concrete, actionable tasks.
2.  **Executor Node**: 
    - Executes tasks sequentially using the LLM.
    - Uses **DuckDuckGo Search** to augment prompts with real-time web context.
    - Incorporates previous critique for refinement.
3.  **Verifier Node**:
    - Evaluates results using a weighted rubric (**Completeness**, **Accuracy**, **Clarity**).
    - Assigns a score (0.0 - 1.0).
    - **Quality Gate**: Only approves the work if the score is **0.7 or higher**.
    - Returns a JSON response with the score and detailed critique.
    - Triggers a re-execution loop if the quality is insufficient (capped at 3 iterations).

---

## How to Run

Execute the main script:

```bash
python main.py
```

### Sample Output
The terminal will display the logs from each agent:
- `[Planner]` listing the roadmap.
- `[Execution Agent]` showing task completions and search context.
- `[Verifier]` providing a numeric score and structured feedback.