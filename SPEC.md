# AI Software Product Factory: Spec for an Agentic Harness

## Prologue: Echoes of Traditional Modular Design
This spec reads like a blueprint for building a large-scale application in C++ (e.g., via abstract base classes and PIMPL idiom) or C#/.NET (e.g., via interfaces and dependency injection). Here's how it maps:

| Aspect in Spec                  | Traditional C++/C# Equivalent                                                                 | Why It Feels Familiar |
|---------------------------------|------------------------------------------------------------------------------------------------|-----------------------|
| **Interface Contracts** (e.g., inputs/outputs/errors/effects/modes) | C++ abstract classes or header files defining pure virtual methods; C# interfaces (e.g., `IEnumerable<T>`) with explicit method signatures, exceptions, and async patterns. | Every class/module declares a public contract upfront. Implementations hide details (e.g., private members), and consumers only care about the "what," not the "how." Omitting error handling? That's a contract violation, just like undefined behavior in C++. |
| **Black-Box Reuse** (treat modules as closed-source services) | Encapsulation in OOP: Use `include` or `using` directives to link against compiled libraries (.dll/.so) without peeking inside. Dependency injection (e.g., via .NET's IServiceCollection) wires them via contracts. | You don't "read" another assembly's code—you trust its ABI (application binary interface) or contract. Reuse via NuGet packages or vcpkg mirrors the Code Index. |
| **Micro-Module Decomposition** | Single Responsibility Principle (SRP) and microservices: Break into tiny, testable units (e.g., a C# class for "UserValidator" or C++ functor for "ParseJson"). Hierarchical folders like `src/components/ui/button/` align with project structures. | Tasks/stories/epics feel like user stories in Agile, but the leaf-level "micro modules" are like atomic funcs/classes that you unit-test in isolation. |
| **Immutable Deliverables** | Versioned libraries (e.g., semantic versioning in .NET packages) or const-correctness in C++. Once shipped, you don't mutate— you fork or supersede. | Prevents "hotfix hell"; changes propagate via new builds, not in-place edits. |
| **Dependency Management** | `#include` guards or .csproj references: Only expose what you need, resolve via build tools (MSBuild/CMake). Circular deps? Build fails. | The state machine's `depends_on` array is like a DAG in your build graph—dispatch only when upstream is ready. |

In short, yes: This is basically "OOP as code" but scaled to an AI workflow. A C# app with interfaces for every service (e.g., `IUserRepository`) and black-box mocks for testing enforces the same "contract-only handoffs." The spec just makes it *explicitly auditable* across agent boundaries, like if your IDE enforced "no peeking at implementation" via access modifiers on steroids.

### Key Differences: Why It's More Than Just "Fancy OOP"
While the analogy is apt, this harness isn't *just* describing a conventional app—it's architecting an *AI factory* to *produce* apps reliably, where humans aren't always in the loop. Traditional dev relies on human discipline (e.g., "don't violate SRP"), but here it's enforced via process invariants to combat AI hallucinations or drift. Some standout twists:

- **Agentic Orchestration vs. Human-Driven Builds**: In C++/C#, your IDE/build tool dispatches compilation sequentially (or in parallel via MSBuild tasks). Here, the Project Loop is an orchestrator dispatching *one task at a time* to AI agents, with a 3-agent debate per micro-task. It's like if every `gcc` invocation required a code review debate before linking—rigorous, but slow by design (non-goal: speed).

- **Fresh Contexts & No Memory**: Humans carry context across modules (e.g., "I remember how that class works"). Agents get zero history—only `{task_id}.md` and contracts. This prevents "conversational anchoring" (AI bias from prior chats), akin to compiling each .cpp in isolation, but stricter than even modular builds.

- **Halt-on-Failure & Escalation**: In .NET/C++, a failed test might just break the build—you fix and retry. Here, debate failure *halts the entire project* for human intervention, producing an "escalation artifact" like a bug report with debate transcripts. It's pessimistic: Better to stop and debug than ship subtly wrong code.

- **Debate Mechanism**: No direct analog—closest is pair programming or PR reviews, but mandatory *Propose → Challenge → Adjudicate* with binary PASS/FAIL is like adversarial testing baked into every commit. Challenger must "falsify" the proposal, echoing TDD but agent-ified.

- **Code Index as Institutional Memory**: Like a corporate NuGet feed, but append-only and queryable for reuse reports. Traditional apps have this via package managers, but the spec mandates justifying *why not reuse* before creating new modules—enforcing DRY at the factory level.

Overall, it's like taking the best of SOLID principles, interface segregation, and CI/CD pipelines, then wrapping them in an AI guardrail system to make software production *deterministic* despite flaky agents. If you built a C# app this way, it'd be over-engineered for a solo dev but gold for a massive, distributed team (or AI swarm).

## 1) Intent

Define an **AI Software Product Factory** — an agentic harness that produces reliable software through three orchestrated loops:

1. **Product Loop** — researches and refines a human-authored spec into a complete, structured product specification.
2. **Project Loop** — decomposes the spec into a hierarchical project folder, dispatches tasks one at a time, and drives execution to completion.
3. **Engineering Loop** — implements a single micro task in a fresh context with a mandatory 3-agent debate; **halts the entire project on failure** so a human can intervene.

The system enforces:

1. **Context discipline**: fresh contexts + contract-only handoffs; zero conversational memory; all external code is treated as closed-source.
2. **Mandatory debate**: *proposal → challenge → adjudication* for every engineering contribution.
3. **Micro‑micro decomposition**: build in tiny, independently verifiable micro modules.
4. **Immutable, shipped deliverables**: "working" is proven with evidence, then frozen.
5. **Black‑box reuse**: other coding agents integrate by contract, never by reading code.
6. **Institutional reuse**: a **Code Index** of all shipped micro modules is the default starting point for new work.

This system prioritizes **correctness, composability, reuse, and traceability** over speed.

### Implementation Approach

The factory is realized as a **Hybrid**: custom LangGraph `StateGraph` instances wrapping deepagents SDK components. This is not building from scratch — roughly 60% of the infrastructure is reused from LangGraph and the deepagents SDK (file tools, middleware, checkpointing, `create_agent`). Nor is it a thin customization of the deepagents SDK, whose `create_deep_agent()` is a single-agent pattern incapable of modeling multi-graph orchestration. The correct framing is: **LangGraph orchestration layer wrapping deepagents components.**

* **LLM-free orchestration**: all control flow between and within the three loops (dispatch, routing, retry counting, halt propagation) is deterministic Python logic. LLMs operate only inside agent nodes where actual work happens — planning, implementation, and debate.

---

## 2) Non-goals

* No attempt to minimize number of agent invocations; this harness is intentionally rigorous.

---

## 3) Factory architecture: three loops

### 3.1 Overview

```
Human writes initial SPEC.md (high-level)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  OUTER GRAPH (Entrypoint StateGraph)                      │
│                                                           │
│  ┌─────────────────────────────────────────┐              │
│  │  PRODUCT LOOP                           │              │
│  │  (create_agent with research tools)     │              │
│  │  Researches, expands, structures spec.  │              │
│  │  Output: complete structured spec.      │              │
│  └──────────────────┬──────────────────────┘              │
│                     │ Human approves spec                  │
│                     ▼                                      │
│  ┌─────────────────────────────────────────┐              │
│  │  PROJECT LOOP  (Custom StateGraph)      │              │
│  │  Nodes: init_state_machine → dispatch   │              │
│  │         → engineering → update_state    │              │
│  │  Dispatch is pure Python (no LLM).      │              │
│  │  Loops until all tasks complete or HALT.│              │
│  │                                         │              │
│  │  ┌───────────────────────────────────┐  │              │
│  │  │ engineering (wrapper node)        │  │              │
│  │  │ Invokes compiled subgraph:       │  │              │
│  │  │                                   │  │              │
│  │  │ ┌─────────────────────────────┐   │  │              │
│  │  │ │ ENGINEERING LOOP            │   │  │              │
│  │  │ │ (Custom StateGraph)         │   │  │              │
│  │  │ │ micro_plan → per-module     │   │  │              │
│  │  │ │ iteration → Debate → ship   │   │  │              │
│  │  │ │                             │   │  │              │
│  │  │ │ Debate Subgraph (StateGraph)│   │  │              │
│  │  │ │ propose → challenge → route │   │  │              │
│  │  │ │ → (revise | adjudicate)     │   │  │              │
│  │  │ │ → (ship | halt)            │   │  │              │
│  │  │ └─────────────────────────────┘   │  │              │
│  │  └───────────────────────────────────┘  │              │
│  └─────────────────────────────────────────┘              │
└───────────────────────────────────────────────────────────┘
```

### 3.2 Product Loop

**Trigger:** Human creates an initial `SPEC.md` at a high level (intent, constraints, scope).

**Responsibility:** An agent cycle researches, expands, and structures the spec into a complete product specification with pillars, epics, stories, and tasks. The Product Loop may use web research, domain knowledge, and iterative refinement.

**Output:** A structured spec ready for human approval, containing:

* Pillars (strategic themes)
* Epics (major capabilities per pillar)
* Stories (user-facing behaviors per epic)
* Tasks (implementable units per story), each with subtasks and acceptance criteria

**Gate:** Human reviews and approves (or rejects/amends) the complete spec before the Project Loop begins.

#### 3.2.1 Agent Tool Inventory

The Product Loop agent receives the following tools. Tools marked **(middleware)** are provided by the deepagents middleware stack. Tools marked **(custom)** are purpose-built for the Product Loop. Tools marked **(shared)** are reused from other factory subsystems.

| Tool | Source | Purpose | Inputs | Outputs | When Used |
|---|---|---|---|---|---|
| `web_search` | Custom | Search the web for domain knowledge, competitive analysis, technical feasibility, and industry standards. | `query: str` — natural-language search query; `max_results: int` (default 10) | `list[SearchResult]` — ranked list of `{url: str, title: str, snippet: str}` objects | During research phases: expanding pillars with domain context, validating epic feasibility, identifying edge cases for stories, sourcing industry standards for acceptance criteria. |
| `read_file` | Middleware (`FilesystemMiddleware`) | Read files from the project workspace (initial SPEC.md, reference documents, current draft). | `path: str` — relative file path | `content: str` — file contents | At loop start to read the human-authored `SPEC.md`; during iteration to re-read the current draft or referenced documents. |
| `write_file` | Middleware (`FilesystemMiddleware`) | Write or overwrite files in the project workspace. | `path: str`, `content: str` | `success: bool` | To persist the structured spec draft after each refinement iteration. |
| `validate_spec` | Custom | Validate the current spec draft against the `ProductSpec` schema (§3.2.2) and the completeness criteria (§3.2.3). Returns a deficiency report. | `spec_path: str` — path to the current spec draft JSON | `ValidationReport` — `{errors: list[Deficiency], warnings: list[Deficiency]}` where each `Deficiency` is `{path: str, field: str, severity: "ERROR" \| "WARNING", message: str}` | After each refinement iteration to check whether the spec meets termination criteria. The agent uses the report to guide further research and refinement. |
| `search_code_index` | Shared (§14) | Query the Code Index for existing shipped modules relevant to the product scope. | `query: str` — natural-language description of capability needed | `list[CodeIndexEntry]` — matching entries with contract summaries and I/O shapes | When expanding epics and stories, to identify existing shipped modules that can be reused. Informs task decomposition and `depends_on` planning. |
| `emit_structured_spec` | Custom | Produce the final structured spec as a validated JSON artifact conforming to the `ProductSpec` schema (§3.2.2). Validates against all completeness criteria before accepting. | `spec: ProductSpec` — the complete structured spec object | `artifact_ref: str` — reference to the persisted spec artifact; or raises `ValidationError` with deficiency list if validation fails | Called once when the agent determines the spec is complete. If validation fails, returns the deficiency list and the agent must continue iterating. This is a **terminal tool** — successful invocation signals the agent is done. |

#### 3.2.2 Structured Spec Output Schema

The Product Loop must produce a structured spec conforming to the `ProductSpec` schema below. This is a Pydantic-validated model — the `emit_structured_spec` tool rejects specs that fail validation.

**`ProductSpec` schema:**

```json
{
  "spec_id": "string (required — unique identifier, format: SPEC-NNN)",
  "spec_version": "string (required — semver, e.g., '1.0.0')",
  "title": "string (required — product title)",
  "description": "string (required — high-level product description)",
  "created_at": "ISO-8601 (required)",
  "updated_at": "ISO-8601 (required)",
  "pillars": [
    {
      "pillar_id": "string (required — format: PIL-NNN)",
      "name": "string (required)",
      "description": "string (required — strategic theme this pillar addresses)",
      "rationale": "string (required — why this pillar exists and what gap it fills)",
      "epics": [
        {
          "epic_id": "string (required — format: EPC-NNN)",
          "name": "string (required)",
          "description": "string (required — major capability this epic delivers)",
          "success_criteria": ["string (required — at least 1, measurable outcomes)"],
          "stories": [
            {
              "story_id": "string (required — format: STR-NNN)",
              "name": "string (required)",
              "description": "string (required)",
              "user_facing_behavior": "string (required — observable behavior from the user's perspective)",
              "tasks": [
                {
                  "task_id": "string (required — format: TSK-NNN, globally unique across entire spec)",
                  "name": "string (required)",
                  "description": "string (required — what this task implements)",
                  "subtasks": ["string (required — at least 2 subtasks)"],
                  "acceptance_criteria": ["string (required — at least 2, testable conditions)"],
                  "depends_on": ["task_id (optional — references to other TSK-NNN IDs within this spec)"],
                  "io_contract_sketch": {
                    "inputs": "string (required — preliminary input types and constraints)",
                    "outputs": "string (required — preliminary output types and invariants)",
                    "error_surfaces": "string (required — anticipated error conditions and failure modes)",
                    "effects": "string (required — expected side effects: writes, calls, mutations)",
                    "modes": "string (required — sync, async, or both, with rationale)"
                  }
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

**Field requirements by hierarchy level:**

| Level | Required Fields | Optional Fields | Minimum Children |
|---|---|---|---|
| **ProductSpec** | `spec_id`, `spec_version`, `title`, `description`, `created_at`, `updated_at`, `pillars` | — | 1 pillar |
| **Pillar** | `pillar_id`, `name`, `description`, `rationale`, `epics` | — | 1 epic |
| **Epic** | `epic_id`, `name`, `description`, `success_criteria`, `stories` | — | 1 story; 1 success criterion |
| **Story** | `story_id`, `name`, `description`, `user_facing_behavior`, `tasks` | — | 1 task |
| **Task** | `task_id`, `name`, `description`, `subtasks`, `acceptance_criteria`, `io_contract_sketch` | `depends_on` | 2 subtasks; 2 acceptance criteria |
| **I/O Contract Sketch** | `inputs`, `outputs`, `error_surfaces`, `effects`, `modes` | — | All five dimensions must be non-empty |

**Note on `io_contract_sketch`:** This is a preliminary sketch, not the final micro module contract (§11.1). The Engineering Loop's `micro_plan` node (§10.1) refines these sketches into full `MicroModuleContract` Pydantic models with typed fields and formal validation. The sketch must be detailed enough that an engineer can understand the task's scope and integration points, but it uses prose descriptions rather than typed schemas. All five I/O contract dimensions (§5.5) must be addressed — no dimension may be left empty, placeholder ("TBD"), or marked not applicable.

**Note on `depends_on`:** Task dependencies define execution order in the Project Loop's state machine (§4.2). The dependency graph must be a directed acyclic graph (DAG). Circular dependencies are a validation error. A task with no dependencies is eligible for immediate dispatch.

#### 3.2.3 Completeness Criteria (Termination Condition)

The Product Loop agent iterates until the spec satisfies **all** of the following criteria. The `validate_spec` tool checks these programmatically. The agent must resolve all `ERROR`-level deficiencies before calling `emit_structured_spec`.

**Structural completeness (ERROR if violated — blocks emission):**

1. Every pillar has at least 1 epic.
2. Every epic has at least 1 story and at least 1 success criterion.
3. Every story has at least 1 task.
4. Every task has at least 2 subtasks.
5. Every task has at least 2 acceptance criteria.
6. Every task's `io_contract_sketch` addresses all five I/O contract dimensions (§5.5): `inputs`, `outputs`, `error_surfaces`, `effects`, `modes`. No dimension may be empty or contain placeholder text (e.g., "TBD", "N/A", "TODO").
7. All `task_id` values are globally unique across the entire spec.
8. All `depends_on` references resolve to existing `task_id` values within the spec.
9. The task dependency graph is a DAG — no circular dependencies.
10. All required fields in §3.2.2 are present and non-empty at every hierarchy level.

**Content quality (WARNING if violated — surfaced to human but does not block emission):**

11. Every `description` field is at least 20 characters (prevents stub descriptions).
12. Acceptance criteria are phrased as testable conditions (contain a verb indicating observable behavior: "returns", "displays", "raises", "writes", "emits", "rejects", "validates", etc.).
13. Subtasks within the same task are semantically distinct (no near-duplicates).
14. `io_contract_sketch.error_surfaces` identifies at least one specific error condition per task (not just "errors may occur").

**Termination rule:** The agent may call `emit_structured_spec` only after `validate_spec` returns zero `ERROR`-level deficiencies. `WARNING`-level deficiencies are included in the output artifact and surfaced to the human at the approval gate (§3.2.4) but do not block emission.

#### 3.2.4 Human Approval Gate

After the Product Loop agent emits the structured spec, the Outer Graph pauses for human review. The human is presented with:

1. The complete structured spec (rendered as readable markdown).
2. Any `WARNING`-level deficiencies from the final `validate_spec` run.
3. A summary of research sources consulted (URLs and queries from `web_search` invocations).

The human selects one of three actions:

| Action | Payload | Behavior | Effect on Agent State |
|---|---|---|---|
| **APPROVE** | None | The spec is accepted as-is. The Outer Graph advances to the Project Loop (§3.3). | Product Loop terminates. The approved `ProductSpec` JSON is passed as input to the Project Loop's `init_state_machine` node. |
| **REJECT** | `reasons: str` (required — explanation of why the spec is unacceptable) | The spec is discarded. The Product Loop restarts from scratch with the original `SPEC.md` as input. | Agent receives a **fresh context** containing the original human-authored `SPEC.md` plus the rejection reasons appended as a `## Rejection Feedback` section. The prior draft is **not** carried forward — this is a clean restart. Prior research and iteration history are discarded. |
| **AMEND** | `feedback: str` (required — specific changes or additions requested) | The spec is not yet acceptable but has a viable foundation. The Product Loop continues iterating from the current draft. | The amendment feedback is appended to the agent's context as a `HumanMessage`. The agent must address every point in the feedback before re-emitting. The `validate_spec` tool re-runs after amendments are incorporated. |

**Gate implementation:** The approval gate uses LangGraph `interrupt()` to pause the Outer Graph and surface the spec to the human operator. The human's response (action + payload) is injected as a `Command` that routes accordingly:

* **APPROVE** → routes to the Project Loop node.
* **REJECT** → restarts the Product Loop node with reset state (fresh context + rejection reasons).
* **AMEND** → resumes the Product Loop node with the amendment feedback appended.

**Bounded amendment cycles:** To prevent infinite amendment loops, the Outer Graph tracks the count of consecutive AMEND responses. After **3 consecutive AMENDs** without an APPROVE, the system surfaces an advisory warning to the human suggesting either APPROVE with known limitations or REJECT to restart with revised initial requirements. This is advisory only — the human may continue amending beyond this threshold.

#### Implementation Binding

The Product Loop is realized as a `create_agent` invocation — not a full `StateGraph` — because its behavior is a single-agent research-and-refine cycle rather than a multi-node workflow.

* **Agent toolset:** The agent receives the tools enumerated in §3.2.1. `read_file` and `write_file` are provided by `FilesystemMiddleware`. `web_search`, `validate_spec`, and `emit_structured_spec` are custom tool implementations registered on the agent. `search_code_index` is the shared Code Index query tool (§14.3).
* **Middleware:** `SummarizationMiddleware` for context window management during extended research sessions; `TodoListMiddleware` for the agent's internal planning.
* **Custom tool details:**
  * `web_search` wraps an HTTP search API (e.g., Tavily, SerpAPI, or equivalent) and returns structured `SearchResult` objects. The tool enforces a maximum of 10 results per query to limit context window consumption.
  * `validate_spec` deserializes the spec draft from disk, validates it against the `ProductSpec` Pydantic model, and runs the completeness checks from §3.2.3. Returns a `ValidationReport` Pydantic model.
  * `emit_structured_spec` validates the spec one final time against the `ProductSpec` schema and all completeness criteria, then persists it as a JSON artifact via `FilesystemBackend`. Returns the artifact reference on success; raises `ValidationError` with the full deficiency list on failure.
* **Output persistence:** The structured spec is persisted in two forms: `state_store/product/spec.json` (machine-readable `ProductSpec` JSON, consumed by the Project Loop's `init_state_machine` node) and `state_store/product/spec.md` (human-readable markdown rendering, presented at the approval gate).
* The human-approval gate sits in the Outer Graph between the Product Loop node and the Project Loop node, implemented via LangGraph `interrupt()`. The Outer Graph state tracks the AMEND count for the bounded amendment cycle advisory (§3.2.4).

### 3.3 Project Loop

**Trigger:** Human-approved spec.

**Responsibility:** The orchestrator that converts the spec into the project folder structure (see §4) and drives execution.

#### Task ID Scheme

Every task in the project receives a canonical task ID with the following format:

```
T-{pillar_slug}-{epic_slug}-{story_slug}-{seq}
```

**Slug generation rules:**

* Derived from the human-readable name by: lowercasing, replacing whitespace and non-alphanumeric characters with a single hyphen, collapsing consecutive hyphens, and trimming leading/trailing hyphens.
* Permitted character set: `[a-z0-9-]` only.
* Maximum slug length: 24 characters. When truncation is necessary, truncate at the last hyphen-delimited word boundary that fits within the limit (do not split mid-word).
* Format: kebab-case (e.g., `user-authentication`, not `user_authentication` or `UserAuthentication`).

**Sequential number (`{seq}`):**

* Three-digit zero-padded integer starting at `001` within each story scope (i.e., unique per pillar+epic+story combination).
* Assigned in declaration order — the order tasks appear in the approved spec within their parent story.
* If a story contains more than 999 tasks, this is a spec defect (the story must be decomposed further before the Project Loop begins).

**Examples:**

* `T-core-auth-login-001` — first task in the "Login" story under the "Auth" epic in the "Core" pillar.
* `T-core-auth-login-002` — second task in the same story.
* `T-platform-deploy-ci-pipeline-001` — first task in the "CI Pipeline" story under the "Deploy" epic in the "Platform" pillar.

**Uniqueness guarantee:** The full task ID must be globally unique across the entire project. The `init_state_machine` node must validate uniqueness across all generated task IDs after generation but before writing any files to disk. On collision, `init_state_machine` must fail immediately with a descriptive error identifying both colliding tasks by their original human-readable names and the generated ID that collided. Collisions can occur when distinct names produce identical slugs after truncation (e.g., "User Authentication Flow" and "User Authentication Form" both truncating to `user-authentication`). In a collision scenario, the implementation must append a disambiguating suffix (e.g., `-a`, `-b`) to the shorter slug before the `{seq}` component and re-validate.

**Behavior:**

1. Create the full project folder hierarchy from the approved spec.
2. Generate task IDs for every task using the Task ID Scheme above, validate global uniqueness, and populate every `{task_id}.md` file with its context: pillar, epic, story, task description, subtasks, acceptance criteria, and dependency contracts.
3. Initialize the state machine (see §4.2) with all tasks in `PENDING` status. Assign each task a `declaration_order` integer reflecting its position in a depth-first traversal of the approved spec (pillar → epic → story → task), starting at 0. The `declaration_order` field is immutable after initialization.
4. **Dispatch loop:**
   a. Read the state machine to determine the next eligible task using the Dispatch Algorithm (see below).
   b. Send **only** the `{task_id}.md` file to the Engineering Loop.
   c. Receive the result (SHIPPED or HALTED).
   d. If SHIPPED: update the state machine, register the module in the Code Index, proceed to next task.
   e. If HALTED: execute BLOCKED cascading (see below), then stop dispatching. Surface the escalation artifact for human intervention.
5. Loop until all tasks reach SHIPPED status.

#### Dispatch Algorithm

When the `dispatch` node reads the state machine, it must select the next task **deterministically**. Given the same state machine contents, dispatch must always produce the same selection.

**Algorithm:**

1. **Filter:** Collect all tasks where `status == PENDING` and every entry in `depends_on` has `status == SHIPPED`.
2. **Sort:** Order the eligible tasks by `declaration_order` ascending. Because `declaration_order` is assigned via depth-first spec traversal (pillar → epic → story → task), this naturally prioritizes: earlier pillars first, then earlier epics within a pillar, then earlier stories within an epic, then earlier tasks within a story.
3. **Select:** Pick the first task in the sorted list. If the filtered list is empty (all tasks are complete, blocked, or have unsatisfied dependencies), dispatch returns no task and the loop evaluates its termination condition.

**Determinism invariant:** The dispatch algorithm is a pure function of the state machine JSON. It uses no randomness, no timestamps, and no external state. Two invocations with identical state machine contents must return the same task ID (or both return no task).

**State machine requirement:** Each task entry in the state machine JSON (§4.2) must include a `declaration_order` field (non-negative integer) populated by `init_state_machine`. This field is immutable after initialization and must not be modified by any subsequent state machine operation.

#### BLOCKED Cascading

When a task is marked `HALTED`, all tasks that depend on it — directly or transitively — must be marked `BLOCKED`. When a halted task is resolved, blocked dependents must be re-evaluated. Both operations are deterministic graph traversals with no LLM involvement.

**On HALT of task X (forward cascade):**

1. Initialize a queue with all tasks that list X in their `depends_on` array (direct dependents of X).
2. Initialize a visited set containing X.
3. For each task Y dequeued:
   a. Add Y to the visited set.
   b. If Y's status is `PENDING`, set Y's status to `BLOCKED`.
   c. If Y's status is `BLOCKED`, `SHIPPED`, `IN_PROGRESS`, or `HALTED`, do not change it.
   d. Enqueue all tasks that list Y in their `depends_on` array, provided they are not already in the visited set.
4. Continue until the queue is empty. This is a breadth-first traversal of the dependency graph forward from X.

**On resolution of task X (HALTED → PENDING for re-execution, or HALTED → SHIPPED if resolved externally):**

1. Collect all tasks with status `BLOCKED` into a candidate set.
2. For each task Y in the candidate set: check every entry in Y's `depends_on`. If **any** direct dependency of Y has status `HALTED`, Y remains `BLOCKED`.
3. If **none** of Y's direct dependencies have status `HALTED`, revert Y to `PENDING`.
4. Repeat steps 1–3 until a full pass produces no status changes (fixed-point convergence). This is necessary because reverting Y to `PENDING` does not itself unblock Y's dependents — but it removes Y from the `BLOCKED` set, and Y's dependents may have had Y as their only blocking cause indirectly (via a chain where a deeper ancestor was `HALTED`). The fixed-point loop re-evaluates the transitive closure.

**Invariant:** After every state machine mutation, a task's status is `BLOCKED` if and only if at least one task in its transitive dependency set has status `HALTED`. The cascading algorithms (forward cascade on HALT, fixed-point re-evaluation on resolution) must preserve this invariant.

#### Implementation Binding

The Project Loop is a custom `StateGraph` with four nodes:

| Node | Responsibility |
|---|---|
| `init_state_machine` | Generate task IDs per the Task ID Scheme (§3.3), validate global uniqueness (fail fast on collision with a descriptive error), populate the state machine from the approved spec with all tasks set to `PENDING`, and assign `declaration_order` to each task via depth-first spec traversal. |
| `dispatch` | Read the state machine, execute the Dispatch Algorithm (§3.3): filter eligible tasks, sort by `declaration_order`, select the first. Pure Python, deterministic. |
| `engineering` | **Wrapper node** — invokes the compiled Engineering Loop subgraph, providing full state isolation between loops. |
| `update_state_machine` | Write back the result (`SHIPPED` or `HALTED`), execute BLOCKED cascading on HALT (§3.3), update the Code Index on ship. |

* **State machine dispatch is pure Python** — no LLM decision-making for orchestration flow. Dependency resolution, dispatch ordering, and BLOCKED cascading are deterministic logic. (Key Decision #3)
* The `engineering` node uses the **wrapper-node pattern** (Key Decision #1): it calls the Engineering Loop as a compiled subgraph, so debate state (proposals, challenges, retry counters) never leaks into the Project Loop's state.
* **Conditional edges** route between `dispatch` (loop) and `END` based on an all-tasks-complete check. If any task is `HALTED`, the loop terminates immediately.
* **Task ID uniqueness** is enforced at initialization time by `init_state_machine`. The node generates all task IDs in a single pass, collects them into a set, and asserts `len(id_set) == len(id_list)`. On failure, it identifies the colliding IDs and their source task names, then raises an error before any files are written to disk.

### 3.4 Engineering Loop

**Trigger:** Receipt of a single `{task_id}.md` from the Project Loop.

**Responsibility:** Implement the micro module described in the task file. This is where the mandatory 3-agent debate occurs.

**Context rule (critical):** The Engineering Loop starts with a **completely fresh context**. It receives:

* The `{task_id}.md` file (which includes subtasks, acceptance criteria, and dependency *contracts*)
* Harness schema definitions — the Pydantic model JSON schemas for all debate artifacts and contracts:
  * `Proposal` (§9.2)
  * `Challenge` (§9.2)
  * `Adjudication` (§9.2)
  * `MicroModuleContract` (§11.1)
  * `MicroPlan` (§10.1)
  * `ShipEvidence` (§11.2)
  * `ReuseSearchReport` (§14.3)
* The decomposition rubric (§10.4) — the criteria the Challenger uses to validate that each micro module is truly "micro"
* Code Index access via the `search_code_index` tool (§14.3) — enabling reuse-first governance during micro-planning and implementation

It does **NOT** receive:

* Any other module's source code
* Any conversation history from prior loops
* Any artifact contents — only the contracts describing how to use them

**Every module not being actively implemented is treated as a closed-source service.** The engineer integrates against contracts and examples, never against implementation details.

**Debate sequence (mandatory):**

1. **Proposer** — implements the module, writes tests, claims it satisfies the contract.
2. **Challenger** — adversarially evaluates. Must be binary: PASS or FAIL. A "mild review" is non-compliant. The Challenger's job is to attempt to falsify correctness and completeness.
3. **Arbiter** — makes the final call: APPROVE (ship), APPROVE_WITH_AMENDMENTS (revise and ship), or REJECT.

**Context Pack construction per debate role:**

Each debate role receives a distinct Context Pack, scoped to enforce the minimum necessary visibility. Context Packs are constructed by the Engineering Loop before each role invocation and enforced at the backend level via `ContextPackBackend` (§6.3).

* **Proposer** receives:
  * The `{task_id}.md` (full task context)
  * The current micro module's contract (from the micro plan, §10.1)
  * Dependency contracts for upstream modules (`CONTRACT_ONLY` — `contract.json`, `examples.md`, `ship.json` only; no implementation source)
  * Code Index search results via the `search_code_index` tool (for reuse discovery)
  * The full harness schema definitions (all Pydantic model JSON schemas listed above)
  * **Access level:** `FULL` for the current module's workspace (implementation directory, tests, local files). `CONTRACT_ONLY` for all other modules.

* **Challenger** receives:
  * The Proposal artifact (the Proposer's claim and deliverable reference — **not** the Proposer's implementation source code)
  * The current micro module's contract (from the micro plan)
  * Dependency contracts for upstream modules (`CONTRACT_ONLY`)
  * The decomposition rubric (§10.4)
  * **Access level:** `CONTRACT_ONLY` for everything. The Challenger never sees implementation source — it evaluates the Proposal's claim against the contract and rubric, not the code itself.

* **Arbiter** receives:
  * The Proposal artifact
  * The Challenge artifact (including verdict, failures, and evidence)
  * The current micro module's contract (from the micro plan)
  * The `{task_id}.md` (full task context, for grounding the final decision)
  * **Access level:** The Arbiter sees both sides of the debate (Proposal + Challenge) and the original task requirements. It does not receive implementation source or Code Index access — its role is adjudication, not implementation review.

**On success:** The module is shipped, the state machine is updated, and a Code Index entry is created.

**On failure:** After bounded retries (1–2 revisions), if the debate still fails, **the Engineering Loop halts the entire project**. An escalation artifact is produced and surfaced for human decision-making.

**Per-module iteration order and failure propagation:**

Modules within a micro plan are executed in **dependency order** — a topological sort of the plan's internal `depends_on` graph (each module's `depends_on` field in the `MicroPlan` schema, §10.1). This ensures that every module's upstream dependencies have been shipped before it is attempted.

If module N fails (HALTED after exhausting debate retries):

1. All modules in the micro plan that **depend on module N** — directly or transitively — are **abandoned**. They are never attempted.
2. Modules that do **not** depend on module N (i.e., they have no path to N in the dependency graph) **may still be attempted**, provided all of their own dependencies have reached SHIPPED status.
3. Once all non-abandoned modules have been attempted (or there are no remaining eligible modules), the Engineering Loop reports **HALTED** for the entire task. The escalation artifact includes the identity of the failed module, the abandoned modules, and any modules that were successfully shipped before the failure.

This partial-execution policy maximizes the useful work produced before halting, while preventing execution of any module whose upstream contract is unresolved.

#### Implementation Binding

The Engineering Loop is a custom `StateGraph` invoked as a **compiled subgraph** from the Project Loop's `engineering` wrapper node. It contains the following nodes:

| Node | Responsibility |
|---|---|
| `micro_plan` | Decompose the incoming `{task_id}.md` into micro modules with explicit I/O contracts. |
| Per-module iteration | For each micro module, run an implementation agent followed by the Debate Subgraph. |
| **Debate Subgraph** | A nested `StateGraph` enforcing the mandatory debate (see below). |
| `ship` | Seal the module, write ship evidence, update the Code Index. |

**Debate Subgraph** is itself a `StateGraph` with this flow:

`propose` → `challenge` → `route` → (`revise` | `adjudicate`) → (`ship` | `halt`)

* **Routing** uses `Command(goto=...)` within nodes (Key Decision #2), keeping retry counter management explicit and co-located with state transitions.
* **On HALT**: the debate node issues `Command(graph=PARENT)` to propagate the halt signal up to the Project Loop, which stops dispatching.
* Each debate role (Proposer, Challenger, Arbiter) uses `model.with_structured_output(Schema)` to produce **Pydantic-validated JSON** (Key Decision #7). This leverages provider-native structured output (Anthropic/OpenAI) rather than post-hoc JSON parsing.

---

## 4) Project folder structure and state machine

### 4.1 Folder hierarchy

The Project Loop creates the following structure from the approved spec:

```
project/
  {pillar_slug}/
    {epic_slug}/
      {story_slug}/
        {task_slug}/
          {task_id}.md
```

All directory names in the hierarchy use **slugified** forms of their human-readable names (see naming conventions below). The `{task_id}.md` filename uses the canonical task ID format, not the slug.

#### Naming conventions (slugification)

Every pillar, epic, story, and task name is converted to a filesystem-safe **slug** before use as a directory name. The slugification algorithm is applied identically at every level of the hierarchy:

1. Convert to lowercase.
2. Replace any character not in `[a-z0-9-]` (including spaces, underscores, dots, and other special characters) with a hyphen.
3. Collapse consecutive hyphens to a single hyphen.
4. Strip leading and trailing hyphens.
5. If the resulting slug exceeds 64 characters, truncate to 56 characters and append a hyphen followed by a 7-character hash suffix (first 7 hex characters of SHA-256 of the full pre-truncation slug), producing a maximum segment length of 64 characters.

**Allowed characters in slugs:** `[a-z0-9-]`

**Examples:**

| Human-readable name | Slug |
|---|---|
| "User Authentication" | `user-authentication` |
| "API v2.0 Integration" | `api-v2-0-integration` |
| "Setup DB & Cache Layer" | `setup-db-cache-layer` |
| "  Leading Spaces  " | `leading-spaces` |
| A 70-character name | Truncated to 56 chars + `-` + 7-char SHA-256 prefix |

#### Task ID format (canonical definition)

The task ID is the globally unique identifier for a task across the entire project. It is canonically defined here; all other sections (including §3.3) reference this definition.

**Format:** `T-{pillar_slug}-{epic_slug}-{story_slug}-{seq}`

* `T-` — fixed prefix identifying this as a task ID.
* `{pillar_slug}` — slugified pillar name.
* `{epic_slug}` — slugified epic name.
* `{story_slug}` — slugified story name.
* `{seq}` — a zero-padded 3-digit sequence number (001, 002, ..., 999) assigned in declaration order within the story.

**Example:** `T-core-infra-auth-service-setup-001`

**Constraints:**

* Maximum total length: 128 characters. If the concatenated ID exceeds 128 characters (due to long slug segments), the `init_state_machine` node must fail with an error identifying the offending task and its computed ID length.
* **Uniqueness:** guaranteed by the combination of the hierarchical path (pillar + epic + story) and the per-story sequence number. The `init_state_machine` node MUST validate uniqueness of all task IDs across the entire project before creating any files. On collision, the node fails immediately with an error identifying both colliding tasks and their computed IDs.

#### Collision handling

Because slugification is lossy (e.g., "Setup DB & Cache" and "Setup DB + Cache" produce the same slug), collisions are possible at each level of the hierarchy.

**Within the same parent directory (same-level collision):**

If two siblings under the same parent produce identical slugs after slugification, the second (and subsequent) occurrences receive a numeric suffix: `-2`, `-3`, etc. For example, if an epic contains two stories that both slugify to `login`, the resulting directories are `login/` and `login-2/`. Suffix assignment follows the declaration order from the approved spec.

**Cross-parent collisions:**

Collisions across different parent directories (e.g., two stories named "Login" under different epics) are naturally resolved by the folder hierarchy — each resides under a different parent path. No additional handling is required.

**Task ID collisions:**

Because the task ID embeds the full hierarchical path (pillar + epic + story slugs), same-level slug deduplication (described above) propagates into the task ID, keeping all task IDs unique. The `init_state_machine` node performs a final uniqueness check across all generated task IDs as a safety net and fails fast on any collision.

**`{task_id}.md` contents:**

Each task file is self-contained and includes everything the Engineering Loop needs. **All fields in the template below are REQUIRED.** If any field cannot be populated from the approved spec, the `init_state_machine` node must fail with a specific error identifying the missing field(s) and the task (by task ID and human-readable name). Partial task files are never written to disk.

```markdown
# Task: {task_name}
## Task ID: {task_id}

## Context
- **Pillar:** {pillar_name} — {pillar_description}
- **Epic:** {epic_name} — {epic_description}
- **Story:** {story_name} — {story_description}

## Description
{task_description}

## Subtasks
1. {subtask_1}
2. {subtask_2}
...

## Acceptance Criteria
- {criterion_1}
- {criterion_2}
...

## Micro Module Contract
- **Inputs:** {types + constraints}
- **Outputs:** {types + invariants}
- **Error surfaces:** {conditions + types/codes}
- **Effects:** {POST/PUT/PATCH/writes/calls/mutations}
- **Modes:** {sync/async}

## Dependency Contracts
References to upstream modules (CONTRACT_ONLY — no source code):
- {dependency_module_id}: {contract summary + examples}
...

## Error Cases
- {error_case_1}
- {error_case_2}
...
```

#### Implementation Binding

* Folder creation and all file I/O use the deepagents `FilesystemBackend`, which implements the `BackendProtocol` interface.
* `BackendProtocol` provides file CRUD (read, write, edit, delete), grep, and glob operations — all filesystem interactions go through this abstraction.
* The folder hierarchy is created programmatically by the Project Loop's `init_state_machine` node, which translates the approved spec structure into the on-disk layout in a single deterministic pass.
* The slugification algorithm and task ID generation are deterministic pure Python functions invoked by `init_state_machine`. They are not LLM-generated — the same approved spec always produces the same folder hierarchy and task IDs.
* `init_state_machine` performs three validation passes before writing any files: (1) slugify all names and apply collision suffixes, (2) generate all task IDs and verify global uniqueness, (3) verify all required fields in the approved spec can populate every task file template. Any validation failure aborts the entire operation — no partial output is produced.

### 4.2 State machine

The project state machine is a file-based JSON structure that mirrors the project folder hierarchy:

```json
{
  "project_id": "string",
  "spec_version": "string",
  "updated_at": "ISO-8601",
  "tasks": {
    "{task_id}": {
      "pillar": "string",
      "epic": "string",
      "story": "string",
      "task": "string",
      "status": "PENDING | IN_PROGRESS | SHIPPED | HALTED | BLOCKED",
      "depends_on": ["{task_id}", "..."],
      "module_ref": "MM-...@version (populated on ship)",
      "shipped_at": "ISO-8601 | null",
      "halted_reason": "string | null",
      "escalation_ref": "ESC-... | null",
      "declaration_order": "integer (non-negative, assigned during init by depth-first spec traversal, immutable after initialization)"
    }
  }
}
```

**`declaration_order` field:** A non-negative integer assigned to each task during `init_state_machine` by performing a depth-first traversal of the approved spec (pillar → epic → story → task), starting at 0. This field is **immutable after initialization** — no subsequent state machine operation may modify it. The dispatch algorithm (§3.3) uses `declaration_order` as the deterministic tiebreaker when multiple tasks are eligible for execution: the eligible `PENDING` task with the lowest `declaration_order` is selected.

**Status transitions:**

* `PENDING` → `IN_PROGRESS` (dispatched to Engineering Loop)
* `IN_PROGRESS` → `SHIPPED` (debate passed, module shipped)
* `IN_PROGRESS` → `HALTED` (debate failed after retries, project stops)
* `PENDING` → `BLOCKED` (upstream dependency transitions to HALTED; applied via breadth-first forward cascade — see BLOCKED propagation rules below)
* `BLOCKED` → `PENDING` (upstream halt resolved; reverts when **none** of the task's transitive dependencies remain in HALTED status — see BLOCKED propagation rules below)
* `HALTED` → `PENDING` (human resolves escalation and re-queues the task for execution; the human's resolution is injected via `Command` after `interrupt()` — see §17)

**Illegal transitions (invariants):**

* `SHIPPED` is a terminal status — no outbound transitions. A shipped task is immutable.
* `BLOCKED` → `IN_PROGRESS` is illegal — a BLOCKED task must first revert to `PENDING` (via the clear rule below), then be dispatched normally.
* `HALTED` → `SHIPPED` may only occur via external human resolution (the human attests the task is resolved without re-execution). This bypasses the Engineering Loop and must include a human-provided rationale persisted in the escalation artifact.

#### BLOCKED propagation rules

BLOCKED status is **derived** — it reflects the reachability of a HALTED task in the dependency graph. Both the set and clear operations are deterministic graph traversals with no LLM involvement.

**Set rule (forward cascade on HALT):**

When task X transitions to HALTED, all tasks that depend on X — directly or transitively — must be evaluated for BLOCKED status. The algorithm is a breadth-first forward walk from X through the dependency graph:

1. Initialize a FIFO queue with all tasks whose `depends_on` array contains X (direct dependents of X).
2. Initialize a visited set containing X.
3. For each task Y dequeued:
   a. Add Y to the visited set.
   b. If Y's status is `PENDING`, transition Y to `BLOCKED`.
   c. If Y's status is `BLOCKED`, `SHIPPED`, `IN_PROGRESS`, or `HALTED`, do **not** change it. (BLOCKED tasks are already blocked; SHIPPED and IN_PROGRESS tasks are unaffected; HALTED tasks have their own independent halt.)
   d. Enqueue all tasks whose `depends_on` array contains Y, provided they are not already in the visited set.
4. Continue until the queue is empty.

**Clear rule (fixed-point re-evaluation on resolution):**

When a HALTED task is resolved — either moved to `PENDING` for re-execution or to `SHIPPED` by human resolution — all BLOCKED tasks must be re-evaluated. The algorithm is a fixed-point computation:

1. Collect all tasks with status `BLOCKED` into a candidate set.
2. For each task Y in the candidate set: inspect every entry in Y's `depends_on` array. If **any** direct dependency of Y has status `HALTED`, Y remains `BLOCKED`.
3. If **none** of Y's direct dependencies have status `HALTED`, revert Y to `PENDING`.
4. **Repeat steps 1–3** until a full pass produces no status changes (fixed-point convergence). This iteration is necessary because reverting Y to `PENDING` does not directly unblock Y's dependents — but it removes Y from the `BLOCKED` set, which may allow Y's own dependents to clear if Y was the sole transitive cause of their blockage.

The fixed-point loop is guaranteed to terminate because each iteration can only change tasks from `BLOCKED` → `PENDING` (monotonically shrinking the BLOCKED set), and the set is finite.

**BLOCKED invariant (must hold after every state machine mutation):**

> A task's status is `BLOCKED` if and only if at least one task in its transitive dependency closure has status `HALTED`.

Both the forward cascade (set rule) and the fixed-point re-evaluation (clear rule) are designed to preserve this invariant. Any state machine mutation function must verify this invariant holds after execution during development and testing.

#### Dispatch priority

The dispatch algorithm is defined in §3.3. Its key property relevant to the state machine schema is:

* **Eligibility:** A task is eligible for dispatch when its status is `PENDING` and every task in its `depends_on` array has status `SHIPPED`.
* **Ordering:** Among eligible tasks, the one with the lowest `declaration_order` value is selected.
* **Determinism:** The algorithm is a pure function of the state machine JSON. Given identical state machine contents, dispatch always selects the same task. No randomness, timestamps, or external state are consulted.

#### Implementation Binding

* The state machine is a JSON file managed by a pure Python module — no LLM involvement in dispatch decisions (Key Decision #3).
* Stored on disk at `state_store/project/state_machine.json` via `FilesystemBackend`.
* Only current-task metadata is loaded into LangGraph graph state to avoid state size bloat (Risk Register mitigation for projects with hundreds of tasks).
* Concurrent access safety is provided by `fcntl.flock` file locking on reads and writes.
* Status transition functions are deterministic Python: dependency resolution checks `depends_on` arrays and picks the next eligible `PENDING` task whose dependencies have all reached `SHIPPED`.

---

## 5) Core definitions

### 5.1 Artifact

A persisted, schema-validated object in the shared state store.
Artifacts are the *only* inter-role communication medium.

**Critical distinction:** Artifacts themselves are never injected into agent contexts. Only their **contracts** — the I/O specifications describing how to use them — are provided. Every agent invocation treats all code outside its current task as a closed-source service.

**Enforcement:** Artifacts are Pydantic-validated at creation via the `ArtifactEnvelope` model. After ship, artifacts are stored via `ImmutableArtifactBackend`, which structurally prevents in-place modification.

### 5.2 Contribution

A single atomic unit of progress produced by one role invocation (e.g., a contract, a micro plan, a shipped micro module entry).

**Schema:**

```json
{
  "contribution_id": "string (required — format: CONTRIB-{id})",
  "type": "CONTRACT | MICRO_PLAN | IMPLEMENTATION | CHALLENGE | ADJUDICATION | SHIP_EVIDENCE",
  "produced_by": {
    "role": "string (required — e.g., 'proposer', 'challenger', 'arbiter', 'micro_planner')",
    "run_id": "string (required — unique identifier for this role invocation)"
  },
  "artifact_ref": "string (required — reference to the artifact produced by this contribution)",
  "task_ref": "string (required — reference to the parent task, format: T-...)",
  "created_at": "ISO-8601 (required)",
  "context_pack_ref": "string (required — reference to the Context Pack used during this invocation, format: CP-{id})"
}
```

**Lifecycle:** Contributions are created during role invocations and are **immutable after creation** — they cannot be edited, deleted, or retroactively amended. Each role invocation produces exactly one Contribution. The ordered sequence of Contributions for a given task forms a complete audit trail of all work performed within the harness, enabling post-hoc tracing from any shipped module back through its debate history, planning decisions, and context boundaries.

### 5.3 Context Pack

A machine-readable manifest of exactly what references were provided to a role invocation (and what was excluded).
Context Packs are **auditable** and must enforce boundaries (especially "no code reading, no artifact contents — contracts only").

**Schema:**

```json
{
  "context_pack_id": "CP-{id}",
  "created_for": {
    "role": "string (required — the role this pack was constructed for, e.g., 'proposer', 'challenger', 'arbiter')",
    "run_id": "string (required — unique identifier for this role invocation)",
    "task_ref": "string (required — the task being executed, format: T-...)"
  },
  "included_refs": [
    {
      "ref": "string (required — format: id@version)",
      "permission": "FULL | CONTRACT_ONLY | SUMMARY_ONLY | METADATA_ONLY",
      "reason": "string (required — why this reference is included and at this permission level)"
    }
  ],
  "excluded_refs": [
    {
      "ref": "string (required — format: id@version)",
      "reason": "string (required — why this reference is excluded)"
    }
  ],
  "default_permission": "CONTRACT_ONLY",
  "created_at": "ISO-8601"
}
```

**Persistence and creation:** Context Packs are persisted to `state_store/context_packs/{cp_id}.json` as an audit trail. They are created by the **orchestration layer** (not by agents) before each role invocation. Agents never construct or modify their own Context Packs — the Engineering Loop's per-role setup logic builds each Context Pack based on the role table (§3.4), the current task's dependency graph, and the harness-wide default permission of `CONTRACT_ONLY`.

**Enforcement:** Context Packs are enforced at the **data layer** via `ContextPackBackend`, a wrapper around `FilesystemBackend` that intercepts all file operations and enforces viewing permissions (`FULL | CONTRACT_ONLY | SUMMARY_ONLY | METADATA_ONLY`). This enforcement cannot be bypassed by prompt injection — it operates at the storage API level, not in system prompts.

### 5.4 Micro module

The smallest shippable unit of software in this harness.

A micro module must:

* implement a **single responsibility**
* expose a **single public contract surface**
* have explicit **inputs / outputs / errors**
* have black-box tests meeting the minimum coverage requirements below
* be **shippable** and **indexable** for reuse

**Minimum test coverage requirements:**

Every micro module must include tests that satisfy **all** of the following minimums. These are not guidelines — they are enforced by the Challenger during debate.

1. **Input constraint coverage:** At least one test case per input constraint declared in the contract. For example, if an input is declared as "non-empty string", there must be a test that supplies an empty string and verifies the documented error behavior.
2. **Output invariant coverage:** At least one test case per output invariant declared in the contract. For example, if an output is declared as "always sorted ascending", there must be a test that verifies the sort order of the result.
3. **Error surface coverage:** At least one test case per error surface declared in the contract. For example, if the contract declares a `ValidationError` when input exceeds 1000 characters, there must be a test that triggers that specific exception.
4. **Happy-path coverage:** At least one end-to-end test exercising the primary input→output behavior described in the contract's purpose field, using valid inputs and verifying the expected output.
5. **Black-box only:** All tests must invoke the module's public contract surface exclusively. Tests must never import, call, or assert against internal (non-exported) functions, classes, or state. If a test requires internal access to verify behavior, that is a signal that the contract is incomplete — the correct response is to amend the contract, not to break encapsulation.
6. **Challenger validation:** The Challenger validates test coverage against these minimums during debate. A Proposal whose tests fail to meet any minimum results in a `FAIL` verdict with a specific failure entry identifying the missing coverage category (e.g., "no test for error surface: ValidationError on oversized input").

**Enforcement:** After ship, micro module implementation directories are sealed via `OpaqueEnforcementBackend`, making source code structurally unreadable to all subsequent agent invocations.

### 5.5 Interface contract (I/O contract)

The canonical specification of "what goes in / what comes out." Every contract must define all five dimensions:

* **inputs** (types + constraints)
* **outputs** (types + invariants)
* **error surfaces** (conditions + types/codes)
* **effects** (POST/PUT/PATCH/writes/calls/mutations)
* **modes** (sync/async)

These five dimensions are the mandatory shape of every micro module contract. Omitting any dimension is a contract defect.

**Enforcement:** Contracts are Pydantic models covering all five mandatory dimensions. Contract fingerprinting uses RFC 8785 JSON Canonicalization + SHA-256 hashing, producing a deterministic fingerprint for change detection and ship verification.

### 5.6 Ship ("shipped as working")

An attestation supported by recorded evidence that the artifact satisfies acceptance requirements.

* For code: tests passed + public interface matches contract fingerprint.
* For non-code artifacts: debate adjudication approved + validators passed.

**Enforcement:** Ship evidence includes the `interface_fingerprint` hash (RFC 8785 + SHA-256) computed at ship time and compared against the locked contract. The fingerprint proves the shipped implementation's public surface matches the approved contract.

### 5.7 Immutable deliverable

Any artifact that has been shipped. **Immutable means no in-place edits**.
Changes require a new version (or successor artifact), with explicit superseding/deprecation metadata.

**Enforcement:** Immutability is enforced structurally by `ImmutableArtifactBackend`, not by convention or prompt instructions.

### 5.8 Opaque implementation rule

After a micro module is shipped:

* other **coding agents** must not read its implementation
* integration must occur via contract + ship evidence + examples
* the *only* allowed cross-module "feedback" is raising an interface change exception

**Enforcement:** Enforced by `OpaqueEnforcementBackend` which makes sealed `implementation/` directories return access-denied errors for all read operations.

### 5.9 Code Index

A global registry of all **shipped micro modules** (append-only + snapshotted).
It is the primary discovery and reuse mechanism for planning and implementation.

**Enforcement:** Implemented as an embedded vector store with append-only semantics enforced at the application layer. Each entry is stored with structured metadata and a semantic embedding (computed from the entry's purpose, tags, and I/O summary), enabling both metadata filtering and semantic similarity search for reuse discovery.

---

## 6) Harness-wide invariants (non-negotiable)

### 6.1 Contract-only communication

Roles never communicate via shared conversation history or by sharing artifact contents.
They receive **contracts** describing upstream modules and produce new artifacts.

**Why:** prevents anchoring, makes critique independent and honest, and enforces the closed-source treatment of all code outside the current task.

**Enforced by:** `ContextPackBackend` intercepts all file operations at the data layer, returning only contract surfaces (`contract.json`, `examples.md`, `ship.json`) for cross-module references. Implementation files are structurally inaccessible.

### 6.2 Fresh context per role invocation

Each role invocation starts from a clean slate and receives only:

* the task definition (`{task_id}.md`)
* a Context Pack (contracts only — never source code or artifact internals)
* required schemas and constraints

**Why:** context contamination is the main source of downstream error propagation.

**Enforced by:** LangGraph subgraph isolation via wrapper-node pattern (Key Decision #1). Each subgraph has private state. **Known risk:** the deepagents `files` state key can leak to subagents via `SubAgentMiddleware`. Mitigation: add `files` to `_EXCLUDED_STATE_KEYS` or build subagent state explicitly per role invocation.

### 6.3 Context Packs must be explicit and enforceable

A Context Pack must declare:

* included references (by ID+version)
* excluded references (with reasons)
* viewing permissions (FULL | CONTRACT_ONLY | SUMMARY_ONLY | METADATA_ONLY)

The default for all cross-module references is **CONTRACT_ONLY**. FULL access is only granted for the module currently being implemented.

**Why:** makes context decisions auditable and prevents accidental leakage.

**Enforced by:** `ContextPackBackend` wraps `FilesystemBackend` per role invocation. Viewing permissions are enforced at the storage API level — backend-level enforcement cannot be bypassed by prompt injection, unlike prompt-only instructions.

### 6.4 Mandatory debate loop in the Engineering Loop

No engineering contribution becomes a shipped deliverable without passing:

**Propose → Challenge → Adjudicate → Ship (or Revise/Halt)**

**On debate failure after bounded retries: the project HALTS for human intervention.**

**Why:** eliminates single-agent failure modes and forces binary decisions. The halt-on-failure policy ensures humans are in the loop for unresolvable issues rather than allowing the factory to produce incorrect software.

**Enforced by:** Debate is a dedicated `StateGraph` subgraph with `Command(goto=...)` routing (Key Decision #2). Each debate role uses `model.with_structured_output(Schema)` for Pydantic-validated structured JSON output (Key Decision #7), using provider-native structured output (Anthropic/OpenAI). Retry counter is explicit in debate state.

### 6.5 Micro‑micro decomposition is required

Planning must continue until the leaf nodes are micro modules with explicit I/O contracts covering all five dimensions (inputs, outputs, error surfaces, effects, modes).

**Why:** smaller units reduce blast radius and maximize parallelism and reuse.

**Enforced by:** The Challenger node in the Debate Subgraph validates decomposition against the rubric (§10.4). Insufficient decomposition results in a FAIL verdict.

### 6.6 Immutability after ship

Shipped artifacts cannot change. Only operations allowed:

* reference
* supersede with a new version
* deprecate with a replacement pointer
* compose into higher-level deliverables

**Why:** makes the system deterministic, cacheable, and safe for downstream dependencies.

**Enforced by:** `ImmutableArtifactBackend` wrapping `FilesystemBackend`. Write/delete operations on shipped artifacts return errors at the storage layer.

### 6.7 Contract-first is mandatory

For code deliverables:

* contract must be debated and locked before "implementation shipped" can occur.
* the contract must specify all five dimensions (inputs, outputs, error surfaces, effects, modes).

**Why:** avoids "implementation-defined APIs" and enables black-box reuse.

**Enforced by:** Contract is a Pydantic model validated against the five-dimension schema. Contract fingerprint (RFC 8785 + SHA-256) must be locked before implementation can be shipped. The `ship` node verifies fingerprint match.

### 6.8 Black-box integration only (closed-source assumption)

Every agent invocation treats all code outside its current micro task as a **closed-source service**. Integration happens through contracts and examples only.

If behavior is missing/ambiguous, that is a **contract defect**, not a reason to read code.

**Why:** enforces modularity, long-term stability, and substitution.

**Enforced by:** `OpaqueEnforcementBackend` prevents file reads of sealed `implementation/` directories. `ContextPackBackend` serves only contract surfaces to downstream consumers. If behavior is missing/ambiguous, that is a contract defect — the backend physically cannot provide implementation source.

---

## 7) State store (logical layout)

The state store must support:

* immutable versioned artifacts
* debate records
* sealed micro module implementations
* a global Code Index
* the project state machine

Example logical layout:

```
{project_root}/state_store/
  project/
    state_machine.json           # ProjectState — task statuses and dispatch state
  artifacts/
    {pillar}/{artifact_id}/{version}/
      envelope.json              # ArtifactEnvelope (Pydantic-validated)
      body/                      # artifact content
  modules/
    {module_id}/{version}/
      contract.json              # MicroModuleContract (Pydantic model)
      examples.md                # usage examples and expected behaviors
      ship.json                  # ShipEvidence with interface fingerprint
      implementation/            # sealed post-ship (OpaqueEnforcementBackend)
      tests/                     # sealed; test summaries are shareable
  code_index/                      # embedded vector store (append-only, single source of truth)
    chroma.sqlite3               # vector store persistence (entries + embeddings)
    embeddings/                  # embedding data files
  debates/{artifact_id}/
    proposal.json                # Proposal (Pydantic model)
    challenge.json               # Challenge (Pydantic model)
    adjudication.json            # Adjudication (Pydantic model)
  context_packs/{cp_id}.json     # ContextPack declarations (audit trail)
  exceptions/{exception_id}.json # InterfaceChangeException records
  escalations/{escalation_id}.json # Escalation artifacts for human review
```

**Why:** immutability + sealing must be enforceable structurally, not by convention.

### Version directory management

Version directories (`{version}/`) appear under both `artifacts/{pillar}/{artifact_id}/` and `modules/{module_id}/` in the logical layout. They use **semantic versioning** (`major.minor.patch`).

**Version format:** Semantic versioning per [semver.org](https://semver.org/) — `major.minor.patch`, e.g., `1.0.0`.

**Initial version:** All new modules start at `1.0.0`.

**Version bumping rules:**

| Bump level | Version change | Trigger condition | Contract fingerprint changes? |
|---|---|---|---|
| **Patch** | `1.0.0` → `1.0.1` | Non-breaking changes that do not alter the contract fingerprint (e.g., internal implementation improvements, documentation updates, additional examples) | No |
| **Minor** | `1.0.0` → `1.1.0` | Backward-compatible contract additions (new optional inputs, new outputs, new error surfaces that do not remove existing behavior) | Yes |
| **Major** | `1.0.0` → `2.0.0` | Breaking contract changes (removed inputs, changed output types, changed error surfaces, altered effects, changed sync/async modes) | Yes |

**Who determines the bump level:** The version is determined by the Interface Change Exception (ICE) resolution process (§13). When an ICE is raised, the ICE resolution subgraph computes the appropriate bump level based on the `compatibility_expectation` field in the `InterfaceChangeException` schema:

* `compatibility_expectation: BACKWARD_COMPATIBLE` → minor bump
* `compatibility_expectation: BREAKING` → major bump
* Patch bumps do not require an ICE — they are applied by the `ship` node when the contract fingerprint is unchanged from the previous version.

**Directory creation:** A new version directory is created by the `ship` node in the Engineering Loop. The old version directory remains untouched (immutability invariant, §6.6). The `ship` node writes the new version's `contract.json`, `examples.md`, `ship.json`, `implementation/`, and `tests/` into the new directory, then seals `implementation/` and `tests/` via `OpaqueEnforcementBackend`.

### Implementation Binding

* **Primary storage:** deepagents `FilesystemBackend` (Key Decision #4 — chosen over LangGraph `BaseStore` because artifacts are structured files organized in a deep directory hierarchy, not flat key-value pairs).
* **Path routing:** `CompositeBackend` routes file operations to different storage backends based on path prefixes. The concrete routing table is:

  | Path prefix | Backend | Behavior |
  |---|---|---|
  | `project/` | `FilesystemBackend` | Standard read/write for state machine |
  | `artifacts/` | `ImmutableArtifactBackend` wrapping `FilesystemBackend` | Write-once after ship; edits/deletes rejected |
  | `modules/` | `OpaqueEnforcementBackend` wrapping `ImmutableArtifactBackend` wrapping `FilesystemBackend` | Post-ship: `implementation/` and `tests/` sealed; `contract.json`, `examples.md`, `ship.json` remain readable |
  | `code_index/` | Managed by `AppendOnlyCodeIndex` wrapper (not directly accessed via `CompositeBackend`) | Append-only; no direct file operations |
  | `debates/` | `FilesystemBackend` | Standard read/write during debate; immutable after adjudication |
  | `context_packs/` | `FilesystemBackend` | Write-once (audit trail) |
  | `exceptions/` | `FilesystemBackend` | Standard read/write |
  | `escalations/` | `FilesystemBackend` | Standard read/write |
* **Immutability:** `ImmutableArtifactBackend` wraps `FilesystemBackend` for shipped artifacts, preventing in-place edits. Changes require creating a new version with explicit superseding metadata.
* **Sealed directories:** `OpaqueEnforcementBackend` makes `implementation/` and `tests/` directories unreadable after ship, enforcing the opaque implementation rule (§12) at the storage layer.
* **Optional metadata indexing:** LangGraph `BaseStore` can be layered on top of the filesystem for cross-thread metadata queries (e.g., "find all shipped modules tagged X"), without replacing the primary file-based store.
* **Code Index:** Backed by an embedded vector store (see §14.1) with append-only semantics enforced at the application layer. The vector store handles its own persistence and concurrency.

---

## 8) Universal artifact envelope (required metadata)

All artifacts must be wrapped in a shared envelope:

```json
{
  "artifact_id": "string (required — format: {TYPE_PREFIX}-{uuid4_short}, see §8.2)",
  "artifact_type": "ArtifactType enum (required — see §8.1)",
  "version": "string (required — semver, e.g., '1.0.0', see §8.3)",
  "status": "ArtifactStatus enum (required — see §8.4)",
  "created_at": "ISO-8601 (required)",
  "created_by": { "role": "string (required)", "run_id": "string (required)" },
  "context_pack_ref": "string (required — format: CP-{id})",
  "inputs": [{ "ref": "string (required — format: id@version)", "purpose": "string (required)" }],
  "outputs": [{ "ref": "string (required — format: id@version)", "purpose": "string (required)" }],
  "notes": "string (optional — default empty)"
}
```

**Why:** makes provenance explicit and reproducible.

### 8.1 `artifact_type` enum

The `artifact_type` field is a **closed enum**. Only the values listed below are valid. Any artifact with a type not in this enum is rejected at validation time.

| Enum Value | Description | Producing Section |
|---|---|---|
| `MICRO_PLAN` | Decomposition plan splitting a task into micro modules | §10 |
| `CONTRACT` | Micro module contract defining the five I/O dimensions | §11 |
| `IMPLEMENTATION` | Code or deliverable produced by the Proposer | §9 |
| `PROPOSAL` | Debate proposal artifact (Proposer's claim + deliverable reference) | §9 |
| `CHALLENGE` | Debate challenge artifact (Challenger's verdict + failures) | §9 |
| `ADJUDICATION` | Debate adjudication artifact (Arbiter's decision + rationale) | §9 |
| `SHIP_EVIDENCE` | Ship verification record proving a module satisfies its contract | §11 |
| `ESCALATION` | Escalation artifact surfaced for human review on project halt | §17 |
| `CONTEXT_PACK` | Context pack declaration defining what a role invocation received | §5.3 |
| `INTERFACE_CHANGE_EXCEPTION` | Interface change exception (ICE) record requesting a contract change | §13 |
| `REUSE_SEARCH_REPORT` | Reuse governance report justifying create-new vs. reuse-existing | §14 |
| `PRODUCT_SPEC` | Structured product specification output of the Product Loop | §3.2 |
| `EXAMPLES` | Usage examples for shipped micro modules consumed by downstream agents | §12 |

This enum is exhaustive. Adding a new artifact type requires a spec amendment — implementations must not accept ad-hoc type strings.

### 8.2 `artifact_id` generation scheme

Every artifact receives a globally unique `artifact_id` with the following canonical format:

```
{TYPE_PREFIX}-{uuid4_short}
```

**Components:**

* `{TYPE_PREFIX}` — a 2–4 character abbreviation derived from the `artifact_type` enum. The mapping is fixed and defined below.
* `-` — literal hyphen separator.
* `{uuid4_short}` — the first 8 hexadecimal characters of a UUID4 string (lowercase). This provides ~4.3 billion unique values per type prefix, which is sufficient for any single project.

**Type prefix mapping:**

| `artifact_type` | Type Prefix | Example `artifact_id` |
|---|---|---|
| `MICRO_PLAN` | `MP` | `MP-a1b2c3d4` |
| `CONTRACT` | `CTR` | `CTR-e5f6g7h8` |
| `IMPLEMENTATION` | `IMPL` | `IMPL-c3d4e5f6` |
| `PROPOSAL` | `PROP` | `PROP-i9j0k1l2` |
| `CHALLENGE` | `CHAL` | `CHAL-m3n4o5p6` |
| `ADJUDICATION` | `ADJ` | `ADJ-q7r8s9t0` |
| `SHIP_EVIDENCE` | `SHIP` | `SHIP-u1v2w3x4` |
| `ESCALATION` | `ESC` | `ESC-y5z6a7b8` |
| `CONTEXT_PACK` | `CP` | `CP-d4e5f6g7` |
| `INTERFACE_CHANGE_EXCEPTION` | `ICE` | `ICE-h8i9j0k1` |
| `REUSE_SEARCH_REPORT` | `RSR` | `RSR-l2m3n4o5` |
| `PRODUCT_SPEC` | `SPEC` | `SPEC-p6q7r8s9` |
| `EXAMPLES` | `EX` | `EX-t0u1v2w3` |

**Generation rules:**

* Artifact IDs are generated by the **orchestration layer** (the graph nodes managing artifact lifecycle), never by agents themselves. Agent tools (`create_artifact`, etc.) accept the artifact type and return the generated ID.
* The UUID4 hex source must be generated using a cryptographically secure random source (e.g., Python's `uuid.uuid4()`).
* IDs are immutable after creation — an artifact's ID never changes across status transitions or version bumps.
* The `artifact_id` is scoped to a single artifact version. Different versions of the same logical artifact have different `artifact_id` values. Cross-version lineage is tracked via the `inputs` and `outputs` reference arrays.

**Validation regex:** `^(MP|CTR|IMPL|PROP|CHAL|ADJ|SHIP|ESC|CP|ICE|RSR|SPEC|EX)-[0-9a-f]{8}$`

### 8.3 `version` format

The `version` field uses **semantic versioning** (`major.minor.patch`):

* **Format:** Three dot-separated non-negative integers (e.g., `1.0.0`, `2.1.3`).
* **Initial version:** All newly created artifacts start at `1.0.0`.
* **Version bumping:** Follows the rules defined in §7 (state store) — when a new version of an artifact is created (e.g., via superseding or interface change exception resolution), the version is incremented according to the nature of the change:
  * **Patch** (`x.y.Z`): backward-compatible fixes (e.g., corrected examples, clarified documentation).
  * **Minor** (`x.Y.0`): backward-compatible additions (e.g., new optional input, additional error surface).
  * **Major** (`X.0.0`): breaking changes (e.g., removed input, changed output type, altered semantics).
* **Monotonic invariant:** Versions always increase and never decrease. A version `2.0.0` cannot be followed by `1.5.0` for the same logical artifact lineage. The `ArtifactStoreService` must reject any version that does not exceed the highest existing version for the same artifact lineage.
* **Validation regex:** `^\d+\.\d+\.\d+$`

### 8.4 Status values and transition rules

The `status` field is a closed enum with five valid values. Transitions between statuses follow a strict state machine — any transition not listed below is invalid and must be rejected by the `ArtifactStoreService`.

**Valid statuses:**

| Status | Meaning |
|---|---|
| `DRAFT` | Artifact has been created but not yet evaluated by the debate loop. |
| `CHALLENGED` | A Challenger has evaluated the artifact and produced a Challenge artifact. |
| `ADJUDICATED` | An Arbiter has rendered a decision on the artifact. |
| `SHIPPED` | Artifact has been approved, verified, and sealed as an immutable deliverable. |
| `DEPRECATED` | Artifact has been superseded by a newer version and should not be used for new work. |

**Valid transitions:**

```
           ┌──────────────────────────────────────────┐
           │                                          │
           ▼                                          │
        ┌───────┐       ┌────────────┐       ┌──────────────┐
  ○────►│ DRAFT │──────►│ CHALLENGED │──────►│ ADJUDICATED  │
        └───────┘       └────────────┘       └──────┬───────┘
           ▲                                        │
           │            (rejected during            │
           │             bounded retry)             │
           └────────────────────────────────────────┘
                                                    │
                                        (approved)  │
                                                    ▼
                                              ┌──────────┐       ┌────────────┐
                                              │ SHIPPED  │──────►│ DEPRECATED │
                                              └──────────┘       └────────────┘
```

| From | To | Trigger | Conditions |
|---|---|---|---|
| `DRAFT` | `CHALLENGED` | Challenger evaluates the artifact | Challenger produces a valid `CHALLENGE` artifact referencing this artifact. |
| `CHALLENGED` | `ADJUDICATED` | Arbiter renders a decision | Arbiter produces a valid `ADJUDICATION` artifact referencing this artifact. |
| `ADJUDICATED` | `SHIPPED` | Artifact approved and ship evidence verified | Adjudication decision is `APPROVE` or `APPROVE_WITH_AMENDMENTS` (after amendments applied), and `ShipEvidence` verification result is `PASS` with matching `interface_fingerprint`. |
| `ADJUDICATED` | `DRAFT` | Artifact rejected, returned for revision | Adjudication decision is `REJECT`. Only permitted during the bounded retry window of the debate loop (§9.3). After retries are exhausted, the artifact cannot return to `DRAFT` — the project halts instead. |
| `SHIPPED` | `DEPRECATED` | Artifact superseded by a new version | A successor artifact has been shipped. The deprecated artifact's Code Index entry is updated with `superseded_by` pointing to the replacement. |

**Invalid transitions (must be rejected):**

| Attempted Transition | Reason |
|---|---|
| `DRAFT` → `SHIPPED` | Cannot skip the mandatory debate loop (§6.4). Every artifact must pass through `CHALLENGED` → `ADJUDICATED` before shipping. |
| `DRAFT` → `ADJUDICATED` | Cannot skip the challenge phase. The Challenger must evaluate before the Arbiter decides. |
| `SHIPPED` → `DRAFT` | Shipped artifacts are immutable (§5.7, §6.6). Modification requires creating a new artifact version, not reverting the existing one. |
| `SHIPPED` → `CHALLENGED` | Shipped artifacts are immutable. Re-evaluation requires a new artifact version via an interface change exception (§13). |
| `DEPRECATED` → any status | `DEPRECATED` is a terminal status. Deprecated artifacts cannot be revived — they can only be referenced or replaced by a successor. |
| `CHALLENGED` → `DRAFT` | The Challenger does not return artifacts to draft. Only the Arbiter (via `ADJUDICATED` → `DRAFT`) can send an artifact back for revision. |
| `CHALLENGED` → `SHIPPED` | Cannot skip adjudication. The Arbiter must render a decision before shipping. |

**Enforcement:** Status transitions are validated by `ArtifactStoreService` before any status mutation is persisted. The valid transition map is a deterministic lookup table — no LLM involvement in transition decisions.

### Implementation Binding

* The envelope is a Pydantic model (`ArtifactEnvelope`) validated at write time — invalid envelopes are rejected before storage, not caught downstream.
* `artifact_type` is a `StrEnum` (`ArtifactType`) with exactly 13 members. The Pydantic model rejects any value not in the enum.
* `artifact_id` generation is encapsulated in a deterministic factory function that takes an `ArtifactType`, generates a UUID4, and returns the formatted ID. This function lives in the orchestration layer and is never exposed to agents directly.
* `version` is validated via a regex-constrained `str` field (`pattern=r"^\d+\.\d+\.\d+$"`). The initial version `1.0.0` is set as the default for new artifacts.
* `status` is a `StrEnum` (`ArtifactStatus`) with exactly 5 members. The valid transition map is a `dict[ArtifactStatus, set[ArtifactStatus]]` checked before every status mutation.
* All artifact creation goes through `ArtifactStoreService`, which orchestrates the full write path: envelope validation → ID generation → body storage → Code Index registration.
* Agent-facing tools: `create_artifact`, `ship_artifact`, `query_artifacts`, `search_code_index` — these are the only way agents interact with the artifact store.
* Envelope validation is enforced at the storage layer, not by prompt instructions. Agents cannot bypass validation regardless of prompt content.

---

## 9) Mandatory debate loop (Engineering Loop detail)

### 9.1 Roles (model-agnostic)

* **Proposer**: produces a candidate implementation and claims it satisfies the contract from the task file.
* **Challenger**: adversarially evaluates against invariants and the contract's five dimensions; must be binary (PASS/FAIL).
* **Arbiter**: final decision-maker; may approve with amendments; triggers ship.

> A "mild review" is non-compliant.
> The Challenger's job is to attempt to falsify correctness and completeness.

#### Implementation Binding

Each role (Proposer, Challenger, Arbiter) is a node in the Debate `StateGraph`. Each node uses `model.with_structured_output(Schema)` to produce Pydantic-validated JSON (Key Decision #7 — provider-native structured output is more reliable than post-hoc JSON parsing). The debate graph is model-agnostic: any LLM provider supporting structured output can fill any role.

### 9.2 Debate artifacts

**Proposal**

```json
{
  "proposal_id": "string",
  "target_artifact_id": "string",
  "claim": "string — what it delivers and why",
  "deliverable_ref": "ref",
  "acceptance_checks": ["string — objective checks"]
}
```

**Challenge**

```json
{
  "challenge_id": "string",
  "target_artifact_id": "string",
  "verdict": "PASS | FAIL",
  "failures": [
    {
      "invariant": "string",
      "evidence": "string",
      "required_change": "string"
    }
  ],
  "optional_alternative_ref": "ref"
}
```

**Adjudication**

```json
{
  "adjudication_id": "string",
  "target_artifact_id": "string",
  "decision": "APPROVE | APPROVE_WITH_AMENDMENTS | REJECT",
  "amendments": [
    { "action": "MODIFY | ADD | REMOVE", "target": "string", "detail": "string" }
  ],
  "rationale": "string",
  "ship_directive": "SHIP | NO_SHIP"
}
```

All debate artifacts (Proposal, Challenge, Adjudication) are Pydantic models with schema validation at creation time. Invalid debate artifacts are rejected before they enter the state store.

### 9.3 Retry + halt policy

* If FAIL: bounded revisions (1–2 attempts).
* If still FAIL: Arbiter either amends and approves, rejects and splits scope, or **halts the project**.
* On HALT: an escalation artifact is produced (see §17) and the state machine marks the task as `HALTED`. The Project Loop stops dispatching.

**Why:** prevents endless loops; keeps decisions binary and explicit. Unresolvable failures require human judgment, not infinite retries.

#### Implementation Binding

* Bounded retries = max 2 revision attempts.
* Routing uses `Command(goto=...)` with an explicit `retry_count` field in the debate `StateGraph` state.
* Route logic:
  * FAIL + retries left > 0 → `Command(goto="revise")`
  * FAIL + retries exhausted → `Command(goto="adjudicate")`
* On HALT: `Command(graph=PARENT)` propagates the halt signal to the Project Loop, which stops dispatching.
* Escalation artifact is persisted to `state_store/escalations/{id}.json`.

### 9.4 Challenger evaluation rubric

The Challenger must evaluate every Proposal against the following rubric. Each criterion is assessed as **MET** or **NOT_MET**. If **any** criterion is NOT_MET, the overall verdict **must** be FAIL.

| # | Criterion | What the Challenger checks | Reference |
|---|-----------|---------------------------|-----------|
| R1 | **Contract conformance** | The implementation satisfies all five I/O dimensions declared in the module contract: inputs (types + constraints), outputs (types + invariants), error surfaces (conditions + types/codes), effects (writes/calls/mutations), and modes (sync/async). Every dimension must be verified — omitting a dimension is an automatic NOT_MET. | §5.5 |
| R2 | **Test coverage** | The test suite meets the minimum coverage requirements: at least one test per input constraint, at least one test per output invariant, at least one test per error surface, and at least one happy-path test. All tests must be black-box (public contract surface only). | §5.4 |
| R3 | **Error handling completeness** | Every error surface declared in the contract has a corresponding handler in the implementation. No undeclared exceptions or failure modes are possible during normal or boundary operation. If the Challenger can identify a reachable error path that is not declared in the contract and not handled, this criterion is NOT_MET. | §5.5 |
| R4 | **Dependency usage** | All dependencies are consumed via their published contracts only (CONTRACT_ONLY access). The implementation must not read, import, or assume any internal detail of a dependency's implementation. Contract fingerprints of consumed dependencies must match the locked versions. | §5.8, §6.1 |
| R5 | **Acceptance criteria** | Every acceptance criterion listed in the task file (`task_id.md`) is demonstrably met. "Demonstrably" means the Challenger can trace each criterion to a specific test, assertion, or verifiable artifact output. Criteria that are met only by claim (without evidence) are NOT_MET. | Task file |
| R6 | **Decomposition compliance** | The module meets the micro-module rubric: single input→output behavior describable in one sentence, at most one public contract surface, independently shippable with local black-box tests, and dependencies expressed only via contracts. | §10.4 |

The Challenger must produce one assessment entry per criterion in its Challenge artifact. Each entry must cite **specific evidence** (e.g., a test name, a contract field, a code path) supporting its MET or NOT_MET determination. A Challenge that omits any criterion is itself non-compliant (see §9.5).

#### Implementation Binding

* The `Challenge` Pydantic model is extended with a `rubric_assessments` field: a list of `{ criterion: str, assessment: "MET" | "NOT_MET", evidence: str }` entries. Schema validation enforces exactly six entries (R1–R6).
* The `challenge_node` system prompt includes the full rubric table and instructs the Challenger to evaluate each criterion sequentially before determining the overall verdict.
* A post-validation check in the `challenge_node` enforces: if any `assessment == "NOT_MET"` then `verdict` must be `"FAIL"`. A Challenge with NOT_MET assessments and a PASS verdict is rejected as schema-invalid and the Challenger is re-invoked.

### 9.5 "Mild review" compliance standard

A "mild review" is any Challenge that avoids rigorous adversarial evaluation. Mild reviews are **non-compliant** and must be detected and rejected. The following are concrete examples:

**Non-compliant challenges (mild reviews):**

1. **Vague approval.** A Challenge that says "looks good, minor suggestions" or similar without citing specific invariants from the contract or rubric criteria. Any PASS verdict that does not include evidence for all six rubric criteria (§9.4) is non-compliant.
2. **Incomplete rubric coverage.** A Challenge that checks only some of the five I/O dimensions (e.g., checks inputs and outputs but skips error surfaces, effects, or modes). Every dimension must be explicitly evaluated.
3. **PASS despite known defects.** A Challenge that identifies issues (e.g., missing error handling, incomplete test coverage) but assigns a PASS verdict anyway. If any rubric criterion is NOT_MET, the verdict must be FAIL — no exceptions.
4. **Missing evidence.** A Challenge whose rubric assessments contain MET determinations without specific evidence (e.g., `"evidence": "looks correct"` instead of citing a specific test case, contract field, or behavior).

**Compliant challenges must:**

* Evaluate every rubric criterion (R1–R6) from §9.4, producing a MET/NOT_MET assessment with specific cited evidence for each.
* Check all five I/O dimensions of the contract (inputs, outputs, error surfaces, effects, modes) as part of R1.
* Assign FAIL if **any** criterion is NOT_MET.
* For PASS verdicts: every rubric assessment must be MET, and each must cite specific evidence demonstrating satisfaction.

#### Implementation Binding

* The `challenge_node` includes a post-hoc compliance gate: after the Challenger produces its artifact, the node checks (1) all six rubric criteria are present, (2) evidence fields are non-empty and exceed a minimum token length, and (3) verdict consistency (no NOT_MET with PASS). Non-compliant challenges trigger a re-invocation of the Challenger with an explicit compliance failure message, up to 2 retries. After 2 compliance retries, the challenge is auto-assigned FAIL with a `mild_review_violation` failure entry.

### 9.6 APPROVE_WITH_AMENDMENTS flow

When the Arbiter's decision is `APPROVE_WITH_AMENDMENTS`, the following procedure applies:

1. **Arbiter specifies amendments.** The `amendments` field in the Adjudication artifact contains a list of structured entries, each with the shape: `{ "action": "MODIFY | ADD | REMOVE", "target": "string", "detail": "string" }`. The `target` identifies the specific element to change (e.g., a function name, a contract field, a test case). The `detail` describes the exact change required. Amendments must be specific and actionable — vague amendments (e.g., `"detail": "improve error handling"`) are non-compliant and the Adjudication is schema-rejected.
2. **Proposer applies amendments.** The Proposer receives the Adjudication artifact and applies each amendment to produce a revised deliverable. The Proposer must apply all amendments; selective application is not permitted.
3. **No re-debate.** The revised deliverable is **not** re-debated. The Arbiter's approval is final for this round. The flow proceeds directly to the ship node.
4. **Ship node validates amendments.** The ship node checks that every amendment in the `amendments` list was applied by comparing the revised deliverable against the amendment list. For each amendment, the ship node verifies: (a) `MODIFY` targets exist and were changed, (b) `ADD` targets are present in the revised deliverable, (c) `REMOVE` targets are absent from the revised deliverable.
5. **Failure to apply.** If the Proposer cannot apply an amendment (e.g., the target does not exist, the change is contradictory), the task **HALTs**. The escalation artifact (§17) must include the unapplied amendment and the Proposer's explanation of why it could not be applied.

#### Implementation Binding

* The `apply_amendments` node in the debate `StateGraph` invokes the Proposer with the amendment list and the current deliverable. The Proposer produces a revised deliverable and a `{ amendment_index: int, applied: bool, note: str }` record for each amendment.
* After the `apply_amendments` node, the graph routes to a `validate_amendments` node that checks all `applied` flags. If any `applied == false`, the graph routes to `halt`.
* The `validate_amendments` node also performs structural checks: for `ADD` amendments, it verifies the target is present; for `REMOVE` amendments, it verifies the target is absent; for `MODIFY` amendments, it verifies the target differs from the pre-amendment version.

### 9.7 Role-specific context

Each debate role receives a defined set of inputs. Roles must **not** receive information outside their specified context. This table is normative.

| Input | Proposer | Challenger | Arbiter |
|-------|----------|------------|---------|
| Task file (`task_id.md`) | ✅ FULL | ✗ | ✅ FULL |
| Module contract | ✅ FULL | ✅ FULL | ✅ FULL |
| Dependency contracts | ✅ CONTRACT_ONLY | ✅ CONTRACT_ONLY | ✗ |
| Code Index search results | ✅ | ✗ | ✗ |
| Harness schemas (artifact schemas, state schemas) | ✅ | ✗ | ✗ |
| Current module workspace (source, tests, config) | ✅ FULL access | ✗ | ✗ |
| Proposal artifact (claim + acceptance checks) | — (self) | ✅ FULL | ✅ FULL |
| Challenge artifact (verdict + failures) | ✗ (receives failures only during revise) | — (self) | ✅ FULL |
| Adjudication artifact | ✅ (only during APPROVE_WITH_AMENDMENTS) | ✗ | — (self) |
| Implementation source code | ✅ FULL (own module) | ✗ **Never** | ✗ **Never** |
| Evaluation rubric (§9.4) | ✗ | ✅ FULL | ✗ |
| Decomposition rubric (§10.4) | ✗ | ✅ FULL | ✗ |
| Full debate trail (all rounds) | ✗ | ✅ (all prior proposals + challenges) | ✅ (all proposals + challenges + adjudications) |

**Key access constraints:**

* **Challenger never sees implementation source.** The Challenger evaluates the Proposal artifact (claim + acceptance checks) against the contract and rubrics. This enforces contract-only verification (§6.1) — the Challenger proves or disproves claims without implementation bias.
* **Arbiter never sees implementation source.** The Arbiter adjudicates based on the Proposal, the Challenge, and the contract. This ensures the Arbiter's decision is based on the structured debate record, not on implementation details.
* **Proposer does not see the evaluation rubric.** The Proposer builds the implementation and claims. It should not "teach to the test" by optimizing for rubric criteria rather than genuine contract satisfaction.
* **Dependency contracts are CONTRACT_ONLY.** Both Proposer and Challenger see dependency contracts but never dependency implementations, enforcing the opaque implementation rule (§5.8).

#### Implementation Binding

* Each debate node constructs its prompt from a role-specific context builder function. The context builder for each role filters the `DebateState` to include only the fields listed in the table above.
* The Challenger's context builder explicitly excludes `artifact_context` (which contains source code) and provides only the latest Proposal artifact, the module contract, dependency contracts (contract fields only), both rubrics, and the prior debate trail.
* The Arbiter's context builder provides the full debate trail, the module contract, and the task file, but excludes source code, dependency contracts, and rubrics.

---

## 10) Micro‑micro decomposition (planning requirement)

### 10.1 Micro Plan artifact

Every Task must be decomposed into a **Micro Plan** whose leaves are micro modules.

```json
{
  "micro_plan_id": "MP-...",
  "parent_task_ref": "T-...@...",
  "modules": [
    {
      "module_id": "MM-...",
      "name": "string",
      "purpose": "string",
      "io_contract": {
        "inputs": ["type + constraints"],
        "outputs": ["type + invariants"],
        "error_surfaces": ["condition + type/code"],
        "effects": ["writes/calls/mutations"],
        "modes": ["sync | async"]
      },
      "error_cases": ["string"],
      "depends_on": ["MM-...@..."],
      "reuse_candidate_refs": ["optional MM-...@... from Code Index"],
      "reuse_decision": "REUSE | CREATE_NEW",
      "reuse_justification": "string"
    }
  ]
}
```

#### Implementation Binding

* Micro-planning is a dedicated node (`micro_plan`) in the Engineering Loop `StateGraph`. It runs once per task, before any per-module iteration begins.
* The micro plan is produced by a **single planning agent invocation** and validated programmatically as a Pydantic model (`MicroPlan`). The micro plan is **not** put through the 3-agent Propose→Challenge→Adjudicate debate cycle. The debate is reserved for implementation artifacts (code, tests, contracts). Plans are validated structurally, not adversarially.
* Reuse search (§14.3) is executed as part of micro-planning: the agent uses the `search_code_index` tool to query the Code Index before proposing new modules, populating the `reuse_candidate_refs`, `reuse_decision`, and `reuse_justification` fields for each module in the plan.

#### Micro plan validation and failure handling

The `micro_plan` node validates the produced plan against the `MicroPlan` Pydantic model immediately after agent generation. Validation checks include:

1. **Schema conformance** — all required fields present, types correct, ID patterns valid (`MM-\w+`, `MP-\w+`, semver strings).
2. **I/O dimension completeness** — every module's `io_contract` must have non-empty `inputs` and `outputs`. No dimension may be empty, placeholder ("TBD"), or marked not applicable.
3. **Dependency well-formedness** — all `depends_on` references must use valid `ModuleRef` format (`MM-...@...`).
4. **Unique module IDs** — no duplicate `module_id` values within the plan (`MicroPlan.validate_unique_module_ids`).
5. **DAG constraint** — the `depends_on` graph within the plan must be a directed acyclic graph. The `micro_plan` node validates this by performing a topological sort on the intra-plan dependency edges; if a cycle is detected, validation fails immediately. Cycles are a structural error in the plan, not a runtime error.
6. **Decomposition rubric (§10.4)** — each module is checked programmatically against the rubric criteria where machine-checkable (e.g., non-empty purpose, non-empty I/O contract). Rubric criteria that require semantic judgment (e.g., "single input→output behavior") are deferred to the Challenger during per-module debate.

**Retry policy on validation failure:**

* If validation fails, the planning agent is **re-prompted** with the specific Pydantic `ValidationError` messages (field paths, error descriptions, and the failing values). The re-prompt includes the original task file and the failed plan for context.
* Up to **2 retries** are permitted (3 total attempts including the initial generation).
* If validation still fails after all retries, the Engineering Loop **HALTs** the task. An escalation artifact (§17) is produced containing:
  * The task ID and task file reference
  * The last failed plan (raw JSON)
  * The full list of validation errors from the final attempt
  * A recommended human resolution: "revise task decomposition" or "simplify task scope"

### 10.2 Module ordering within a plan

Modules within a validated micro plan are executed in **topological sort order** of their `depends_on` graph:

* Modules with no intra-plan dependencies (roots of the DAG) are processed first, in **declaration order** within the `modules` array of the `MicroPlan`.
* A module is eligible for processing only after all of its intra-plan dependencies have reached `SHIPPED` status.
* External dependencies (references to modules outside the current plan, identified by `ModuleRef` pointing to a module ID not present in this plan's `modules` list) are assumed to be already shipped — they were resolved by the Project Loop's dispatch ordering before this task was dispatched.
* If a module fails debate (per the failure propagation rules in §3.4), all modules that depend on it — directly or transitively — are abandoned. Modules with no dependency path to the failed module may still be attempted.

### 10.3 Plan revision rules

The micro plan is **locked** after successful validation. It cannot be revised during module execution:

* Once the `micro_plan` node produces a validated `MicroPlan`, it is persisted as an immutable artifact in the state store and the Engineering Loop proceeds to per-module iteration.
* No node in the Engineering Loop may modify the locked plan. Module execution follows the plan exactly as validated.
* If a module fails debate and the Challenger's `failures` evidence indicates the plan decomposition itself was wrong (e.g., the module's scope is too broad, responsibilities are misassigned between modules, or a missing module is needed), the Engineering Loop **HALTs** the entire task. The escalation artifact must include:
  * The locked micro plan
  * The failed module's debate trail
  * The Challenger's evidence identifying the plan-level issue
  * A recommended human resolution: **"split task"**, **"revise plan"**, or **"merge modules"**
* A human may resolve a plan-level escalation by providing a revised plan. This restarts the Engineering Loop from the `micro_plan` node with the human-provided plan as input, bypassing agent generation and proceeding directly to validation. If the human-provided plan passes validation, it becomes the new locked plan and module execution begins from scratch (no partial state is carried over from the previous attempt).

### 10.4 Decomposition rubric (must be enforced by Challenger)

A leaf node is only "micro" if:

* it can be described as a single input→output behavior in one sentence
* it introduces at most one public contract surface
* it can be shipped independently (tests are local and black-box)
* it has clear dependencies (only via contracts)

**Why:** ensures atomicity and shippability.

**Enforced by:** The Challenger node in the Debate Subgraph validates that each leaf module meets the decomposition rubric. A module that fails any rubric criterion (single behavior, single contract surface, independent shippability, contract-only dependencies) results in a `FAIL` verdict during debate.

---

## 11) Micro module contract + ship evidence (engineering currency)

### 11.1 Micro module contract schema (canonical)

The contract must cover all five dimensions of an I/O contract. No dimension may be omitted.

```json
{
  "module_id": "MM-...",
  "module_version": "x.y.z",
  "name": "string",
  "purpose": "string",
  "tags": ["string"],
  "inputs": [
    { "name": "string", "type": "string", "constraints": ["string"] }
  ],
  "outputs": [
    { "name": "string", "type": "string", "invariants": ["string"] }
  ],
  "error_surfaces": [
    { "name": "string", "when": "string", "surface": "exception | code | variant" }
  ],
  "effects": [
    { "type": "POST | PUT | PATCH | WRITE | CALL | MUTATION", "target": "string", "description": "string" }
  ],
  "modes": {
    "sync": "boolean",
    "async": "boolean",
    "notes": "string"
  },
  "error_cases": ["string — specific scenarios the implementation must handle"],
  "dependencies": [{ "ref": "MM-...@range", "why": "string" }],
  "compatibility": {
    "backward_compatible_with": ["MM-...@range"],
    "breaking_change_policy": "string"
  },
  "runtime_budgets": {
    "latency_ms_p95": "optional number",
    "memory_mb_max": "optional number"
  }
}
```

#### Implementation Binding

* The contract schema is a Pydantic model (`MicroModuleContract`) — Phase 1.1 of the implementation roadmap.
* Contract fingerprinting uses RFC 8785 JSON Canonicalization Scheme + SHA-256 hash (Phase 1.6). The fingerprint is computed on the canonical JSON form of the contract's five I/O dimensions (inputs, outputs, error_surfaces, effects, modes).
* `rfc8785>=0.1.4` (Trail of Bits, Apache-2.0, zero dependencies) is the only new external dependency required for canonicalization.
* Fingerprints enable automated detection of contract drift between planning and shipping — if the shipped interface diverges from the locked contract, the `ship` node rejects it.

#### Fingerprint scope

The interface fingerprint captures the "shape" of the public API surface. Only contract fields that define observable behavior at the call boundary are hashed. Metadata that does not affect how callers integrate with the module is excluded so that renaming, retagging, or updating documentation never triggers a spurious fingerprint mismatch.

**INCLUDED in fingerprint** (the five I/O dimensions):

| Field | Why included |
|-------|-------------|
| `inputs` | Defines what callers must provide. |
| `outputs` | Defines what callers receive. |
| `error_surfaces` | Defines how failures are communicated to callers. |
| `effects` | Defines side-effects callers must account for. |
| `modes` | Defines whether callers use sync, async, or both. |

**EXCLUDED from fingerprint** (metadata that does not affect the public interface):

| Field | Why excluded |
|-------|-------------|
| `module_id` | Registry identity, not part of the call surface. |
| `module_version` | Versioning metadata, not an observable behavior change. |
| `name` | Human label; renaming has no integration impact. |
| `purpose` | Documentation; does not change the call boundary. |
| `tags` | Discoverability metadata; no caller-visible effect. |
| `error_cases` | Implementation-level scenarios; callers observe only `error_surfaces`. |
| `dependencies` | Internal wiring; invisible to callers (opaque implementation rule). |
| `compatibility` | Governance metadata for the registry, not the call surface. |
| `runtime_budgets` | Advisory performance hints; not a contractual call-boundary property. |

The fingerprint is a `computed_field` on the Pydantic model: it serializes the five included fields to JSON via `model_dump(mode="json")`, canonicalizes with `rfc8785.dumps()`, and returns the SHA-256 hex digest (64 lowercase hex characters).

### 11.2 Ship evidence schema

```json
{
  "module_id": "MM-...",
  "module_version": "x.y.z",
  "ship_id": "SHIP-...",
  "verified_at": "ISO-8601",
  "verification": {
    "result": "PASS | FAIL",
    "interface_fingerprint": "hash of public I/O surface",
    "evidence_ref": "ref to logs and outputs"
  },
  "environment": {
    "repo_revision": "string",
    "dependency_lock_ref": "string",
    "runner_id": "string"
  }
}
```

**Why:** "working" must be falsifiable and reproducible.

#### Implementation Binding

* The `interface_fingerprint` in ship evidence is computed automatically at ship time by the `ship` node in the Engineering Loop.
* The ship node computes the fingerprint (RFC 8785 + SHA-256, same algorithm as contract fingerprinting) and compares it against the locked contract fingerprint — a mismatch blocks shipping and forces a contract reconciliation.
* Ship evidence is persisted to `state_store/modules/{module_id}/{version}/ship.json` via `ImmutableArtifactBackend`.

#### Fingerprint mismatch reconciliation

When the `ship` node detects that the implementation's public surface does not match the locked contract fingerprint, the following deterministic flow applies:

1. **Rejection.** The `ship` node rejects the ship attempt with a structured error:
   `"Interface fingerprint mismatch: expected {locked_fingerprint}, got {actual_fingerprint}"`
   where both values are the full 64-character SHA-256 hex digests.

2. **Debate FAIL.** The mismatch is treated as a debate FAIL outcome. The Proposer must either:
   * **Revise the implementation** so its public surface matches the locked contract fingerprint exactly, OR
   * **Raise an Interface Change Exception** (§13) if the contract itself needs to change — the ICE workflow produces a new contract version with a new fingerprint, after which the Proposer targets the updated fingerprint.

3. **Bounded retries.** Fingerprint-mismatch rejections count against the bounded retry budget (maximum 2 revisions per ship attempt, as defined by the debate loop in §9). Each retry re-computes the actual fingerprint and re-compares.

4. **HALT on exhaustion.** If all retries are exhausted and the fingerprint still does not match, the task HALTs. The ship node emits an escalation artifact containing:
   * The locked fingerprint and the actual fingerprint.
   * A per-dimension diff identifying which of the five I/O dimensions (`inputs`, `outputs`, `error_surfaces`, `effects`, `modes`) diverged.
   * The `module_id` and `module_version` of the failed ship attempt.

   This artifact is written to `state_store/modules/{module_id}/{version}/escalation.json` and surfaced to the orchestrator for human review.

### 11.3 Test evidence requirements

The `evidence_ref` field in the ship evidence record must point to a programmatic test execution log — not a narrative summary or manually logged assertions. The `ship` node owns test execution; the Proposer agent does not run or report tests itself.

#### Execution model

* The `ship` node invokes the test runner as a **deterministic step** in the ship pipeline, after fingerprint verification passes and before writing the final `ShipEvidence` record.
* Tests are discovered from the module's implementation directory (`state_store/modules/{module_id}/{version}/impl/tests/`).
* The test runner executes all discovered tests programmatically (e.g., via `pytest` subprocess or equivalent harness). Tests must be executable — stub assertions, pseudocode, or "tests" that only log messages are not valid.

#### Execution log schema

The `evidence_ref` field points to:

```
state_store/modules/{module_id}/{version}/tests/execution_log.json
```

The execution log must contain the following fields:

```json
{
  "module_id": "MM-...",
  "module_version": "x.y.z",
  "executed_at": "ISO-8601",
  "tests": [
    {
      "name": "test_function_or_case_name",
      "result": "PASS | FAIL",
      "duration_ms": 42
    }
  ],
  "summary": {
    "total": 5,
    "passed": 5,
    "failed": 0
  }
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `module_id` | Yes | The module under test. |
| `module_version` | Yes | The version under test. |
| `executed_at` | Yes | ISO-8601 timestamp of when the test run started. |
| `tests` | Yes | Array of per-test results. At least one entry required. |
| `tests[].name` | Yes | Unique name identifying the test case. |
| `tests[].result` | Yes | `"PASS"` or `"FAIL"` — no other values permitted. |
| `tests[].duration_ms` | No | Wall-clock duration of the individual test in milliseconds. |
| `summary.total` | Yes | Total number of tests executed. Must equal `len(tests)`. |
| `summary.passed` | Yes | Count of tests with `result == "PASS"`. |
| `summary.failed` | Yes | Count of tests with `result == "FAIL"`. |

#### Ship-blocking policy

* **Zero-tolerance:** Ship is blocked if `summary.failed > 0`. Any single test failure prevents the `ShipEvidence` record from being written.
* A blocked ship due to test failure is treated identically to a debate FAIL — the Proposer must fix the implementation and retry within the bounded retry budget (max 2 revisions).
* If retries are exhausted with tests still failing, the task HALTs with an escalation artifact that includes the full `execution_log.json` as evidence.

---

## 12) Opaque implementation rule (black-box reuse)

### 12.1 Rule

For any **shipped** micro module, other coding agents:

* cannot view implementation text
* cannot "review" implementation details
* can only integrate through contract + examples + ship evidence

Every agent treats every other module as a **closed-source service**.

**Enforced by:** `OpaqueEnforcementBackend` (Phase 3.1) structurally makes sealed `implementation/` directories unreadable to all agents. Read operations on sealed paths return access-denied errors at the storage layer — this is not a prompt instruction but a backend enforcement.

### 12.2 Allowed surfaces for reuse

A shipped module must expose, and downstream may consume:

* `contract.json` (full I/O contract with all five dimensions)
* `examples.md` (usage, expected behaviors)
* `ship.json` (evidence pointer + fingerprint)
* Code Index entry (summary + pointers)

These are the only files served by `ContextPackBackend` when downstream agents reference a shipped module. The backend physically cannot return implementation source — only contract surfaces.

### 12.3 `examples.md` ownership and lifecycle

The **Proposer** (the agent implementing the module) is responsible for writing `examples.md`. It is a required deliverable of the Proposal artifact — written **before** the debate begins, not after.

**Lifecycle:**

1. **Proposal phase.** The Proposer authors `examples.md` alongside `contract.json` as part of the implementation deliverable. Both are committed to the module directory before the debate starts.
2. **Debate phase.** The Challenger validates `examples.md` completeness as part of the evaluation rubric. An incomplete or missing `examples.md` is grounds to reject the proposal.
3. **Post-ship.** `examples.md` becomes a permanent public surface alongside `contract.json` and `ship.json`. It is served by `ContextPackBackend` to all downstream agents and indexed via the `examples_ref` field in the Code Index entry.

### 12.4 Minimum `examples.md` content

Every `examples.md` must meet the following floor. A file that fails any of these requirements is considered incomplete and the Challenger must reject it.

**Entry point coverage:**

* At least **one example per public entry point** (function, method, or endpoint) declared in `contract.json`.
* Each example must include: the input (as a fenced code block), the expected output (as a fenced code block), and a brief prose description of what the example demonstrates.

**Error case coverage:**

* At least **one error case example per declared error surface** in `contract.json`.
* Each error case example must include: the input that triggers the error (as a fenced code block) and the expected error response (as a fenced code block).

**Effect coverage:**

* If the module declares effects (writes, calls, mutations) in `contract.json`, at least **one example showing the effect in context** — demonstrating the before/after state or the observable side effect.

**Concreteness and self-containment:**

* All examples must use **concrete values** — actual strings, numbers, objects. Placeholders like `"some_input"`, `"..."`, or `<your_value_here>` are not permitted.
* All examples must be **self-contained** — runnable or verifiable without external context beyond the module's declared dependencies.

**Format:**

* Markdown with fenced code blocks (` ``` `) for all inputs and outputs.
* Each example should be a separate subsection or list item with its description, input block, and output block grouped together.

### 12.5 Error message on blocked access

When `OpaqueEnforcementBackend` blocks an operation that would expose sealed content, it returns an error with the following exact format:

```
ACCESS DENIED: '{path}' is a sealed implementation directory. This module is a closed-source service — integrate via contract.json, examples.md, and ship.json only. See §12 (Opaque Implementation Rule) and §6.8 (Black-Box Integration) for rationale. If the contract is insufficient, raise an Interface Change Exception (§13).
```

Where `{path}` is the literal path that was blocked.

**Required error content.** Every blocked-access error message must include all three of:

1. **The path that was blocked** — so the caller knows which specific access was denied.
2. **The alternative files to use** — `contract.json`, `examples.md`, and `ship.json` — so the caller has an immediate remediation path.
3. **A reference to the ICE process (§13)** — so the caller knows how to escalate if the contract surfaces are insufficient for their integration need.

**Covered operations.** The same error format applies uniformly to all operations that would expose sealed content:

| Operation | Behavior on sealed path |
|-----------|------------------------|
| `read` | Returns the error message string instead of file content |
| `edit` | Returns an `EditResult` with the error message |
| `grep` / `grep_raw` | Silently filters sealed paths from results (no error — sealed files simply do not appear) |
| `glob` / `glob_info` | Silently filters sealed paths from results (no error — sealed files simply do not appear) |
| `ls` / `ls_info` | Silently filters sealed entries from results (no error — sealed directories simply do not appear) |

> **Design note:** `read` and `edit` return explicit error messages because the caller targeted a specific path and needs to know why it failed. `grep`, `glob`, and `ls` silently filter because they are discovery operations — exposing that sealed files *exist* without revealing their content is acceptable, but the simpler and safer design is to omit them entirely so agents cannot even enumerate implementation file names.

### 12.6 Why this is mandatory

* Prevents hidden coupling
* Forces contracts to be complete across all five dimensions
* Makes modules replaceable as long as the contract holds
* Improves reuse: agents stop re-implementing after reading internals "to be safe"

---

## 13) Interface change exception (the only valid cross-module interrupt)

### 13.1 Trigger conditions

A downstream contributor may raise an exception only when:

* the current I/O contract cannot support the required workflow
* the contract is ambiguous/incomplete such that black-box integration is unsafe
* a required contract dimension (inputs/outputs/error surfaces/effects/modes) is missing

### 13.2 Eligible raisers

Only three actors in the system may raise an ICE. Each has a distinct context in which the need for a contract change becomes apparent.

| Actor | When | Context |
|---|---|---|
| **Proposer agent** (§3.4) | During implementation | The Proposer discovers that an upstream dependency's contract is insufficient for the required workflow — e.g., a needed input field is missing, an error surface is undeclared, or the contract's output invariants are too weak to satisfy the current module's post-conditions. |
| **Challenger agent** (§3.4) | During debate | The Challenger identifies a contract defect in an upstream dependency (not the module currently being debated). For example, the Challenger determines that the Proposer's integration with dependency `MM-0042@1.0.0` relies on an output invariant that `MM-0042`'s contract does not actually guarantee. |
| **Ship node** (§3.4, §11.2) | During fingerprint mismatch reconciliation | The ship node detects a fingerprint mismatch (§11.2) whose root cause is an upstream dependency's contract — not a defect in the current module's implementation. The Proposer, during a retry after mismatch rejection, may raise the ICE on behalf of the ship pipeline. |

**Cannot raise an ICE:**

* The **dispatch node** (§3.3) — dispatch is pure Python with no LLM involvement. It executes the Dispatch Algorithm deterministically and has no capacity to evaluate contract sufficiency.
* The **Arbiter agent** (§3.4) — the Arbiter adjudicates the current debate; it does not evaluate upstream dependencies. If the Arbiter identifies a dependency concern, it issues a `REJECT` decision with rationale, and the Proposer may then raise an ICE during revision.

**Directional invariant:** An ICE is always raised against an **upstream** dependency's contract (`target_module`), never against the module currently being implemented. If the current module's own contract needs revision, that is handled through the normal debate mechanism (§9) — the Challenger rejects the proposal, the Arbiter issues `REJECT` or `APPROVE_WITH_AMENDMENTS`, and the Proposer revises. The ICE workflow is exclusively for cross-module contract defects in dependencies that the current module cannot control.

### 13.3 Exception schema

```json
{
  "exception_id": "ICE-...",
  "type": "CANNOT_SUPPORT | AMBIGUOUS | INCOMPLETE | MISSING_DIMENSION",
  "raised_by": { "artifact_ref": "ref", "role": "string", "run_id": "string" },
  "target_module": "MM-...@x.y.z",
  "reason": "string",
  "evidence": ["string (min 1)"],
  "proposed_contract_delta": {
    "added_inputs": [],
    "removed_inputs": [],
    "modified_inputs": [],
    "added_outputs": [],
    "removed_outputs": [],
    "modified_outputs": [],
    "added_error_surfaces": [],
    "added_effects": [],
    "mode_changes": null
  },
  "compatibility_expectation": "BACKWARD_COMPATIBLE | BREAKING | UNKNOWN",
  "urgency": "BLOCKING | HIGH | NORMAL | LOW"
}
```

Exception records are validated as Pydantic models (`InterfaceChangeException`) and persisted to `state_store/exceptions/{exception_id}.json`. The `InterfaceChangeException` model enforces that `proposed_contract_delta` contains at least one change (empty deltas are schema-rejected) and that `evidence` contains at least one entry.

### 13.4 Detection mechanism

Agents raise ICEs via a dedicated tool provided to eligible raisers (§13.2).

**Tool signature:**

```
raise_interface_change_exception(
    target_module: str,     # ModuleRef (e.g., "MM-0001@1.0.0") — must reference a shipped upstream dependency
    reason: str,            # Why the current contract is insufficient
    evidence: list[str],    # Concrete evidence: failing test cases, workflow gaps, unmet post-conditions
    proposed_delta: dict,   # ProposedContractDelta as a JSON-serializable dict (§13.3 schema)
    compatibility: str,     # "BACKWARD_COMPATIBLE" | "BREAKING" | "UNKNOWN"
    urgency: str            # "BLOCKING" | "HIGH" | "NORMAL" | "LOW"
) -> dict
```

**Invocation behavior (step by step):**

1. **ID generation.** The tool generates an `exception_id` with the `ICE-` prefix per the artifact ID scheme (§8.2).
2. **Model construction.** The tool constructs an `InterfaceChangeException` Pydantic model from the input arguments. The `raised_by` field is populated automatically from the current role context (the calling agent's `artifact_ref`, `role`, and `run_id`).
3. **Validation.** The constructed model is validated against the `InterfaceChangeException` schema. If validation fails, the tool returns an error to the calling agent with the Pydantic validation errors. The tool invocation is considered failed — the agent may retry with corrected inputs. No state changes occur on validation failure.
4. **Persistence.** On successful validation, the ICE is persisted to `state_store/exceptions/{exception_id}.json` via `ImmutableArtifactBackend`. Once persisted, the ICE record is immutable.
5. **Return value.** The tool returns: `{ "exception_id": "ICE-...", "status": "RAISED", "task_halted": true }`.
6. **Task halt.** The current task is HALTED. The agent that raised the ICE cannot proceed — no further tool calls or implementation work is permitted for this task in this invocation. The ICE `exception_id` is attached to the task's escalation artifact (§17) as the halt reason.

**Ship node invocation:** When the ship node needs to raise an ICE during fingerprint mismatch reconciliation (§11.2), it invokes the same validation and persistence logic programmatically (not via the tool interface, since the ship node is deterministic Python). The effect is identical: the ICE is persisted to `state_store/exceptions/{exception_id}.json` and the task HALTs.

**Tool availability:** The `raise_interface_change_exception` tool is included in the tool inventory for the Proposer and Challenger roles (§3.4 Context Pack construction). The Arbiter does not receive this tool. The dispatch node does not invoke tools.

### 13.5 Processing flow

The ICE lifecycle proceeds through three phases: raise (automated), human decision (manual), and resolution (automated but human-initiated).

#### Phase 1 — Raise (automated)

1. An eligible agent (§13.2) invokes `raise_interface_change_exception` (§13.4).
2. The ICE is validated and persisted to `state_store/exceptions/{exception_id}.json`.
3. The current task transitions to `HALTED` status. The Engineering Loop reports `HALTED` to the Project Loop.
4. The escalation artifact (§17) is produced, containing: the ICE `exception_id`, the `target_module`, the `reason`, the `evidence`, and the full `proposed_contract_delta`.
5. The Project Loop receives the `HALTED` result and executes standard halt behavior (§13.6).

#### Phase 2 — Human decision (manual)

6. The human reviews the ICE via the escalation artifact. The artifact contains all information needed to evaluate the request: the reason, evidence, proposed delta, compatibility expectation, and urgency.
7. The human decides one of three outcomes:

| Decision | Meaning | Next step |
|---|---|---|
| `APPROVE_CHANGE` | The proposed contract delta is accepted for debate | Proceed to Phase 3 |
| `REJECT_CHANGE` | The proposed contract delta is rejected | The halted task must find an alternative implementation approach that works within the existing contract, or remain `HALTED` for further human guidance |
| `DEFER` | The ICE is acknowledged but no immediate action is taken | The ICE is logged with status `DEFERRED`; the task remains `HALTED`; the human may revisit later |

#### Phase 3 — Resolution (automated, human-initiated)

8. On `APPROVE_CHANGE`, the human invokes the ICE resolution subgraph. The subgraph executes:

   a. **Generate candidate contract.** Apply the `proposed_contract_delta` to the target module's current `MicroModuleContract` (§11.1) to produce a candidate contract with a bumped version — minor bump for `BACKWARD_COMPATIBLE`, major bump for `BREAKING` (per §7.4 versioning rules).

   b. **Debate the candidate contract.** The candidate contract undergoes a full Propose → Challenge → Adjudicate debate (§9). The debate evaluates:
      * Whether the new contract is well-formed and internally consistent.
      * Whether the proposed delta is justified by the evidence in the ICE.
      * Whether the backward compatibility claim is accurate (i.e., existing consumers of the old contract are not broken if `BACKWARD_COMPATIBLE` is claimed).

   c. **On debate approval:** Three outputs are produced:
      1. A new module version directory is created at `state_store/modules/{module_id}/{new_version}/` containing the approved `contract.json` (validated as a `MicroModuleContract`).
      2. A debate record for the new contract (Proposal → Challenge → Adjudication artifacts).
      3. Code Index updates — the old version's entry is marked `SUPERSEDED` (with `superseded_by` pointing to `{module_id}@{new_version}`) and the new version is appended as `SHIPPED`.

   d. **On debate rejection:** The ICE resolution fails. The task remains `HALTED`. The escalation artifact is updated with the debate transcript, and the human must decide next steps (e.g., revise the proposed delta, reject the ICE, or provide alternative guidance).

9. After successful resolution, the halted task is re-queued as `PENDING`. The human resumes the Project Loop, and the task is re-evaluated for dispatch via the standard BLOCKED cascading fixed-point re-evaluation (§3.3).

10. On `REJECT_CHANGE`: the escalation artifact is updated with the human's rejection reason. The human may either:
    * Re-queue the task as `PENDING` for re-execution — the Proposer must find an alternative implementation approach within the existing contract.
    * Leave the task `HALTED` for further human guidance (e.g., if the task is fundamentally infeasible without the contract change).

11. On `DEFER`: no state machine changes occur. The ICE record in `state_store/exceptions/{exception_id}.json` is annotated with status `DEFERRED`. The task remains `HALTED`. The human may revisit the deferred ICE at any future point.

### 13.6 Dispatch interaction

When an ICE is raised, the interaction with the Project Loop's dispatch mechanism follows the standard halt protocol defined in §3.3. No ICE-specific dispatch logic exists — the dispatch node sees only the consequence (a `HALTED` task), not the cause (the ICE).

**Sequence:**

1. **Task halts.** The task that raised the ICE transitions to `HALTED` status. The Engineering Loop reports `HALTED` to the Project Loop's `update_state_machine` node.
2. **BLOCKED cascading.** The `update_state_machine` node executes the forward cascade (§3.3): all tasks that depend on the halted task — directly or transitively — are marked `BLOCKED`.
3. **Dispatch stops.** The Project Loop's conditional edge evaluates the halt condition (§3.3: "If any task is `HALTED`, the loop terminates immediately") and terminates the dispatch loop. No further tasks are dispatched.
4. **Escalation surfaces.** The escalation artifact — containing the ICE record — is surfaced for human review via `LangGraph interrupt()` (§17).
5. **Human initiates resolution.** ICE resolution is a **human-initiated** process. The dispatch loop does not automatically invoke the ICE resolution subgraph. The human reviews the escalation artifact, makes a decision (§13.5 Phase 2), and if `APPROVE_CHANGE`, manually triggers the ICE resolution subgraph.
6. **Resume after resolution.** After ICE resolution (or human decision to re-queue the task), the human resumes the Project Loop via the LangGraph checkpoint (§17). The `dispatch` node re-reads the state machine:
   * If the previously halted task is now `PENDING`, the BLOCKED cascading fixed-point re-evaluation (§3.3) unblocks eligible dependents, and dispatch resumes normally.
   * If the task remains `HALTED` (e.g., `DEFER` or `REJECT_CHANGE` without re-queue), dispatch remains stopped.

**Dispatch node invariant:** The dispatch node never evaluates, processes, or routes ICEs. It observes only task statuses (`PENDING`, `SHIPPED`, `HALTED`, `BLOCKED`, `IN_PROGRESS`). The entire ICE workflow — detection, persistence, human review, resolution subgraph — operates outside the dispatch node's scope.

### 13.7 Resolution rule

Interface change requests must result in:

* a new module version, or a successor module
* a debate loop on the new contract
* explicit superseding/deprecation metadata in the Code Index

**Why:** preserves immutability and keeps dependency graphs stable.

#### Implementation Binding

* Exception processing runs as a dedicated subgraph (Phase 3.3 of the implementation roadmap) that invokes a debate loop on the proposed contract delta.
* The `raise_interface_change_exception` tool is implemented as a `@tool`-decorated function registered on the Proposer and Challenger agents. The tool validates inputs against the `InterfaceChangeException` Pydantic model and persists via `ImmutableArtifactBackend`.
* The ICE resolution subgraph is a `StateGraph` with nodes: `validate_ice` → `assess_impact` → `propose_change` → `debate_round` → `check_resolved` → `finalize`. The `debate_round` ↔ `check_resolved` edge loops until consensus or max rounds (3), with human escalation on non-convergence.
* Resolution produces three outputs:
  1. A new module version with the updated contract (validated as a `MicroModuleContract`).
  2. A debate record for the new contract (Proposal → Challenge → Adjudication).
  3. Updated Code Index entries — the old version is marked `SUPERSEDED` (with `superseded_by` pointing to the new version) and the new version is appended.
* Human decisions (`APPROVE_CHANGE`, `REJECT_CHANGE`, `DEFER`) are surfaced via `LangGraph interrupt()` and resolved via checkpoint resume, consistent with the escalation mechanism (§17).

---

## 14) Code Index (reuse system)

### 14.1 Index is append-only

The Code Index is an append-only ledger of shipped modules + periodic snapshots.

#### Implementation Binding

* The Code Index is an **embedded vector store** — a single source of truth for both structured metadata and semantic search. This replaces the earlier JSONL + snapshot design to eliminate dual-store maintenance.
* Each Code Index entry is stored with its structured metadata (module_id, version, tags, status, I/O summary, etc.) **and** a semantic embedding computed from the entry's `purpose`, `tags`, and `io_summary` fields.
* **Append-only semantics** are enforced at the application layer by the `AppendOnlyCodeIndex` wrapper: entries are inserted but never deleted. Status transitions (e.g., `SHIPPED` → `SUPERSEDED`) are the only permitted mutations, and they are one-way — a status can only advance, never regress.
* **Search modes:** The vector store supports both metadata filtering (by tags, status, module_id, version range) and semantic similarity search (by natural-language purpose description or I/O contract shape). Reuse discovery (§14.3) uses semantic search as the primary mechanism.
* **Persistence and concurrency:** The embedded vector store handles its own disk persistence and write safety — no manual `fcntl.flock` or `fsync` management required.
* **Technology:** An embedded vector store such as ChromaDB (persistent mode, no separate server process) or equivalent. The `AppendOnlyCodeIndex` wrapper abstracts the specific backend behind a clean interface.

#### Embedding model

* The embedding model is a **deployment configuration**, not a spec constraint — implementations may use any model that meets all of the following criteria:
  1. Produces **dense vector embeddings** (not sparse / bag-of-words).
  2. Supports text inputs of **at least 512 tokens**.
  3. Output dimensionality between **256 and 1536** (inclusive).
* **Default recommendation:** use the same embedding model already available in the LangChain/LangGraph ecosystem (e.g., OpenAI `text-embedding-3-small` or an equivalent open-source model such as `sentence-transformers/all-MiniLM-L6-v2`).
* **Embedding input construction:** each entry's embedding is computed from a **single concatenated string** built from three fields of the `CodeIndexEntry`:
  ```
  {purpose}\n{tags joined by ", "}\n{io_summary serialized as JSON string}
  ```
  Specifically: the entry's `purpose` field, followed by a newline, followed by `tags` joined with `", "`, followed by a newline, followed by the `io_summary` object serialized to a JSON string (compact, no extra whitespace). This concatenation is the sole input to the embedding model.
* Changing the embedding model requires **re-indexing all existing entries** — the `AppendOnlyCodeIndex` must expose a `reindex()` operation for this purpose.

### 14.2 Index entry schema

```json
{
  "module_id": "MM-...",
  "version": "x.y.z",
  "name": "string",
  "purpose": "string",
  "tags": ["string"],
  "contract_ref": "ref",
  "examples_ref": "ref",
  "ship_ref": "ref",
  "io_summary": {
    "inputs": ["..."],
    "outputs": ["..."],
    "error_surfaces": ["..."],
    "effects": ["..."],
    "modes": ["..."]
  },
  "dependencies": ["MM-...@range"],
  "status": "SHIPPED | DEPRECATED | SUPERSEDED",
  "superseded_by": "optional MM-...@...",
  "notes": "string"
}
```

### 14.3 Reuse-first governance

Before proposing a new module, a role must:

1. search the Code Index
2. include a **Reuse Search Report**
3. justify creation if reuse is not viable

**Why:** forces compounding leverage.

**Reuse Search Report**

```json
{
  "query": "string",
  "candidates_considered": [{ "module_ref": "MM-...@...", "why_rejected": "string" }],
  "conclusion": "REUSE_EXISTING | CREATE_NEW",
  "justification": "string"
}
```

#### Implementation Binding

* Reuse search is implemented as agent tools: `search_code_index` (Phase 3.2, 3.5 of the implementation roadmap).
* Tools are exposed to both planning nodes (micro-planning in the Engineering Loop) and implementation nodes (Proposer in the Debate Subgraph).
* The `search_code_index` tool queries the vector store using **semantic similarity** (natural-language purpose descriptions, I/O contract shapes) combined with **metadata filters** (tags, status, dependency constraints). This makes reuse discovery effective even when naming conventions differ across modules.
* The `ReuseSearchReport` is a Pydantic model validated at creation — agents must produce a valid report before a `CREATE_NEW` decision is accepted.

#### `search_code_index` tool input schema

The `search_code_index` tool accepts the following input:

```json
{
  "query": "string — natural language description of the desired module purpose",
  "tags": ["optional list of tags to filter by (metadata filter, not semantic)"],
  "input_types": ["optional list of expected input type strings"],
  "output_types": ["optional list of expected output type strings"],
  "include_inactive": "boolean, default false — whether to include DEPRECATED/SUPERSEDED entries",
  "top_k": "integer, default 10 — number of results to return"
}
```

**Search execution order:**

1. **Metadata pre-filter.** If any of `tags`, `input_types`, or `output_types` are provided, the vector store filters the candidate set to entries whose metadata matches **before** any semantic ranking occurs. Each filter is applied as a set-intersection (entry must match at least one value in each provided filter list).
2. **Status filter.** By default, entries with status `DEPRECATED` or `SUPERSEDED` are excluded. Setting `include_inactive=true` overrides this and includes all statuses.
3. **Semantic ranking.** The `query` string is embedded using the same embedding model as the index (see §14.1 — Embedding model). The filtered candidate set is ranked by **cosine similarity** between the query embedding and each candidate's stored embedding.
4. **Top-K selection.** The top `top_k` results (default: 10) are returned, ordered by descending similarity score.

**Result format:** each result includes the full `CodeIndexEntry` metadata **plus** a `similarity_score` field (float, 0.0–1.0). There is **no hard similarity threshold** — all top-K results are returned regardless of score. The agent (or the Reuse Search Report logic) decides whether any candidate is viable based on the scores and metadata.

#### Superseded and deprecated entry handling

* When a module is superseded (e.g., via the Interface Change Exception workflow in §13), the old entry's `status` changes to `SUPERSEDED` and its `superseded_by` field is set to the new version's `ModuleRef` (e.g., `MM-0001@2.0.0`).
* When a module is deprecated without a replacement, its `status` changes to `DEPRECATED` and `superseded_by` remains `null`.
* Both old and new entries **remain in the Code Index** — the index is append-only and entries are never deleted.
* By default, `search_code_index` **excludes** entries with status `DEPRECATED` or `SUPERSEDED`. This ensures agents discover only active, current modules during reuse search.
* Agents can set `include_inactive=true` to include inactive entries. This is useful for understanding module evolution, auditing lineage, or finding the predecessor of a current module.

---

## 15) End-to-end workflow summary

### 15.1 Product stage

* **Input:** human-authored high-level `SPEC.md`
* **Process:** Product Loop researches, expands, structures
* **Output:** complete structured spec with pillars/epics/stories/tasks
* **Gate:** human approval

### 15.2 Project stage

* **Input:** approved spec + Code Index snapshot
* **Process:** Project Loop creates folder hierarchy, populates `{task_id}.md` files, initializes state machine
* **Output:** project folder + state machine, ready for dispatch
* **Ongoing:** dispatches tasks, updates state machine, loops until complete or halted
* **Implementation:** The Project Loop is a custom `StateGraph` with a dispatch cycle (`init_state_machine` → `dispatch` → `engineering` → `update_state_machine`). The `engineering` wrapper node invokes the Engineering Loop as a compiled subgraph, ensuring full state isolation between the orchestration layer and the debate layer.

### 15.3 Engineering stage (per task)

* **Input:** single `{task_id}.md` (contains contract, subtasks, dependency contracts — no source code)
* **Process:** 3-agent debate (Propose → Challenge → Adjudicate)
* **Output on success:** sealed implementation + tests + ship evidence + Code Index entry
* **Output on failure:** escalation artifact + project HALT
* **Ship means:** black-box tests pass + interface fingerprint matches contract
* **Implementation:** The Engineering Loop is a custom `StateGraph` containing `micro_plan`, per-module iteration, a Debate Subgraph (`StateGraph`: `propose` → `challenge` → `route` → `revise`/`adjudicate` → `ship`/`halt`), and a `ship` node. Each debate role produces Pydantic-validated structured output via `model.with_structured_output(Schema)`.

### 15.4 Release stage

* **Input:** set of shipped module versions
* **Output:** release manifest + system-level evidence
* **Ship means:** integration gates pass for the set

#### Trigger

The release stage is triggered in one of two ways:

1. **Automatic (full release):** ALL tasks in the state machine reach `SHIPPED` status. The Project Loop detects that no tasks remain in `PENDING`, `IN_PROGRESS`, `BLOCKED`, or `HALTED` status and automatically initiates the release stage for the complete set of shipped modules.
2. **Manual (partial release):** A human explicitly triggers a partial release for a subset of shipped modules. The human specifies the set of module references (`MM-...@x.y.z`) to include. Every module in the specified set must have `SHIPPED` status in the state machine; if any module is not `SHIPPED`, the partial release request is rejected.

#### Integration gates

Integration gates are a set of verification checks run against the complete set of modules included in the release. Each gate produces a binary `PASS` or `FAIL` result. The gates are:

1. **Dependency check** — Every `depends_on` reference across all included modules points to a module that is also included in the release set with `SHIPPED` status. No dangling or unresolved dependencies are permitted.
2. **Fingerprint check** — For every producer-consumer pair (where module A declares a dependency on module B), the contract fingerprint that module A was built against matches the current contract fingerprint of module B's shipped version. Any mismatch indicates contract drift — the consumer was built against a contract that no longer matches the producer's shipped artifact.
3. **Deprecation check** — No non-`DEPRECATED` module in the release set references (via `dependencies` in its Code Index entry) a module whose Code Index status is `DEPRECATED` or `SUPERSEDED`. This prevents shipping a release that contains active modules depending on deprecated capabilities.
4. **Code Index check** — Every module in the release set has a corresponding Code Index entry with status `SHIPPED`. This confirms that the Code Index is consistent with the state machine and that all ship evidence has been properly registered.

#### Release manifest schema

A successful release stage produces a release manifest conforming to this schema:

```json
{
  "release_id": "REL-{uuid4_short}",
  "created_at": "ISO-8601",
  "spec_version": "string (the spec_version from the ProductSpec that drove this project)",
  "modules": [
    {
      "module_ref": "MM-...@x.y.z",
      "ship_ref": "SHIP-..."
    }
  ],
  "integration_gates": {
    "dependency_check": "PASS | FAIL",
    "fingerprint_check": "PASS | FAIL",
    "deprecation_check": "PASS | FAIL",
    "code_index_check": "PASS | FAIL"
  },
  "overall_result": "PASS | FAIL",
  "notes": "string (human-readable summary; includes failure details when overall_result is FAIL)"
}
```

* `release_id` uses a short UUID4 prefix (first 8 hex characters) for uniqueness without excessive length.
* `modules` is the ordered list of all modules included in this release, each paired with its ship evidence reference.
* `integration_gates` records the individual result of each gate.
* `overall_result` is `PASS` if and only if ALL four integration gates are `PASS`. If any gate is `FAIL`, `overall_result` is `FAIL`.

#### Gate enforcement

ALL integration gates must `PASS` for a successful release. If any gate returns `FAIL`:

1. The release is **blocked** — no release manifest with `overall_result: PASS` is produced.
2. A **failure report** is generated identifying each failing gate, the specific modules and references that caused the failure, and a human-readable explanation of the violation.
3. The failure report is persisted alongside the release manifest (which has `overall_result: FAIL`) so that the failure is auditable.
4. Resolution requires fixing the underlying issue (e.g., re-shipping a module with an updated contract, resolving a deprecated dependency) and re-triggering the release stage.

### 15.5 Crash recovery

* **Mechanism:** LangGraph checkpointing persists the state of every graph and subgraph after each node execution. If the system crashes mid-debate (or at any other point), it can resume from the last checkpoint without losing progress.
* **Implication:** a crash during a Challenger evaluation, for example, does not require re-running the Proposer — the system restores the post-proposal checkpoint and re-enters the Challenger node.
* **Requirement:** checkpoint persistence must be tested explicitly as part of integration testing (see Risk Register in the decision document).

---

## 16) Deprecation and compatibility

* No breaking edits in place.
* Breaking changes require new versions/successor modules.
* Deprecation must include replacement pointers and a migration note in the Code Index.

### Deprecation notification

When a module is deprecated, the following occurs:

1. **Downstream identification:** All modules in the Code Index whose `dependencies` field references the deprecated module are identified via a metadata filter query against the Code Index.
2. **Deprecated module annotation:** A deprecation notice is appended to the deprecated module's `notes` field in its Code Index entry, including the deprecation reason and the ISO-8601 timestamp of the deprecation. The `superseded_by` field is set to point to the replacement module (e.g., `MM-...@x.y.z`).
3. **Discovery-based notification:** There is no active push-notification mechanism. Downstream consumers discover deprecations passively — when an agent queries the Code Index via `search_code_index`, deprecated entries appear with their updated `status`, `superseded_by`, and `notes` fields. The `search_code_index` tool excludes `DEPRECATED` and `SUPERSEDED` entries from results by default (see "Continued referencing of deprecated modules" below), so agents are steered toward the replacement without encountering the deprecated module in normal reuse searches.

### Grace period

There is no enforced grace period for deprecated modules. Specifically:

* **Indefinite retention:** Deprecated modules remain in the Code Index indefinitely. The append-only semantics (§14.1) guarantee that no Code Index entry is ever deleted — deprecation is a status transition, not a removal.
* **Release-time enforcement:** The release stage's deprecation check (§15.4) flags any non-`DEPRECATED` module in the release set that depends on a `DEPRECATED` or `SUPERSEDED` module. This forces resolution before release — either the downstream module must be updated to depend on the replacement, or it must itself be deprecated.
* **No time-based expiration:** There is no countdown or deadline after which deprecated modules become inaccessible. The pressure to migrate comes from the release gate, not from a timer.

### Continued referencing of deprecated modules

Agents **can** still reference `DEPRECATED` modules — their `contract.json`, `examples.md`, and `ship.json` remain accessible in the artifact store. The behavioral rules are:

1. **Default exclusion from reuse search:** The `search_code_index` tool excludes entries with status `DEPRECATED` or `SUPERSEDED` from results by default. This steers agents toward active replacements during normal reuse discovery (§14.3).
2. **Explicit reference permitted:** If an agent explicitly references a deprecated module by its `module_id` and version (bypassing the default search filter), the reference is allowed. The module's artifacts are fully accessible.
3. **Challenger flagging:** If an agent's proposal includes a dependency on a `DEPRECATED` module, the Challenger (§6.4) should flag this during debate as a potential issue — questioning whether the replacement module (`superseded_by`) would be a better fit and whether the deprecation creates a release-blocking risk (per the deprecation check in §15.4).

### Implementation Binding

* Deprecation metadata is stored in Code Index entries: the `status` field transitions from `SHIPPED` to `DEPRECATED` or `SUPERSEDED`.
* The `superseded_by` field in the Code Index entry points to the replacement module (e.g., `MM-...@x.y.z`).
* Migration notes are stored in the Code Index entry's `notes` field, providing downstream consumers with guidance on how to adopt the replacement.

---

## 17) Escalation (human intervention)

The project HALTS when the Engineering Loop's bounded debate cannot converge.

Escalation artifacts must include:

* the full debate trail (proposal/challenge/adjudication)
* failed invariants and evidence
* the task_id and its context
* the state machine snapshot at time of halt
* the minimal decision required from a human (e.g., approve new contract, split module, amend spec)
* a recommended resolution action and supporting context to help the human decide

The Project Loop does not dispatch further tasks until the human resolves the escalation and the halted task exits `HALTED` status — either to `PENDING` (for re-execution), `SHIPPED` (if the human resolved it externally), or `ABANDONED` (if the human decides the task is not needed). After resolution, BLOCKED cascading re-evaluation runs (§3.3) to unblock dependents.

### 17.1 Escalation artifact schema

Each escalation artifact conforms to the following schema, persisted as a Pydantic-validated `EscalationArtifact` model:

```json
{
  "escalation_id": "ESC-{uuid4_short}",
  "task_id": "string",
  "task_ref": "string — path to the {task_id}.md file",
  "created_at": "ISO-8601",
  "debate_trail": {
    "proposals": ["Proposal artifacts (§9)"],
    "challenges": ["Challenge artifacts (§9)"],
    "adjudications": ["Adjudication artifacts (§9)"]
  },
  "failed_invariants": [
    {
      "invariant": "string — the invariant that was violated",
      "evidence": "string — evidence demonstrating the violation"
    }
  ],
  "state_machine_snapshot": "object — the full state machine JSON at time of halt",
  "minimal_decision_required": "string — plain-language description of what the human needs to decide",
  "recommended_resolution": "APPROVE_OVERRIDE | AMEND_SPEC | SPLIT_TASK | REVISE_PLAN | PROVIDE_FIX | ABANDON_TASK",
  "resolution_context": "string — additional context to help the human decide (e.g., which invariants failed most persistently, what the Arbiter's final rationale was)"
}
```

The `escalation_id` uses the prefix `ESC-` followed by an 8-character lowercase hex string derived from `uuid4()`. The `recommended_resolution` is the system's best-guess suggestion; the human is not bound by it.

### 17.2 Human resolution options

The human resolves an escalation by selecting exactly one resolution action from the enum below and providing the required payload. Resolutions are structured inputs — free-text-only responses are not accepted.

| Resolution Action | Required Payload | Effect on Task | Effect on State Machine |
|---|---|---|---|
| **`APPROVE_OVERRIDE`** | `rationale: str` — why the human approves despite debate failure. | The task proceeds to ship. The human's approval replaces the Arbiter's adjudication as the final decision. The rationale is persisted in the escalation artifact. | Task transitions `HALTED → IN_PROGRESS`, re-enters the Engineering Loop at the `ship` node (skipping the debate), then transitions `IN_PROGRESS → SHIPPED` on success. |
| **`AMEND_SPEC`** | `amended_acceptance_criteria: list[str]` — the revised acceptance criteria; `amendment_rationale: str` — why the criteria changed. | The `{task_id}.md` file is updated with the new acceptance criteria. The original criteria are preserved in a `## Amendment History` section for traceability. | Task transitions `HALTED → PENDING`. The task re-enters the dispatch queue and will be re-executed from scratch with a fresh Engineering Loop context. |
| **`SPLIT_TASK`** | `new_tasks: list[{name: str, description: str, acceptance_criteria: list[str], depends_on: list[str]}]` — the decomposed tasks. Each entry follows the same schema as a task definition in the approved spec. | The original task is marked as superseded. New `{task_id}.md` files are created for each subtask, with task IDs generated per the Task ID Scheme (§3.3). | Original task transitions `HALTED → ABANDONED` with `superseded_by` references to the new task IDs. New tasks are added to the state machine with status `PENDING`, `declaration_order` values assigned sequentially after the current maximum, and `depends_on` as specified by the human (which may reference the original task's dependencies or sibling new tasks). |
| **`REVISE_PLAN`** | `revised_micro_plan: list[{module_id: str, description: str, depends_on: list[str]}]` — the human-provided micro plan. | The task restarts from the `micro_plan` node in the Engineering Loop, using the human-provided plan instead of generating one. The prior micro plan and all module-level progress are discarded. | Task transitions `HALTED → IN_PROGRESS`. The Engineering Loop resumes at the `micro_plan` node with the human-provided plan injected into state. |
| **`PROVIDE_FIX`** | `fix_description: str` — what the fix does and why; `fix_artifacts: dict[str, str]` — map of file paths to file contents for the human's code fix. | The human's fix is written to disk. The fix then enters the debate subgraph at the `challenge` node (skipping the Proposer). The Challenger evaluates the fix against the contract; the Arbiter adjudicates. If the debate passes, the fix ships. If it fails, the task re-escalates. | Task transitions `HALTED → IN_PROGRESS`. The Engineering Loop resumes at the debate subgraph with the human's fix as the proposal artifact. On debate success: `IN_PROGRESS → SHIPPED`. On debate failure: `IN_PROGRESS → HALTED` (new escalation produced). |
| **`ABANDON_TASK`** | `rationale: str` — why the task is no longer needed. | The task is permanently closed. No further execution is attempted. The rationale is persisted in the escalation artifact. | Task transitions `HALTED → ABANDONED`. `ABANDONED` is a terminal status (no outbound transitions), equivalent to `SHIPPED` for the purpose of dependency evaluation — dependents of an `ABANDONED` task are **not** unblocked automatically. The human must explicitly re-evaluate dependents (they may also need to be abandoned or amended). |

**Status addition:** `ABANDONED` is added to the task status enum (`PENDING | IN_PROGRESS | SHIPPED | HALTED | BLOCKED | ABANDONED`). It is a terminal status with no outbound transitions. Unlike `SHIPPED`, an `ABANDONED` task does **not** satisfy the `depends_on` requirement of downstream tasks — dependents remain `BLOCKED` until the human explicitly resolves them (via `SPLIT_TASK`, `AMEND_SPEC`, or `ABANDON_TASK` on each dependent).

### 17.3 Resume flow

The resume flow defines how the system pauses for human intervention and resumes after the human provides a resolution.

**Pause mechanics:**

1. When the debate subgraph produces an escalation (the `halt` node fires), the `update_state_machine` node in the Project Loop transitions the task to `HALTED`, executes BLOCKED forward-cascading (§3.3), and persists the escalation artifact.
2. The Project Loop then calls LangGraph `interrupt()`, which pauses graph execution at the current checkpoint. The escalation artifact is included in the interrupt payload so it is surfaced to the human operator.
3. The graph remains suspended at the checkpoint. No further nodes execute. The dispatch loop does not advance.

**Human interaction:**

4. The human reviews the escalation artifact — including the debate trail, failed invariants, state machine snapshot, and recommended resolution — via the LangGraph API or a client application built on it.
5. The human provides their resolution as a structured `Command` injected via the LangGraph API (`graph.update_state(config, values, as_node=...)`) — **not** by editing `state_machine.json` or any persisted file directly. The `Command` contains the resolution action (from the enum in §17.2) and its required payload.

**Resume mechanics:**

6. The graph resumes from the LangGraph checkpoint at the point of interruption. The `Command` is processed by the `update_state_machine` node (or a dedicated `process_human_resolution` node invoked from it).
7. The resolution handler performs the action corresponding to the resolution type:
   * **`APPROVE_OVERRIDE`**: updates the task status to `IN_PROGRESS`, injects the human's rationale as the adjudication, and routes to the `ship` node in the Engineering Loop.
   * **`AMEND_SPEC`**: rewrites the `{task_id}.md` file with updated acceptance criteria, transitions the task to `PENDING`, and resumes the dispatch loop.
   * **`SPLIT_TASK`**: generates new task IDs per the Task ID Scheme (§3.3), creates `{task_id}.md` files for each new task, adds the new tasks to the state machine as `PENDING`, transitions the original task to `ABANDONED`, and resumes the dispatch loop.
   * **`REVISE_PLAN`**: transitions the task to `IN_PROGRESS` and routes to the `micro_plan` node in the Engineering Loop with the human-provided plan injected into state.
   * **`PROVIDE_FIX`**: writes the fix artifacts to disk, transitions the task to `IN_PROGRESS`, and routes to the debate subgraph's `challenge` node with the fix as the proposal.
   * **`ABANDON_TASK`**: transitions the task to `ABANDONED`, persists the rationale, and resumes the dispatch loop. Dependents remain `BLOCKED` (they are not auto-unblocked; see §17.2).
8. After resolution processing, BLOCKED cascading re-evaluation runs (§3.3) for resolutions that change the halted task's status to a non-`HALTED` value.
9. The dispatch loop resumes normally. If new tasks were created (via `SPLIT_TASK`), they participate in the standard dispatch algorithm (§3.3) — filtered by dependency readiness, sorted by `declaration_order`.

### Implementation Binding

* Escalation artifacts are persisted to `state_store/escalations/{escalation_id}.json` as Pydantic-validated models (`EscalationArtifact`).
* The Project Loop's `dispatch` node checks for any `HALTED` task before selecting the next task — if a halted task exists, dispatch stops and does not advance the loop.
* LangGraph `interrupt()` is used to pause graph execution when an escalation is produced. The interrupt payload contains the serialized `EscalationArtifact` so the human operator can review it without querying the filesystem.
* The human injects their resolution via `graph.update_state(config, {"resolution": resolution_command}, as_node="update_state_machine")` using the LangGraph API. The `resolution_command` is a Pydantic-validated model (`HumanResolution`) containing the `action` enum value and the action-specific payload.
* The graph resumes from the LangGraph checkpoint at the point of interruption. The `update_state_machine` node (or a dedicated `process_human_resolution` sub-node) reads the injected resolution from state and executes the corresponding handler.
* Resolution handlers are deterministic Python functions — no LLM involvement. They perform file I/O (creating/updating `{task_id}.md` files), state machine mutations (status transitions, new task insertion), and routing decisions (which Engineering Loop node to resume at).
* For `SPLIT_TASK`: new task IDs are generated via the Task ID Scheme (§3.3) in a single pass, with uniqueness validated against all existing task IDs (including `ABANDONED` tasks). The `declaration_order` for new tasks is assigned as `max(existing_declaration_order) + 1, + 2, ...` to ensure they sort after all originally planned tasks.
* For `PROVIDE_FIX`: the debate subgraph is invoked starting at the `challenge` node. A synthetic `Proposal` artifact is constructed from the human's fix (with the human identified as the proposer) so the Challenger and Arbiter operate on the same artifact schema. If the debate fails, a new escalation is produced (the human may need to iterate).
* The `ABANDONED` status is added to the state machine's status enum. Transition rules: `HALTED → ABANDONED` (via `ABANDON_TASK` or `SPLIT_TASK` on the original task). No outbound transitions from `ABANDONED`.

---

## 18) Technology stack and dependencies

This section records the technology choices agreed upon in the framework decision (see `DECISION.md`). These are binding for implementation.

### 18.1 Orchestration

* **LangGraph `StateGraph`** — three custom graph definitions (Product Loop, Project Loop, Engineering Loop) plus a Debate subgraph
* **`Command(goto=...)`** — intra-graph routing for debate flow and retry logic
* **`Command(graph=PARENT)`** — cross-graph signaling (halt propagation from Engineering to Project Loop)
* **Subgraph composition** — wrapper-node pattern (Key Decision #1): the Project Loop invokes the Engineering Loop as a compiled subgraph within a wrapper node, providing full state isolation
* **LangGraph checkpointing** — crash recovery and resume-from-checkpoint for all graphs and subgraphs

### 18.2 Agent framework

* **`create_agent`** (LangChain) — individual agents within graph nodes (Product Loop, Proposer, Challenger, Arbiter, micro-planner)
* **`model.with_structured_output(Schema)`** — provider-native structured output (Anthropic/OpenAI) for Pydantic-validated debate artifacts (Key Decision #7)

### 18.3 Storage

* **`FilesystemBackend`** (deepagents) — primary disk persistence for the state store layout (§7)
* **`BackendProtocol`** (deepagents) — clean interface for file CRUD, grep, glob operations
* **`CompositeBackend`** (deepagents) — path-based routing to different storage backends
* **`ImmutableArtifactBackend`** (custom) — wraps `FilesystemBackend` to enforce immutability on shipped artifacts
* **`OpaqueEnforcementBackend`** (custom) — seals `implementation/` directories post-ship, returning access-denied on reads
* **`ContextPackBackend`** (custom) — wraps `FilesystemBackend` per role invocation with viewing permission enforcement at the data layer

### 18.4 Context management

* **`SummarizationMiddleware`** (deepagents) — context window management for long tasks
* **`TodoListMiddleware`** (deepagents) — agent planning within nodes
* **`PatchToolCallsMiddleware`** (deepagents) — tool call normalization
* **`FilesystemMiddleware`** (deepagents) — file operation tools for agents

### 18.5 Schema and validation

* **Pydantic models** for all artifact types: `ArtifactEnvelope`, `MicroModuleContract`, `ShipEvidence`, `CodeIndexEntry`, `DebateState`, `Proposal`, `Challenge`, `Adjudication`, `EscalationArtifact`, `ContextPack`, `InterfaceChangeException`, `ReuseSearchReport`, `MicroPlan`
* Validation occurs at the storage layer (write-time), not by prompt instructions

### 18.6 Contract fingerprinting

* **RFC 8785 JSON Canonicalization** — deterministic serialization of contract I/O dimensions
* **SHA-256** — hash of the canonical form produces the interface fingerprint
* **Dependency:** `rfc8785>=0.1.4` (Trail of Bits, Apache-2.0, zero transitive dependencies) — the only new external dependency

### 18.7 Code Index storage

* **Embedded vector store** — single source of truth for Code Index entries (structured metadata + semantic embeddings)
* **Semantic embeddings** — computed from each entry's `purpose`, `tags`, and `io_summary` fields at insert time
* **`AppendOnlyCodeIndex`** — application-layer wrapper enforcing append-only semantics and one-way status transitions
* **Technology:** ChromaDB (persistent mode, embedded, no separate server) or equivalent embedded vector store
* **Dependency:** `chromadb` (or chosen vector store library) — new external dependency

### 18.8 Dependencies

**New (required by this spec):**

* `rfc8785>=0.1.4` — RFC 8785 JSON Canonicalization (contract fingerprinting)
* `chromadb` (or equivalent embedded vector store) — Code Index storage and semantic search

**Existing (already present):**

* Pydantic (transitive via LangChain/LangGraph)
* LangGraph
* LangChain
* deepagents SDK (imported as library, not used as-is)

---

## 19) Risk mitigations (spec-level requirements)

These mitigations are derived from the Risk Register in the framework decision (`DECISION.md`) and are mandatory implementation requirements.

### 19.1 State machine size

* **Risk:** State machine JSON grows too large for LangGraph state when projects have hundreds of tasks.
* **Requirement:** State machine JSON must be stored externally via `FilesystemBackend`. Only current-task metadata (the single `{task_id}` entry being dispatched) is loaded into LangGraph graph state.

### 19.2 Mid-execution crash recovery

* **Risk:** Engineering Loop crashes mid-debate, losing progress.
* **Requirement:** LangGraph checkpointing must be enabled for all graphs and subgraphs. The crash-and-resume path must be tested explicitly in integration tests.

### 19.3 Dispatch loop recursion

* **Risk:** LangGraph recursion limit exceeded in long dispatch loops.
* **Requirement:** `recursion_limit` must be set to at least 1000 (deepagents default). Projects with more than 500 tasks must be monitored for recursion depth.

### 19.4 State key leakage

* **Risk:** The deepagents `files` state key leaks to subagents via `SubAgentMiddleware`, violating context isolation (§6.2).
* **Requirement:** `files` must be added to `_EXCLUDED_STATE_KEYS` in `SubAgentMiddleware`, or subagent state must be built explicitly per role invocation. This must be verified in unit tests.

### 19.5 Code Index integrity

* **Risk:** Code Index entries are mutated or deleted, violating the append-only invariant.
* **Requirement:** The `AppendOnlyCodeIndex` wrapper must be the sole interface for Code Index writes. It must enforce: no deletions, no key reuse, and one-way status transitions only (`SHIPPED` → `DEPRECATED`/`SUPERSEDED`, never backward). The embedded vector store handles its own persistence and concurrency; direct database access bypassing the wrapper is prohibited.

### 19.6 Debate convergence

* **Risk:** Debate loop never converges, causing infinite retries.
* **Requirement:** Debate loop retries are bounded at exactly 2 revision attempts. After 2 failures, the Arbiter must render a final verdict (APPROVE_WITH_AMENDMENTS or REJECT). On REJECT, the project HALTS — no further retries are permitted. This bound must be enforced in the debate `StateGraph` routing logic, not by prompt instructions.

### 19.7 Prompt injection bypass of context isolation

* **Risk:** An agent ignores Context Pack restrictions by hallucinating file contents or referencing implementation details from training data.
* **Requirement:** Context isolation must be enforced at the backend level (`ContextPackBackend`, `OpaqueEnforcementBackend`), not solely by prompt instructions. Additionally, prompt reinforcement should remind agents that all external code is closed-source. Error messages from failed reads should reference the contract-first design principle.
