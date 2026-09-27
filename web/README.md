# TicTacPro — Web App

A fully static, frontend-only TicTacPro game. The rules engine **and** the AI
(tree search with alpha-beta pruning + the opening book distilled from the
exact game-tree solve) run entirely in the browser — the search runs in a Web
Worker so the UI never blocks (with a main-thread fallback if workers are
unavailable). No backend, no build step, no dependencies.

**Interaction model — radial slot targeting.** Every cell shows a faint guide
for each piece you could actually place there (Small disc / Medium ring /
Large ring, minus taken slots and exhausted pieces). Hover a guide's zone and
it lights up; click and that exact piece is placed. No size selector, no dead
clicks — clicking a taken or unavailable spot explains why instead of silently
failing. Touch devices get a tap flow instead: tap a cell, and if more than
one of your pieces fits, a mini chooser pops up.

## Run locally

ES modules require an HTTP server (opening `index.html` via `file://` won't work):

```bash
cd web
python3 -m http.server 8080     # or: npx serve .
# open http://localhost:8080
```

## Deploy

### Cloudflare Pages
1. Cloudflare dashboard → **Workers & Pages → Create → Pages → Connect to Git**, pick this repo.
2. Framework preset: **None** · Build command: *(leave empty)* · Build output directory: **`web`**.
3. Deploy. Done.

(Or without git: `npx wrangler pages deploy web` from the repo root.)

### Vercel
The repo-root `vercel.json` already points Vercel at the `web/` folder with no
build step, so just:
1. vercel.com → **Add New → Project**, import this repo.
2. Deploy with default settings. Done.

(Or from the CLI: `npx vercel deploy` at the repo root.)

## Structure

```
web/
├── index.html        app shell
├── styles.css        design system (light default, dark via toggle)
└── js/
    ├── engine.js     game rules — port of game/tictacpro.py (parity-tested, see tests_web/)
    ├── heuristic.js  easy/medium policies — port of pick_rollout_move + OptimalAgent
    ├── book.js       opening book — port of the exact-solve-derived plans
    ├── minimax.js    iterative-deepening negamax + alpha-beta + TT — port of rl/minimax_agent.py
    ├── ai.js         difficulty dispatcher (easy / medium / hard / expert)
    ├── worker.js     Web Worker wrapper so search never blocks the UI
    ├── serialize.js  board → text, for describing a position to a language model
    └── app.js        UI logic

api/                  (only needed for the LLM tier)
├── _llm.js           the Claude call: prompt, strict tool schema, retry policy
└── llm-move.js       POST /api/llm-move — thin HTTP wrapper
```

## AI difficulties

| Tier | Engine | Notes |
|---|---|---|
| Easy | win-taking + 50% blocks + random | misses on purpose |
| Medium | win > block > threat heuristic | the MCTS rollout policy, with slips |
| Hard | alpha-beta search, ~350 ms | no opening book |
| Expert | alpha-beta search, ~1 s + opening book | plays the proven Red winning lines |
| Claude (LLM) | Claude reads the position as text and reasons | needs a server — see below |

### The LLM tier

Every other tier runs entirely in the browser. This one cannot: an API key must
never ship to the client, so it goes through a serverless function at
`api/llm-move.js`. Everything else stays static.

The board is described in words (per-cell slot listing, both inventories, and
the explicit list of legal move ids like `CC-S`), the rules prompt is sent as a
cached prefix, and the reply comes back through a `strict` tool schema so the
move is always well-formed. An illegal move gets one corrective round-trip, and
that retry is counted rather than hidden — the illegal-move rate is a result.

To enable it after deploying, set the key in Vercel:

```bash
npx vercel env add ANTHROPIC_API_KEY production
npx vercel deploy --prod          # redeploy so the function picks it up
```

Without the key the other four tiers work exactly as before; picking the LLM
tier just reports that it is unavailable.

**Cost:** roughly 10-15¢ per full game on `claude-opus-5`. The rules prompt is
~670 tokens, which clears that model's 512-token caching minimum, so every move
after the first in a game re-reads it at ~0.1× input price. (On Haiku 4.5 the
minimum is 4096 tokens, so the same prefix would silently not cache.)

### Benchmark

`tests_web/llm_bench.mjs` measures how well the model actually plays, sharing
the exact prompt and retry policy the deployed game uses:

```bash
export ANTHROPIC_API_KEY=...
node tests_web/llm_bench.mjs tactics        # 5 positions with provable answers
node tests_web/llm_bench.mjs match expert 4 # head-to-head vs a search tier
node tests_web/llm_bench.mjs sweep          # accuracy across effort levels
```

The tactics suite covers taking a line win, taking a bullseye win, blocking a
line, blocking a bullseye, and blocking into a cell that already holds a piece
of another size (the rule models get wrong most often). Every position is
verified against the engine before any billable call, so a wrong suite fails
loudly instead of producing meaningless numbers. Each run prints accuracy,
illegal-move rate, latency, cache hit rate, and estimated cost.

TicTacPro is exactly solved: **the first player (Red) wins with perfect play
from every opening move**. Expert as Red is meant to be practically unbeatable;
to win as Red yourself, you'll have to out-play the search.

## Tests

Engine parity against the Python reference and AI self-tests live in
`../tests_web/` (Node, no framework):

```bash
node tests_web/parity.mjs        # replays Python-generated traces through the JS engine
node tests_web/engine_unit.mjs
```
