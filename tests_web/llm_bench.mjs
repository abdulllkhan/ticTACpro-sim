/**
 * How well does a language model play TicTacPro?
 *
 * Usage (needs ANTHROPIC_API_KEY):
 *   node tests_web/llm_bench.mjs tactics [effort]        constructed positions with provable answers
 *   node tests_web/llm_bench.mjs match <tier> [games]    head-to-head vs easy|medium|hard|expert
 *   node tests_web/llm_bench.mjs sweep                   tactics accuracy across effort levels
 *
 * Every run costs real money. Each tactics pass is 5 moves; a match is roughly
 * 9 LLM moves per game. Cost is estimated and printed at the end of each run.
 */
import {
  newGame, makeMove, legalMoves, wouldWin, otherPlayer, idx, RED, BLUE,
} from '../web/js/engine.js';
import { moveId } from '../web/js/serialize.js';
import { chooseMove } from '../web/js/ai.js';
import { chooseLlmMove, DEFAULT_MODEL, EFFORTS } from '../api/_llm.js';

// claude-opus-5 list price, $ per 1M tokens. Cache reads are 0.1x input,
// cache writes 1.25x (5-minute TTL).
const PRICE = { input: 5.0, output: 25.0 };
const cost = (u) =>
  (u.input * PRICE.input + u.cacheRead * PRICE.input * 0.1 + u.cacheWrite * PRICE.input * 1.25
    + u.output * PRICE.output) / 1e6;

const totals = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
const track = (u) => { for (const k of Object.keys(totals)) totals[k] += u[k] ?? 0; };
const spent = () => cost(totals);

/** Replay a move list onto a fresh game. Each entry is [r, c, s]. */
function position(moves) {
  const g = newGame();
  for (const [r, c, s] of moves) {
    if (!makeMove(g, r, c, s)) throw new Error(`bad setup move ${r},${c},${s}`);
  }
  return g;
}

// ── Tactics suite ────────────────────────────────────────────────────────────
// Positions where the correct move is provable, so a miss is unambiguous.
// `accept` lists every move that counts as correct.
const TACTICS = [
  {
    name: 'take the same-size line win',
    // Red Smalls on TL and TC; TR-S completes the top row.
    setup: [[0, 0, 0], [1, 0, 1], [0, 1, 0], [1, 2, 1]],
    accept: ['TR-S'],
    tests: 'sees a one-move win along a line',
  },
  {
    name: 'take the bullseye win',
    // Red Small + Medium on CC; CC-L completes the stack.
    setup: [[1, 1, 0], [0, 0, 1], [1, 1, 1], [1, 2, 2]],
    accept: ['CC-L'],
    tests: 'sees that three sizes in one cell wins',
  },
  {
    name: 'block the line threat',
    // Blue Smalls on TL and TC; Red must occupy TR-S.
    setup: [[0, 0, 2], [0, 0, 0], [1, 2, 2], [0, 1, 0]],
    accept: ['TR-S'],
    tests: 'blocks an opponent line before building its own',
  },
  {
    name: 'block the bullseye threat',
    // Blue Small + Medium on CC; Red must take CC-L.
    setup: [[0, 0, 2], [1, 1, 0], [1, 2, 2], [1, 1, 1]],
    accept: ['CC-L'],
    tests: 'blocks a stack, not just a line',
  },
  {
    name: 'block into an occupied cell',
    // Blue Mediums on ML and MR. The blocking square CC already holds a Red
    // Small — its Medium slot is still free. Tests slot independence.
    setup: [[1, 1, 0], [1, 0, 1], [0, 0, 2], [1, 2, 1]],
    accept: ['CC-M'],
    tests: 'understands a cell with a piece in it still accepts other sizes',
  },
];

/**
 * Guard against a silently wrong suite: every "win" position must have exactly
 * one winning move and it must be the accepted one; every "block" position must
 * have no win available and exactly one opponent threat. Runs before any
 * billable API call.
 */
function validateSuite() {
  for (const t of TACTICS) {
    const g = position(t.setup);
    const me = g.current;
    const opp = otherPlayer(me);
    const wins = legalMoves(g).filter((m) => wouldWin(g.board, m.r, m.c, m.s, me)).map(moveId);
    const threats = [];
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        for (let s = 0; s < 3; s++) {
          if (g.board[idx(r, c, s)] === 0 && g.pieces[opp][s] > 0
              && wouldWin(g.board, r, c, s, opp)) threats.push(moveId({ r, c, s }));
        }
      }
    }
    const isWin = wins.length > 0;
    const problem = isWin
      ? (wins.length !== 1 || wins[0] !== t.accept[0]
        ? `expected the unique win ${t.accept[0]}, engine says [${wins}]` : null)
      : (threats.length !== 1 || threats[0] !== t.accept[0]
        ? `expected the unique threat ${t.accept[0]}, engine says [${threats}]` : null);
    if (problem) throw new Error(`tactics suite is wrong — "${t.name}": ${problem}`);
  }
}

async function runTactics(effort) {
  console.log(`\n=== TACTICS  (model ${DEFAULT_MODEL}, effort ${effort}) ===\n`);
  let correct = 0, retries = 0, totalMs = 0;

  for (const t of TACTICS) {
    const game = position(t.setup);
    let line;
    try {
      const r = await chooseLlmMove(game, { effort });
      track(r.usage);
      totalMs += r.ms;
      if (r.retried) retries++;
      const hit = t.accept.includes(r.moveId);
      if (hit) correct++;
      line = `${hit ? 'PASS' : 'FAIL'}  ${t.name}\n        played ${r.moveId}`
        + `${hit ? '' : `, expected ${t.accept.join(' or ')}`}`
        + `  (${(r.ms / 1000).toFixed(1)}s${r.retried ? ', needed a retry' : ''})`
        + `\n        "${r.reasoning}"`;
    } catch (err) {
      line = `ERROR ${t.name}\n        ${err.message}`;
    }
    console.log(line + '\n');
  }

  console.log(`Score: ${correct}/${TACTICS.length}   illegal-move retries: ${retries}`
    + `   mean latency: ${(totalMs / TACTICS.length / 1000).toFixed(1)}s`);
  return { correct, total: TACTICS.length, retries };
}

// ── Head-to-head ─────────────────────────────────────────────────────────────
async function runMatch(tier, games, effort) {
  console.log(`\n=== MATCH  LLM (${effort}) vs ${tier}   ${games} games ===\n`);
  const score = { llm: 0, tier: 0, draw: 0 };
  let retries = 0, llmMoves = 0;

  for (let i = 0; i < games; i++) {
    const llmSide = i % 2 === 0 ? RED : BLUE;   // alternate colours
    const game = newGame();
    let failed = false;

    while (!game.gameOver) {
      if (game.current === llmSide) {
        try {
          const r = await chooseLlmMove(game, { effort });
          track(r.usage);
          llmMoves++;
          if (r.retried) retries++;
          makeMove(game, r.move.r, r.move.c, r.move.s);
        } catch (err) {
          console.log(`  game ${i + 1}: LLM failed — ${err.message}`);
          failed = true;
          break;
        }
      } else {
        const { move } = chooseMove(game, tier);
        makeMove(game, move.r, move.c, move.s);
      }
    }
    if (failed) continue;

    const result = game.winner === 0 ? 'draw' : game.winner === llmSide ? 'llm' : 'tier';
    score[result]++;
    console.log(`  game ${i + 1}: LLM as ${llmSide === RED ? 'Red ' : 'Blue'}`
      + ` → ${result === 'draw' ? 'draw' : result === 'llm' ? 'LLM wins' : `${tier} wins`}`
      + `  (${game.history.length} plies)`);
  }

  console.log(`\nLLM ${score.llm}  –  ${score.tier} ${tier}   (${score.draw} draws)`);
  console.log(`illegal-move retries: ${retries} of ${llmMoves} moves`
    + ` (${llmMoves ? (100 * retries / llmMoves).toFixed(1) : '0'}%)`);
  return score;
}

// ── Entry point ──────────────────────────────────────────────────────────────
const [mode = 'tactics', arg1, arg2] = process.argv.slice(2);

if (!process.env.ANTHROPIC_API_KEY) {
  console.error('ANTHROPIC_API_KEY is not set — this benchmark calls the Claude API.');
  process.exit(1);
}

const validEffort = (e, fallback = 'medium') => (EFFORTS.includes(e) ? e : fallback);

try {
  validateSuite();   // never spend money on a suite that is itself wrong
  if (mode === 'tactics') {
    await runTactics(validEffort(arg1));
  } else if (mode === 'match') {
    const tier = arg1 || 'expert';
    if (!['easy', 'medium', 'hard', 'expert'].includes(tier)) {
      throw new Error(`unknown tier "${tier}" — use easy|medium|hard|expert`);
    }
    await runMatch(tier, Number(arg2) || 4, 'medium');
  } else if (mode === 'sweep') {
    const results = [];
    for (const effort of ['low', 'medium', 'high']) {
      results.push([effort, await runTactics(effort)]);
    }
    console.log('\n=== SWEEP SUMMARY ===');
    for (const [effort, r] of results) {
      console.log(`  ${effort.padEnd(7)} ${r.correct}/${r.total}   retries: ${r.retries}`);
    }
  } else {
    throw new Error(`unknown mode "${mode}" — use tactics | match | sweep`);
  }

  const cacheTotal = totals.cacheRead + totals.cacheWrite;
  console.log(`\ntokens  in ${totals.input}  out ${totals.output}`
    + `  cache read ${totals.cacheRead}  cache write ${totals.cacheWrite}`);
  if (cacheTotal > 0) {
    console.log(`cache hit rate: ${(100 * totals.cacheRead / cacheTotal).toFixed(0)}%`
      + ' of cacheable prefix tokens served from cache');
  } else {
    console.log('cache: no entries created — the prefix may be under the model minimum');
  }
  console.log(`estimated cost: $${spent().toFixed(4)}`);
} catch (err) {
  console.error(`\n${err.message}`);
  process.exit(1);
}
