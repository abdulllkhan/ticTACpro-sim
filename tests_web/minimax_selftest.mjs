/**
 * Self-test for web/js/minimax.js + web/js/book.js.
 * Run: node tests_web/minimax_selftest.mjs [quick|gauntlet-red|gauntlet-blue|all]
 */

import {
  RED, BLUE, newGame, makeMove, legalMoves, isLegal,
} from '../web/js/engine.js';
import { randomMove } from '../web/js/heuristic.js';
import { bookMove } from '../web/js/book.js';
import { minimaxMove } from '../web/js/minimax.js';

let pass = 0, fail = 0;
function check(name, cond, extra = '') {
  if (cond) { pass++; console.log(`  ok   ${name}${extra ? '  (' + extra + ')' : ''}`); }
  else { fail++; console.log(`  FAIL ${name}${extra ? '  (' + extra + ')' : ''}`); }
}

const mvEq = (m, r, c, s) => m !== null && m.r === r && m.c === c && m.s === s;
const mvStr = (m) => m ? `(${m.r},${m.c},${m.s})` : 'null';

// Deterministic RNG (mulberry32).
function rng32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function playSeq(seq) {
  const g = newGame();
  for (const [r, c, s] of seq) {
    if (!makeMove(g, r, c, s)) throw new Error(`illegal setup move (${r},${c},${s})`);
  }
  return g;
}

// ── 1. Determinism & no-crash ────────────────────────────────────────────────
function testBasics() {
  console.log('1. determinism & no-crash');
  const g = newGame();
  const snap = JSON.stringify({ b: Array.from(g.board), cur: g.current, pl: g.piecesLeft });

  const r1 = minimaxMove(g, { timeLimitMs: 300 });
  check('fresh game returns a legal move (book on)',
    r1.move !== null && isLegal(g, r1.move.r, r1.move.c, r1.move.s), mvStr(r1.move));
  check('book opener is TC-L', mvEq(r1.move, 0, 1, 2));

  const r2 = minimaxMove(g, { timeLimitMs: 300, useBook: false });
  check('fresh game returns a legal move (search, 300ms)',
    r2.move !== null && isLegal(g, r2.move.r, r2.move.c, r2.move.s),
    `${mvStr(r2.move)} depth=${r2.depth} nodes=${r2.nodes} score=${r2.score}`);

  check('caller game not mutated',
    JSON.stringify({ b: Array.from(g.board), cur: g.current, pl: g.piecesLeft }) === snap);

  // Fixed depth + generous time → identical results across runs.
  const d1 = minimaxMove(g, { timeLimitMs: 60000, maxDepth: 4, useBook: false });
  const d2 = minimaxMove(g, { timeLimitMs: 60000, maxDepth: 4, useBook: false });
  check('deterministic at fixed depth 4',
    mvEq(d1.move, d2.move.r, d2.move.c, d2.move.s)
      && d1.score === d2.score && d1.nodes === d2.nodes && d1.depth === 4,
    `${mvStr(d1.move)} score=${d1.score} nodes=${d1.nodes}`);
}

// ── 2. Tactical ──────────────────────────────────────────────────────────────
function testTactics() {
  console.log('2. tactical');

  // (a) Immediate win: RED has TL-L + TC-L; (0,2,2) completes top-row Large.
  const ga = playSeq([[0, 0, 2], [1, 0, 0], [0, 1, 2], [2, 1, 0]]);
  const wa = minimaxMove(ga, { timeLimitMs: 500, useBook: false });
  check('takes immediate line win (search)', mvEq(wa.move, 0, 2, 2), mvStr(wa.move));
  const wb = minimaxMove(ga, { timeLimitMs: 500 });
  check('takes immediate line win (book path)', mvEq(wb.move, 0, 2, 2), mvStr(wb.move));

  // (b) Forced block: BLUE has TL-S + TC-S → RED must take (0,2,0).
  const gb = playSeq([[1, 0, 2], [0, 0, 0], [2, 1, 1], [0, 1, 0]]);
  const ba = minimaxMove(gb, { timeLimitMs: 500, useBook: false });
  check('blocks line threat (search)', mvEq(ba.move, 0, 2, 0), mvStr(ba.move));
  const bb = minimaxMove(gb, { timeLimitMs: 500 });
  check('blocks line threat (book path)', mvEq(bb.move, 0, 2, 0), mvStr(bb.move));

  // (c) Bullseye block: BLUE has BR-S + BR-M → RED must take (2,2,2).
  const gc = playSeq([[0, 1, 0], [2, 2, 0], [1, 0, 1], [2, 2, 1]]);
  const ca = minimaxMove(gc, { timeLimitMs: 500, useBook: false });
  check('blocks bullseye threat (search)', mvEq(ca.move, 2, 2, 2), mvStr(ca.move));
  const cb = minimaxMove(gc, { timeLimitMs: 500 });
  check('blocks bullseye threat (book path)', mvEq(cb.move, 2, 2, 2), mvStr(cb.move));
}

// ── 4. Book sanity ───────────────────────────────────────────────────────────
function testBook() {
  console.log('4. book sanity');
  const g0 = newGame();
  check('empty board RED book = TC-L', mvEq(bookMove(g0, RED), 0, 1, 2));

  const g1 = playSeq([[0, 1, 2], [1, 1, 2]]);          // RED TC-L, BLUE CC-L
  check('RED book vs CC-L = TC-S', mvEq(bookMove(g1, RED), 0, 1, 0),
    mvStr(bookMove(g1, RED)));

  const g2 = playSeq([[0, 1, 2]]);                      // RED TC-L → BLUE to move
  check('BLUE book vs TC-L = CC-S', mvEq(bookMove(g2, BLUE), 1, 1, 0));

  const g3 = playSeq([[1, 1, 2]]);                      // RED CC-L → §124 counter-book
  check('BLUE book vs CC-L = CC-M', mvEq(bookMove(g3, BLUE), 1, 1, 1));

  const g4 = playSeq([[0, 0, 1]]);                      // RED TL-M → §119 corner counter
  check('BLUE book vs corner-M = corner-S', mvEq(bookMove(g4, BLUE), 0, 0, 0));

  const g5 = playSeq([[0, 1, 2], [1, 1, 1]]);           // RED TC-L, BLUE CC-M
  check('RED book vs CC-M = CC-S (VS_CCM plan)', mvEq(bookMove(g5, RED), 1, 1, 0));

  const g6 = playSeq([[0, 1, 2], [0, 2, 0]]);           // BLUE non-center → §131
  check('RED book vs non-center BLUE = null (search)', bookMove(g6, RED) === null);
}

// ── Speed probe ──────────────────────────────────────────────────────────────
function testSpeed() {
  console.log('speed probe (400ms midgame, book off)');
  // Quiet midgame position: 6 plies, no same-size pairs → no immediate tactics.
  const g = playSeq([[0, 0, 2], [1, 1, 0], [1, 2, 1], [0, 1, 1], [2, 1, 0], [2, 2, 2]]);
  const r = minimaxMove(g, { timeLimitMs: 400, useBook: false });
  check('midgame 400ms reaches depth >= 6', r.depth >= 6,
    `depth=${r.depth} nodes=${r.nodes} score=${r.score} move=${mvStr(r.move)}`);
}

// ── 3. Strength gauntlets ────────────────────────────────────────────────────
function gauntlet(mmPlayer, seed, games, label) {
  const rng = rng32(seed);
  let wins = 0, draws = 0;
  const depths = [], nodesList = [];
  for (let i = 0; i < games; i++) {
    const g = newGame();
    while (!g.gameOver) {
      let m;
      if (g.current === mmPlayer) {
        const r = minimaxMove(g, { timeLimitMs: 400 });
        m = r.move;
        if (r.nodes > 0) { depths.push(r.depth); nodesList.push(r.nodes); }
      } else {
        m = randomMove(g, rng);
      }
      if (m === null) break;
      makeMove(g, m.r, m.c, m.s);
    }
    if (g.winner === mmPlayer) wins++;
    else if (g.winner === 0) draws++;
  }
  depths.sort((a, b) => a - b);
  const dStats = depths.length
    ? `search depths min/med/max = ${depths[0]}/${depths[(depths.length / 2) | 0]}/${depths[depths.length - 1]}, ` +
      `avg nodes = ${Math.round(nodesList.reduce((a, b) => a + b, 0) / nodesList.length)}`
    : 'all moves from book';
  console.log(`  ${label}: ${wins} wins, ${draws} draws, ${games - wins - draws} losses over ${games} games`);
  console.log(`  ${dStats}`);
  return wins;
}

function testGauntletRed() {
  console.log('3a. gauntlet: RED=minimax(400ms, book) vs BLUE=random');
  const wins = gauntlet(RED, 12345, 20, 'RED minimax');
  check('RED minimax wins >= 19/20', wins >= 19, `${wins}/20`);
}

function testGauntletBlue() {
  console.log('3b. gauntlet: RED=random vs BLUE=minimax(400ms, book)');
  const wins = gauntlet(BLUE, 67890, 20, 'BLUE minimax');
  check('BLUE minimax wins >= 15/20', wins >= 15, `${wins}/20`);
}

// ── Main ─────────────────────────────────────────────────────────────────────
const mode = process.argv[2] || 'all';
const t0 = performance.now();
if (mode === 'quick' || mode === 'all') {
  testBasics();
  testTactics();
  testBook();
  testSpeed();
}
if (mode === 'gauntlet-red' || mode === 'all') testGauntletRed();
if (mode === 'gauntlet-blue' || mode === 'all') testGauntletBlue();
console.log(`\n${pass} passed, ${fail} failed  (${((performance.now() - t0) / 1000).toFixed(1)}s)`);
process.exit(fail === 0 ? 0 : 1);
