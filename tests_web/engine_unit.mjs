/**
 * Unit tests for web/js/engine.js.
 *
 * Run:  node tests_web/engine_unit.mjs   (draw test reads traces.json)
 *
 * Covers: win detection on all 8 lines × 3 sizes, bullseye win, a full
 * 18-piece draw, undoMove round-trip over 50 random plies, searchPush/
 * searchPop LIFO round-trip, winInfo cells, and isLegal rejections.
 */
import { readFileSync } from 'node:fs';
import {
  newGame, legalMoves, isLegal, makeMove, undoMove,
  searchPush, searchPop, winInfo, otherPlayer,
  LINES, idx, RED, BLUE, EMPTY,
} from '../web/js/engine.js';

let passed = 0, failed = 0;
function check(cond, msg) {
  if (cond) { passed++; } else { failed++; console.error(`FAIL: ${msg}`); }
}
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// Deterministic PRNG (mulberry32) so failures are reproducible.
function rng32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), a | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function snapshot(g) {
  return {
    board: Array.from(g.board),
    pieces: [g.pieces[1].slice(), g.pieces[2].slice()],
    current: g.current, winner: g.winner, gameOver: g.gameOver,
    piecesLeft: g.piecesLeft, histLen: g.history.length,
  };
}
const sameState = (g, snap) => eq(snapshot(g), snap);

// ── Win detection: all 8 line orientations × 3 sizes ─────────────────────────
for (let li = 0; li < LINES.length; li++) {
  const line = LINES[li];
  for (let s = 0; s < 3; s++) {
    const g = newGame();
    const s2 = (s + 1) % 3;                 // BLUE filler size ≠ RED's, so slots never clash
    makeMove(g, line[0][0], line[0][1], s); // RED
    makeMove(g, 0, 0, s2);                  // BLUE filler
    makeMove(g, line[1][0], line[1][1], s); // RED
    makeMove(g, 0, 1, s2);                  // BLUE filler
    check(!g.gameOver, `line ${li} size ${s}: premature game over`);
    check(makeMove(g, line[2][0], line[2][1], s), `line ${li} size ${s}: winning move rejected`);
    check(g.gameOver && g.winner === RED, `line ${li} size ${s}: RED win not detected`);
    check(g.current === RED, `line ${li} size ${s}: mover should stay current on win`);
    const info = winInfo(g);
    check(info && info.type === 'line' && info.size === s && eq(info.cells, line),
          `line ${li} size ${s}: winInfo wrong: ${JSON.stringify(info)}`);
  }
}

// ── Bullseye win ─────────────────────────────────────────────────────────────
{
  const g = newGame();
  makeMove(g, 1, 1, 0);  // RED
  makeMove(g, 0, 0, 0);  // BLUE filler
  makeMove(g, 1, 1, 1);  // RED
  makeMove(g, 0, 1, 0);  // BLUE filler
  check(!g.gameOver, 'bullseye: premature game over');
  check(makeMove(g, 1, 1, 2), 'bullseye: winning move rejected');
  check(g.gameOver && g.winner === RED, 'bullseye: RED win not detected');
  check(eq(winInfo(g), { type: 'bullseye', cells: [[1, 1]] }),
        `bullseye: winInfo wrong: ${JSON.stringify(winInfo(g))}`);
}

// ── Draw: full 18-piece game with no winner, taken from the Python traces ────
{
  const traces = JSON.parse(readFileSync(new URL('./traces.json', import.meta.url), 'utf8'));
  const drawn = traces.games.find(t => t.plies.at(-1).winner === 0);
  check(!!drawn, 'draw: no drawn game found in traces.json');
  if (drawn) {
    check(drawn.plies.length === 18, `draw: expected 18 plies, got ${drawn.plies.length}`);
    const g = newGame();
    for (const { move: [r, c, s] } of drawn.plies) {
      check(makeMove(g, r, c, s), `draw: move ${r},${c},${s} rejected`);
    }
    check(g.gameOver && g.winner === EMPTY && g.piecesLeft === 0,
          `draw: bad end state gameOver=${g.gameOver} winner=${g.winner} piecesLeft=${g.piecesLeft}`);
    check(winInfo(g) === null, 'draw: winInfo should be null');
    check(legalMoves(g).length === 0, 'draw: no legal moves after draw');
  }
}

// ── undoMove round-trip across 50 random plies ───────────────────────────────
{
  const rand = rng32(1234);
  let plies = 0;
  while (plies < 50) {
    const g = newGame();
    const snaps = [];
    while (!g.gameOver) {
      const moves = legalMoves(g);
      const m = moves[(rand() * moves.length) | 0];
      snaps.push(snapshot(g));
      makeMove(g, m.r, m.c, m.s);
    }
    plies += snaps.length;
    for (let i = snaps.length - 1; i >= 0; i--) {
      check(undoMove(g), `undo: undoMove failed at depth ${i}`);
      check(sameState(g, snaps[i]),
            `undo: state not restored at depth ${i}: ${JSON.stringify(snapshot(g))} vs ${JSON.stringify(snaps[i])}`);
    }
    check(!undoMove(g), 'undo: undoMove on empty history should return false');
  }
}

// ── searchPush/searchPop LIFO round-trip ─────────────────────────────────────
{
  const rand = rng32(5678);
  for (let trial = 0; trial < 20; trial++) {
    const g = newGame();
    for (let i = 0; i < 4 && !g.gameOver; i++) {   // reach a mid-game position
      const moves = legalMoves(g);
      const m = moves[(rand() * moves.length) | 0];
      makeMove(g, m.r, m.c, m.s);
    }
    const snap = snapshot(g);
    const tokens = [];
    for (let d = 0; d < 6 && !g.gameOver; d++) {
      const moves = legalMoves(g);
      const m = moves[(rand() * moves.length) | 0];
      tokens.push(searchPush(g, m.r, m.c, m.s));
    }
    while (tokens.length) searchPop(g, tokens.pop());
    check(sameState(g, snap), `search trial ${trial}: state not restored after pops`);
  }

  // Winning push still switches player (negamax invariant), and pop restores.
  const g = newGame();
  makeMove(g, 0, 0, 0); makeMove(g, 2, 2, 1);
  makeMove(g, 0, 1, 0); makeMove(g, 2, 1, 1);   // RED threatens row 0 small
  const snap = snapshot(g);
  const tok = searchPush(g, 0, 2, 0);           // RED completes the line
  check(g.gameOver && g.winner === RED && g.current === BLUE,
        'searchPush: winning push must set winner yet switch current player');
  searchPop(g, tok);
  check(sameState(g, snap), 'searchPop: winning push not undone exactly');
}

// ── isLegal rejections ───────────────────────────────────────────────────────
{
  const g = newGame();
  check(!isLegal(g, -1, 0, 0), 'isLegal: r=-1 accepted');
  check(!isLegal(g, 3, 0, 0), 'isLegal: r=3 accepted');
  check(!isLegal(g, 0, -1, 0), 'isLegal: c=-1 accepted');
  check(!isLegal(g, 0, 3, 0), 'isLegal: c=3 accepted');
  check(!isLegal(g, 0, 0, -1), 'isLegal: s=-1 accepted');
  check(!isLegal(g, 0, 0, 3), 'isLegal: s=3 accepted');
  check(isLegal(g, 0, 0, 0), 'isLegal: valid opening move rejected');

  // Exhaust RED's smalls (non-winning cells), leave it RED's turn.
  makeMove(g, 0, 0, 0); makeMove(g, 2, 2, 0);
  makeMove(g, 0, 1, 0); makeMove(g, 2, 1, 0);
  makeMove(g, 1, 0, 0); makeMove(g, 2, 0, 1);
  check(!g.gameOver && g.current === RED && g.pieces[RED][0] === 0,
        'isLegal setup: expected live game, RED to move with 0 smalls');
  check(!isLegal(g, 1, 1, 0), 'isLegal: exhausted-size move accepted');
  check(isLegal(g, 1, 1, 1), 'isLegal: medium move should be legal');
  check(!isLegal(g, 0, 0, 0), 'isLegal: occupied slot accepted');
}

console.log(failed === 0
  ? `UNIT OK: ${passed} checks passed`
  : `UNIT FAIL: ${failed} of ${passed + failed} checks failed`);
process.exit(failed === 0 ? 0 : 1);
