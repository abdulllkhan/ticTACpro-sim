/**
 * Heuristic move policies — ports of pick_rollout_move (game/tictacpro.py)
 * and OptimalAgent (game/optimal_agent.py).
 *
 * All sizes here are 0-indexed (SMALL=0..LARGE=2), per engine.js. The Python
 * sources are 1-indexed; every ported (r, c, size) tuple has size - 1 applied.
 * Scan order matches Python exactly: size-major (S, M, L), cells row-major 0..8.
 * Policies never mutate the game state.
 */

import {
  RED, BLUE, SMALL, WIN_PAIRS, idx, otherPlayer,
  legalMoves, isLegal, wouldWin,
} from './engine.js';

/** Uniform random legal move, or null if none. */
export function randomMove(game, rng = Math.random) {
  const moves = legalMoves(game);
  if (moves.length === 0) return null;
  return moves[(rng() * moves.length) | 0];
}

/**
 * Faithful port of pick_rollout_move: win > bullseye-block > line-block >
 * threat (2-in-a-row setup) > first-available, deterministic.
 * Returns {move, isWin}; {move: null, isWin: false} if no legal moves.
 */
export function rolloutMove(game) {
  if (game.gameOver) return { move: null, isWin: false };
  const bf = game.board;
  const cur = game.current;
  const opp = 3 - cur;
  const pr = game.pieces[cur];
  const oppPr = game.pieces[opp];

  let bullBlock = null;   // bullseye 2/3 block (higher urgency)
  let lineBlock = null;   // standard same-size line block
  let threat = null;      // first 2-in-a-row setup found
  let rest = null;

  for (let s = 0; s < 3; s++) {
    if (pr[s] === 0) continue;
    const oppHasSize = oppPr[s] > 0;  // skip block if opp can't threaten this size
    const wl = WIN_PAIRS[s];

    for (let ci = 0; ci < 9; ci++) {
      if (bf[ci * 3 + s]) continue;   // slot occupied
      const r = (ci / 3) | 0, c = ci % 3;
      const pairs = wl[ci];

      if (bullBlock === null) {
        // Full scan: wins, bullseye blocks, line blocks, threats, rest.
        let isBull = false, isLine = false, isThreat = false;
        for (let k = 0; k < pairs.length; k++) {
          const i1 = pairs[k][0], i2 = pairs[k][1];
          const v1 = bf[i1];
          if (v1 === cur) {
            if (bf[i2] === cur) return { move: { r, c, s }, isWin: true };
            else if (bf[i2] === 0) isThreat = true;
          } else if (v1 === opp) {
            if (bf[i2] === opp && oppHasSize) {
              // Same cell (i/3 = cellIdx) → bullseye threat, else line threat.
              if (((i1 / 3) | 0) === ((i2 / 3) | 0)) isBull = true;
              else isLine = true;
            }
          } else if (bf[i2] === cur) {  // v1 === 0
            isThreat = true;
          }
        }
        // Exact Python elif chain, fall-through quirks included.
        if (isBull) bullBlock = { r, c, s };
        else if (isLine && lineBlock === null) lineBlock = { r, c, s };
        else if (isThreat && threat === null) threat = { r, c, s };
        else if (rest === null) rest = { r, c, s };
      } else {
        // Bullseye block already found — only scan for wins.
        for (let k = 0; k < pairs.length; k++) {
          if (bf[pairs[k][0]] === cur && bf[pairs[k][1]] === cur) {
            return { move: { r, c, s }, isWin: true };
          }
        }
      }
    }
  }

  const move = bullBlock ?? lineBlock ?? threat ?? rest;
  return { move, isWin: false };
}

// Anti-diagonal Small slots for the winning plan, as [r, c, s] (0-indexed).
const ANTIDIAG_S = [[0, 2, SMALL], [1, 1, SMALL], [2, 0, SMALL]];

// _RED_PLAN with Python 1-indexed sizes converted to 0-indexed.
const RED_PLAN = [
  [0, 1, 2],  // TC-L  — opening
  [0, 1, 1],  // TC-M
  [2, 0, 1],  // BL-M
  [1, 1, 2],  // CC-L
  [2, 2, 1],  // BR-M
  [2, 2, 2],  // BR-L
  [1, 1, 0],  // CC-S  ← anti-diag 1/3
  [0, 2, 0],  // TR-S  ← anti-diag 2/3
  [2, 0, 0],  // BL-S  ← anti-diag 3/3  WIN
];

/**
 * Faithful port of OptimalAgent.get_action: win > block (bullseye priority) >
 * anti-diag denial > anti-diag plan > _RED_PLAN > first legal. Null if no moves.
 */
export function optimalMove(game) {
  if (game.gameOver) return null;
  const me = game.current;
  const opp = 3 - me;
  const bf = game.board;
  const pr = game.pieces[me];
  const prOpp = game.pieces[opp];

  // Single-pass scan: first win returns immediately; collect blocks + legal list.
  let bullseyeBlock = null;
  let lineBlock = null;
  const legal = [];

  for (let s = 0; s < 3; s++) {
    if (pr[s] === 0) continue;
    const hasOppSz = prOpp[s] > 0;
    const wl = WIN_PAIRS[s];
    for (let ci = 0; ci < 9; ci++) {
      if (bf[ci * 3 + s]) continue;   // slot occupied
      const r = (ci / 3) | 0, c = ci % 3;
      legal.push({ r, c, s });
      const pairs = wl[ci];
      for (let k = 0; k < pairs.length; k++) {
        const i1 = pairs[k][0], i2 = pairs[k][1];
        const v1 = bf[i1], v2 = bf[i2];
        if (v1 === me && v2 === me) return { r, c, s };
        if (hasOppSz && v1 === opp && v2 === opp) {
          if (((i1 / 3) | 0) === ((i2 / 3) | 0)) {  // same cell → bullseye threat
            if (bullseyeBlock === null) bullseyeBlock = { r, c, s };
          } else if (lineBlock === null) {
            lineBlock = { r, c, s };
          }
        }
      }
    }
  }

  if (legal.length === 0) return null;
  const block = bullseyeBlock ?? lineBlock;
  if (block !== null) return block;

  const legalSet = new Set(legal.map(m => idx(m.r, m.c, m.s)));
  const has = (r, c, s) => legalSet.has(idx(r, c, s));

  // 3. Deny anti-diagonal S cells when opponent has 2/3 and still has Smalls.
  let oppAnti = 0;
  for (const [r, c] of [[0, 2], [1, 1], [2, 0]]) {
    if (bf[idx(r, c, SMALL)] === opp) oppAnti++;
  }
  if (oppAnti >= 2 && prOpp[SMALL] > 0) {
    for (const [r, c, s] of ANTIDIAG_S) {
      if (has(r, c, s) && bf[idx(r, c, SMALL)] === 0) return { r, c, s };
    }
  }

  // 4. Anti-diagonal Small pieces (the winning plan).
  for (const [r, c, s] of ANTIDIAG_S) {
    if (has(r, c, s)) return { r, c, s };
  }

  // 5. Follow the planned sequence in order.
  for (const [r, c, s] of RED_PLAN) {
    if (has(r, c, s)) return { r, c, s };
  }

  // 6. Any legal move.
  return legal[0];
}

/**
 * Easy policy: immediate win if any; else block opponent's immediate win
 * (line or bullseye) with 50% probability; else uniform random legal.
 */
export function easyMove(game, rng = Math.random) {
  const moves = legalMoves(game);
  if (moves.length === 0) return null;
  const me = game.current;
  const opp = otherPlayer(me);
  const oppPieces = game.pieces[opp];

  for (const m of moves) {
    if (wouldWin(game.board, m.r, m.c, m.s, me)) return m;
  }
  let block = null;
  for (const m of moves) {
    // Opponent must still hold a piece of this size to actually threaten it.
    if (oppPieces[m.s] > 0 && wouldWin(game.board, m.r, m.c, m.s, opp)) {
      block = m;
      break;
    }
  }
  if (block !== null && rng() < 0.5) return block;
  return moves[(rng() * moves.length) | 0];
}

/**
 * Medium policy: immediate win always; forced block always (bullseye priority,
 * via rolloutMove); otherwise rolloutMove's pick, with a 15% chance of a
 * uniform random legal move instead.
 */
export function mediumMove(game, rng = Math.random) {
  const { move, isWin } = rolloutMove(game);
  if (move === null) return null;
  if (isWin) return move;

  // Forced block? Opponent has an immediate win at a slot we can play.
  const opp = otherPlayer(game.current);
  const oppPieces = game.pieces[opp];
  const moves = legalMoves(game);
  for (const m of moves) {
    if (oppPieces[m.s] > 0 && wouldWin(game.board, m.r, m.c, m.s, opp)) {
      return move;  // rolloutMove already picked a bullseye-priority block
    }
  }
  if (rng() < 0.15) return moves[(rng() * moves.length) | 0];
  return move;
}
