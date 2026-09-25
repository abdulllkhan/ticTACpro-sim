/**
 * Opening book — port of the book tables and _book_move from rl/minimax_agent.py.
 *
 * All sizes here are 0-indexed (SMALL=0..LARGE=2), per engine.js; the Python
 * source is 1-indexed (PieceSize.SMALL=1..LARGE=3), so every ported (r, c, size)
 * tuple has size - 1 applied. Plans are arrays of {r, c, s} in priority order.
 */

import {
  RED, BLUE, SMALL, MEDIUM, LARGE, idx,
  legalMoves, wouldWin,
} from './engine.js';

// Book is consulted when total pieces on board <= the mover's book depth.
export const RED_BOOK_DEPTH = 10;  // RED: first 6 moves (pp 0, 2, 4, 6, 8, 10)
export const BLUE_BOOK_DEPTH = 1;  // BLUE: first move only (pp 1)

const mv = (r, c, s) => ({ r, c, s });

// RED opening: exact 6-move winning sequence from TC-L (used when BLUE does NOT
// open CC-L/CC-M). TL-S threatens col-0-S, TL-M forks TL-bullseye vs main-diag-M.
export const _RED_BOOK_PLAN = [
  mv(0, 1, LARGE),    // TC-L  (pp=0)
  mv(1, 1, MEDIUM),   // CC-M  (pp=2)  — sets up main-diag-M fork
  mv(2, 0, SMALL),    // BL-S  (pp=4)  — anti-diag anchor; col-0-S 1/3
  mv(0, 0, SMALL),    // TL-S  (pp=6)  — col-0-S 2/3; forces BLUE ML-S block
  mv(0, 0, MEDIUM),   // TL-M  (pp=8)  — fork: TL bullseye vs main-diag-M
  mv(0, 0, LARGE),    // TL-L  (pp=10) — bullseye win at TL
  mv(2, 2, MEDIUM),   // BR-M  — fallback: main-diag-M win
  mv(0, 2, SMALL),    // TR-S  — fallback
  mv(2, 2, LARGE),    // BR-L  — fallback
];

// RED counter-plan vs BLUE CC-L: proven top-row-L fork win at pp=8.
export const _RED_BOOK_PLAN_VS_CCL = [
  mv(0, 1, LARGE),    // TC-L  (pp=0)
  mv(0, 1, SMALL),    // TC-S  (pp=2)  — threatens TC bullseye; forces BLUE TC-M
  mv(0, 2, SMALL),    // TR-S  (pp=4)  — anchors TR
  mv(0, 2, LARGE),    // TR-L  (pp=6)  — dual fork: TR bullseye vs top-row-L
  mv(0, 0, LARGE),    // TL-L  (pp=8)  — top-row-L win
];

// RED counter-plan vs BLUE CC-M: proven TC-bullseye win at pp=8.
export const _RED_BOOK_PLAN_VS_CCM = [
  mv(0, 1, LARGE),    // TC-L  (pp=0)
  mv(1, 1, SMALL),    // CC-S  (pp=2)  — contests CC
  mv(1, 1, LARGE),    // CC-L  (pp=4)  — CC anchor
  mv(0, 1, SMALL),    // TC-S  (pp=6)  — threatens TC bullseye
  mv(0, 1, MEDIUM),   // TC-M  (pp=8)  — TC bullseye win
];

// BLUE counter-book: contest CC immediately, then pursue the anti-diagonal.
export const _BLUE_BOOK_PLAN = [
  mv(1, 1, SMALL),    // CC-S  — take center, block both diagonals
  mv(1, 1, MEDIUM),   // CC-M  — if RED took CC-S: poisons CC bullseye
  mv(1, 1, LARGE),    // CC-L  — if RED took CC-S + CC-M: still deny the cell
  mv(0, 2, SMALL),    // TR-S  — anti-diag 2/3
  mv(2, 0, SMALL),    // BL-S  — anti-diag win
  mv(0, 0, SMALL),    // TL-S  — start main-diag
  mv(2, 2, SMALL),    // BR-S  — main-diag win
  mv(0, 1, LARGE),    // TC-L  — column staking
  mv(2, 1, LARGE),    // BC-L  — bottom row control
];

// BLUE counter-book vs RED CC-L (§124): poison CC bullseye, then anti-diagonal.
export const _BLUE_BOOK_VS_CCL = [
  mv(1, 1, MEDIUM),   // CC-M  — poisons CC bullseye AND denies center Medium (§127)
  mv(0, 2, SMALL),    // TR-S  — anti-diagonal anchor
  mv(2, 0, SMALL),    // BL-S  — anti-diagonal win
  mv(0, 0, SMALL),    // TL-S  — corner / main-diag
  mv(2, 2, SMALL),    // BR-S  — corner / main-diag
  mv(0, 1, LARGE),    // TC-L  — column staking
  mv(2, 1, LARGE),    // BC-L  — bottom row control
];

const CORNERS = [[0, 0], [0, 2], [2, 0], [2, 2]];
const DIAG_ENDPOINTS = [[[0, 0], [2, 2]], [[0, 2], [2, 0]]];

/**
 * Faithful port of _book_move. Returns the next in-book move for `player`
 * (must be game.current), or null to fall through to search.
 * Priority: immediate win → forced 1-move block → conditional plan selection.
 */
export function bookMove(game, player) {
  const legal = legalMoves(game);
  if (legal.length === 0) return null;
  const b = game.board;
  const opp = 3 - player;

  // Immediate win
  for (const m of legal) {
    if (wouldWin(b, m.r, m.c, m.s, player)) return m;
  }
  // Forced 1-move block — only if opponent still has pieces of that size
  for (const m of legal) {
    if (game.pieces[opp][m.s] > 0 && wouldWin(b, m.r, m.c, m.s, opp)) return m;
  }

  const legalSet = new Set(legal.map(m => idx(m.r, m.c, m.s)));
  const has = (r, c, s) => legalSet.has(idx(r, c, s));
  let pp = 0;
  for (let i = 0; i < 27; i++) if (b[i] !== 0) pp++;

  let plan;
  if (player === RED) {
    // All counter-plans were designed for the TC-L opener (§132): if RED never
    // played TC-L, they don't apply — fall to search.
    const redHasTCL = b[idx(0, 1, LARGE)] === RED;
    if (b[idx(1, 1, LARGE)] === BLUE && b[idx(1, 1, MEDIUM)] !== RED) {
      // BLUE has CC-L and RED hasn't committed CC-M → top-row-L fork plan
      if (!redHasTCL) return null;
      plan = _RED_BOOK_PLAN_VS_CCL;
    } else if (b[idx(1, 1, MEDIUM)] === BLUE) {
      // BLUE has CC-M → TC-bullseye plan
      if (!redHasTCL) return null;
      plan = _RED_BOOK_PLAN_VS_CCM;
    } else {
      // §131/§136: the main plan's fork is only proven vs passive/CC-S BLUE.
      // Fall to search when BLUE occupies a non-center cell.
      if (pp > 0 && b[idx(1, 1, SMALL)] !== BLUE) return null;
      plan = _RED_BOOK_PLAN;
    }
  } else {
    if (b[idx(1, 1, LARGE)] === RED) {
      // §124: RED has CC-L anywhere → defensive counter-book
      plan = _BLUE_BOOK_VS_CCL;
    } else if (pp === 1) {
      // §119: RED opened a corner with Medium → contest that corner's Small
      // slot immediately to poison the bullseye and deny the corner-M fork.
      for (const [cr, cc] of CORNERS) {
        if (b[idx(cr, cc, MEDIUM)] === RED && has(cr, cc, SMALL)) {
          return mv(cr, cc, SMALL);
        }
      }
      plan = _BLUE_BOOK_PLAN;
    } else if (pp === 3) {
      // §125: RED holds both endpoints of a diagonal, one endpoint Small-only →
      // block Medium there to prevent the bullseye + same-size diagonal fork.
      for (const [[r1, c1], [r2, c2]] of DIAG_ENDPOINTS) {
        const red1 = b[idx(r1, c1, 0)] === RED || b[idx(r1, c1, 1)] === RED || b[idx(r1, c1, 2)] === RED;
        const red2 = b[idx(r2, c2, 0)] === RED || b[idx(r2, c2, 1)] === RED || b[idx(r2, c2, 2)] === RED;
        if (!(red1 && red2)) continue;
        const onlyS1 = b[idx(r1, c1, 0)] === RED && b[idx(r1, c1, 1)] === 0 && b[idx(r1, c1, 2)] === 0;
        const onlyS2 = b[idx(r2, c2, 0)] === RED && b[idx(r2, c2, 1)] === 0 && b[idx(r2, c2, 2)] === 0;
        let target = null;
        if (onlyS2 && !onlyS1) target = [r2, c2];
        else if (onlyS1 && !onlyS2) target = [r1, c1];
        if (target && has(target[0], target[1], MEDIUM)) {
          return mv(target[0], target[1], MEDIUM);
        }
      }
      plan = _BLUE_BOOK_PLAN;
    } else {
      plan = _BLUE_BOOK_PLAN;
    }
  }

  for (const m of plan) {
    if (!has(m.r, m.c, m.s)) continue;
    // BLUE: skip any non-S center move if BLUE already has CC-S, so the plan
    // advances to TR-S rather than re-stacking the center.
    if (player === BLUE && m.r === 1 && m.c === 1 && m.s !== SMALL
        && b[idx(1, 1, SMALL)] === BLUE) {
      continue;
    }
    return m;
  }
  return null;
}
