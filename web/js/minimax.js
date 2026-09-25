/**
 * Minimax agent — port of rl/minimax_agent.py MinimaxAgent.
 *
 * Iterative-deepening negamax with alpha-beta, a depth-preferred transposition
 * table (EXACT/LOWER/UPPER), killer moves, aspiration windows, and late-move
 * reduction, all mirroring the Python source. Sizes are 0-indexed per engine.js.
 * Search uses engine searchPush/searchPop (always switches player — negamax
 * invariant). Time limit via performance.now() (global in browsers and Node 16+).
 */

import {
  RED, LINES, idx, cloneGame, legalMoves, wouldWin, searchPush, searchPop,
} from './engine.js';
import { RED_BOOK_DEPTH, BLUE_BOOK_DEPTH, bookMove } from './book.js';

const WIN = 1_000_000;
const LOSS = -1_000_000;
const DRAW = 0;

const EXACT = 0, LOWER = 1, UPPER = 2;

// Flat board index of each line cell's Small slot: LINE_BASE[line][cell] + s
// addresses board[r][c][s].
const LINE_BASE = LINES.map(line => line.map(([r, c]) => r * 9 + c * 3));

// §110 per-size line-threat weights [S, M, L]: Small participates in more
// winning mechanisms and is scarcer, so its threats are worth more.
const SZ_THREAT_W = [70, 50, 30];   // 2-of-3
const SZ_THREAT_W1 = [14, 10, 6];   // 1-of-3

// Anti-diagonal is LINES[7] (TR, CC, BL); main diagonal is LINES[6].
const AD_LINE = 7, MD_LINE = 6;

/**
 * Heuristic from `me`'s perspective (the player to move at the leaf).
 * §110 size-weighted line threats + diagonal-S bonuses + bullseye setups,
 * §124 fork early-return (±200000), §120 beachhead penalty, multi-cell control.
 */
export function evaluate(g, me) {
  const b = g.board;
  const opp = 3 - me;
  let score = 0;

  // 1. Same-size line threats (8 lines × 3 sizes) + fork counters.
  let meLine2 = 0, themLine2 = 0;
  for (let li = 0; li < 8; li++) {
    const base = LINE_BASE[li];
    for (let s = 0; s < 3; s++) {
      let m = 0, t = 0;
      for (let k = 0; k < 3; k++) {
        const v = b[base[k] + s];
        if (v === me) m++;
        else if (v === opp) t++;
      }
      if (t === 0) {
        if (m === 2) { score += SZ_THREAT_W[s]; meLine2++; }
        else if (m === 1) score += SZ_THREAT_W1[s];
      }
      if (m === 0) {
        if (t === 2) { score -= SZ_THREAT_W[s]; themLine2++; }
        else if (t === 1) score -= SZ_THREAT_W1[s];
      }
    }
  }

  // 2. Anti-diagonal and main-diagonal Small bonuses (additive).
  let adMe = 0, adOpp = 0, mdMe = 0, mdOpp = 0;
  for (let k = 0; k < 3; k++) {
    const va = b[LINE_BASE[AD_LINE][k]];
    const vm = b[LINE_BASE[MD_LINE][k]];
    if (va === me) adMe++; else if (va === opp) adOpp++;
    if (vm === me) mdMe++; else if (vm === opp) mdOpp++;
  }
  if (adOpp === 0) score += adMe === 2 ? 500 : adMe === 1 ? 100 : 0;
  if (adMe === 0) score -= adOpp === 2 ? 500 : adOpp === 1 ? 100 : 0;
  if (mdOpp === 0) score += mdMe === 2 ? 300 : mdMe === 1 ? 60 : 0;
  if (mdMe === 0) score -= mdOpp === 2 ? 300 : mdOpp === 1 ? 60 : 0;

  // 4. Bullseye setups; also record per-cell occupancy for step 7.
  let meBull2 = 0, themBull2 = 0, meBull1 = 0, themBull1 = 0;
  const meAny = new Uint8Array(9), oppAny = new Uint8Array(9);
  for (let ci = 0; ci < 9; ci++) {
    const f = ci * 3;
    let m = 0, t = 0;
    for (let s = 0; s < 3; s++) {
      const v = b[f + s];
      if (v === me) m++;
      else if (v === opp) t++;
    }
    meAny[ci] = m > 0 ? 1 : 0;
    oppAny[ci] = t > 0 ? 1 : 0;
    if (t === 0) {
      if (m === 2) { meBull2++; score += 80; }
      else if (m === 1) { meBull1++; score += 25; }
    } else if (m === 0) {
      if (t === 2) { themBull2++; score -= 80; }
      else if (t === 1) { themBull1++; score -= 25; }
    }
  }

  // 5. §124 fork early-return — ≥2 simultaneous 2-of-3 threats is decisive;
  // overrides the rest so diagonal bonuses can't mask opponent forks.
  if (meLine2 + meBull2 >= 2) return 200_000;
  if (themLine2 + themBull2 >= 2) return -200_000;

  // 6. §120 pre-fork beachhead penalty: 2+ uncontested 1-of-3 cells.
  if (themBull1 >= 2) score -= 45 * (themBull1 - 1);
  if (meBull1 >= 2) score += 45 * (meBull1 - 1);

  // 7. Multi-cell line control: pieces (any size) in 2+ cells of a line the
  // opponent is absent from — latent fork threat.
  for (let li = 0; li < 8; li++) {
    const base = LINE_BASE[li];
    let mc = 0, tc = 0;
    for (let k = 0; k < 3; k++) {
      const ci = base[k] / 3;
      mc += meAny[ci];
      tc += oppAny[ci];
    }
    if (tc >= 2 && mc === 0) score -= 40;
    if (mc >= 2 && tc === 0) score += 40;
  }

  return score;
}

// ── Move ordering ────────────────────────────────────────────────────────────
// Center > corners > edges (by cellIdx); Small > Medium > Large.
const POS_SCORE = [20, 0, 20, 0, 40, 0, 20, 0, 20];
const SZ_SCORE = [15, 10, 5];
// Anti-diagonal Small slots as flat board indices.
const AD_SMALL_IDX = [idx(0, 2, 0), idx(1, 1, 0), idx(2, 0, 0)];

const sameMove = (a, b) => a !== null && a.r === b.r && a.c === b.c && a.s === b.s;

/** Port of _order_moves: TT > win > block > killers > positional heuristics. */
function orderMoves(g, moves, player, ttMove, killers) {
  const b = g.board;
  const opp = 3 - player;
  let adMe = 0;
  for (const fi of AD_SMALL_IDX) if (b[fi] === player) adMe++;

  const scored = [];
  for (const m of moves) {
    let pri;
    if (ttMove && sameMove(ttMove, m)) {
      pri = 10_000_000;
    } else if (wouldWin(b, m.r, m.c, m.s, player)) {
      pri = 9_000_000;
    } else if (g.pieces[opp][m.s] > 0 && wouldWin(b, m.r, m.c, m.s, opp)) {
      pri = 8_000_000;
    } else if (killers && killers.some(k => sameMove(k, m))) {
      pri = 7_500_000;
    } else {
      const fi0 = m.r * 9 + m.c * 3;
      let base = POS_SCORE[m.r * 3 + m.c] + SZ_SCORE[m.s];
      // Anti-diagonal Small bonus, scaled by pieces already gathered there.
      if (m.s === 0 && b[fi0] === 0 && AD_SMALL_IDX.includes(fi0)) {
        base += 35 * (adMe + 1);
      }
      // §129/§133 anti-bullseye: contest a cell where ONLY the opponent has
      // pieces — occupying a remaining slot denies bullseye completion.
      const oppIn = (b[fi0] === opp) + (b[fi0 + 1] === opp) + (b[fi0 + 2] === opp);
      const meIn = (b[fi0] === player) + (b[fi0 + 1] === player) + (b[fi0 + 2] === player);
      if (oppIn > 0 && meIn === 0 && b[fi0 + m.s] === 0) base += 200 * oppIn;
      pri = base;
    }
    scored.push({ pri, m });
  }
  scored.sort((a, b2) => b2.pri - a.pri);   // stable in modern JS
  return scored.map(x => x.m);
}

// Cheap TT key: 27 board bytes + player, as a string (mirrors _hash_bf).
function hashKey(g) {
  return String.fromCharCode.apply(null, g.board) + g.current;
}

const TIME_UP = Symbol('timeUp');

// ── Search ───────────────────────────────────────────────────────────────────
const ASP_DELTA = 500;   // aspiration window half-width
const LMR_IDX = 3;       // first 3 moves always searched at full depth
const LMR_DEPTH = 3;     // minimum depth for reduction

class Search {
  constructor(maxDepth, deadline) {
    this.maxDepth = maxDepth;
    this.deadline = deadline;
    this.tt = new Map();               // key → {score, flag, depth, move}
    this.killers = Array.from({ length: maxDepth + 2 }, () => []);
    this.nodes = 0;
  }

  /** Iterative deepening; returns {move, score, depth, nodes}. */
  idSearch(g) {
    const moves = legalMoves(g);
    if (moves.length === 0) return { move: null, score: DRAW, depth: 0, nodes: 0 };
    if (moves.length === 1) return { move: moves[0], score: 0, depth: 1, nodes: 1 };

    let bestMove = moves[0];
    let bestScore = LOSS;
    let depthDone = 0;
    let prevScore = DRAW;

    for (let depth = 1; depth <= this.maxDepth; depth++) {
      try {
        let score, move;
        if (depth <= 2 || Math.abs(prevScore) >= WIN / 2) {
          [score, move] = this.root(g, depth, LOSS - 1, WIN + 1);
        } else {
          // Aspiration window: narrow band around the previous score.
          const a = Math.max(LOSS - 1, prevScore - ASP_DELTA);
          const bnd = Math.min(WIN + 1, prevScore + ASP_DELTA);
          [score, move] = this.root(g, depth, a, bnd);
          if (score <= a) [score, move] = this.root(g, depth, LOSS - 1, bnd);
          else if (score >= bnd) [score, move] = this.root(g, depth, a, WIN + 1);
        }
        bestMove = move;
        bestScore = score;
        prevScore = score;
        depthDone = depth;
        if (Math.abs(score) >= WIN - 100_000) break;  // terminal win, not fork heuristic
      } catch (e) {
        if (e === TIME_UP) break;
        throw e;
      }
    }
    return { move: bestMove, score: bestScore, depth: depthDone, nodes: this.nodes };
  }

  root(g, depth, alpha, beta) {
    let bestScore = LOSS - 1;
    let bestMove = null;
    const h = hashKey(g);
    const entry = this.tt.get(h);
    const moves = orderMoves(g, legalMoves(g), g.current, entry ? entry.move : null, null);
    const origAlpha = alpha;

    for (const m of moves) {
      if (performance.now() > this.deadline) throw TIME_UP;
      const tok = searchPush(g, m.r, m.c, m.s);
      const score = -this.negamax(g, depth - 1, -beta, -alpha, 1);
      searchPop(g, tok);
      if (score > bestScore) {
        bestScore = score;
        bestMove = m;
      }
      if (score > alpha) alpha = score;
      // Cut off on fail-high or a found forced win.
      if (alpha >= beta || bestScore >= WIN - 100_000) break;
    }

    if (bestMove !== null) {
      const flag = bestScore <= origAlpha ? UPPER : bestScore >= beta ? LOWER : EXACT;
      this.tt.set(h, { score: bestScore, flag, depth, move: bestMove });
    }
    return [bestScore, bestMove];
  }

  negamax(g, depth, alpha, beta, ply) {
    this.nodes++;

    if (g.gameOver) {
      return g.winner !== 0 ? LOSS + ply : DRAW;  // mover already switched → loser
    }
    if (depth === 0) return evaluate(g, g.current);

    const moves = legalMoves(g);
    if (moves.length === 0) return DRAW;   // stuck player

    const h = hashKey(g);
    let ttMove = null;
    const entry = this.tt.get(h);
    if (entry) {
      ttMove = entry.move;
      if (entry.depth >= depth) {
        if (entry.flag === EXACT) return entry.score;
        if (entry.flag === LOWER && entry.score > alpha) alpha = entry.score;
        if (entry.flag === UPPER && entry.score < beta) beta = entry.score;
        if (alpha >= beta) return entry.score;
      }
    }

    // Throttled time check.
    if ((this.nodes & 4095) === 0 && performance.now() > this.deadline) throw TIME_UP;

    const killers = ply < this.killers.length ? this.killers[ply] : null;
    const ordered = orderMoves(g, moves, g.current, ttMove, killers);
    const origAlpha = alpha;
    let best = LOSS - 1;
    let bestMv = null;
    const doLmr = depth >= LMR_DEPTH;

    for (let i = 0; i < ordered.length; i++) {
      const m = ordered[i];
      const tok = searchPush(g, m.r, m.c, m.s);
      let score;
      if (doLmr && i >= LMR_IDX && Math.abs(best) < WIN / 2) {
        // LMR: null-window at reduced depth; re-search full on cutthrough.
        score = -this.negamax(g, depth - 2, -alpha - 1, -alpha, ply + 1);
        if (score > alpha) score = -this.negamax(g, depth - 1, -beta, -alpha, ply + 1);
      } else {
        score = -this.negamax(g, depth - 1, -beta, -alpha, ply + 1);
      }
      searchPop(g, tok);
      if (score > best) {
        best = score;
        bestMv = m;
      }
      if (score > alpha) alpha = score;
      if (alpha >= beta) {
        if (Math.abs(score) < WIN / 2 && killers && !killers.some(k => sameMove(k, m))) {
          killers.unshift(m);
          killers.length = Math.min(killers.length, 2);
        }
        break;
      }
    }

    const flag = best <= origAlpha ? UPPER : best >= beta ? LOWER : EXACT;
    this.tt.set(h, { score: best, flag, depth, move: bestMv });
    return best;
  }
}

/**
 * Pick a move for game.current. Port of MinimaxAgent.get_action/get_info.
 * Does not mutate `game`. Book is consulted first when useBook and the piece
 * count is within the mover's book depth; book moves report depth 0, nodes 0.
 *
 * @param {object} game engine.js game state
 * @param {object} opts { timeLimitMs=1000, maxDepth=18, useBook=true }
 * @returns {{move: {r,c,s}|null, score: number, depth: number, nodes: number}}
 */
export function minimaxMove(game, opts = {}) {
  const { timeLimitMs = 1000, maxDepth = 18, useBook = true } = opts;
  if (game.gameOver) return { move: null, score: DRAW, depth: 0, nodes: 0 };

  const player = game.current;
  if (useBook) {
    let pp = 0;
    for (let i = 0; i < 27; i++) if (game.board[i] !== 0) pp++;
    const bookDepth = player === RED ? RED_BOOK_DEPTH : BLUE_BOOK_DEPTH;
    if (pp <= bookDepth) {
      const bm = bookMove(game, player);
      if (bm !== null) return { move: bm, score: 0, depth: 0, nodes: 0 };
    }
  }

  const g = cloneGame(game);
  const search = new Search(maxDepth, performance.now() + timeLimitMs);
  return search.idSearch(g);
}
