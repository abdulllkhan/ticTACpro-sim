/**
 * Self-test for web/js/heuristic.js. Run: node tests_web/heuristic_selftest.mjs
 * Exit code 0 = all checks passed.
 */
import {
  newGame, makeMove, legalMoves, isLegal, RED, BLUE,
} from '../web/js/engine.js';
import {
  randomMove, rolloutMove, optimalMove, easyMove, mediumMove,
} from '../web/js/heuristic.js';

let failures = 0;
function check(name, cond, extra = '') {
  if (cond) console.log(`ok   ${name}`);
  else { failures++; console.log(`FAIL ${name} ${extra}`); }
}

// Deterministic RNG (mulberry32) so match results are reproducible.
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const sameMove = (m, r, c, s) => m && m.r === r && m.c === c && m.s === s;
const fmt = (m) => m ? `{r:${m.r},c:${m.c},s:${m.s}}` : 'null';

// ---------- 1. rolloutMove spot-checks ----------

// (a) Win available: RED has (0,0,S) and (0,1,S); win at (0,2,S).
{
  const g = newGame();
  makeMove(g, 0, 0, 0);  // R
  makeMove(g, 2, 2, 0);  // B
  makeMove(g, 0, 1, 0);  // R
  makeMove(g, 2, 1, 1);  // B (Medium — no BLUE line threat in Smalls)
  const { move, isWin } = rolloutMove(g);
  check('rollout: win pick (0,2,S)', sameMove(move, 0, 2, 0) && isWin === true, fmt(move));
}

// (b) Bullseye block: BLUE holds (1,1,S) and (1,1,M); RED must take (1,1,L).
{
  const g = newGame();
  makeMove(g, 0, 0, 0);  // R
  makeMove(g, 1, 1, 0);  // B
  makeMove(g, 2, 2, 2);  // R
  makeMove(g, 1, 1, 1);  // B — bullseye 2/3 at center
  const { move, isWin } = rolloutMove(g);
  check('rollout: bullseye block (1,1,L)', sameMove(move, 1, 1, 2) && isWin === false, fmt(move));
}

// (c) Quiet fresh board: no wins/blocks/threats anywhere, so the deterministic
// "rest" tier fires at the first free slot in scan order: (0,0,SMALL).
{
  const { move, isWin } = rolloutMove(newGame());
  check('rollout: fresh board rest pick (0,0,S)', sameMove(move, 0, 0, 0) && isWin === false, fmt(move));
  check('rollout: fresh board pick is legal', isLegal(newGame(), move.r, move.c, move.s));
}

// ---------- 2. optimalMove spot-checks ----------

// Empty board: no win/block, no denial; step 4 anti-diag plan tries
// _ANTIDIAG_S in order [(0,2,S),(1,1,S),(2,0,S)] => (0,2,SMALL).
{
  const m = optimalMove(newGame());
  check('optimal: empty board => (0,2,S)', sameMove(m, 0, 2, 0), fmt(m));
}

// Bullseye-block priority (same position as 1b).
{
  const g = newGame();
  makeMove(g, 0, 0, 0);
  makeMove(g, 1, 1, 0);
  makeMove(g, 2, 2, 2);
  makeMove(g, 1, 1, 1);
  const m = optimalMove(g);
  check('optimal: bullseye block (1,1,L)', sameMove(m, 1, 1, 2), fmt(m));
}

// Immediate win beats everything (same position as 1a).
{
  const g = newGame();
  makeMove(g, 0, 0, 0);
  makeMove(g, 2, 2, 0);
  makeMove(g, 0, 1, 0);
  makeMove(g, 2, 1, 1);
  const m = optimalMove(g);
  check('optimal: win pick (0,2,S)', sameMove(m, 0, 2, 0), fmt(m));
}

// ---------- 3. Match strength ----------

function playMatch(redPolicy, bluePolicy) {
  const g = newGame();
  while (!g.gameOver) {
    const m = g.current === RED ? redPolicy(g) : bluePolicy(g);
    if (!m) break;
    makeMove(g, m.r, m.c, m.s);
  }
  return g.winner;  // 0 = draw
}

{
  const rng = mulberry32(1234);
  let wins = 0;
  for (let i = 0; i < 50; i++) {
    if (playMatch(g => mediumMove(g, rng), g => randomMove(g, rng)) === RED) wins++;
  }
  check(`medium(RED) vs random: ${wins}/50 wins (need >= 40)`, wins >= 40);
}

{
  const rng = mulberry32(5678);
  let wins = 0, losses = 0;
  for (let i = 0; i < 50; i++) {
    // Alternate colors so first-move advantage doesn't decide the check.
    const easyIsRed = i % 2 === 0;
    const w = easyIsRed
      ? playMatch(g => easyMove(g, rng), g => randomMove(g, rng))
      : playMatch(g => randomMove(g, rng), g => easyMove(g, rng));
    const easyColor = easyIsRed ? RED : BLUE;
    if (w === easyColor) wins++;
    else if (w !== 0) losses++;
  }
  check(`easy vs random (alternating colors): ${wins}W-${losses}L (need W > L)`, wins > losses);
}

// ---------- 4. Legality sweep ----------

{
  const rng = mulberry32(42);
  const policies = [
    ['random', g => randomMove(g, rng)],
    ['rollout', g => rolloutMove(g).move],
    ['optimal', g => optimalMove(g)],
    ['easy', g => easyMove(g, rng)],
    ['medium', g => mediumMove(g, rng)],
  ];
  let queries = 0, bad = 0;
  for (let game = 0; game < 200; game++) {
    const g = newGame();
    while (!g.gameOver) {
      for (const [name, pol] of policies) {
        const m = pol(g);
        queries++;
        if (!m || !isLegal(g, m.r, m.c, m.s)) {
          bad++;
          console.log(`  illegal from ${name} at ply ${g.history.length}: ${fmt(m)}`);
        }
      }
      const moves = legalMoves(g);
      const m = moves[(rng() * moves.length) | 0];
      makeMove(g, m.r, m.c, m.s);
    }
  }
  check(`legality sweep: ${queries} queries across 200 games, 0 illegal`, bad === 0, `${bad} illegal`);
}

console.log(failures === 0 ? '\nALL CHECKS PASSED' : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
