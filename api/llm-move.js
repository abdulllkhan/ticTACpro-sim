/**
 * POST /api/llm-move — ask Claude for a TicTacPro move.
 *
 * Body:  { state, effort?, model? }   state is a serialized engine game
 * Reply: { move: {r,c,s}, moveId, reasoning, retried, usage, ms }
 *
 * The API key never reaches the browser; it lives in the ANTHROPIC_API_KEY
 * environment variable on Vercel. All the model logic is in ./_llm.js so the
 * benchmark exercises exactly the same code path.
 */
import { reviveGame } from '../web/js/engine.js';
import { chooseLlmMove, EFFORTS } from './_llm.js';

export const config = { maxDuration: 60 };

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).json({ error: 'Use POST' });
  }
  if (!process.env.ANTHROPIC_API_KEY) {
    return res.status(500).json({ error: 'ANTHROPIC_API_KEY is not configured on the server' });
  }

  try {
    const body = typeof req.body === 'string' ? JSON.parse(req.body) : req.body;
    const game = reviveGame(body.state);
    if (game.gameOver) return res.status(400).json({ error: 'Game is already over' });

    const result = await chooseLlmMove(game, {
      effort: EFFORTS.includes(body.effort) ? body.effort : 'medium',
      model: body.model,
    });
    return res.status(200).json(result);
  } catch (err) {
    console.error('llm-move failed:', err);
    const status = err?.status >= 400 && err.status < 600 ? err.status : 502;
    return res.status(status).json({
      error: err?.message || 'LLM request failed',
      attempted: err?.attempted ?? undefined,
    });
  }
}
