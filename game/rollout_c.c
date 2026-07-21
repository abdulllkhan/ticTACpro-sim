/*
 * rollout_c.c — C implementation of greedy rollout for TicTacPro.
 *
 * Build:
 *   gcc -O3 -march=native -fPIC -shared -o game/rollout_c.so game/rollout_c.c
 *
 * Exported functions:
 *
 *   run_rollout_c(bf, cur, pr_flat, pieces_left) -> int (winner)
 *     Runs an entire greedy rollout in C; returns 1=RED, 2=BLUE, 0=DRAW.
 *     bf and pr_flat are modified in place.
 *     Blocks only when opponent actually has pieces of the relevant size remaining.
 *
 *   sort_legal_moves_c(bf, pr_cur, cur, opp, out_moves, pr_opp) -> int (count)
 *   sort_legal_moves_ex_c(bf, pr_cur, cur, opp, out_moves, out_n_winning, pr_opp) -> int
 *     pr_opp[3]: opponent's [S,M,L] piece counts — prevents phantom block classification.
 *
 *   pick_rollout_move_c(bf, cur, opp, pr_flat, out_r, out_c, out_size, out_is_win)
 *     Original single-move selector (kept for compatibility).
 *
 * Board layout:
 *   bf[r*9 + c*3 + sz_idx] = player occupying cell (r,c) at size sz_idx
 *   sz_idx: 0=SMALL, 1=MEDIUM, 2=LARGE
 *   player: 0=empty, 1=RED, 2=BLUE
 */

/* WIN_LINES[sz_idx*9 + ci][k]: pairs (i1,i2) of flat board indices that
 * form a 3-in-a-row with cell ci at size sz_idx. Terminated by -1. */
static const int WIN_LINES[27][12] = {
    {3,6,9,18,12,24,1,2,-1,-1,-1,-1},
    {0,6,12,21,4,5,-1,-1,-1,-1,-1,-1},
    {0,3,15,24,12,18,7,8,-1,-1,-1,-1},
    {12,15,0,18,10,11,-1,-1,-1,-1,-1,-1},
    {9,15,3,21,0,24,6,18,13,14,-1,-1},
    {9,12,6,24,16,17,-1,-1,-1,-1,-1,-1},
    {21,24,0,9,6,12,19,20,-1,-1,-1,-1},
    {18,24,3,12,22,23,-1,-1,-1,-1,-1,-1},
    {18,21,6,15,0,12,25,26,-1,-1,-1,-1},
    {4,7,10,19,13,25,0,2,-1,-1,-1,-1},
    {1,7,13,22,3,5,-1,-1,-1,-1,-1,-1},
    {1,4,16,25,13,19,6,8,-1,-1,-1,-1},
    {13,16,1,19,9,11,-1,-1,-1,-1,-1,-1},
    {10,16,4,22,1,25,7,19,12,14,-1,-1},
    {10,13,7,25,15,17,-1,-1,-1,-1,-1,-1},
    {22,25,1,10,7,13,18,20,-1,-1,-1,-1},
    {19,25,4,13,21,23,-1,-1,-1,-1,-1,-1},
    {19,22,7,16,1,13,24,26,-1,-1,-1,-1},
    {5,8,11,20,14,26,0,1,-1,-1,-1,-1},
    {2,8,14,23,3,4,-1,-1,-1,-1,-1,-1},
    {2,5,17,26,14,20,6,7,-1,-1,-1,-1},
    {14,17,2,20,9,10,-1,-1,-1,-1,-1,-1},
    {11,17,5,23,2,26,8,20,12,13,-1,-1},
    {11,14,8,26,15,16,-1,-1,-1,-1,-1,-1},
    {23,26,2,11,8,14,18,19,-1,-1,-1,-1},
    {20,26,5,14,21,22,-1,-1,-1,-1,-1,-1},
    {20,23,8,17,2,14,24,25,-1,-1,-1,-1},
};

/* CELL_SLOT[sz_idx*9 + ci]: flat board index for cell ci at size sz_idx */
static const int CELL_SLOT[27] = {
    0,3,6,9,12,15,18,21,24,   /* sz=0 (SMALL) */
    1,4,7,10,13,16,19,22,25,  /* sz=1 (MEDIUM) */
    2,5,8,11,14,17,20,23,26   /* sz=2 (LARGE) */
};

static const int CELL_R[9] = {0,0,0,1,1,1,2,2,2};
static const int CELL_C[9] = {0,1,2,0,1,2,0,1,2};
static const int SIZES[3]  = {1, 2, 3};


/*
 * run_rollout_c — run a complete greedy rollout from the given state.
 *
 * bf:          signed char[27] — board, modified in place
 * cur:         starting player (1 or 2)
 * pr_flat:     int[6] — [R_S,R_M,R_L,B_S,B_M,B_L], modified in place
 * pieces_left: total pieces remaining for both players
 *
 * Returns: winner (1=RED, 2=BLUE, 0=DRAW)
 */
int run_rollout_c(signed char *bf, int cur, int *pr_flat, int pieces_left) {
    while (pieces_left > 0) {
        int opp   = 3 - cur;
        int p_off = (cur - 1) * 3;

        /* §343: dual-track — bullseye blocks take priority over line blocks. */
        int bull_slot   = -1, bull_sz   = -1;   /* bullseye 2/3 block (same cell) */
        int block_slot  = -1, block_sz  = -1;   /* standard same-size line block   */
        /* §183: threat tier — 2-of-3 setup (one piece + one empty in a win line) */
        int threat_slot = -1, threat_sz = -1;
        int rest_slot   = -1, rest_sz   = -1;

        for (int sz_idx = 0; sz_idx < 3; sz_idx++) {
            if (pr_flat[p_off + sz_idx] == 0) continue;
            int base = sz_idx * 9;

            for (int ci = 0; ci < 9; ci++) {
                int slot = CELL_SLOT[base + ci];
                if (bf[slot] != 0) continue;

                const int *wl = WIN_LINES[base + ci];

                int is_bull = 0, is_block = 0, is_threat = 0;
                for (int k = 0; k < 12; k += 2) {
                    int i1 = wl[k];
                    if (i1 < 0) break;
                    int i2 = wl[k + 1];
                    int v1 = bf[i1], v2 = bf[i2];
                    if (v1 == cur) {
                        if (v2 == cur) {
                            /* Immediate win: place piece and return */
                            bf[slot] = (signed char)cur;
                            return cur;
                        }
                        if (v2 == 0) is_threat = 1;
                    } else if (v1 == 0 && v2 == cur) {
                        is_threat = 1;
                    } else if (v1 == opp && v2 == opp && pr_flat[(opp-1)*3 + sz_idx] > 0) {
                        if (i1 / 3 == i2 / 3) is_bull  = 1;  /* same cell → bullseye */
                        else                   is_block = 1;
                    }
                }

                if (is_bull) {
                    if (bull_slot < 0) { bull_slot = slot; bull_sz = sz_idx; }
                } else if (is_block) {
                    if (block_slot < 0) { block_slot = slot; block_sz = sz_idx; }
                } else if (is_threat) {
                    if (threat_slot < 0) { threat_slot = slot; threat_sz = sz_idx; }
                } else {
                    if (rest_slot < 0) { rest_slot = slot; rest_sz = sz_idx; }
                }
            }
        }

        /* Apply chosen non-winning move: bullseye_block > line_block > threat > rest */
        int chosen_slot, chosen_sz;
        if (bull_slot >= 0) {
            chosen_slot = bull_slot; chosen_sz = bull_sz;
        } else if (block_slot >= 0) {
            chosen_slot = block_slot; chosen_sz = block_sz;
        } else if (threat_slot >= 0) {
            chosen_slot = threat_slot; chosen_sz = threat_sz;
        } else if (rest_slot >= 0) {
            chosen_slot = rest_slot; chosen_sz = rest_sz;
        } else {
            break;  /* No legal moves */
        }
        bf[chosen_slot] = (signed char)cur;
        pr_flat[p_off + chosen_sz]--;
        pieces_left--;
        cur = opp;
    }
    return 0;  /* Draw */
}


/*
 * sort_legal_moves_c — generate AND categorize legal moves in one pass.
 *
 * Output order: [rest..., blocking..., winning...] — pop() returns winning first.
 *
 * bf:       signed char[27]
 * pr_cur:   int[3] — current player's [S, M, L] piece counts (0-indexed)
 * cur/opp:  int (1 or 2)
 * out_moves unsigned char[27] — output move indices in pop()-friendly order
 *
 * Returns: total legal moves. The last out_nwb moves are winning+blocking (at end).
 */
int sort_legal_moves_c(
    const signed char *bf,
    const int *pr_cur,
    int cur,
    int opp,
    unsigned char *out_moves,
    const int *pr_opp
) {
    /* §341: bullseye blocks (same-cell threats) separated from line blocks.
     * Layout: [rest, line_blocking, bullseye_blocking, winning]
     * pop() returns winning first, then bullseye blocks, then line blocks. */
    unsigned char rest[27], line_blk[27], bull_blk[27], winning[27];
    int nr = 0, nl = 0, nbb = 0, nw = 0;

    for (int sz_idx = 0; sz_idx < 3; sz_idx++) {
        if (pr_cur[sz_idx] == 0) continue;
        int base = sz_idx * 9;
        int opp_has_sz = pr_opp[sz_idx] > 0;

        for (int ci = 0; ci < 9; ci++) {
            int slot = CELL_SLOT[base + ci];
            if (bf[slot]) continue;

            unsigned char idx = (unsigned char)(base + ci);
            const int *wl = WIN_LINES[base + ci];

            int is_win = 0, is_bull = 0, is_line = 0;
            for (int k = 0; k < 12; k += 2) {
                int i1 = wl[k];
                if (i1 < 0) break;
                int i2 = wl[k + 1];
                int v1 = bf[i1];
                if (v1 == cur) {
                    if (bf[i2] == cur) { is_win = 1; break; }
                } else if (v1 == opp && bf[i2] == opp && opp_has_sz) {
                    if (i1 / 3 == i2 / 3) is_bull = 1; /* same cell → bullseye */
                    else                   is_line = 1;
                }
            }

            if (is_win)       winning[nw++]   = idx;
            else if (is_bull) bull_blk[nbb++] = idx;
            else if (is_line) line_blk[nl++]  = idx;
            else              rest[nr++]       = idx;
        }
    }

    /* Concatenate [rest, line_blocking, bullseye_blocking, winning] */
    for (int i = 0; i < nr;  i++) out_moves[i]                 = rest[i];
    for (int i = 0; i < nl;  i++) out_moves[nr + i]             = line_blk[i];
    for (int i = 0; i < nbb; i++) out_moves[nr + nl + i]        = bull_blk[i];
    for (int i = 0; i < nw;  i++) out_moves[nr + nl + nbb + i]  = winning[i];
    return nr + nl + nbb + nw;
}


/*
 * sort_legal_moves_ex_c — same as sort_legal_moves_c but also writes n_winning.
 *
 * Allows Python callers to distinguish winning moves from blocking/rest without
 * re-checking win lines, so _push_known_undo can be used instead of _push_undo.
 */
int sort_legal_moves_ex_c(
    const signed char *bf,
    const int *pr_cur,
    int cur,
    int opp,
    unsigned char *out_moves,
    int *out_n_winning,
    const int *pr_opp
) {
    /* §341: same bullseye-block priority fix as sort_legal_moves_c.
     * Layout: [rest, line_blocking, bullseye_blocking, winning] */
    unsigned char rest[27], line_blk[27], bull_blk[27], winning[27];
    int nr = 0, nl = 0, nbb = 0, nw = 0;

    for (int sz_idx = 0; sz_idx < 3; sz_idx++) {
        if (pr_cur[sz_idx] == 0) continue;
        int base = sz_idx * 9;
        int opp_has_sz = pr_opp[sz_idx] > 0;

        for (int ci = 0; ci < 9; ci++) {
            int slot = CELL_SLOT[base + ci];
            if (bf[slot]) continue;

            unsigned char idx = (unsigned char)(base + ci);
            const int *wl = WIN_LINES[base + ci];

            int is_win = 0, is_bull = 0, is_line = 0;
            for (int k = 0; k < 12; k += 2) {
                int i1 = wl[k];
                if (i1 < 0) break;
                int i2 = wl[k + 1];
                int v1 = bf[i1];
                if (v1 == cur) {
                    if (bf[i2] == cur) { is_win = 1; break; }
                } else if (v1 == opp && bf[i2] == opp && opp_has_sz) {
                    if (i1 / 3 == i2 / 3) is_bull = 1;
                    else                   is_line = 1;
                }
            }

            if (is_win)       winning[nw++]   = idx;
            else if (is_bull) bull_blk[nbb++] = idx;
            else if (is_line) line_blk[nl++]  = idx;
            else              rest[nr++]       = idx;
        }
    }

    for (int i = 0; i < nr;  i++) out_moves[i]                = rest[i];
    for (int i = 0; i < nl;  i++) out_moves[nr + i]            = line_blk[i];
    for (int i = 0; i < nbb; i++) out_moves[nr + nl + i]       = bull_blk[i];
    for (int i = 0; i < nw;  i++) out_moves[nr + nl + nbb + i] = winning[i];
    *out_n_winning = nw;
    return nr + nl + nbb + nw;
}


/*
 * get_legal_moves_c — fill out_moves with legal move indices (0-26) for cur player.
 *
 * bf:       signed char[27] — board bytes
 * pr_cur:   int[3] — current player's piece counts [S, M, L] (1-indexed sizes)
 * out_moves unsigned char[27] — output: flat move indices (sz_idx*9 + ci)
 *
 * Returns: number of legal moves written to out_moves.
 * Index encoding: sz_idx*9 + ci where ci = r*3+c (0-8).
 */
int get_legal_moves_c(
    const signed char *bf,
    const int *pr_cur,
    unsigned char *out_moves
) {
    int n = 0;
    for (int sz_idx = 0; sz_idx < 3; sz_idx++) {
        if (pr_cur[sz_idx] == 0) continue;
        int base = sz_idx * 9;
        for (int ci = 0; ci < 9; ci++) {
            if (!bf[CELL_SLOT[base + ci]]) {
                out_moves[n++] = (unsigned char)(base + ci);
            }
        }
    }
    return n;
}


/* ── Original single-move selector (kept for compatibility) ─────────────── */
void pick_rollout_move_c(
    const signed char *bf,
    int cur,
    int opp,
    const int *pr_flat,
    int *out_r,
    int *out_c,
    int *out_size,
    int *out_is_win
) {
    /* §343: dual-track — bullseye block takes priority over line block. */
    int bull_r = -1, bull_c = -1, bull_size = -1;
    int block_r = -1, block_c = -1, block_size = -1;
    int rest_r  = -1, rest_c  = -1, rest_size  = -1;
    int p_off = (cur - 1) * 3;

    for (int sz_idx = 0; sz_idx < 3; sz_idx++) {
        if (pr_flat[p_off + sz_idx] == 0) continue;
        int wl_base = sz_idx * 9;
        int cs_base = sz_idx * 9;

        for (int ci = 0; ci < 9; ci++) {
            if (bf[CELL_SLOT[cs_base + ci]] != 0) continue;

            int r = CELL_R[ci];
            int c = CELL_C[ci];
            const int *wl = WIN_LINES[wl_base + ci];

            if (bull_r == -1) {
                /* Full scan until bullseye block found: wins, bull blocks, line blocks. */
                int is_bull = 0, is_block = 0;
                for (int k = 0; k < 12; k += 2) {
                    int i1 = wl[k];
                    if (i1 < 0) break;
                    int i2 = wl[k + 1];
                    int v1 = bf[i1];
                    if (v1 == cur) {
                        if (bf[i2] == cur) {
                            *out_r = r; *out_c = c;
                            *out_size = SIZES[sz_idx]; *out_is_win = 1;
                            return;
                        }
                    } else if (v1 == opp && bf[i2] == opp && pr_flat[(opp-1)*3 + sz_idx] > 0) {
                        if (i1 / 3 == i2 / 3) is_bull  = 1;  /* bullseye threat */
                        else                   is_block = 1;
                    }
                }
                if (is_bull) {
                    bull_r = r; bull_c = c; bull_size = SIZES[sz_idx];
                } else if (is_block && block_r == -1) {
                    block_r = r; block_c = c; block_size = SIZES[sz_idx];
                } else if (rest_r == -1) {
                    rest_r = r; rest_c = c; rest_size = SIZES[sz_idx];
                }
            } else {
                /* Bullseye block found — only scan for wins now. */
                for (int k = 0; k < 12; k += 2) {
                    int i1 = wl[k];
                    if (i1 < 0) break;
                    if (bf[i1] == cur && bf[wl[k + 1]] == cur) {
                        *out_r = r; *out_c = c;
                        *out_size = SIZES[sz_idx]; *out_is_win = 1;
                        return;
                    }
                }
            }
        }
    }

    if (bull_r >= 0) {
        *out_r = bull_r; *out_c = bull_c; *out_size = bull_size; *out_is_win = 0;
    } else if (block_r >= 0) {
        *out_r = block_r; *out_c = block_c; *out_size = block_size; *out_is_win = 0;
    } else if (rest_r >= 0) {
        *out_r = rest_r; *out_c = rest_c; *out_size = rest_size; *out_is_win = 0;
    } else {
        *out_r = -1; *out_c = -1; *out_size = -1; *out_is_win = 0;
    }
}
