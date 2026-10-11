"""Grader-interpretation sensitivity (my own code only): does any plausible alternative reading flip a verdict?"""
import sys, re, json
sys.dont_write_bytecode = True
sys.path.insert(0, 'C:/Users/clclc/AppData/Local/Temp/claude/bwm_5c_scratch')
import recompute as RCMP

base_primary, base_lenient = RCMP.grade_primary, RCMP.grade_lenient


def v_no_units(gold, text, term):
    if gold['kind'] == 'number':
        s = RCMP.normalise(text, gold)
        if ' ' in s:
            return False
    return base_primary(gold, text, term)


def v_len_exact_ci(gold, text):
    if gold['kind'] == 'exact':
        return base_lenient({**gold, 'case': False}, text)
    return base_lenient(gold, text)


def v_len_exact_whole(gold, text):
    if gold['kind'] == 'exact':
        return base_lenient(gold, text) and base_primary(gold, text, True)
    return base_lenient(gold, text)


def v_len_kv_nodistractor(gold, text):
    if gold['kind'] == 'kv':
        return RCMP.word_present(gold['value'], RCMP.n2(RCMP.n1(text)), False)
    return base_lenient(gold, text)


variants = {'baseline': (base_primary, base_lenient), 'primary_rejects_unit_alias': (v_no_units, base_lenient),
            'lenient_exact_case_insensitive': (base_primary, v_len_exact_ci),
            'lenient_exact_whole_answer_only': (base_primary, v_len_exact_whole),
            'lenient_kv_ignores_other_passage_values': (base_primary, v_len_kv_nodistractor),
            'lenient_equals_primary': (base_primary, lambda g, t: base_primary(g, t, True))}
for name, (gp, gl) in variants.items():
    RCMP.grade_primary, RCMP.grade_lenient = gp, gl
    res, _, _ = RCMP.main()
    out = []
    for cn in ('x_bwf_vs_off', 'x_bwpa_vs_off'):
        c = res['comparisons'][cn]
        pp, pl = c['graders']['primary'], c['graders']['lenient']
        minh = min(float(r['p_holm']) for g in (pp, pl) for r in g['categories'].values() if r['lost'] > r['gained']) if any(r['lost'] > r['gained'] for g in (pp, pl) for r in g['categories'].values()) else None
        minnet = min(r['net'] for g in (pp, pl) for r in g['categories'].values())
        out.append(f"{cn}: {c['verdict']} pooled {pp['pooled']['net']:+d}/{pl['pooled']['net']:+d} min_cat_net {minnet:+d} min_pHolm(loss-side) {minh}")
    print(f'{name:42s}', ' | '.join(out))
RCMP.grade_primary, RCMP.grade_lenient = base_primary, base_lenient
RCMP.main()  # restore baseline outputs on disk
