"""Trajectory signatures (`verb:label[="value"]|...`) and terminal outcome keys."""
from __future__ import annotations

import re

_ELEM_RE = re.compile(r"\[(\d+)\]\s+\S+\s+'([^']*)'")          # `[bid] role 'label'`
_ACTION_RE = re.compile(r"^\s*([a-z_]+)\s*\((.*)\)\s*$", re.DOTALL)
_BID_RE = re.compile(r"\[?(\d+)\]?")
_STR_RE = re.compile(r'"([^"]*)"|\'([^\']*)\'')
_DROP_VERBS = {"infeasible", "noop"}
_NORM_TOK_RE = re.compile(r"^[a-z_]+[:=]")
_SELECT_VAL_RE = re.compile(r'="([^"]*)"\s*$')
_FINAL_CLICK_TERMINALS = {"send_msg_to_user", "infeasible", "noop"}


def _label_for_bid(page_state: str, bid: str) -> str | None:
    for m in _ELEM_RE.finditer(page_state or ""):
        if m.group(1) == bid:
            return m.group(2)
    return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def normalize_turn(action: str | None, page_state: str) -> str | None:
    """One turn -> `verb:target[="value"]` (bid resolved to its label), or None for no-op/give-up.
    send_msg_to_user is reduced to the bare verb (its text is dropped)."""
    if not action:
        return None
    m = _ACTION_RE.match(action)
    if not m:
        return None
    verb, args = m.group(1).lower(), m.group(2)
    if verb in _DROP_VERBS:
        return None
    if verb == "send_msg_to_user":
        return verb
    sval = _STR_RE.search(args)
    value = next((g for g in sval.groups() if g is not None), None) if sval else None
    if verb in ("click", "fill", "hover", "select_option"):
        bm = _BID_RE.search(args)
        if bm:
            tok = f"{verb}:{_norm(_label_for_bid(page_state, bm.group(1)) or bm.group(1))}"
            if value is not None and verb in ("fill", "select_option"):
                tok += f'="{_norm(value)}"'
            return tok
        return f'{verb}="{_norm(value)}"' if value is not None else verb
    if value is not None:
        return f'{verb}="{_norm(value)}"'
    return f"{verb}:{_norm(args)}" if args.strip() else verb


def trajectory_signature(turns: list[dict]) -> str:
    toks = [normalize_turn(t.get("action"), t.get("state_shown") or "") for t in turns]
    return "|".join(t for t in toks if t is not None)


def _final_click_label(t: str) -> str | None:
    if t.startswith("click:"):
        return t[len("click:"):]
    if t.startswith("select_option:"):
        body = t[len("select_option:"):]
        m = _SELECT_VAL_RE.search(body)
        return m.group(1) if m else body
    return None


def _outcome_final_click_from_tokens(toks: list[str]) -> str | None:
    """Label of the literal last click / select_option."""
    for t in reversed(toks):
        if t not in _FINAL_CLICK_TERMINALS and (label := _final_click_label(t)) is not None:
            return label
    return None


def _fill_pair(t: str) -> tuple[str, str] | None:
    for verb in ("fill", "select_option"):
        if t.startswith(verb + ":"):
            body = t[len(verb) + 1:]
            m = _SELECT_VAL_RE.search(body)
            return (body[:m.start()], m.group(1)) if m else (body, "")
        if t.startswith(verb + "="):
            m = _SELECT_VAL_RE.search(t)
            return ("", m.group(1) if m else "")
    return None


def _outcome_final_state_from_tokens(toks: list[str]) -> str | None:
    """`<final click>#<field=value,...>`: last click plus the last value written to each field."""
    final = _outcome_final_click_from_tokens(toks)
    if final is None:
        return None
    state = {}
    for t in toks:
        if (pair := _fill_pair(t)) is not None:
            state[pair[0]] = pair[1]
    return final + "#" + ",".join(sorted(f"{k}={v}" for k, v in state.items())) if state else final


def _outcome_for_mode(toks: list[str], mode: str) -> str | None:
    return _outcome_final_click_from_tokens(toks) if mode == "final_click" else _outcome_final_state_from_tokens(toks)


def extract_outcome(turns: list[dict], mode: str = "final_state") -> str | None:
    toks = []
    for t in turns:
        action = t.get("action")
        tok = normalize_turn(action, t.get("state_shown") or "")
        if tok is None and action and _NORM_TOK_RE.match(action):
            tok = _norm(action)
        if tok is not None:
            toks.append(tok)
    return _outcome_for_mode(toks, mode)


def outcome_from_signature(signature: str, mode: str = "final_state") -> str | None:
    return _outcome_for_mode(signature.split("|"), mode) if signature else None
