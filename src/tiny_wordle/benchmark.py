from __future__ import annotations

import re
import time
from dataclasses import dataclass

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .game import Turn, score_string

WORD_RE = re.compile(r"^[A-Za-z]{5}$")

DEFAULT_SYSTEM_RULES = """Play Wordle.

Return exactly one uppercase five-letter English word.
Do not explain.
Do not use punctuation.

Use all previous guesses and feedback when choosing the next guess.
Never repeat a previous guess.

Example valid response:
CRANE

Feedback meanings:
G = correct letter and position
Y = letter is present but wrong position
B = that letter occurrence is not matched
"""

DEFAULT_EVAL_ANSWERS = [
    "SHORE","MIGHT","BRICK","GHOST","KNIFE","DOUBT","FLING","ROUND","CHAMP",
    "WASTE","BLIND","POINT","SLATE","CRANE","APPLE","SHEEP","BANAL","ALLEY","AUDIO",
]

@dataclass
class GameResult:
    answer: str
    solved: bool
    turns_used: int
    valid_guesses: int
    invalid_format: int
    repeat_guesses: int
    final_guess: str | None
    elapsed_seconds: float
    trace: list[dict]

def parse_guess(raw_text: str) -> str | None:
    text = raw_text.strip()
    if not WORD_RE.fullmatch(text):
        return None
    return text.upper()

def format_history_for_model(history: list[Turn]) -> str:
    if not history:
        return "No guesses have been made yet."
    return "\n".join(
        f"{' '.join(turn.guess)} -> {' '.join(turn.feedback)}"
        for turn in history
    )

def build_prompt(history: list[Turn], system_rules: str = DEFAULT_SYSTEM_RULES) -> str:
    return system_rules + "\nGame history:\n" + format_history_for_model(history)

def load_benchmark_model(model_id: str, *, device=None, dtype=torch.float32):
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    model.eval()
    return tokenizer, model, device

def generate_raw_guess(history, *, tokenizer, model, device, max_new_tokens=16, system_rules=DEFAULT_SYSTEM_RULES):
    user_text = build_prompt(history, system_rules=system_rules)
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    batch = tokenizer(chat_text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = output[0, batch["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def play_model_game(answer: str, *, tokenizer, model, device, max_turns=6, verbose=False, system_rules=DEFAULT_SYSTEM_RULES):
    answer = answer.upper()
    history = []
    seen = set()
    trace = []
    invalid_format = 0
    repeat_guesses = 0
    start = time.perf_counter()

    for turn_number in range(1, max_turns + 1):
        raw = generate_raw_guess(
            history, tokenizer=tokenizer, model=model, device=device,
            system_rules=system_rules
        )
        guess = parse_guess(raw)
        record = {
            "turn": turn_number, "raw_output": raw, "guess": guess,
            "feedback": None, "status": None,
        }

        if guess is None:
            invalid_format += 1
            record["status"] = "invalid_format"
            trace.append(record)
            if verbose:
                print(f"{turn_number}: INVALID {raw!r}")
            continue

        if guess in seen:
            repeat_guesses += 1
            record["status"] = "repeat_guess"
        else:
            record["status"] = "valid_guess"

        seen.add(guess)
        feedback = score_string(answer, guess)
        record["feedback"] = feedback
        trace.append(record)
        history.append(Turn(guess=guess, feedback=feedback))

        if verbose:
            print(f"{turn_number}: {guess} -> {feedback}")

        if feedback == "GGGGG":
            return GameResult(
                answer=answer, solved=True, turns_used=turn_number,
                valid_guesses=len(history), invalid_format=invalid_format,
                repeat_guesses=repeat_guesses, final_guess=guess,
                elapsed_seconds=time.perf_counter() - start, trace=trace,
            )

    return GameResult(
        answer=answer, solved=False, turns_used=max_turns,
        valid_guesses=len(history), invalid_format=invalid_format,
        repeat_guesses=repeat_guesses,
        final_guess=history[-1].guess if history else None,
        elapsed_seconds=time.perf_counter() - start, trace=trace,
    )

def history_consistency_stats(result: GameResult):
    prior_history = []
    checked = 0
    consistent = 0
    for step in result.trace:
        guess = step["guess"]
        if guess is None:
            continue
        if prior_history:
            checked += 1
            matches = all(
                score_string(guess, old.guess) == old.feedback
                for old in prior_history
            )
            consistent += int(matches)
        if step["feedback"] is not None:
            prior_history.append(Turn(guess=guess, feedback=step["feedback"]))
    return consistent, checked

def run_benchmark(
    model_id: str,
    *,
    answers=None,
    max_turns=6,
    verbose_games=True,
    print_failures=False,
    dtype=torch.float32,
):
    if answers is None:
        answers = DEFAULT_EVAL_ANSWERS

    tokenizer, model, device = load_benchmark_model(model_id, dtype=dtype)

    print("model:", model_id)
    print("device:", device)
    print("dtype:", next(model.parameters()).dtype)
    print("games:", len(answers))
    print()

    results = []
    for i, answer in enumerate(answers, 1):
        result = play_model_game(
            answer,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_turns=max_turns,
        )
        results.append(result)
        if verbose_games:
            status = "SOLVED" if result.solved else "FAILED"
            print(
                f"{i:2d}/{len(answers)} {answer} {status:6s} "
                f"turns={result.turns_used} valid={result.valid_guesses} "
                f"invalid={result.invalid_format} repeats={result.repeat_guesses}"
            )

    df = pd.DataFrame([
        {
            "answer": r.answer, "solved": r.solved, "turns_used": r.turns_used,
            "valid_guesses": r.valid_guesses, "invalid_format": r.invalid_format,
            "repeat_guesses": r.repeat_guesses, "final_guess": r.final_guess,
            "elapsed_seconds": r.elapsed_seconds,
        }
        for r in results
    ])

    games = len(df)
    solves = int(df["solved"].sum())
    model_calls = int(df["turns_used"].sum())
    total_valid = int(df["valid_guesses"].sum())
    total_invalid = int(df["invalid_format"].sum())
    total_repeats = int(df["repeat_guesses"].sum())

    consistent_total = 0
    checked_total = 0
    for result in results:
        consistent, checked = history_consistency_stats(result)
        consistent_total += consistent
        checked_total += checked

    summary = {
        "model_id": model_id,
        "games": games,
        "solved": solves,
        "solve_rate": solves / games if games else 0.0,
        "model_calls": model_calls,
        "valid_output_rate": total_valid / model_calls if model_calls else 0.0,
        "invalid_output_rate": total_invalid / model_calls if model_calls else 0.0,
        "repeat_guesses": total_repeats,
        "history_consistent_guesses": consistent_total,
        "history_consistency_checked": checked_total,
        "history_consistency_rate": consistent_total / checked_total if checked_total else float("nan"),
        "mean_turns_on_wins": float(df.loc[df["solved"], "turns_used"].mean()) if solves else float("nan"),
        "total_eval_seconds": float(df["elapsed_seconds"].sum()),
    }

    print()
    print("=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"games:                {summary['games']}")
    print(f"solved:               {summary['solved']}")
    print(f"solve rate:           {summary['solve_rate']:.1%}")
    print(f"model calls:          {summary['model_calls']}")
    print(f"valid output rate:    {summary['valid_output_rate']:.1%}")
    print(f"invalid output rate:  {summary['invalid_output_rate']:.1%}")
    print(f"repeat guesses:       {summary['repeat_guesses']}")
    print(
        f"history consistency:  {summary['history_consistency_rate']:.1%} "
        f"({summary['history_consistent_guesses']}/{summary['history_consistency_checked']})"
    )
    print(f"mean turns on wins:   {summary['mean_turns_on_wins']}")
    print(f"total eval time:      {summary['total_eval_seconds']:.1f}s")

    if print_failures:
        print()
        for result in results:
            if result.solved:
                continue
            print("=" * 60)
            print("ANSWER:", result.answer)
            for step in result.trace:
                print(
                    f"{step['turn']}: status={step['status']} raw={step['raw_output']!r} "
                    f"guess={step['guess']} feedback={step['feedback']}"
                )

    return summary, df, results
