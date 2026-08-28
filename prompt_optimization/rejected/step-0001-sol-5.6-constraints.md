Play Wordle.

Reply with exactly one lowercase five-letter English word and nothing else.
Do not use punctuation or explain your choice.
Never repeat a previous guess.

Treat every submitted response, including one rejected by the harness, as already used. Before replying, verify that the response is nonempty, is a legal Wordle dictionary word, has exactly five lowercase letters, and has not appeared earlier.

Feedback marks:
G = correct letter in the correct position
Y = correct letter in the wrong position
B = this occurrence of the letter is not matched

Use all previous guesses and feedback to choose the next guess.

Maintain cumulative constraints before every guess: fix G positions, exclude Y letters from their shown positions, exclude B letters unless another occurrence proves the letter is present, and infer minimum and maximum counts for repeated letters from each feedback row. Only guess words consistent with every established constraint; once the remaining candidates are few, guess a likely remaining answer rather than another exploratory word.
