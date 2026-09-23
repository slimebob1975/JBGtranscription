# Backlog - JBGTransciption

## Known issues
- [ ] `call_openai_simple` is no longer used by anything and could be removed.

## Planned improvments
- [ ] Let the user edit the instructions for the other analyses too
      (suspected errors, follow-up questions, speaker identification), reusing
      the summary prompt editor.
- [ ] Let the user save their own edited instruction as a named option,
      alongside the ones defined in the policy file.
- [ ] Show an estimated cost or segment count before upload, now that the
      segment count is predictable from the transcription length.

- [ ] Replace the inline textarea with a floating panel that opens over the page,
      so the form height does not change when the instruction is shown.

## Solved
- [x] Give every OpenAI call an output limit, negotiating `max_completion_tokens`
      against `max_tokens` per model, and detect answers cut off by that limit
- [x] Segment `find_suspicious_phrases` and `suggest_follow_up_questions`
      instead of sending the whole transcription in one unbudgeted call
- [x] Size text-rewriting work by how much the model can answer rather than by
      the context window, so long interviews are not silently truncated
- [x] Steer if encryption is optional from environmental variable
- [x] Make the summary instruction editable in the GUI, with Enkel and Utförlig
      as starting points
- [x] Make the set of summary types configurable from the policy file, with a
      dropdown and per-option tooltips instead of two fixed radio buttons
- [x] Stop segmenting transcriptions that fit in a single call. The old fixed
      budget of 3500 tokens was sized for 4k-context models and split a one-hour
      interview into several pieces unnecessarily.
- [x] Split between sentences instead of at raw token offsets, and merge partial
      summaries into one coherent text instead of concatenating them.
- [x] Fix `"\n".join(...)` applied to `short_summary`, which is a string in the
      policy file and was therefore joined character by character.
- [x] Fix `logger.debug("...", instructions)`, which passed a second argument as
      a `%`-format argument to a string with no placeholders.
- [x] Load `policy/prompt_policy.json` relative to the package instead of the
      current working directory, so the policy is not silently missed when the
      app is started from another directory.
- [x] Fall back to an approximate token estimate when tiktoken cannot download
      its BPE data, instead of failing the whole summary step.
