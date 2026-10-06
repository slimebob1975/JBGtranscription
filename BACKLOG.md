# Backlog - JBGTransciption

## Known issues

## Planned improvments

- [ ] Consider summarising the marked-up text rather than the original when
  "Med markering av misstänkta fel" is chosen, so the summary works from the
  corrected reading instead of the suspected mistranscription. Marking
  already runs before the summary, so the text is available. Two things to
  settle first:
  - The prompts are not hidden: what is in `policy/prompt_policy.json` is
    what the model receives, apart from the map and reduce instructions the
    code appends when a transcription has to be split. An instruction about
    `[FEL?]` markers should therefore be appended by the code only when the
    marked text is actually being summarised, so the other summary types do
    not carry a rule about markers that are not there. Wording along the
    lines of: the text may contain `[FEL?]...[/FEL?]` around a suspected
    mistranscription followed by `(Kanske: ...)` with a suggested reading;
    use whichever reading is most plausible and do not reproduce the
    markers.
  - The suggestions inside the markers are the model's own guesses, which
    nobody has reviewed. Summarising them means a second model builds on a
    first model's corrections. For a supervisory file that may argue for
    keeping the original as the summary input, or for making it a choice
    rather than automatic.
- [ ] Offer the speaker-labelled text as a fourth presentation of the
  transcription ("Med talarangivelser") rather than as a separate section,
  once the per-speaker analysis is in place.
- [ ] Add a factual per-speaker analysis on top of stable identities: which
  topics each speaker returned to, what they stated, roughly how much they
  spoke. It must stay descriptive, never evaluative - "Intervjuperson 1
  återkom ofta till behovet av säkra rutiner för AI-utveckling", not
  "Intervjuperson 1 verkar orolig för att rutinerna inte fungerar". The
  transcript does not support inferences about a person's state of mind,
  and the subjects are identifiable staff in a supervisory file.
- [ ] Let the user edit the instructions for the other analyses too
  (suspected errors, follow-up questions, speaker identification), reusing
  the summary prompt editor.
- [ ] Let the user save their own edited instruction as a named option,
  alongside the ones defined in the policy file.
- [ ] Show an estimated cost or segment count before upload, now that the
  segment count is predictable from the transcription length.
- [ ] Investigate if batching and moving window can improve execution time for the transcription step without big loss of accuracy
- [ ] Investigare if a faster-Whisper model can improve execution time without big loss of accuracy

## Solved

- [X] Take speaker identification out of beta. Two runs of a seven-segment
  recording, with six register hand-offs each, produced the same five
  speakers and no verbatim repetition at any seam; the single-call runs it
  replaced had given 7, 6 and 6. `_deduplicate_blocks` is removed with it,
  since it had nothing left to clean up
- [X] Make marking of suspected errors return the whole text. The prompt asked
  the model to mark the text but never to return all of it, and on an 8,700
  word interview it answered with 23% of it. Rewriting work now also has a
  practical segment cap, and a short answer makes the step retry with
  smaller segments instead of being discarded
- [X] Make `start_local_service` configure itself on first run: it finds the
  installed Python versions, suggests the code and work directories, saves
  the answers, and re-asks with `-Reinstall`
- [X] Remove the participant labels from the Fråga–svar summary, so that saying
  who spoke belongs to speaker identification and saying what was said
  belongs to the summary
- [X] Remove `call_openai_simple`, which nothing had called since the analyses
  were given token budgets
- [X] Give the card one centred content column, so the grey panels have the same
  gap on both sides and the headings line up with their left edge
- [X] Correct the GUI tooltips: several described behaviour that had since
  changed, most of all the marked-errors option, which promised `[FEL?]`
  tags that the document renderer turns into highlighting
- [X] Check that a rewriting step gave the whole text back, by comparing word
  counts, instead of trusting the model to follow the instruction
- [X] Move "Om transkriberingen" to the end of the document and add a "Statistik"
  table of execution time and tokens sent and received per model
- [X] Configure logging once per process, so a run no longer leaves empty log
  files behind, and keep third-party request logging out of the log file
  since it contained the whole transcription in clear text
- [X] Let the user set a ceiling for the Whisper model with a "Välj noggrannhet:"
  dropdown, in a new "Inställningar för transkriberingen" panel, and record
  in the document which model actually ran
- [X] Carry speaker identity across segments with a register passed from one
  segment to the next, pass the previous segment's last sentences as
  context so boundaries are attributed correctly, and use one input shape
  for both the single-call and segmented paths
- [X] Move the prompt editor into a floating panel opened from a link, so the
  form height no longer changes when the instruction is shown, and centre it
  on the visible window rather than on the iframe's own viewport
- [X] Trim the summary panel: the option description is carried by the tooltip
  alone, and no label is shown while an unedited default is in place
- [X] Present the transcribed text in one of three mutually exclusive forms,
  chosen with a radio group instead of a separate "mark suspected errors"
  checkbox
- [X] Add a "Fråga-Svar" option to the summary types
- [X] Give every OpenAI call an output limit, negotiating `max_completion_tokens`
  against `max_tokens` per model, and detect answers cut off by that limit
- [X] Segment `find_suspicious_phrases` and `suggest_follow_up_questions`
  instead of sending the whole transcription in one unbudgeted call
- [X] Size text-rewriting work by how much the model can answer rather than by
  the context window, so long interviews are not silently truncated
- [X] Steer if encryption is optional from environmental variable
- [X] Make the summary instruction editable in the GUI, with Enkel and Utförlig
  as starting points
- [X] Make the set of summary types configurable from the policy file, with a
  dropdown and per-option tooltips instead of two fixed radio buttons
- [X] Stop segmenting transcriptions that fit in a single call. The old fixed
  budget of 3500 tokens was sized for 4k-context models and split a one-hour
  interview into several pieces unnecessarily.
- [X] Split between sentences instead of at raw token offsets, and merge partial
  summaries into one coherent text instead of concatenating them.
- [X] Fix `"\n".join(...)` applied to `short_summary`, which is a string in the
  policy file and was therefore joined character by character.
- [X] Fix `logger.debug("...", instructions)`, which passed a second argument as
  a `%`-format argument to a string with no placeholders.
- [X] Load `policy/prompt_policy.json` relative to the package instead of the
  current working directory, so the policy is not silently missed when the
  app is started from another directory.
- [X] Fall back to an approximate token estimate when tiktoken cannot download
  its BPE data, instead of failing the whole summary step.
