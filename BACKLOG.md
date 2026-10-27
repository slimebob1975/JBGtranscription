# Backlog - JBGTransciption

The version number is kept only in `VERSION`, and each release is tagged
`v<version>` on GitHub.

## Known issues

## Planned improvments

- [ ] Readable timestamps in the "Med tidsstämplar" presentation. Today each
  Whisper segment gets its own line, written as a Python tuple of seconds:
  `(12.34, 18.9): Hej och välkommen ...`. Segments are only a few seconds
  long, so a timestamp comes every 2-10 seconds, and an unknown end time
  shows as `None`. Proposal:
  - Write the time as `[hh:mm:ss]`, start time only, at the start of a
    paragraph, e.g. `[00:12:34]`.
  - Merge consecutive segments into paragraphs, starting a new one when at
    least a given interval has passed (e.g. 30 or 60 seconds) and the
    previous segment ends a sentence, so paragraphs do not break mid-sentence.
  - Drop the fixed 80-character line breaks in this presentation and let
    Word wrap the paragraphs.
  - Set the timestamp apart visually, e.g. bold or grey.
  Points to settle:
  - The interval: a fixed value, a setting such as
    `JBG_TIMESTAMP_INTERVAL_SECONDS`, or a choice in the GUI.
  - The cache stores the timestamped text already formatted. Storing the raw
    segments (start, end, text) instead and formatting when the document is
    written would let a cached transcription use the new format; old cache
    files expire within `JBG_CACHE_RETENTION_HOURS` anyway.
  - Both engines return segments the same way, but the Transformers engine
    can leave the end of the last segment unknown, which the new format
    avoids by showing start times only.
- [ ] Add a last presentation, e.g. "Ingen transkribering", for users who only
  want the analyses and not the transcribed text in the document. With this
  choice, a summary or the custom analysis must be selected; otherwise the
  start button is disabled, with a short visible note saying why, rather
  than silently not responding. The server must reject the same combination
  for requests that do not come from the page. Points to settle:
  - The audio is still transcribed, since the analyses need the text; only
    the section is left out. The cache, the cost and the statistics work as
    before.
  - "Om transkriberingen" should state that the transcription was left out
    by choice, so a reader of the document knows the analyses cannot be
    checked against the text in the same file.
  - Whether follow-up questions alone should also count as enough.
- [ ] Consider removing speaker identification completely. The "Fråga–svar"
  summary type now covers most of what it was for, and it remains a beta
  whose labels can be wrong. Removing it means the "Med talarangivelser"
  presentation and its editor target, `speaker_diarization` in the policy,
  `SPEAKER_FORMAT_SUFFIX`, the register handling and its constants,
  `JBG_DIARIZATION_SEGMENT_TOKENS`, the README section, and the server's
  mapping of an old `speakers=true` request. Saved speaker instructions in
  users' browsers would simply no longer be used.
- [ ] Investigate if batching and moving window can improve execution time for the transcription step without big loss of accuracy
- [ ] Investigare if timestamp format for transcribed text can be changed from seconds to something like [hh:mm:ss] and perhaps colleted in larger chunks via some postprocessing stel
- [ ] Investigare if marking of suspected errors and timestamps in the transcribed text can be combined

## Solved

- [X] Split `src/JBGtranscriber.py` (2 800 lines, one class with 85 methods)
  into seven mixin modules: prompt policy, language model, segmentation,
  speech to text, analyses, speakers and document. `JBGtranscriber.py` keeps
  the constructor, the step runner and the command-line tool, and re-exports
  the old module-level names, so `main.py` is unchanged. A test suite of 24
  tests was added first and passes before and after. The split was made by a
  script that copied code and comments verbatim; all 85 methods were
  verified identical apart from `JBGtranscriber.` becoming `self.`, `cls.`
  or the owning mixin, and every constant keeps its value
- [X] Remove unused dependencies: pydub and audioop-lts (no longer used since
  decoding moved to PyAV), resampy (imported but never called), ffmpeg-python
  (never imported - ffmpeg is called as a program and installed by
  startup.sh), and whisper 1.1.10, which is Graphite's time-series storage
  library and not OpenAI's Whisper. numpy stays pinned, as torch,
  transformers and faster-whisper depend on it
- [X] Make speaker identification the fourth presentation of the
  transcription ("Med talarangivelser (beta)") instead of a separate
  checkbox and section. The dialogue now replaces the transcription section,
  a failed identification falls back to the raw text with a note, and the
  speaker instruction's Redigera link sits under the presentation choice. The
  per-speaker analysis that was planned on top of it was dropped, as was the
  estimate before upload; the faster-whisper investigation was settled by
  the fast engine
- [X] Base the summary, follow-up questions and custom analysis on the marked
  text when "Med markering av misstänkta fel" is chosen. The marked text holds
  both the original wording and the suggested correction, so the analysis can
  weigh them rather than inherit a transcription error. The code adds an
  instruction about the markers only when they are present, telling the model
  that the suggestions are unreviewed and not to present an uncertain detail
  as certain. Speaker identification keeps the original text, since markers
  do not survive its rewrite reliably. The tooltip and "Om transkriberingen"
  say which analyses were affected
- [X] Let the user save their own edited instructions under a name, for all
  four editable instructions, each with its own list. Kept in the browser
  only, by choice: nothing is stored on the server, and options for the
  whole team still belong in the policy file. The editor dialog now has a
  fixed height, so the text field shrinks rather than pushing the buttons
  out when more rows are shown
- [X] Let the user edit the instructions for follow-up questions and speaker
  identification, and add a custom analysis ("Anpassad") for recordings that
  are not interviews, such as meetings. One editor dialog now serves the
  summary, both analyses and the custom analysis. The custom analysis starts
  with a fixed opening from `custom_analysis_preamble` that cannot be edited;
  checking the box opens the editor, Klar leaves it checked only if something
  was written, and the text is kept in the browser between visits. The speaker
  answer format moved from the editable instruction into the code, so an
  edited instruction cannot break the layout. Suspected errors stay
  non-editable: their instruction defines the marker format the document
  depends on
- [X] Show the cost of the OpenAI calls in the result document. Prices per
  model are kept in `policy/model_prices.json`, with input, cached input and
  output priced separately, since most input in a long run is cached. The
  cost is computed from the token counts OpenAI reports for each call, shown
  per row in "Statistik" and as a total in "Om transkriberingen". The same
  file now defines the model menu (`_i_menyn`, `_standardmodell`), the server
  rejects models outside it, and the startup log warns about missing prices
  and a check date (`_kontrollerad`) that is missing or older than 90 days
- [X] Give cached transcriptions a lifetime. With encryption off,
  transcriptions were cached in `cache/` but never deleted, so every
  unencrypted transcription stayed on the server in clear text. They are now
  deleted after `JBG_CACHE_RETENTION_HOURS` (24 by default, at least 1) by the
  hourly cleanup. The cache key now includes the accuracy level and engine,
  so a request for higher accuracy is no longer answered from an earlier,
  less accurate run, and the cache file records the model used, so "Om
  transkriberingen" is no longer left out on a cache hit and notes the reuse
- [X] Accept other audio formats than mp3: mp3, m4a, wav, aac, flac and ogg,
  listed in `SUPPORTED_AUDIO_EXTENSIONS`. Video containers are left out on
  purpose. The page and the server check the extension, the upload keeps its
  own suffix, and `JBG_MAX_UPLOAD_MB` (1024 by default) caps the size in both
  places. In encrypted mode the decrypted audio is now decoded in memory with
  PyAV (`faster_whisper.decode_audio`) instead of pydub. pydub passed it to
  ffmpeg and ffprobe through their `cache:` protocol, which buffers the input
  in a temporary file, so the decrypted audio reached the disk in every
  encrypted run, mp3 included. `av` is pinned to 18.1.0, since faster-whisper
  1.1.1 calls `av.open(metadata_errors=...)`, which PyAV 19 removed; unpinned,
  a fresh install broke the fast engine for unencrypted files
- [X] Show the version number and GitHub commit under the application title,
  e.g. "Version 1.1.0 · a1b2c3d (prod)", with the commit linking to GitHub,
  as in Klarspråkshjälpen. The version is in `VERSION`; commit, branch, run
  number and build time are written to `build_info.json` by
  `scripts/write_build_info.py` in the build step of both workflows, because
  the deployed package does not contain `.git`, and locally the commit is
  read from git. `/config` returns the information, and the "Om
  transkriberingen" section of the result document gives it as
  "Programversion"
- [X] Stop uploads and results from staying on the server indefinitely. A
  failure while setting up a job now deletes the upload and ends the job like
  any other error, and files in `uploads/` and `results/` older than
  `JBG_RETENTION_HOURS` (24 by default, at least 3) are deleted at startup
  and once an hour. The rule is age alone rather than emptying `uploads/` at
  startup, since the folders can be shared by several instances. Old entries
  in the in-memory `jobs` dict are pruned by the same sweep, and an error
  without a message is stored as its type name instead of reading as success
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
