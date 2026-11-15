# MS2 materials

| File | What it is |
|---|---|
| `ms2_problem_definition.pptx` | The final MS2 deck, 23 slides, with speaker notes. Presented by Noureldin S. Abdelrahman. **Do not edit**: record corrections in `ERRATA.md` |
| `ms2_outline.md` | Per-slide source map (path + commit for every claim), open questions, and the dated history of deck changes |
| `ERRATA.md` | Where the deck differs from the committed artifacts, and which value the repo keeps |

The deck is the team-facing reference for scope, ownership, status and headline numbers. Exact values come from the artifacts listed in the outline and in [`results/README.md`](../../results/README.md). The order of precedence is in [`docs/README.md`](../README.md#source-of-truth-order).

## Read the deck as text

```bash
python - <<'EOF'
from pptx import Presentation   # pip install python-pptx
for i, s in enumerate(Presentation("docs/ms2/ms2_problem_definition.pptx").slides, 1):
    print(f"=== slide {i}")
    for sh in s.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip():
            print(sh.text_frame.text)
        if sh.has_table:
            for r in sh.table.rows:
                print(" | ".join(c.text for c in r.cells))
    if s.has_notes_slide:
        print("NOTES:", s.notes_slide.notes_text_frame.text)
EOF
```

Without python-pptx: `unzip -p docs/ms2/ms2_problem_definition.pptx 'ppt/slides/slide*.xml'` (notes are in `ppt/notesSlides/`).

## Rebuild

The deck was built with python-pptx 1.0.2 by a script kept outside the repo (`build_deck.py` in the presenter's session scratchpad). It is not committed, so the `.pptx` is the artifact. Render check: LibreOffice → PDF → `pdftoppm -r 80`.

An older 16-slide copy edited by a colleague (`ms2_problem_definition (1).pptx`, 2026-11-07) was partly merged and is **not** a source; it is not in the repo.
