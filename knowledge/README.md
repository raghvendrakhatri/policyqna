Facts that live nowhere in the ingested documents — formulas, definitions,
internal rules, abbreviations.

Every `.md` and `.txt` file in this folder is read at startup and appended to the
system prompt on **every** question. Unlike an ingested document, nothing here
can be missed by retrieval: the model always sees it, whatever was asked.

That guarantee is paid for in context. Keep the folder small — a few thousand
characters total. `rag.py` warns past 6000. Anything longer belongs in a document
you `ingest`.

No restart concept: the files are re-read each time a chain is built, so `ask`
picks up an edit immediately and `chat` picks it up on its next launch.

`README.md` is skipped by name; every other `.md`/`.txt` file is loaded.
