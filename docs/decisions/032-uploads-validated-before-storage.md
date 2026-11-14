# 032. Uploads Are Checked in Full Before Anything Is Stored, Under Names the Server Chooses
Status: Accepted | Date: 2026-11-03 | Records pipeline-track steps 1–3b (2026-10-23)

## Context
An upload writes a documents row and a file, and later the extract worker stores the file's text in `raw_text`. Before steps 1–3b, each could go wrong on its own:
- **Encoding.** Only the first 8 bytes were decoded as UTF-8, so a valid file whose multibyte character crossed byte 8 got a 422. Invalid UTF-8 later in the file was accepted, then failed in the worker. A NUL byte (valid UTF-8, common in UTF-16 files saved by mistake) was rejected by PostgreSQL only when the worker committed `raw_text`.
- **Paths.** The client's filename was the path on disk. `..` caused a 500, and an overlong name failed only after the row was written.
- **Leftovers.** A failed write left a partial file behind, and deleting a workspace removed its rows but none of its files.
- **Display text.** Titles and filenames are shown to people, sent in headers and put into prompts, but they were only length-checked. A bidi override can make `invoice‮fdp.exe` display as `invoiceexe.pdf`.

## Decision
**Everything a later stage would reject is rejected before the row exists (422, no row, no file).** `_inspect_upload` (`src/services/document.py`) reads the whole stream once:
- **Size:** at most `MAX_UPLOAD_SIZE_MB` (25), for every type.
- **PDF:** the first read must start with `%PDF`, so an empty PDF is rejected.
- **Text types:** no NUL byte, and strict UTF-8, decoded incrementally so a character split across two reads still passes.

**Display text** follows one rule for filenames and titles (`clean_display_text`, `src/domain/documents.py`): NFC, trimmed, no control characters, no bidi overrides or isolates, at most 255 characters. A filename also loses any path. `""`, `.` and `..` count as no name. The original filename is kept as provenance, and a missing title defaults to the filename's stem.

**On disk, the server chooses the name: `<UPLOAD_DIR>/<workspace_id>/<document_id><ext>`**, with the extension taken from the content type. The file is opened with exclusive create, so it never overwrites another file. It is written through `FileStoreProtocol` (`LocalFileStore`).

**The commit is the dividing line.**
- **Before it**, any failure removes the file this upload created, and so does a cancellation by an anyio cancel scope, which is how Starlette and the MCP SDK cancel a request. The file and its directories are fsynced before the commit.
- **From the commit on**, the file is kept and a failure is logged. A row without its file is corruption; a file without its row is only garbage.

**Deleting a workspace** removes its directory after the rows' commit, best-effort.

Tests:
- `tests/unit/test_document_service.py`, "content pre-pass", "storage path", "original filename and title" and "after the row" sections.
- `tests/unit/test_domain.py`.
- `tests/unit/test_local_file_store.py`.
- `tests/integration/test_document_upload_failures.py`.
- `tests/integration/test_workspace_delete_files.py`.
- `tests/api/test_documents.py`.

## Consequences
- **What the worker can meet:** text files are always storable UTF-8, so only a PDF can fail extraction because of its content.
- **The upload is read twice**, once to check it and once to write it, and Starlette has already buffered the whole body before either. A request body limit belongs to the abuse plan.
- **Orphan files are possible.** A crash or failed commit after the write can leave a file with no row. So can a native `task.cancel()` during the write: it is delivered once, while the request waits on the writing thread, and the thread finishes after the cleanup was skipped. Only uvicorn's `--timeout-graceful-shutdown` sends one, and our stack doesn't set it; a stopped API is killed instead, which is the crash case. Such a file is only garbage, so the orphan sweeper (pipeline step 9b) is recorded, not built.
- **The stored key is the absolute path** (S2-6). Moving `UPLOAD_DIR` breaks only documents not yet extracted, since only extraction reads the file.

Rejected alternatives:
- **Decoding with `errors="replace"`, or detecting the charset.** The first corrupts what search sees; the second is a guessing heuristic plus a dependency.
- **Checking while writing.** The rejection would come after the row and a partial file exist, so it would need compensation.
- **Sanitizing the client's filename.** That's an endless list of special cases.
- **Content-addressed storage.** Deduplication across workspaces reveals that another tenant uploaded the same file.
- **Deleting the file when the commit fails.** The commit may have succeeded anyway.
- **Storing an invalid name as NULL.** The client would never learn why its filename disappeared.
