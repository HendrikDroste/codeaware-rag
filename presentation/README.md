# Interview presentation

Current deck: [Codeaware_Interview_Definitions.pptx](Codeaware_Interview_Definitions.pptx).

Saved natively from the repaired PowerPoint presentation on 8 November 2026.
Uses `find_definitions`, “code definitions + relationships” and “definition ID”.
Includes the Python-focused analysis decision to keep the prototype simple.
The original opening remains at the end of the appendix as a backup.

Other PPTX files in this directory are earlier revisions. Their tool names and
wording are historical; use the current deck above for the presentation.

The API/database fields `symbol_id` and `symbol_ids` remain unchanged. Reconnect
MCP clients after updating the server to refresh tool discovery; no index rebuild
is required for this rename. Recorded benchmark artifacts also retain their
original tool names so their evidence remains accurate.
