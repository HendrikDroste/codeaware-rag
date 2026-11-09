# Benchmark appendix redesign options

The main presentation already communicates the results. The appendix slide
should explain the experiment rather than repeat the token comparison.
All three designs were requested and generated as separate backup slides A3–A5.

## A. Paired experiment flow — recommended

Title: How the paired benchmark works

One question branches into two lanes: native tools, and native tools plus
Codeaware. Labels above the lanes specify the same pinned repository,
model, reasoning and prompt. Both lanes lead to recorded answers and usage,
then to quality review. A short footer states one run per condition and an
unblinded retrospective review. Use no token-result chart.

Best for explaining experimental controls and fairness during code discussion.

## B. Question coverage map

Title: What the benchmark tests

Group the existing questions by API lookup, implementation tracing and code
relationships. Show one representative question per group, then a small list
of untested dimensions: other repositories, larger codebases and completed
maintenance tasks. Assign categories from the actual dataset before drawing
counts. Do not invent category totals.

Best for discussing representativeness and the expansion plan.

## C. Quality-grading example

Title: How answer quality was reviewed

Use Q15 to connect the question, essential facts and rubric. Show that both
answers identify make_response, while the Codeaware answer omits tuple
status/header handling. Include the caveat that the existing checklist goes
beyond the narrow literal question and a narrower rubric would tie this pair.
The reviewer was retrospective and unblinded.

Best for discussing correctness, grading judgments and limitations.

## Merge status

Complete: `presentation/Codeaware_Interview_Revised.pptx` contains four main
slides, an Appendix divider and six backup slides (11 total), with notes.
The benchmark slides are A3 (method), A4 (coverage) and A5 (grading).
