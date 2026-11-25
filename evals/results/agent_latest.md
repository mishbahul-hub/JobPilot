# Agent eval — 2026-11-24T12:55:24+00:00 (gemini-3.5-flash-lite)

| Metric | Value |
|---|---|
| scenarios | 10 |
| turns | 15 |
| turn_pass_rate | 0.867 |
| scenario_pass_rate | 0.8 |
| check_pass_rate | 0.957 |
| tool_accuracy | 1 |
| grounded_rate | 0.867 |
| judge_relevance | 4.8 |
| judge_faithfulness | 4.3 |
| latency_median_s | 11.3 |
| latency_max_s | 24.8 |
| errors | 2 |

## Turns

| Scenario | # | Pass | Tools | Failed checks | Judge rel/faith | s |
|---|---|---|---|---|---|---|
| backend-intern-blr | 1 | ❌ | - | no_error | - | None |
| data-eng-hyd | 1 | ✅ | update_profile, search_jobs, get_job, get_job, get_job | - | 4/3 | 13.8 |
| java-pune | 1 | ✅ | update_profile, search_jobs | - | 5/5 | 10.1 |
| remote-only | 1 | ✅ | update_profile, search_jobs | - | 5/4 | 9.3 |
| save-ordinal | 1 | ✅ | search_jobs | - | 5/4 | 14.2 |
| save-ordinal | 2 | ✅ | save_job | - | - | 5.3 |
| ordinal-out-of-range | 1 | ✅ | search_jobs, update_profile | - | 5/5 | 24.8 |
| ordinal-out-of-range | 2 | ✅ | - | - | - | 3.2 |
| job-details | 1 | ✅ | search_jobs, update_profile | - | 5/5 | 11.3 |
| job-details | 2 | ✅ | get_job | - | 5/3 | 14.1 |
| off-topic | 1 | ✅ | - | - | - | 3.0 |
| memory-across-conversations | 1 | ✅ | update_profile, search_jobs, search_jobs | - | 4/5 | 13.8 |
| memory-across-conversations | 2 | ✅ | search_jobs | - | 5/5 | 13.3 |
| dislike-then-alternatives | 1 | ✅ | search_jobs | - | 5/4 | 7.6 |
| dislike-then-alternatives | 2 | ❌ | - | no_error | - | None |
