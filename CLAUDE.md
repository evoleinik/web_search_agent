## Web Research

**Use the built-in WebSearch tool** for all web searches. Call multiple WebSearch in parallel for related queries.

For any internet search:

1. Use the `WebSearch` tool directly — no bash, no scripts
2. **Multiple queries: call multiple WebSearch tools in parallel** in a single message
3. For fetching a specific known URL: `~/.claude/tools/scrape.sh <url>`

### Deep fetch fallback
When WebSearch snippets aren't enough and you need full page content:
- `~/.claude/tools/web_search.sh "query"` — fetches 50 full pages (raw text, no summarization)
- Add `-S` flag for Gemini summarization (~10x compression)
- `--cache-only` for instant recall from previous searches
- `--cache-stats` for cache health

### Rules
- SEARCH BY DEFAULT. Your knowledge has a cutoff date.
- The failure mode is NOT "I searched unnecessarily" — it's "I confidently gave outdated advice."
- ALWAYS search before: recommending tools/libraries, comparing options, debugging unfamiliar errors
- MANDATORY search for AI/LLM topics: models, APIs, SDKs, protocols, agent frameworks
- Only SKIP search for: pure logic/algorithms, code you've already read, project-internal questions
