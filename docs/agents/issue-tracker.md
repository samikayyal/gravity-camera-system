# Issue tracker: GitHub

Issues and specs live in GitHub Issues for `samikayyal/gravity-camera-system`.
Use the `gh` CLI from this clone; outside the clone, pass
`--repo samikayyal/gravity-camera-system`.

## Conventions

For multiline issue bodies and comments, write the exact text to a temporary
UTF-8 file and pass `--body-file <path>`.

- Create: `gh issue create --title "..." --body-file <path>`.
- Read: `gh issue view <number> --json number,title,body,labels,comments`.
- List: `gh issue list --state open --json number,title,body,labels,assignees`.
  Add label or state filters as needed; use `--limit` or API pagination when
  the workflow needs the complete set.
- Update body: `gh issue edit <number> --body-file <path>`.
- Comment: `gh issue comment <number> --body-file <path>`.
- Apply or remove labels: `gh issue edit <number> --add-label "<label>"`
  or `--remove-label "<label>"`.
- Close: `gh issue close <number>`.
- Create a sub-issue: `gh issue create --parent <parent> --title "..." --body-file <path>`.
- Link an existing child: `gh issue edit <parent> --add-sub-issue <child>`.
  If sub-issues are unavailable, add the child to a task list in the parent
  and put `Part of #<parent>` at the top of the child body.

When a skill says "publish to the issue tracker", create a GitHub issue.
When it says "fetch the relevant ticket", read the corresponding issue.

## Pull requests as a triage surface

**PRs as a request surface: no.**

GitHub issues and pull requests share a number space. When the object type
is unclear, try `gh pr view <number>` and fall back to `gh issue view <number>`.

## Wayfinding operations

- Map: one issue labelled `wayfinder:map`, containing Notes,
  Decisions-so-far, and Fog.
- Child ticket: a sub-issue of the map, labelled `wayfinder:<type>`, where
  type is `research`, `prototype`, `grilling`, or `task`. Create missing
  wayfinder labels when that workflow is used.
- Blocking: use native issue dependencies:
  `gh issue edit <child> --add-blocked-by <blocker>`.
  If unavailable, put `Blocked by: #<number>, ...` at the top of the child.
- Frontier: inspect the map's open children in map order; select the first
  unassigned child whose blockers are all closed. Follow API pagination
  when listing sub-issues.
- Claim: `gh issue edit <number> --add-assignee "@me"` as the driving
  session's first write.
- Resolve: comment with the result, close the child, then append a concise
  finding and link to the map's Decisions-so-far.
