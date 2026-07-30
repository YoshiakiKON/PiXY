# PiXY Release Notes v1.4.3

Release date: 2026-07-30

## Highlights

- Improved center-table consistency by unifying export/clipboard output with the center numeric model.
- Fixed stale center-table view behavior after Clear/Undo in centroid extraction workflows.
- Added manual group-name input fields directly under each AddToList button in centroid extraction mode.
- Connected manual group names to center auto-name generation while preserving explicit row-level custom names.
- Added automatic manual-group-name inheritance after K-Means regrouping using nearest group color.
- Added collision policy for inherited names: when multiple names map to one target group, the larger source-group size wins.

## Notes

- Group-name overrides are now included in project save/load.
- Existing row-level custom names are preserved and are not overwritten by group-name inheritance.
