# Circle Coverage Problem and Validation

## Problem

The scan circle and the active scan tiles can diverge at the boundary if tile inclusion is based only on tile centers.

That old rule is:

- include tile `T` if the center of `T` lies inside the circle

This is too weak.

A tile can visibly intersect the scan circle while its center is still outside the circle. When that happens, the boundary looks underfilled even though the drawn circle says the area is part of the scan region.

## Correct rule

Use rectangle-circle intersection instead:

- include tile `T` if the tile rectangle of `T` intersects the scan circle

## Validation idea

Let:

- `C` be the scan circle
- `R(T)` be the rectangle of tile `T`
- `S` be the set of active scan tiles

The correct target set is:

- `S* = { T | R(T) intersects C }`

Then the validation obligation is:

- if the UI renders exactly the set `S*`,
- then no tile that should visually belong to the scan circle is missing,
- and no tile outside the scan circle is wrongly included.

So the validation case becomes:

1. compute `S*` from geometry
2. compare it to the actually rendered active tile set `S`
3. pass only if `S = S*`

## Implementation in this project

- the scan circle is defined in real map meters
- each tile rectangle is checked against that circle
- the validation runner includes a `circle-coverage` case that compares expected vs actual tile IDs

Relevant files:

- `web/src/utils.ts`
- `web/src/components/SceneMap.tsx`
- `web/src/useMapValidationRuntime.ts`
