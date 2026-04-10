# Map Validation Manual Protocol

This protocol covers the one part that cannot be proven with high confidence from automation alone on this machine: native macOS trackpad pinch behavior.

## Preconditions

- Start the app with `bun run dev` inside `web/`.
- Open the validation view with:
  - `http://crosswalk-review.localhost:1355/?mapDebug=1`
- Ensure the hidden diagnostics panel is visible.

## Trackpad pinch validation

1. Place the cursor over the map.
2. Perform a short pinch-in gesture.
3. While the gesture is active, check:
   - browser page zoom does not change
   - diagnostics `pageScale` stays at `1.000`
   - grid rectangles stay visually aligned to the map
4. Release the gesture and check:
   - the map stays where it ended
   - no snap-back occurs
   - the overlay still lines up with the same streets or crossings
5. Repeat with pinch-out.

## Acceptance rules

- `pageScale` must remain `1.000`
- no browser zoom UI may appear
- no visible overlay drift is allowed during or after the gesture
- no unexpected recenter or fit may happen after the gesture ends
- diagnostics must not show new console errors

## Evidence to capture

- one screen recording for pinch-in
- one screen recording for pinch-out
- one screenshot of the diagnostics panel after the gesture

## Failure rule

If any one of these happens, the validation fails:

- page zoom changes
- the overlay looks wrong during the gesture and only corrects afterwards
- the map jumps after the gesture ends
- diagnostics show console errors or state churn without input
