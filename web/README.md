# Crosswalk Dataset Explorer

This folder contains the web frontend for inspecting the released crosswalk
metadata dataset. The public deployment is read-only and uses compressed static
metadata from `web/public/static-datasets/`.

## Run

```bash
bun install
bun run dev
bun run lint
bun run dev:url
bun run build
```

The app starts through `portless` on a stable local URL:

```text
http://crosswalk-review.localhost:1355
```

For a static-only production build:

```bash
VITE_CROSSWALK_STATIC_ONLY=1 bun run build
```

If Tailscale is installed, `bun run dev` may also expose the current local dev
server through Tailscale Serve. Set `TAILSCALE_SERVE=0` to skip that local
convenience step.

## What It Does

- shows the released metadata dataset on a Leaflet map
- filters tiles by dataset, city, split, and label
- previews source imagery and mask overlays for selected tiles
- shows model and human label history for each tile
- can call the local backend for CrossMaskNet checks when writable local data is
  available
