# Crosswalk Review App

This folder now contains a browser-first dataset creator for crosswalk labeling. The dataset state lives in the browser, the labeling model runs in the browser, and the finished labeled dataset can be exported as a ZIP.

## Run

```bash
bun install
bun run dev
bun run lint
bun run dev:url
bun run dev:tailnet:url
```

The app starts the frontend through `portless` on a stable local URL:

- `http://crosswalk-review.localhost:1355`

When you start `bun run dev`, the launcher also publishes the current Vite dev port into your Tailscale tailnet with `tailscale serve --bg http://127.0.0.1:<vite-port>`.
This keeps the stable `portless` URL for local development while making the app reachable from your other Tailscale devices through your machine's `*.ts.net` address.
The launcher prints both the stable local URL and the Tailnet URL on startup.
If Serve is not yet enabled for your tailnet, the launcher prints the one-time enable link you need to open.
You can always print the expected tailnet hostname manually with `bun run dev:tailnet:url`.

If you want to bypass `portless` temporarily, use:

```bash
bun run dev:client:direct
```

If you want to skip the automatic Tailscale step for one run:

```bash
TAILSCALE_SERVE=0 bun run dev
```

## What it does

- generates dataset tiles in the browser from built-in Swiss scene presets
- renders the map as the single review canvas
- loads SWISSIMAGE tiles directly in the browser and slices scene tiles on demand
- runs CLIPSeg in the browser with `transformers.js`
- stores labels, config, and map review state in browser storage
- exports a ready-to-use ZIP with `labels.csv`, `tiles.json`, `config.toml`, and tile images
- enforces file-size limits with ESLint:
  - warning above `500` lines
  - error above `700` lines
