# Sentinel M3 hardware console

A browser UI for `sentinel-hwsim`. It runs a simulated fleet (4 IP cameras and 1 Kaaval edge node), monitors it through the driver contract the way M3-C6 would, and lets you trigger every failure mode from the browser.

## Run

Run from the project root, with the venv that has `pip install -e ".[dev]"` in it:

```
python ui/server.py            # open http://127.0.0.1:8765
python ui/server.py --open     # open the browser for you as well
```

| Option | Default | Meaning |
|---|---|---|
| `--port` | 8765 | HTTP port |
| `--host` | 127.0.0.1 | Bind address. On loopback, requests with a foreign `Host` header are refused. |
| `--speed` | 4 | Simulation time compression |
| `--seed` | 7 | Deterministic device behaviour |

There's no build step and nothing to install beyond the package itself. The server uses only the Python standard library (asyncio HTTP + Server-Sent Events). The frontend is three static files.

```
ui/
  server.py          fleet + monitoring + JSON API + SSE
  static/index.html  semantic markup, 5 views (hash routing)
  static/styles.css  design tokens, BEM components
  static/app.js      ES module: live feed, routing, forms
```

## Pages

| Route | Purpose |
|---|---|
| `#/overview` | Key figures, alert history (one per state transition), camera tiles, latest events |
| `#/cameras/CAM-00x` | Identity, live fps/bitrate, snapshot, stream URIs, PTZ and encoder commands, fault injection |
| `#/node` | Layered health check (node → runtime → inference → pipe), telemetry, failure controls, model deploy, watchlist |
| `#/events` | Filterable live AI event feed and the dead-letter queue |
| `#/trace` | Every simulated wire request, with latency and injected faults |

## API

| Method | Path | Body |
|---|---|---|
| GET | `/api/state` | Full snapshot |
| GET | `/api/stream` | SSE: `state`, `camera`, `node`, `ai_event`, `alert`, `dlq` |
| GET | `/api/trace?limit=N` | Last N wire requests |
| GET | `/api/devices/{id}/snapshot` | PNG |
| POST | `/api/devices/{cam}/ptz` · `encoder` | `{pan,tilt,zoom}` · `{profile,fps,bitrate_kbps}` |
| POST | `/api/devices/{id}/fault` · `offline` · `revoke` · `heal` | `{fault}` · `{on}` · `{on}` · `{}` |
| POST | `/api/devices/{node}/crash` · `restart` · `stall` · `unstall` · `cut` · `restore` | `{}` |
| POST | `/api/devices/{node}/ambient` · `model` · `watchlist` | `{celsius}` · `{model_version}` · `{plates:[]}` |

POSTs must be `application/json`. That forces a CORS preflight, so other websites can't drive the console. Device errors come back as `502 {error: <ErrorCode>, message, retryable}`.

## Design system

**Colour.** There are 4 base colours, and every other colour is a `color-mix()` tint of one of them. Dark mode follows `prefers-color-scheme`.

| Token | Light | Dark | Use | Contrast |
|---|---|---|---|---|
| `--ink` | #1b1d22 | #e7e8eb | Text | 16:1 |
| `--paper` | #fafaf7 | #121418 | Background | — |
| `--accent` | #2447c8 | #93a9ff | Interactive, OK state, focus ring | 7.4:1 / 7.3:1 |
| `--alert` | #b42318 | #ff8f84 | Faults, destructive actions | 6.6:1 / 7.0:1 |

Status is never shown by colour alone. Pills carry a glyph (● ok, ▲ warning, ■ critical) and a text label. Warnings use a dashed ink outline rather than a fifth colour.

**Type.** System font stack, 1.25 (major third) scale, 16 px base.

| Role | Size | Weight | Line height |
|---|---|---|---|
| h1 | 25 → 31.25 px (fluid `clamp`) | 600 | 1.2 |
| KPI value | 25 px, tabular numerals | 600 | 1.2 |
| h2 (panel titles) | 20 px | 600 | 1.3 |
| Body | 16 px | 400 | 1.55 |
| Data, mono | 14 px | 400 | — |
| Caption, labels | 12.8 px | 400 / 600 | 1.4 |

**Spacing and layout.** 4 px spacing scale (`--s1`…`--s8`: 4 to 64 px). The container is `min(100% − 2 × gutter, 80rem)`.

| Breakpoint | Width | Gutter | Changes |
|---|---|---|---|
| Base | 320 px+ | 16 px | Single column; nav and camera picker scroll horizontally |
| sm | ≥ 480 px | 16 px | KPIs in 2 columns; brand subtitle appears |
| md | ≥ 768 px | 24 px | Camera detail splits into 2 columns |
| lg | ≥ 1024 px | 32 px | Single-row header; KPIs in 4 columns; overview 5/7 split; vertical camera picker |
| xl | ≥ 1440 px | 32 px | Container widens to 86rem |

Wide tables scroll inside a focusable region, so the page itself never scrolls sideways.

**Motion.** 120 ms for hovers and colour changes, 200 ms for view and toast entrances, `cubic-bezier(0.2, 0, 0, 1)`. Meters use 400 ms. All motion is removed under `prefers-reduced-motion`.

## Accessibility (WCAG 2.1 AA)

- **Structure.** Landmarks (header, nav, main, footer), a skip link, one `h1` per view, labelled panel sections, `scope` on table headers, and scrollable table regions that can be focused and have names.
- **Focus.** On a route change, focus moves to the new view's `h1` and the document title updates. The focus ring is a 2 px accent outline with a 2 px offset, shown only for keyboard focus (`:focus-visible`).
- **Live updates.** Polling never rebuilds a focused control. Updates are keyed, and rows are only prepended. New alerts, command results and errors go to polite/assertive live regions. Visual toasts are `aria-hidden` duplicates of those messages.
- **Forms.** Every input has a `<label>`. Hints and errors are linked with `aria-describedby`. Invalid fields get `aria-invalid` and receive focus. Error text uses `role="alert"`.
- **Loading.** Busy buttons use `aria-busy` + `aria-disabled` rather than `disabled`, so keyboard focus isn't lost. `main` has `aria-busy` until the first data arrives. Skeleton placeholders are shown in the meantime.
- **Toggles and modes.** Offline and revoke buttons expose `aria-pressed`. Windows High Contrast mode is supported via `forced-colors`.

## Performance

- **Payload.** No framework, web fonts or images, apart from on-demand snapshots (blob URLs, fixed 16:9 frame so nothing shifts). About 78 KB of static files uncompressed, gzip-compressed by the server.
- **Critical path.** One render-blocking stylesheet in `<head>`. The script is `type="module"`, so it's deferred. The HTML ships a full skeleton, so first paint needs no JavaScript.
- **Updates.** Server-Sent Events replace polling. DOM updates are batched with `requestAnimationFrame`. Tables are capped at 100 rows and the event buffer at 300. The trace page polls only while it's visible and the tab isn't hidden.
- **Security headers.** A strict CSP (`default-src 'self'`, no inline script or style), `nosniff` and `no-referrer`. All device data is rendered with `textContent`, because garbled payloads can contain HTML.
