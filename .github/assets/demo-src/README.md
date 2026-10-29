# README demo

`stage.html` is the demo at the top of the project README: an HTML mockup of
a coding agent session and the admin UI, drawn with the admin UI's theme,
fonts and icons. It renders one frame for any time `t` through
`window.render(t)`, so every frame is reproducible.

The story follows a new hire's first ticket: without team memory the agent
guesses; with MemForge it answers from the team's docs and cites them; the
admin UI shows the sources syncing and a memory that a changed doc updated.

## Render

Install the admin UI's dependencies first (`cd admin && npm ci`); the
renderer uses its Playwright, lucide-react and Geist fonts. It also needs
`ffmpeg` and `gifsicle`.

```bash
node .github/assets/demo-src/render.mjs
```

This writes `.github/assets/demo-light.gif` and `demo-dark.gif`.

```bash
node .github/assets/demo-src/render.mjs --mp4 <dir>
```

This also writes 1080p MP4s to `<dir>` at the relaxed pace, which holds each
beat longer for a video with controls.

```bash
node .github/assets/demo-src/render.mjs --stills <dir> 2,9.5,20
```

This writes light and dark PNG stills at those seconds, for checking a change
without a full render.

## Editing

- Captions are the `.cap` elements; `CAPTIONS` ties each to a scene and the
  span of that scene's story time it covers.
- `SCENES` sets each scene's story length and its reading holds. A hold
  freezes the scene on a frame worth reading; repeated frames cost almost
  nothing in a GIF. `?pace=relaxed` lengthens every hold.
