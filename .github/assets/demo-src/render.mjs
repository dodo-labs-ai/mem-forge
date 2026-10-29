// Renders the README demo from stage.html.
//
//   node .github/assets/demo-src/render.mjs                 GIFs into .github/assets
//   node .github/assets/demo-src/render.mjs --mp4 <dir>     also 1080p MP4s, relaxed pace
//   node .github/assets/demo-src/render.mjs --stills <dir> 2,9.5,20
//                                                           PNG stills at those seconds
//
// Needs `npm ci` in admin/ (Playwright with Chromium, lucide-react, the Geist
// fonts), plus ffmpeg and gifsicle on PATH.
import { createServer } from "node:http";
import { mkdtemp, mkdir, readFile, rm } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { extname, join, resolve } from "node:path";
import { tmpdir } from "node:os";

const REPO = resolve(new URL("../../..", import.meta.url).pathname);
const ADMIN_MODULES = join(REPO, "admin/node_modules");
const { chromium } = await import(join(ADMIN_MODULES, "playwright/index.mjs"));

/** The lucide icons the stage draws, as the admin UI's own lucide-react version ships them. */
const ICONS = [
  "brain", "tags", "shield-check", "activity", "files", "kanban-square", "settings", "laptop", "plus", "clock",
  "server", "search", "folder-kanban", "pencil", "ellipsis", "git-branch", "arrow-left",
];
async function iconScript() {
  const dir = join(ADMIN_MODULES, "lucide-react/dist/esm/icons");
  const icons = {};
  for (const name of ICONS) {
    let file = join(dir, `${name}.mjs`);
    const source = await readFile(file, "utf8");
    // An alias icon re-exports its canonical module.
    const alias = source.match(/from '\.\/([\w-]+)\.mjs'/);
    if (alias && !source.includes("__iconData")) file = join(dir, `${alias[1]}.mjs`);
    const { __iconData } = await import(file);
    icons[name] = __iconData.node
      .map(([tag, { key, ...attrs }]) => `<${tag} ${Object.entries(attrs).map(([k, v]) => `${k}="${v}"`).join(" ")}/>`)
      .join("");
  }
  return `window.LUCIDE=${JSON.stringify(icons)};`;
}

const TYPES = { ".html": "text/html", ".js": "text/javascript", ".svg": "image/svg+xml", ".woff2": "font/woff2" };
const icons = await iconScript();
const server = createServer(async (req, res) => {
  const path = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
  if (path === "/icons.js") {
    res.writeHead(200, { "content-type": TYPES[".js"] });
    return res.end(icons);
  }
  try {
    const body = await readFile(join(REPO, path));
    res.writeHead(200, { "content-type": TYPES[extname(path)] ?? "application/octet-stream" });
    res.end(body);
  } catch {
    res.writeHead(404);
    res.end();
  }
}).listen(0);
const origin = `http://localhost:${server.address().port}`;
const browser = await chromium.launch();

async function openStage(theme, { scale = 1, pace = "" } = {}) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: scale });
  await page.goto(`${origin}/.github/assets/demo-src/stage.html?theme=${theme}&pace=${pace}`);
  await page.evaluate(() => window.ready);
  return page;
}

/** Renders every frame of one theme into `dir` as numbered PNGs. */
async function renderFrames(theme, fps, dir, options) {
  const page = await openStage(theme, options);
  const duration = await page.evaluate(() => window.DURATION);
  await mkdir(dir, { recursive: true });
  for (let i = 0; i < Math.round(duration * fps); i++) {
    await page.evaluate((t) => window.render(t), i / fps);
    await page.screenshot({ path: join(dir, `f${String(i).padStart(5, "0")}.png`) });
  }
  await page.close();
}

const run = (cmd, args) => execFileSync(cmd, args, { stdio: ["ignore", "ignore", "inherit"] });
const [flag, target, times] = process.argv.slice(2);
const work = await mkdtemp(join(tmpdir(), "memforge-demo-"));
try {
  if (flag === "--stills") {
    await mkdir(target, { recursive: true });
    for (const theme of ["light", "dark"]) {
      const page = await openStage(theme);
      for (const t of times.split(",").map(Number)) {
        await page.evaluate((x) => window.render(x), t);
        await page.screenshot({ path: join(target, `${theme}-${t}.png`) });
      }
      await page.close();
    }
  } else {
    await Promise.all(["light", "dark"].map(async (theme) => {
      const frames = join(work, `gif-${theme}`);
      await renderFrames(theme, 15, frames);
      const raw = join(work, `${theme}.gif`);
      run("ffmpeg", ["-v", "error", "-y", "-framerate", "15", "-i", join(frames, "f%05d.png"),
        "-vf", "split[a][b];[a]palettegen=max_colors=192:stats_mode=full[p];[b][p]paletteuse=dither=none:diff_mode=rectangle",
        "-loop", "0", raw]);
      run("gifsicle", ["-O3", "--lossy=30", raw, "-o", join(REPO, `.github/assets/demo-${theme}.gif`)]);
    }));
    if (flag === "--mp4") {
      await mkdir(target, { recursive: true });
      await Promise.all(["light", "dark"].map(async (theme) => {
        const frames = join(work, `mp4-${theme}`);
        await renderFrames(theme, 30, frames, { scale: 1.5, pace: "relaxed" });
        run("ffmpeg", ["-v", "error", "-y", "-framerate", "30", "-i", join(frames, "f%05d.png"),
          "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
          join(target, `memforge-demo-${theme}.mp4`)]);
      }));
    }
  }
} finally {
  await browser.close();
  server.close();
  await rm(work, { recursive: true, force: true });
}
