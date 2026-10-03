const { app, BrowserWindow } = require("electron");
const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = 4488;
const OUT_DIR = path.join(__dirname, "..", "out");

const MIME_TYPES = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".woff2": "font/woff2",
  ".txt": "text/plain",
};

function resolveFile(urlPath) {
  const clean = decodeURIComponent(urlPath.split("?")[0]);
  const candidates = [
    path.join(OUT_DIR, clean),
    path.join(OUT_DIR, clean, "index.html"),
    path.join(OUT_DIR, `${clean}.html`),
  ];
  for (const candidate of candidates) {
    if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) {
      return candidate;
    }
  }
  return null;
}

function startServer() {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const filePath = resolveFile(req.url === "/" ? "/index.html" : req.url);
      if (!filePath) {
        res.writeHead(404);
        res.end("Not found");
        return;
      }
      res.writeHead(200, {
        "Content-Type": MIME_TYPES[path.extname(filePath)] || "application/octet-stream",
      });
      fs.createReadStream(filePath).pipe(res);
    });
    server.listen(PORT, "127.0.0.1", () => resolve());
  });
}

// The frontend connects to the Python API; start the backend separately.
// TODO: bundle and launch the backend from the packaged desktop app.
async function createWindow() {
  await startServer();
  const win = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 960,
    minHeight: 640,
    backgroundColor: "#FAF9F5",
    title: "ZeroDelay",
    webPreferences: {
      contextIsolation: true,
    },
  });
  // The packaged app skips the marketing landing page entirely and opens
  // straight into the session/discussion view.
  win.loadURL(`http://127.0.0.1:${PORT}/app/`);
}

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
