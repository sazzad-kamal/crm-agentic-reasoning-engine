import express from "express";
import { existsSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = process.env.PORT || 3000;
const AUTH_USER = process.env.AUTH_USER;
const AUTH_PASS = process.env.AUTH_PASS;

// Basic auth middleware — only active when AUTH_USER and AUTH_PASS are set
if (AUTH_USER && AUTH_PASS) {
  app.use((req, res, next) => {
    const header = req.headers.authorization;
    if (!header) {
      res.setHeader("WWW-Authenticate", 'Basic realm="Acme CRM AI"');
      return res.status(401).send("Authentication required");
    }

    const [scheme, encoded] = header.split(" ");
    if (scheme !== "Basic" || !encoded) {
      return res.status(401).send("Invalid auth format");
    }

    const [user, pass] = Buffer.from(encoded, "base64").toString().split(":");
    if (user === AUTH_USER && pass === AUTH_PASS) {
      return next();
    }

    res.setHeader("WWW-Authenticate", 'Basic realm="Acme CRM AI"');
    return res.status(401).send("Invalid credentials");
  });
}

// Serve static files from dist/
const distPath = join(__dirname, "dist");

if (!existsSync(distPath)) {
  console.error("Error: dist/ directory not found. Run 'npm run build' first.");
  process.exit(1);
}

app.use(express.static(distPath));

// SPA fallback — serve index.html for all non-file routes
app.get("*", (_req, res) => {
  res.sendFile(join(distPath, "index.html"));
});

app.listen(PORT, () => {
  const authStatus = AUTH_USER ? "enabled" : "disabled (no AUTH_USER set)";
  console.log(`Server running on port ${PORT} — auth ${authStatus}`);
});
