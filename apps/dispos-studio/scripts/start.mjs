#!/usr/bin/env node
/**
 * Dispos Studio Launcher
 * Builds the React app, then loads it via file:// protocol.
 * Watches for file changes and rebuilds automatically.
 */
import { execSync, spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { existsSync } from 'fs';
import http from 'http';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, '..');
const projectRoot = join(root, '..', '..');
const distIndex = join(root, 'dist', 'index.html');
const electronBin = join(root, 'node_modules', 'electron', 'dist', 'electron.exe');
const viteBin = join(root, 'node_modules', '.bin', 'vite.cmd');
const daemonBin = process.env.DISPOS_DAEMON_PATH || join(projectRoot, 'target', 'release', 'daemon-core.exe');
const modelsDir = join(projectRoot, 'models');

function daemonIsReady() {
  return new Promise(resolve => {
    const request = http.get('http://127.0.0.1:8080/health', response => {
      response.resume();
      resolve(true);
    });
    request.on('error', () => resolve(false));
    request.setTimeout(750, () => { request.destroy(); resolve(false); });
  });
}

async function ensureDaemon() {
  if (await daemonIsReady()) {
    console.log('[LAUNCHER] Reusing local daemon at http://127.0.0.1:8080.');
    return null;
  }
  if (!existsSync(daemonBin)) {
    throw new Error(`daemon-core not found at ${daemonBin}. Set DISPOS_DAEMON_PATH or build the daemon first.`);
  }
  console.log('[LAUNCHER] Starting local daemon...');
  const daemon = spawn(daemonBin, [], {
    cwd: projectRoot,
    stdio: 'inherit',
    env: { ...process.env, DISPOS_MODELS_DIR: modelsDir },
  });
  for (let attempt = 0; attempt < 60; attempt += 1) {
    if (await daemonIsReady()) {
      console.log('[LAUNCHER] Local daemon is ready.');
      return daemon;
    }
    await new Promise(resolve => setTimeout(resolve, 250));
  }
  daemon.kill();
  throw new Error('Timed out waiting for daemon-core on http://127.0.0.1:8080.');
}

// --- Kill stale daemon so cargo can overwrite the binary ---
async function killStaleDaemon() {
  if (!(await daemonIsReady())) return;
  console.log('[LAUNCHER] Stopping stale daemon so the binary can be rebuilt...');
  await new Promise(resolve => {
    http.get('http://127.0.0.1:8080/shutdown', () => {}).on('error', () => {});
    setTimeout(resolve, 1500);
  });
  if (await daemonIsReady()) {
    console.log('[LAUNCHER] Daemon still alive, force-killing daemon-core.exe...');
    try { execSync('taskkill /F /IM daemon-core.exe', { stdio: 'ignore' }); } catch {}
    await new Promise(resolve => setTimeout(resolve, 500));
  }
}

await killStaleDaemon();

// --- Build Rust daemon ---
console.log('[LAUNCHER] Building daemon-core (release)...');
try {
  execSync('cargo build --release --package daemon-core', { cwd: projectRoot, stdio: 'inherit' });
} catch (e) {
  console.error('[LAUNCHER] Cargo build failed:', e.message);
  process.exit(1);
}

let daemon;
try {
  daemon = await ensureDaemon();
} catch (error) {
  console.error(`[LAUNCHER] ${error.message}`);
  process.exit(1);
}

// --- Build React app ---
console.log('[LAUNCHER] Building React app...');
try {
  execSync(`cmd.exe /c "${viteBin}" build`, { cwd: root, stdio: 'inherit' });
} catch (e) {
  console.error('[LAUNCHER] Build failed:', e.message);
  process.exit(1);
}

if (!existsSync(distIndex)) {
  console.error('[LAUNCHER] dist/index.html not found after build!');
  process.exit(1);
}
console.log('[LAUNCHER] Build complete.');

// --- Launch Electron loading from file:// ---
console.log('[LAUNCHER] Launching Electron window...');
const electron = spawn(electronBin, ['electron-main.js'], {
  cwd: root,
  stdio: 'inherit',
  env: { ...process.env, DISPOS_DIST_PATH: distIndex },
});

// --- Watch for source changes and rebuild ---
console.log('[LAUNCHER] Starting file watcher for hot rebuilds...');
const watcher = spawn(
  'cmd.exe',
  ['/c', viteBin, 'build', '--watch'],
  { cwd: root, stdio: 'inherit', shell: false }
);

// --- Cleanup on exit ---
async function shutdown() {
  console.log('\n[LAUNCHER] Shutting down...');
  try { watcher.kill(); } catch {}
  try { electron.kill(); } catch {}
  // On Windows, call daemon's graceful shutdown endpoint before killing.
  // This ensures Job Objects are closed and children are terminated before
  // TerminateProcess abruptly kills the daemon.
  if (daemon && process.platform === 'win32') {
    try {
      await new Promise((resolve, reject) => {
        http.get('http://127.0.0.1:8080/health', response => {
          if (response.statusCode === 200) {
            // Daemon is alive; try to shut it down gracefully
            console.log('[LAUNCHER] Sending graceful shutdown signal to daemon');
            http.get('http://127.0.0.1:8080/shutdown', {}, () => {}).on('error', () => {});
            setTimeout(resolve, 500);
          } else {
            resolve();
          }
        }).on('error', () => resolve());
        setTimeout(resolve, 750);
      });
    } catch {}
  }
  try { daemon?.kill(); } catch {}
  process.exit(0);
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

electron.on('exit', () => {
  console.log('[LAUNCHER] Electron closed.');
  shutdown();
});
