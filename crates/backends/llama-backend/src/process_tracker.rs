use std::collections::HashMap;
use std::process::Child;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tracing::{info, warn};

const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);
const SHUTDOWN_POLL: Duration = Duration::from_millis(100);

// ──────────────────────────────────────────────────────────────────────────
// Windows Job Object
//
// On Unix, when the daemon exits its orphaned children get re-parented to init
// and are reaped by the SIGCHLD handler / the `Drop` impl below. On Windows the
// story is completely different:
//
//   * The Electron launcher (`start.mjs`) tears the daemon down with
//     `daemon.kill()`, which is `TerminateProcess` on Windows — no signal, no
//     `Drop` impls, no graceful shutdown. `drop(children)` in `main.rs` never
//     runs.
//   * `TerminateProcess` does NOT cascade to grandchildren. The
//     `llama-server.exe` processes spawned by the daemon survive as orphans,
//     still holding their port (50052+) and VRAM. On the next app launch the
//     fresh daemon's `loaded_models` registry is empty, so the UI says "No
//     Model Loaded" while the orphan is still alive — that's Bug 2.
//
// The only reliable Windows mechanism that kills a whole process tree when the
// parent dies (for ANY reason — TerminateProcess, crash, normal exit) is a Job
// Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. The kernel owns the kill,
// not userland. We create one job, assign every spawned `llama-server.exe` to
// it, and hold the only handle on the job inside `ChildRegistry`. When the
// registry is dropped gracefully we `CloseHandle` explicitly; when the daemon
// is `TerminateProcess`'d the OS closes the handle as part of tearing the
// process down — in both cases the kernel force-kills every process in the
// job (including any CUDA workers that `llama-server.exe` itself forked).
// ──────────────────────────────────────────────────────────────────────────
#[cfg(windows)]
mod win_job {
    use tracing::{info, warn};
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, SetInformationJobObject,
        JobObjectExtendedLimitInformation, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_SET_QUOTA, PROCESS_TERMINATE};

    /// Wraps a Windows Job Object handle. The handle is a kernel object
    /// reference (just a pointer-sized token); it is safe to move and share
    /// across threads because all access goes through the kernel.
    pub struct JobObject(HANDLE);

    // SAFETY: A Job Object handle is an opaque kernel reference. The kernel
    // serializes all operations on it, so the handle itself is safe to send
    // to other threads and to share by reference. All mutation goes through
    // the FFI calls in this module, which take `&self`.
    unsafe impl Send for JobObject {}
    unsafe impl Sync for JobObject {}

    impl JobObject {
        /// Create a job whose members die when its last handle closes.
        pub fn new() -> Option<Self> {
            unsafe {
                // `CreateJobObjectW(NULL, NULL)` — no security descriptor, no name.
                let handle = CreateJobObjectW(std::ptr::null(), std::ptr::null());
                // Per the API contract a NULL return means failure.
                if handle.is_null() {
                    warn!("CreateJobObjectW returned NULL; child cleanup will rely on Drop only");
                    return None;
                }

                // KILL_ON_JOB_CLOSE: the kernel kills the whole job tree the
                // instant its last handle is closed — whether by our graceful
                // `Drop` or by the OS closing our handle when the daemon is
                // `TerminateProcess`'d. Every other limit field is left zero
                // (no CPU/memory caps, no scheduling constraints). We zero-init
                // the whole extended struct and flip just the LimitFlags bit,
                // which avoids needing to name every nested field (and the
                // `IO_COUNTERS` type, whose module path varies across
                // windows-sys versions).
                let mut ext: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
                ext.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

                let ok = SetInformationJobObject(
                    handle,
                    JobObjectExtendedLimitInformation,
                    &ext as *const _ as *const _,
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                );
                if ok == 0 {
                    warn!("SetInformationJobObject failed; closing job handle");
                    let _ = CloseHandle(handle);
                    return None;
                }
                Some(JobObject(handle))
            }
        }

        /// Attach an already-spawned child (by PID) to the job. The child
        /// must still be alive at the moment of assignment; the caller
        /// (`register`) invokes this immediately after a successful spawn.
        pub fn assign(&self, pid: u32) {
            unsafe {
                let proc = OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, 0, pid);
                if proc.is_null() {
                    warn!(pid, "OpenProcess failed; cannot add child to kill-on-close job");
                    return;
                }
                let ok = AssignProcessToJobObject(self.0, proc);
                let _ = CloseHandle(proc);
                if ok == 0 {
                    warn!(pid, "AssignProcessToJobObject failed; child may survive abrupt daemon exit");
                } else {
                    info!(pid, "child added to kill-on-close job object");
                }
            }
        }
    }

    impl Drop for JobObject {
        fn drop(&mut self) {
            // Closing the last handle to the job triggers
            // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. Any still-assigned processes
            // are terminated by the kernel.
            unsafe {
                let _ = CloseHandle(self.0);
            }
        }
    }
}

/// Tracks child processes spawned by the daemon so they can be terminated
/// when the daemon exits, even on a panic (Rust runs destructors during
/// unwinding).
///
/// `register` is called after a successful spawn. `take` removes an entry
/// and returns the owned `Child` so the caller can decide whether to
/// kill it (used by `unload_model`). `kill_all` is called from `Drop` and
/// from the signal-handler path; it kills every remaining child and
/// waits up to `SHUTDOWN_GRACE` for graceful exit before re-killing.
pub struct ChildRegistry {
    children: Mutex<HashMap<String, Child>>,
    /// On Windows, a single Job Object with KILL_ON_JOB_CLOSE that every
    /// spawned llama-server.exe is assigned to. When the daemon is
    /// `TerminateProcess`'d by the launcher (no `Drop` runs), the OS
    /// closes this handle and the kernel kills the whole job tree —
    /// the only reliable Windows mechanism for cleaning up grandchildren
    /// on abrupt parent termination.
    #[cfg(windows)]
    job: Option<win_job::JobObject>,
}

impl ChildRegistry {
    pub fn new() -> Self {
        Self {
            children: Mutex::new(HashMap::new()),
            #[cfg(windows)]
            job: win_job::JobObject::new(),
        }
    }

    /// Insert `child` under `key`. If a prior child exists under the same key,
    /// kill it first to avoid leaking it.
    pub fn register(&self, key: String, child: Child) {
        // Assign the child to the kill-on-close job BEFORE taking the map lock.
        // On Windows this is what guarantees the child dies with the daemon
        // even under TerminateProcess. We do it eagerly so the assignment is
        // in place before any later kill / drop path runs.
        #[cfg(windows)]
        {
            let pid = child.id();
            if let Some(job) = &self.job {
                job.assign(pid);
            }
        }

        let mut map = self.children.lock().unwrap();
        if let Some(mut prev) = map.remove(&key) {
            let _ = prev.kill();
            let _ = prev.wait();
        }
        map.insert(key, child);
    }

    /// Remove the entry for `key`, terminate the underlying child process,
    /// and return the owned (now defunct) `Child`. `take` is the unload path:
    /// every existing caller uses it precisely because they want the child
    /// gone, so killing is the right default. A future caller that needs
    /// ownership without killing should add a separate method instead of
    /// relying on this one. Returns `None` if no such key exists (e.g. the
    /// child already exited and was reaped).
    pub fn take(&self, key: &str) -> Option<Child> {
        let mut map = self.children.lock().unwrap();
        let mut child = map.remove(key)?;
        let _ = child.kill();
        let _ = child.wait();
        Some(child)
    }

    /// Drain and kill every tracked child. Best-effort: the first
    /// `kill()` is `TerminateProcess` on Windows and `SIGKILL` on Unix
    /// (both delivered via `std::process::Child::kill`). We poll for up to
    /// `SHUTDOWN_GRACE` for the process to actually exit; if it has not,
    /// we re-issue `kill()` and `wait()` to force the zombie reap.
    pub fn kill_all(&self, grace: Duration) {
        let mut map = self.children.lock().unwrap();
        if map.is_empty() {
            return;
        }
        for (key, mut child) in map.drain() {
            let pid = child.id();
            let _ = child.kill();
            let start = Instant::now();
            let exited = loop {
                match child.try_wait() {
                    Ok(Some(_)) => break true,
                    Ok(None) => {
                        if start.elapsed() >= grace {
                            break false;
                        }
                        std::thread::sleep(SHUTDOWN_POLL);
                    }
                    Err(_) => break true,
                }
            };
            if exited {
                info!(pid = ?pid, key = %key, "llama-server child terminated");
            } else {
                warn!(
                    pid = ?pid, key = %key, grace_secs = grace.as_secs(),
                    "llama-server child did not exit in time, force-killing"
                );
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

impl Default for ChildRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ChildRegistry {
    fn drop(&mut self) {
        self.kill_all(SHUTDOWN_GRACE);
        // The `job` field drops after `kill_all` returns, closing the job
        // handle. On Windows this is the safety net for the abrupt-termination
        // case: even though this `Drop` won't run when the daemon is
        // `TerminateProcess`'d, the OS closes the handle for us and the
        // kernel's KILL_ON_JOB_CLOSE does the same work. When we DO drop
        // gracefully, closing the handle is a no-op (kill_all already
        // terminated everything), but harmless.
    }
}