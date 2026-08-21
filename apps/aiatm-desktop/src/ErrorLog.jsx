import React, { createContext, useContext, useState, useCallback, useEffect, useRef } from 'react';
import { XCircle, AlertTriangle, Info, X, Copy, Check } from 'lucide-react';

const ErrorLogContext = createContext(null);

let nextId = 1;

export function ErrorLogProvider({ children }) {
  const [errors, setErrors] = useState([]);

  const pushError = useCallback((message, level = 'error') => {
    setErrors(prev => [
      ...prev,
      { id: nextId++, message, level, timestamp: Date.now() }
    ]);
  }, []);

  const removeError = useCallback((id) => {
    setErrors(prev => prev.filter(e => e.id !== id));
  }, []);

  const clearErrors = useCallback(() => {
    setErrors([]);
  }, []);

  return (
    <ErrorLogContext.Provider value={{ pushError, removeError, errors, clearErrors }}>
      {children}
    </ErrorLogContext.Provider>
  );
}

export function useErrorLog() {
  const ctx = useContext(ErrorLogContext);
  if (!ctx) {
    throw new Error('useErrorLog must be used within an ErrorLogProvider');
  }
  return ctx;
}

const LEVEL_ICON = {
  error: XCircle,
  warn: AlertTriangle,
  info: Info
};

function formatTime(ts) {
  return new Date(ts).toLocaleTimeString();
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false);
  const onClick = () => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };
  return (
    <button className="error-log-copy-btn" onClick={onClick} aria-label="Copy error">
      {copied ? <Check size={14} /> : <Copy size={14} />}
    </button>
  );
}

export function ErrorToasts() {
  const { errors, removeError, clearErrors } = useErrorLog();
  const [dismissedIds, setDismissedIds] = useState(new Set());
  const [showLog, setShowLog] = useState(false);
  const timersRef = useRef({});

  const dismissToast = useCallback((id) => {
    setDismissedIds(prev => new Set(prev).add(id));
    clearTimeout(timersRef.current[id]);
    delete timersRef.current[id];
  }, []);

  const visibleToasts = errors.filter(e => !dismissedIds.has(e.id)).slice(-5);

  useEffect(() => {
    visibleToasts.forEach(err => {
      if (!timersRef.current[err.id]) {
        timersRef.current[err.id] = setTimeout(() => {
          dismissToast(err.id);
        }, 8000);
      }
    });
  }, [visibleToasts, dismissToast]);

  const handleClearLog = () => {
    clearErrors();
    setDismissedIds(new Set());
  };

  return (
    <>
      <div className="error-log-toast-stack">
        {visibleToasts.map(err => {
          const Icon = LEVEL_ICON[err.level] || XCircle;
          return (
            <div key={err.id} className={`error-log-toast error-log-toast-${err.level}`}>
              <Icon size={16} className="error-log-toast-icon" />
              <div className="error-log-toast-message">{err.message}</div>
              <CopyButton text={err.message} />
              <button
                className="error-log-toast-close"
                onClick={() => dismissToast(err.id)}
                aria-label="Dismiss"
              >
                <X size={14} />
              </button>
            </div>
          );
        })}
        {errors.length > 0 && (
          <button className="error-log-view-link" onClick={() => setShowLog(true)}>
            View Error Log ({errors.length})
          </button>
        )}
      </div>

      {showLog && (
        <div className="error-log-panel-overlay" onClick={() => setShowLog(false)}>
          <div className="error-log-panel" onClick={e => e.stopPropagation()}>
            <div className="error-log-panel-header">
              <span>Error Log</span>
              <div className="error-log-panel-header-actions">
                <button className="error-log-panel-clear" onClick={handleClearLog}>Clear</button>
                <button className="error-log-panel-close" onClick={() => setShowLog(false)}>
                  <X size={16} />
                </button>
              </div>
            </div>
            <div className="error-log-panel-body">
              {errors.length === 0 ? (
                <div className="error-log-panel-empty">No errors logged this session.</div>
              ) : (
                [...errors].reverse().map(err => {
                  const Icon = LEVEL_ICON[err.level] || XCircle;
                  return (
                    <div key={err.id} className={`error-log-panel-entry error-log-panel-entry-${err.level}`}>
                      <Icon size={14} className="error-log-panel-entry-icon" />
                      <div className="error-log-panel-entry-content">
                        <div className="error-log-panel-entry-message">{err.message}</div>
                        <div className="error-log-panel-entry-time">{formatTime(err.timestamp)}</div>
                      </div>
                      <CopyButton text={err.message} />
                      <button
                        className="error-log-entry-remove"
                        onClick={() => removeError(err.id)}
                        aria-label="Remove from log"
                      >
                        <X size={14} />
                      </button>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
