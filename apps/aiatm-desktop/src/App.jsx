import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  MessageSquare, Sparkles, Volume2, Video,
  Cpu, HardDrive, Zap, Send, Play, Image, FileAudio, RefreshCw,
  Brain, ChevronDown, ChevronRight, Sliders, Folder, Power, Layers, Settings,
  CheckCircle2, XCircle, PackagePlus, Box, Paperclip, X, Pencil
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './App.css';

function findResponseStart(text) {
  const patterns = [
    /\n\n(?=Hello|Hi|Hey|Greetings|Sure|Certainly|I'm|I am|Here|As an AI)/i,
    /\n\n(?="[A-Z])/i
  ];
  for (const pat of patterns) {
    const match = text.match(pat);
    if (match && match.index > 50) {
      return match.index;
    }
  }
  return -1;
}

function parseThinking(content) {
  if (!content) return { thinking: '', answer: '', hadThinkTag: false };
  const str = content.trim();

  // 1. Check for explicit <think> tag
  const thinkStart = str.indexOf('<think>');
  const thinkEnd = str.indexOf('</think>');

  if (thinkStart !== -1) {
    if (thinkEnd !== -1 && thinkEnd > thinkStart) {
      const thinking = str.substring(thinkStart + 7, thinkEnd).trim();
      const answer = (str.substring(0, thinkStart) + '\n' + str.substring(thinkEnd + 8)).trim();
      return { thinking, answer, hadThinkTag: true };
    } else {
      // <think> opened but never closed — entire remainder is thinking, no answer yet
      const thinking = str.substring(thinkStart + 7).trim();
      const answer = str.substring(0, thinkStart).trim();
      return { thinking, answer, hadThinkTag: true };
    }
  }

  // 2. Check for </think> tag without opening <think>
  if (thinkEnd !== -1) {
    const thinking = str.substring(0, thinkEnd).replace(/^(?:Thinking|Thought)\s+Process:?\s*/i, '').trim();
    const answer = str.substring(thinkEnd + 8).trim();
    return { thinking, answer, hadThinkTag: true };
  }

  // 3. Check for "Thought Process" or "Thinking Process" block without <think> tags
  const tpRegex = /^(?:Thought|Thinking)\s+Process(?:\s*\(\d+\s*words\))?:?\s*/i;
  if (tpRegex.test(str)) {
    const body = str.replace(tpRegex, '').trim();
    const responseStart = findResponseStart(body);
    if (responseStart !== -1) {
      const thinking = body.substring(0, responseStart).trim();
      const answer = body.substring(responseStart).trim();
      return { thinking, answer, hadThinkTag: true };
    }
    return { thinking: body, answer: '', hadThinkTag: true };
  }

  // 4. Default: No reasoning detected, whole text is answer
  return { thinking: '', answer: str, hadThinkTag: false };
}

function ThinkingBlock({ thinkText, hadThinkTag }) {
  const [isOpen, setIsOpen] = useState(false);

  // Only render when there is actual reasoning content
  if (!hadThinkTag || !thinkText || thinkText.trim().length === 0) return null;

  const wordCount = thinkText.trim().split(/\s+/).filter(Boolean).length;

  return (
    <div className="thinking-accordion">
      <button className="thinking-header" onClick={() => setIsOpen(!isOpen)}>
        <div className="thinking-title">
          <Brain size={14} className="thinking-icon" />
          <span>Thought Process ({wordCount} words)</span>
        </div>
        {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
      </button>
      {isOpen && (
        <div className="thinking-content">
          {thinkText}
        </div>
      )}
    </div>
  );
}

// Stable identity for a loaded model. Backend model_ids are timestamped
// (mdl-{millis}-{port}), so reloading the same file yields a new id. The file's
// basename is the only thing that survives across loads; normalize it so
// backslash vs forward slash, casing, and quantization-tag drift all collapse
// to the same key.
function canonicalStudioKey(modelPath, modelName) {
  const raw = (modelPath || modelName || '').toString();
  const basename = raw.split(/[\\/]/).pop() || raw;
  const stripped = basename.replace(/\.(gguf|safetensors|onnx|bin|ckpt)$/i, '');
  const cleaned = stripped.toLowerCase().replace(/\s+/g, '-').trim();
  return cleaned || raw;
}

function AttachmentPreview({ attachment }) {
  if (attachment.type.startsWith('image/')) {
    return <img className="chat-attachment-image" src={attachment.dataUrl} alt={attachment.name} />;
  }
  if (attachment.type.startsWith('audio/')) {
    return <audio className="chat-attachment-media" controls src={attachment.dataUrl} />;
  }
  if (attachment.type.startsWith('video/')) {
    return <video className="chat-attachment-media" controls src={attachment.dataUrl} />;
  }
  return <a className="chat-attachment-file" href={attachment.dataUrl} download={attachment.name}>{attachment.name}</a>;
}

export default function App() {
  const [activeTab, setActiveTab] = useState('models');
  const [modelPath, setModelPath] = useState(
    'models\\Qwen3.5-0.8B-Q8_0.gguf'
  );
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      role: 'assistant',
      content: 'Hello! I am your AIATM Local Inference engine running natively with NVIDIA CUDA GPU acceleration.\n\nEnter your GGUF model path and prompt below to generate real responses at over 140+ tokens/sec!',
      telemetry: null
    }
  ]);
  const [inputPrompt, setInputPrompt] = useState('');
  const [attachments, setAttachments] = useState([]);
  const attachmentInputRef = useRef(null);
  const mmprojInputRef = useRef(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [appSettings, setAppSettings] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('aiatm.general-settings')) ?? {
        autoSelectNewest: true,
        refreshSeconds: 3,
        showThoughtProcess: true,
      };
    } catch {
      return { autoSelectNewest: true, refreshSeconds: 3, showThoughtProcess: true };
    }
  });

  useEffect(() => {
    localStorage.setItem('aiatm.general-settings', JSON.stringify(appSettings));
  }, [appSettings]);

  // Hardware stats state
  const [sysInfo, setSysInfo] = useState({
    cpu_cores: 20,
    total_ram_gb: 32.0,
    free_ram_gb: 20.0,
    total_vram_gb: 16.0,
    free_vram_gb: 12.0
  });

  // Image Studio State
  const [imgPrompt, setImgPrompt] = useState('A futuristic cyberpunk city skyline at sunset, photorealistic, 8k');
  const [imgSrc, setImgSrc] = useState(null);
  const [isGeneratingImg, setIsGeneratingImg] = useState(false);

  // Voice Studio State
  const [ttsInput, setTtsInput] = useState('Welcome to AIATM Standalone Desktop Application.');
  const [audioSrc, setAudioSrc] = useState(null);
  const [isGeneratingTts, setIsGeneratingTts] = useState(false);

  // Default values used by the fit preview and new model configuration.
  const [gpuLayers, setGpuLayers] = useState(99);
  const [contextSize, setContextSize] = useState(4096);
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  // Multi-model state
  const [loadedModels, setLoadedModels] = useState([]); // array of LoadedModelEntry from backend
  const [selectedModelId, setSelectedModelId] = useState(null); // which model chat uses
  const [openStudios, setOpenStudios] = useState([]); // { modelId, name, modality }
  const [chatSessions, setChatSessions] = useState(() => {
    try {
      const stored = JSON.parse(localStorage.getItem('aiatm.chat-sessions'));
      if (!Array.isArray(stored)) return [];
      // One-time migration: stamp studioKey on legacy sessions that predate
      // canonical keying. Older items lack modelPath; fall back to modelName.
      let migrated = false;
      const result = stored.map(item => {
        if (!item) return item;
        if (item.studioKey) return item;
        const studioKey = canonicalStudioKey(item.modelPath, item.modelName || item.modelId);
        const modelName = item.modelName
          || (item.modelPath ? item.modelPath.split(/[\\/]/).pop() : '')
          || (item.modelId ? item.modelId : '')
          || 'Local model';
        migrated = true;
        return { ...item, modelName, studioKey };
      });
      if (migrated) {
        try { localStorage.setItem('aiatm.chat-sessions', JSON.stringify(result)); } catch {}
      }
      return result;
    } catch { return []; }
  });
  const [activeChatId, setActiveChatId] = useState(null);
  const [collapsedStudios, setCollapsedStudios] = useState(() => {
    try { return JSON.parse(localStorage.getItem('aiatm.studio-collapsed')) ?? {}; } catch { return {}; }
  });
  const [editingSessionId, setEditingSessionId] = useState(null);
  const [editValue, setEditValue] = useState('');
  const [pendingDeleteId, setPendingDeleteId] = useState(null);
  const [pendingDeleteModelKey, setPendingDeleteModelKey] = useState(null);

  useEffect(() => {
    if (!pendingDeleteId) return;
    const id = setTimeout(() => setPendingDeleteId(null), 3500);
    return () => clearTimeout(id);
  }, [pendingDeleteId]);

  useEffect(() => {
    if (!pendingDeleteModelKey) return;
    const id = setTimeout(() => setPendingDeleteModelKey(null), 3500);
    return () => clearTimeout(id);
  }, [pendingDeleteModelKey]);

  // Rename animation: keep a displayed-name per session id, separate from
  // chat.title, so we can walk letter-by-letter from old to new on rename.
  const [displayedTitles, setDisplayedTitles] = useState({});
  const prevTitlesRef = useRef({});
  const animIntervalsRef = useRef({});
  // Session ids queued for fade-out before being dropped from chatSessions.
  const [pendingRemoveIds, setPendingRemoveIds] = useState([]);

  // Keep displayedTitles in sync with the chatSessions list (additions/removals).
  useEffect(() => {
    const sessionIds = new Set(chatSessions.map(session => session.id));
    setDisplayedTitles(current => {
      const next = { ...current };
      let changed = false;
      for (const session of chatSessions) {
        if (!(session.id in next)) {
          next[session.id] = session.title;
          changed = true;
        }
      }
      for (const id of Object.keys(next)) {
        if (!sessionIds.has(id)) {
          delete next[id];
          delete prevTitlesRef.current[id];
          if (animIntervalsRef.current[id]) {
            clearInterval(animIntervalsRef.current[id]);
            delete animIntervalsRef.current[id];
          }
          changed = true;
        }
      }
      return changed ? next : current;
    });
    for (const session of chatSessions) {
      if (!(session.id in prevTitlesRef.current)) {
        prevTitlesRef.current[session.id] = session.title;
      }
    }
  }, [chatSessions]);

  // When a session's title changes, walk from the old title to the new one
  // letter-by-letter over ~1.5s. Only fires on actual title changes — initial
  // mount and same-title re-renders are skipped via prevTitlesRef.
  useEffect(() => {
    const prevTitles = prevTitlesRef.current;
    for (const session of chatSessions) {
      const prev = prevTitles[session.id];
      if (prev === undefined || prev === session.title) continue;
      const oldTitle = prev;
      const newTitle = session.title;
      prevTitles[session.id] = session.title;

      if (animIntervalsRef.current[session.id]) {
        clearInterval(animIntervalsRef.current[session.id]);
      }

      const maxLen = Math.max(oldTitle.length, newTitle.length, 1);
      const stepMs = Math.max(40, Math.floor(1500 / maxLen));
      let step = 0;

      const tick = () => {
        step++;
        if (step > maxLen) {
          clearInterval(animIntervalsRef.current[session.id]);
          delete animIntervalsRef.current[session.id];
          setDisplayedTitles(current => ({ ...current, [session.id]: newTitle }));
          return;
        }
        const revealed = newTitle.slice(0, step);
        const remaining = oldTitle.slice(step);
        setDisplayedTitles(current => ({ ...current, [session.id]: revealed + remaining }));
      };

      animIntervalsRef.current[session.id] = setInterval(tick, stepMs);
    }
  }, [chatSessions]);

  // Clear any in-flight rename intervals when the app unmounts.
  useEffect(() => () => {
    for (const id in animIntervalsRef.current) {
      clearInterval(animIntervalsRef.current[id]);
    }
  }, []);

  const [configTarget, setConfigTarget] = useState(null);
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [unloadingModelId, setUnloadingModelId] = useState(null);
  const [modelFitPreview, setModelFitPreview] = useState(null);
  const [detectedModels, setDetectedModels] = useState([]);
  const [modelCards, setModelCards] = useState(() => {
    const saved = localStorage.getItem('antioom-model-cards');
    return saved ? JSON.parse(saved) : [];
  });
  const [showModelPicker, setShowModelPicker] = useState(false);
  const [modelPickerSearch, setModelPickerSearch] = useState('');

  useEffect(() => {
    localStorage.setItem('antioom-model-cards', JSON.stringify(modelCards));
  }, [modelCards]);

  useEffect(() => {
    localStorage.setItem('aiatm.chat-sessions', JSON.stringify(chatSessions));
  }, [chatSessions]);

  useEffect(() => {
    localStorage.setItem('aiatm.studio-collapsed', JSON.stringify(collapsedStudios));
  }, [collapsedStudios]);

  // Model preset catalog
  const MODEL_PRESETS = [
    {
      name: 'Qwen 3.5 0.8B',
      tag: 'Q8_0 · 0.8B',
      path: 'models\\Qwen3.5-0.8B-Q8_0.gguf',
      defaultGpu: 99, defaultCtx: 4096, defaultTemp: 0.7, defaultTopP: 0.9,
    },
    {
      name: 'DeepSeek R1 1.5B',
      tag: 'Q4_K_M · 1.5B',
      path: 'C:\\Users\\adem2\\.lmstudio\\models\\deepseek-ai\\DeepSeek-R1-Distill-Qwen-1.5B-GGUF\\DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf',
      defaultGpu: 99, defaultCtx: 4096, defaultTemp: 0.6, defaultTopP: 0.95,
    },
  ];

  const openConfigPanel = (preset) => {
    const maxContextSize = preset.max_context_size ?? null;
    setConfigTarget({
      model_path: preset.path,
      name: preset.name,
      tag: preset.tag,
      gpu_layers: preset.defaultGpu ?? 99,
      context_size: maxContextSize ? Math.min(preset.defaultCtx ?? 4096, maxContextSize) : (preset.defaultCtx ?? 4096),
      max_context_size: maxContextSize,
      temperature: preset.defaultTemp ?? 0.7,
      top_p: preset.defaultTopP ?? 0.9,
      mmproj_path: preset.mmproj_path || detectedModels.find(m => m.path === preset.path)?.mmproj_path || '',
    });
  };

  const fetchLoadedModels = useCallback(async () => {
    try {
      const res = await fetch('http://127.0.0.1:8080/v1/model/list');
      const data = await res.json();
      if (Array.isArray(data)) {
        setLoadedModels(data);
        // Auto-select first model if none selected
        if (data.length > 0 && !selectedModelId) {
          setSelectedModelId(data[0].model_id);
        }
        if (data.length === 0) setSelectedModelId(null);
      }
    } catch {}
  }, [selectedModelId]);

  const handleLoadModel = async (configuration = configTarget) => {
    if (!configuration?.model_path?.trim() || isLoadingModel) return;
    setIsLoadingModel(true);
    try {
      const res = await fetch('http://127.0.0.1:8080/v1/model/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_path: configuration.model_path,
          gpu_layers: configuration.gpu_layers,
          context_size: configuration.context_size,
          mmproj_path: configuration.mmproj_path || undefined,
        }),
      });
      const data = await res.json();
      if (res.ok && data?.model_id) {
        await fetchLoadedModels();
        fetchSystemInfo();
        setTimeout(fetchSystemInfo, 1500);
        return data.model_id;
      } else {
        alert('Failed to load model: ' + (data?.error || res.statusText));
        return null;
      }
    } catch (err) {
      alert('Error loading model: ' + err.message);
      return null;
    } finally {
      setIsLoadingModel(false);
    }
  };

  const handleUnloadModel = async (modelId) => {
    if (!modelId) return;
    setUnloadingModelId(modelId);
    try {
      const res = await fetch('http://127.0.0.1:8080/v1/model/unload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId }),
      });
      if (res.ok) {
        await fetchLoadedModels();
        setOpenStudios(current => current.filter(studio => studio.modelId !== modelId));
        if (selectedModelId === modelId) {
          setSelectedModelId(null);
        }
        fetchSystemInfo();
        setTimeout(fetchSystemInfo, 1500);
      }
    } catch (err) {
      alert('Error ejecting model: ' + err.message);
    } finally {
      setUnloadingModelId(null);
    }
  };

  useEffect(() => {
    fetchLoadedModels();
    const interval = setInterval(fetchLoadedModels, appSettings.refreshSeconds * 1000);
    return () => clearInterval(interval);
  }, [appSettings.refreshSeconds, fetchLoadedModels]);

  useEffect(() => {
    fetch('http://127.0.0.1:8080/v1/model/catalog')
      .then(res => res.ok ? res.json() : [])
      .then(models => setDetectedModels(Array.isArray(models) ? models : []))
      .catch(() => setDetectedModels([]));
  }, []);

  const selectedModel = loadedModels.find(model => model.model_id === selectedModelId) ?? null;
  const selectedCatalogModel = detectedModels.find(model => model.path === selectedModel?.model_path);
  const acceptsImageInput = selectedCatalogModel?.image_input_available === true;
  const openModelStudio = (model) => {
    const studio = model.modality === 'vision' ? 'chat' : model.modality;
    setActiveTab(['chat', 'image', 'audio', 'video'].includes(studio) ? studio : 'chat');
  };
  const startStudio = (loadedEntry, catalogModel) => {
    const modality = catalogModel?.modality ?? 'text';
    const modelName = catalogModel?.name ?? loadedEntry.model_path?.split('\\').pop() ?? 'Local model';
    const chatId = `chat-${Date.now()}`;
    const initialMessages = [{ id: 'welcome', role: 'assistant', content: `Ready to use ${modelName}.`, telemetry: null }];
    setSelectedModelId(loadedEntry.model_id);
    setOpenStudios(current => current.some(studio => studio.modelId === loadedEntry.model_id)
      ? current
      : [...current, {
          modelId: loadedEntry.model_id,
          name: modelName,
          modality,
        }]);
    const session = { id: chatId, title: 'New chat', modelId: loadedEntry.model_id, modelPath: loadedEntry.model_path, modelName, modality, studioKey: canonicalStudioKey(loadedEntry.model_path, modelName), messages: initialMessages, createdAt: Date.now() };
    setChatSessions(current => [...current, session]);
    setActiveChatId(chatId);
    setMessages(initialMessages);
    openModelStudio({ modality });
  };
  const syncMessages = (updater) => setMessages(previous => {
    const next = typeof updater === 'function' ? updater(previous) : updater;
    if (activeChatId) setChatSessions(current => current.map(session => session.id === activeChatId ? { ...session, messages: next } : session));
    return next;
  });
  const generateChatTitle = async (sessionId, prompt) => {
    try {
      const response = await fetch('http://127.0.0.1:8080/v1/chat/completions/stream', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelPath, model_id: selectedModelId, max_tokens: 16, temperature: 0.2, messages: [{ role: 'user', content: `Give this chat a concise 3-6 word title. Return only the title. Topic: ${prompt}` }] }),
      });
      if (!response.body) return;
      const reader = response.body.getReader(); const decoder = new TextDecoder(); let title = '';
      while (true) { const { done, value } = await reader.read(); if (done) break; for (const line of decoder.decode(value, { stream: true }).split('\n')) { try { const raw = line.replace(/^data:\s*/, ''); if (raw && raw !== '[DONE]') title += JSON.parse(raw)?.choices?.[0]?.delta?.content ?? ''; } catch {} } }
      title = title.replace(/[\n#*_`]/g, ' ').trim().slice(0, 64);
      if (title) setChatSessions(current => current.map(session => session.id === sessionId ? { ...session, title } : session));
    } catch {}
  };
  const activeLoadedModel = loadedModels[0] ?? null;
  const modelStatus = {
    is_loaded: Boolean(activeLoadedModel),
    model_path: activeLoadedModel?.model_path,
    gpu_layers: activeLoadedModel?.gpu_layers,
    context_size: activeLoadedModel?.context_size,
  };


  useEffect(() => {
    fetch('http://127.0.0.1:8080/v1/fit-estimator', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        parameter_count_billions: 0.8,
        quantization: 'Q8_0',
        context_size: parseInt(contextSize, 10),
        modality: 'Text',
      }),
    })
      .then(res => res.json())
      .then(data => setModelFitPreview(data))
      .catch(() => {});
  }, [contextSize, configTarget?.model_path]);

  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sysInfoInFlight = useRef(false);
  const fetchSystemInfo = async () => {
    if (sysInfoInFlight.current) return;
    sysInfoInFlight.current = true;
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 4000);
      const res = await fetch('http://127.0.0.1:8080/v1/system/info', { signal: controller.signal });
      clearTimeout(timeout);
      const data = await res.json();
      if (data) {
        setSysInfo({
          cpu_cores: data.cpu_cores || 20,
          total_ram_gb: (data.total_ram_bytes / (1024 * 1024 * 1024)).toFixed(1),
          free_ram_gb: (data.available_ram_bytes / (1024 * 1024 * 1024)).toFixed(1),
          total_vram_gb: (data.total_vram_bytes / (1024 * 1024 * 1024)).toFixed(1),
          free_vram_gb: (data.free_vram_bytes / (1024 * 1024 * 1024)).toFixed(1)
        });
      }
    } catch {} finally {
      sysInfoInFlight.current = false;
    }
  };

  useEffect(() => {
    fetchSystemInfo();
    const interval = setInterval(fetchSystemInfo, 3000);
    const onVisibility = () => { if (document.visibilityState === 'visible') fetchSystemInfo(); };
    document.addEventListener('visibilitychange', onVisibility);
    return () => { clearInterval(interval); document.removeEventListener('visibilitychange', onVisibility); };
  }, []);

  const handleSendMessage = async () => {
    if ((!inputPrompt.trim() && attachments.length === 0) || isGenerating) return;

    const userText = inputPrompt.trim();
    setInputPrompt('');

    const aiId = 'ai-' + Date.now();
    const userMsg = { id: 'u-' + Date.now(), role: 'user', content: userText, attachments };
    const tempAiMsg = { id: aiId, role: 'assistant', content: '', loading: true };

    syncMessages(prev => [...prev, userMsg, tempAiMsg]);
    setIsGenerating(true);
    setAttachments([]);

    // Accumulators for the two phases
    let thinkingBuf = '';
    let answerBuf = '';
    let inThinkingPhase = true; // reasoning_content comes before content
    let tokenCount = 0;
    let finishReason = null;
    const startTime = Date.now();

    const updateMsg = (done = false) => {
      const elapsed = (Date.now() - startTime) / 1000;
      const speed = elapsed > 0.1 ? (tokenCount / elapsed).toFixed(1) : '0.0';
      const telemetry = `${speed} tok/s | CUDA GPU`;

      // Compose the combined content string that parseThinking() can parse
      const combined = thinkingBuf
        ? `<think>\n${thinkingBuf}\n</think>\n\n${answerBuf}`
        : answerBuf;
      syncMessages(prev =>
        prev.map(m =>
          m.id === aiId
            ? { ...m, content: combined, loading: !done, telemetry: telemetry, truncated: done && finishReason === 'length' }
            : m
        )
      );
    };

    try {
      const res = await fetch('http://127.0.0.1:8080/v1/chat/completions/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: modelPath,
          model_id: selectedModelId,
          messages: [{ role: 'user', content: attachments.length ? [
            { type: 'text', text: userText },
            ...attachments.map(attachment => ({ type: 'image_url', image_url: { url: attachment.dataUrl } })),
          ] : userText }],
          max_tokens: 4096,
          temperature: parseFloat(temperature),
          top_p: parseFloat(topP),
        }),
      });

      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value, { stream: true });
        // SSE lines look like: data: {...json...}\n
        const lines = text.split('\n');
        for (const line of lines) {
          const raw = line.startsWith('data:') ? line.slice(5).trim() : line.trim();
          if (!raw || raw === '[DONE]') continue;

          try {
            const chunk = JSON.parse(raw);
            const choice = chunk?.choices?.[0] ?? {};
            const delta = choice.delta ?? {};
            if (choice.finish_reason) finishReason = choice.finish_reason;

            let chunkAdded = false;
            if (delta.reasoning_content != null) {
              thinkingBuf += delta.reasoning_content;
              inThinkingPhase = true;
              tokenCount++;
              chunkAdded = true;
            }
            if (delta.content != null && delta.content !== '') {
              if (inThinkingPhase) inThinkingPhase = false;
              answerBuf += delta.content;
              tokenCount++;
              chunkAdded = true;
            }
            if (chunkAdded) {
              updateMsg(false);
            }
          } catch {
            // partial JSON chunk, skip
          }
        }
      }

      // Stream finished — mark done
      updateMsg(true);
      if (activeChatId && chatSessions.find(item => item.id === activeChatId)?.title === 'New chat') {
        generateChatTitle(activeChatId, userText);
      }
    } catch (err) {
      syncMessages(prev =>
        prev.map(m =>
          m.id === aiId
            ? { ...m, content: 'Error: ' + err.message, loading: false }
            : m
        )
      );
    } finally {
      setIsGenerating(false);
    }
  };

  const addAttachments = async (files) => {
    if (!acceptsImageInput) return;
    const imageFiles = Array.from(files).filter(file => file.type.startsWith('image/'));
    const accepted = await Promise.all(imageFiles.slice(0, 4).map(file => new Promise(resolve => {
      const reader = new FileReader();
      reader.onload = () => resolve({ name: file.name, type: file.type, dataUrl: reader.result });
      reader.readAsDataURL(file);
    })));
    setAttachments(current => [...current, ...accepted]);
  };

  const handleGenerateImage = async () => {
    setIsGeneratingImg(true);
    try {
      const res = await fetch('http://127.0.0.1:8080/v1/images/generations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: imgPrompt })
      });
      const data = await res.json();
      if (data?.data?.[0]?.b64_json) {
        setImgSrc('data:image/png;base64,' + data.data[0].b64_json);
      }
    } catch (e) {
      alert('Image generation error: ' + e);
    } finally {
      setIsGeneratingImg(false);
    }
  };

  const handleSynthesizeSpeech = async () => {
    setIsGeneratingTts(true);
    try {
      const res = await fetch('http://127.0.0.1:8080/v1/audio/speech', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'kokoro', input: ttsInput })
      });
      const data = await res.json();
      if (data?.audio_b64) {
        setAudioSrc('data:audio/wav;base64,' + data.audio_b64);
      }
    } catch (e) {
      alert('TTS error: ' + e);
    } finally {
      setIsGeneratingTts(false);
    }
  };

  const totalRam = parseFloat(sysInfo.total_ram_gb) || 16.0;
  const freeRam = parseFloat(sysInfo.free_ram_gb) || 10.0;
  const usedRam = totalRam - freeRam;
  const ramPct = Math.min(100, Math.max(0, (usedRam / totalRam) * 100)).toFixed(0);

  const totalVram = parseFloat(sysInfo.total_vram_gb) || 8.0;
  const freeVram = parseFloat(sysInfo.free_vram_gb) || 6.0;
  const usedVram = totalVram - freeVram;
  const vramPct = Math.min(100, Math.max(0, (usedVram / totalVram) * 100)).toFixed(0);

  return (
    <div className="desktop-app">
      {/* Top Header Bar */}
      <header className="desktop-header">
        <div className="logo-group">
          <div className="logo-icon">AI</div>
          <div className="logo-title">AIATM Desktop</div>
          <span className="version-tag">Standalone Native</span>
        </div>

        <div className="header-stats">
          <div className="stat-item">
            <Cpu size={14} style={{ color: 'var(--accent-blue)' }} />
            <span>VRAM:</span>
            <div className="stat-progress">
              <div className="stat-bar" style={{ width: `${vramPct}%` }}></div>
            </div>
            <span>{usedVram.toFixed(1)} / {totalVram.toFixed(1)} GB</span>
          </div>

          <div className="stat-item">
            <HardDrive size={14} style={{ color: 'var(--accent-purple)' }} />
            <span>RAM:</span>
            <div className="stat-progress">
              <div className="stat-bar" style={{ width: `${ramPct}%`, background: 'var(--accent-purple)' }}></div>
            </div>
            <span>{usedRam.toFixed(1)} / {totalRam.toFixed(1)} GB</span>
          </div>

          <div className="status-badge">
            <div className="status-dot"></div>
            <span>CUDA GPU Active</span>
          </div>
        </div>
      </header>

      {/* Main Content Body */}
      <div className="app-container">
        {/* Left Sidebar */}
        <div className="sidebar">
          <div className="nav-section">MODEL CENTER</div>

          <button
            className={`nav-btn ${activeTab === 'models' ? 'active' : ''}`}
            onClick={() => setActiveTab('models')}
          >
            <Box size={17} /> Models
          </button>

          <button
            className={`nav-btn ${activeTab === 'settings' ? 'active' : ''}`}
            onClick={() => setActiveTab('settings')}
          >
            <Settings size={17} /> General Settings
          </button>

          <div className="nav-section" style={{ marginTop: '1rem' }}>STUDIOS</div>
          {chatSessions.length === 0 ? <div style={{ color: 'var(--text-secondary)', fontSize: '0.78rem', padding: '0.25rem 0.75rem 0.75rem' }}>Start a studio from a loaded model in Model Center.</div>
          : Object.entries(chatSessions.reduce((groups, chat) => {
              const modelKey = chat.studioKey || canonicalStudioKey(chat.modelPath, chat.modelName);
              if (!modelKey) return groups;
              groups[modelKey] ??= { name: chat.modelName, modelId: chat.modelId, chats: [] };
              groups[modelKey].chats.push(chat);
              return groups;
            }, {})).map(([modelKey, model]) => {
              const modelOpen = !(collapsedStudios[modelKey] ?? false);
              const isPendingDeleteModel = pendingDeleteModelKey === modelKey;
              const modelSessionIds = model.chats.map(c => c.id);
              const handleDeleteModelClick = (e) => {
                e.stopPropagation();
                if (isPendingDeleteModel) {
                  setPendingRemoveIds(ids => {
                    const additions = modelSessionIds.filter(id => !ids.includes(id));
                    return additions.length ? [...ids, ...additions] : ids;
                  });
                  setTimeout(() => {
                    setChatSessions(current => current.filter(item => {
                      const k = item.studioKey || canonicalStudioKey(item.modelPath, item.modelName);
                      return k !== modelKey;
                    }));
                    setPendingRemoveIds(ids => ids.filter(id => !modelSessionIds.includes(id)));
                    if (activeChatId && modelSessionIds.includes(activeChatId)) {
                      setActiveChatId(null);
                      setMessages([]);
                    }
                    setPendingDeleteModelKey(null);
                  }, 350);
                } else {
                  setPendingDeleteModelKey(modelKey);
                }
              };
              return <div key={modelKey} style={{ paddingLeft: '0.55rem' }}>
                <div className={`studio-group-row ${isPendingDeleteModel ? 'pending-delete' : ''}`}>
                  <button className="nav-btn studio-group-toggle" onClick={() => setCollapsedStudios(current => ({ ...current, [modelKey]: !(current[modelKey] ?? false) }))}>
                    {modelOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />} {model.name}
                  </button>
                  <div className="studio-group-actions">
                    <button type="button" className={`chat-session-action delete ${isPendingDeleteModel ? 'confirm' : ''}`} onClick={handleDeleteModelClick} title={isPendingDeleteModel ? 'Click again to confirm delete all studios' : 'Delete all studios for this model'} aria-label="Delete all studios for this model">
                      {isPendingDeleteModel ? <span className="confirm-label">Delete?</span> : <X size={13} />}
                    </button>
                  </div>
                </div>
                {modelOpen && model.chats.map(chat => {
                  const isEditing = editingSessionId === chat.id;
                  const isPendingDelete = pendingDeleteId === chat.id;
                  const commitRename = () => {
                    const trimmed = editValue.trim();
                    const targetId = editingSessionId;
                    setEditingSessionId(null);
                    setEditValue('');
                    if (!trimmed || !targetId) return;
                    setChatSessions(current => current.map(item => item.id === targetId && trimmed !== item.title ? { ...item, title: trimmed } : item));
                  };
                  const cancelRename = () => {
                    setEditingSessionId(null);
                    setEditValue('');
                  };
                  const startRename = (e) => {
                    e.stopPropagation();
                    setEditValue(chat.title);
                    setEditingSessionId(chat.id);
                  };
                  const handleDeleteClick = (e) => {
                    e.stopPropagation();
                    if (isPendingDelete) {
                      // Mark the row for fade-out, then drop it from state
                      // once the CSS transition has played.
                      setPendingRemoveIds(ids => ids.includes(chat.id) ? ids : [...ids, chat.id]);
                      setTimeout(() => {
                        setChatSessions(current => current.filter(item => item.id !== chat.id));
                        setPendingRemoveIds(ids => ids.filter(id => id !== chat.id));
                        if (activeChatId === chat.id) {
                          setActiveChatId(null);
                          setMessages([]);
                        }
                        setPendingDeleteId(null);
                      }, 350);
                    } else {
                      setPendingDeleteId(chat.id);
                    }
                  };
                  return <div key={chat.id} className={`nav-btn chat-session-row ${activeChatId === chat.id ? 'active' : ''} ${isEditing ? 'editing' : ''} ${isPendingDelete ? 'pending-delete' : ''} ${pendingRemoveIds.includes(chat.id) ? 'fading-out' : ''}`} style={{ paddingLeft: '2.1rem', fontSize: '0.8rem' }}>
                    <button type="button" className="chat-session-main" onClick={() => {
                      // Backend model_ids include a load timestamp (mdl-{ts}-{port}),
                      // so a re-loaded model has a fresh id even when the file
                      // is the same. Resolve chat.modelId against the current
                      // daemon registry (by id, then by stable modelPath) so
                      // the UI shows the actually-loaded model instead of "No
                      // Model Loaded" for studios created under a previous
                      // daemon instance. Falls back to the stored id so
                      // messages still route through the daemon's port-50052
                      // fallback when nothing matches.
                      const liveMatch = loadedModels.find(loaded => loaded.model_id === chat.modelId)
                        ?? (chat.modelPath ? loadedModels.find(loaded => loaded.model_path === chat.modelPath) : null)
                        ?? null;
                      setSelectedModelId(liveMatch?.model_id ?? chat.modelId);
                      setActiveChatId(chat.id);
                      setMessages(chat.messages ?? []);
                      openModelStudio(chat);
                    }} title={chat.title}>
                      <MessageSquare size={14} />
                      {isEditing ? (
                        <input
                          type="text"
                          className="chat-session-edit-input"
                          value={editValue}
                          onChange={e => setEditValue(e.target.value)}
                          onClick={e => e.stopPropagation()}
                          onKeyDown={e => {
                            if (e.key === 'Enter') { e.preventDefault(); commitRename(); }
                            else if (e.key === 'Escape') { e.preventDefault(); cancelRename(); }
                          }}
                          onBlur={commitRename}
                          autoFocus
                          onFocus={e => e.target.select()}
                          aria-label="Rename studio"
                        />
                      ) : (
                        <span className="chat-session-title">{displayedTitles[chat.id] ?? chat.title}</span>
                      )}
                    </button>
                    <div className="chat-session-actions">
                      <button type="button" className="chat-session-action" onClick={startRename} title="Rename studio" aria-label="Rename studio">
                        <Pencil size={13} />
                      </button>
                      <button type="button" className={`chat-session-action delete ${isPendingDelete ? 'confirm' : ''}`} onClick={handleDeleteClick} title={isPendingDelete ? 'Click again to confirm delete' : 'Delete studio'} aria-label="Delete studio">
                        {isPendingDelete ? <span className="confirm-label">Delete?</span> : <X size={13} />}
                      </button>
                    </div>
                  </div>;
                })}
              </div>;
            })}

          {false && (<>
          {/* Legacy model controls: management now lives in the Models page. */}
          <div className="nav-section" style={{ marginTop: '1rem' }}>MODEL CONTROLS</div>

          <div className="sidebar-section">
            <label className="section-label">Model Presets</label>
            <select
              className="control-select"
              value={modelPath}
              onChange={e => {
                setModelPath(e.target.value);
                setConfigTarget(current => ({ ...current, model_path: e.target.value }));
              }}
            >
              <option value="C:\Users\adem2\.lmstudio\models\unsloth\Qwen3.5-0.8B-GGUF\Qwen3.5-0.8B-Q8_0.gguf">
                Qwen 3.5 0.8B (Q8_0)
              </option>
              <option value="C:\Users\adem2\.lmstudio\models\deepseek-ai\DeepSeek-R1-Distill-Qwen-1.5B-GGUF\DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf">
                DeepSeek R1 1.5B (Q4_K_M)
              </option>
            </select>
            <input
              type="text"
              className="control-input"
              value={modelPath}
              onChange={e => {
                setModelPath(e.target.value);
                setConfigTarget(current => ({ ...current, model_path: e.target.value }));
              }}
              placeholder="C:\path\to\model.gguf"
              style={{ marginTop: '0.4rem' }}
            />
          </div>

          <div className="sidebar-section">
            <div className="slider-header">
              <label className="section-label">GPU Layers</label>
              <span className="badge-value">{gpuLayers} / 99</span>
            </div>
            <input
              type="range" min="0" max="99"
              value={gpuLayers}
              onChange={e => {
                const value = parseInt(e.target.value, 10);
                setGpuLayers(value);
                setConfigTarget(current => ({ ...current, gpu_layers: value }));
              }}
              className="control-slider"
            />
            <div className="slider-hint">
              {gpuLayers === 99 ? '🔥 Full GPU' : gpuLayers === 0 ? '💻 CPU Only' : `${gpuLayers} layers → VRAM`}
            </div>
          </div>

          <div className="sidebar-section">
            <label className="section-label">Context Window</label>
            <select
              className="control-select"
              value={contextSize}
              onChange={e => {
                const value = parseInt(e.target.value, 10);
                setContextSize(value);
                setConfigTarget(current => ({ ...current, context_size: value }));
              }}
            >
              <option value={2048}>2048 – Minimal VRAM</option>
              <option value={4096}>4096 – Standard</option>
              <option value={8192}>8192 – Extended</option>
              <option value={16384}>16384 – Deep Reasoning</option>
            </select>
          </div>

          <div className="sidebar-section">
            <div className="slider-header">
              <label className="section-label">Temperature</label>
              <span className="badge-value">{temperature}</span>
            </div>
            <input
              type="range" min="0.0" max="1.5" step="0.05"
              value={temperature}
              onChange={e => {
                const value = parseFloat(e.target.value);
                setTemperature(value);
                setConfigTarget(current => ({ ...current, temperature: value }));
              }}
              className="control-slider"
            />
            <div className="slider-header" style={{ marginTop: '0.6rem' }}>
              <label className="section-label">Top-P</label>
              <span className="badge-value">{topP}</span>
            </div>
            <input
              type="range" min="0.1" max="1.0" step="0.05"
              value={topP}
              onChange={e => {
                const value = parseFloat(e.target.value);
                setTopP(value);
                setConfigTarget(current => ({ ...current, top_p: value }));
              }}
              className="control-slider"
            />
          </div>

          <div className="vram-preview-card">
            <div className="preview-title"><Cpu size={13} /> VRAM Impact</div>
            <div className="preview-grid">
              <div className="preview-item">
                <span>Est. Required:</span>
                <strong>{((modelFitPreview?.total_required_vram_bytes || 1.2e9) / 1e9).toFixed(2)} GB</strong>
              </div>
              <div className="preview-item">
                <span>Fits in GPU:</span>
                <span className="status-yes"><CheckCircle2 size={12} /> YES</span>
              </div>
            </div>
          </div>

          <div className="control-actions">
            <button className="btn-load-model" onClick={handleLoadModel} disabled={isLoadingModel}>
              <Zap size={15} /> {isLoadingModel ? 'Loading...' : modelStatus.is_loaded ? 'Reload' : 'Load to VRAM'}
            </button>
            {modelStatus.is_loaded && (
              <button className="btn-unload-model" onClick={() => handleUnloadModel(selectedModelId)} disabled={unloadingModelId === selectedModelId}>
                <Power size={15} /> {unloadingModelId === selectedModelId ? 'Ejecting...' : 'Eject Model'}
              </button>
            )}
          </div>
          </>)}

        </div>

        {/* Main Tab Panels */}
        <main>
          {/* 1. Chat Studio */}
          {activeTab === 'chat' && (
            <div className="tab-panel chat-studio-panel">
              {/* Model Status Strip */}
              <div className="model-status-strip">
                <div className="status-info">
                  <div className={`status-dot-lg ${modelStatus.is_loaded ? 'online' : 'offline'}`}></div>
                  <div className="status-text">
                    <span className="model-name">
                      {modelStatus.is_loaded
                        ? modelStatus.model_path?.split('\\').pop() || 'Model Loaded'
                        : 'No Model Loaded'}
                    </span>
                    <span className="model-meta">
                      {modelStatus.is_loaded
                        ? `CUDA • ${modelStatus.gpu_layers ?? gpuLayers} GPU layers • ${modelStatus.context_size ?? contextSize} ctx`
                        : 'Use the sidebar to configure and load a model'}
                    </span>
                  </div>
                </div>
                {modelStatus.is_loaded && (
                  <button className="btn-eject-mini" onClick={() => handleUnloadModel(selectedModelId)} disabled={unloadingModelId === selectedModelId}>
                    <Power size={13} /> {unloadingModelId === selectedModelId ? 'Ejecting...' : 'Eject'}
                  </button>
                )}
              </div>

              {/* Full-width chat workspace */}
              <div className="chat-workspace">
                <div className="chat-history">
                  {messages.map(msg => (
                    <div key={msg.id} className={`msg-row ${msg.role}`}>
                      {msg.role === 'assistant' && <div className="avatar ai">AI</div>}
                      <div className="bubble">
                        {msg.role === 'assistant' ? (
                          (() => {
                            const { thinking, answer, hadThinkTag } = parseThinking(msg.content);
                            return (
                              <>
                                {appSettings.showThoughtProcess && <ThinkingBlock thinkText={thinking} hadThinkTag={hadThinkTag} />}
                                <div className="answer-content">
                                  {answer ? (
                                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer}</ReactMarkdown>
                                  ) : msg.loading ? (
                                    <em style={{ opacity: 0.6, fontSize: '0.85rem', color: '#94a3b8' }}>
                                      Thinking…
                                    </em>
                                  ) : msg.truncated ? (
                                    <em style={{ opacity: 0.6, fontSize: '0.85rem', color: '#94a3b8' }}>
                                      [Model reached output token limit during reasoning — try re-prompting]
                                    </em>
                                  ) : null}
                                </div>
                              </>
                            );
                          })()
                        ) : (
                          <div className="answer-content">
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                          </div>
                        )}
                        {msg.attachments?.length > 0 && <div className="chat-attachments">
                          {msg.attachments.map(attachment => <AttachmentPreview key={attachment.dataUrl} attachment={attachment} />)}
                        </div>}
                        {msg.telemetry && (
                          <div className="meta-info">
                            <span className="tag-speed"><Zap size={11} /> {msg.telemetry}</span>
                          </div>
                        )}
                      </div>
                      {msg.role === 'user' && <div className="avatar user">YOU</div>}
                    </div>
                  ))}
                  <div ref={messagesEndRef} />
                </div>

                <div className="input-card">
                  {attachments.length > 0 && <div className="composer-attachments">
                    {attachments.map(attachment => <div key={attachment.dataUrl} className="composer-attachment">
                      <AttachmentPreview attachment={attachment} />
                      <button onClick={() => setAttachments(current => current.filter(item => item.dataUrl !== attachment.dataUrl))} title={`Remove ${attachment.name}`}><X size={13} /></button>
                    </div>)}
                  </div>}
                  <div className="prompt-area">
                    <input ref={attachmentInputRef} type="file" accept="image/*" multiple hidden onChange={event => { addAttachments(event.target.files); event.target.value = ''; }} />
                    <button className="btn-attach" onClick={() => attachmentInputRef.current?.click()} disabled={!acceptsImageInput} title={acceptsImageInput ? 'Attach image' : 'The selected model does not have an available image-input backend'}>
                      <Paperclip size={18} />
                    </button>
                    <textarea
                      value={inputPrompt}
                      onChange={e => setInputPrompt(e.target.value)}
                      onKeyDown={e => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault();
                          handleSendMessage();
                        }
                      }}
                      placeholder={acceptsImageInput ? 'Ask about an image or type a message...' : modelStatus.is_loaded ? 'Ask AntiOOM AI anything...' : 'Load a model to start chatting...'}
                    />
                    <button className="btn-send" onClick={handleSendMessage} disabled={isGenerating}>
                      <Send size={18} />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'models' && (
            <div className="tab-panel">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                <h2 style={{ fontSize: '1.4rem' }}>Models</h2>
                <button className="btn-load-model" onClick={() => setShowModelPicker(true)}>
                  <PackagePlus size={14} /> Add Model
                </button>
              </div>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
                Load multiple local GGUF models and choose which one powers the chat.
              </p>
              <div className="grid-2">
                <div className="card">
                  <h3 style={{ marginBottom: '1rem' }}>My Models</h3>
                  {modelCards.map(card => {
                    const loadedEntry = loadedModels.find(loaded => loaded.model_id === card.loadedModelId);
                    const isLoaded = Boolean(loadedEntry);
                    return (
                      <div key={card.id} style={{ borderBottom: '1px solid var(--border-color)', padding: '0.85rem 0' }}>
                        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
                          <strong>{card.name}</strong>
                          <button
                            className="btn-remove-card"
                            title="Remove model card"
                            onClick={async () => {
                              if (isLoaded) await handleUnloadModel(card.loadedModelId);
                              setModelCards(current => current.filter(item => item.id !== card.id));
                            }}
                          >
                            <X size={13} />
                          </button>
                        </div>
                        <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', margin: '0.25rem 0 0.65rem' }}>{card.modality?.toUpperCase()} · {(card.size_bytes / 1e9).toFixed(2)} GB</div>
                        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', alignItems: 'center', marginTop: '0.25rem' }}>
                          <button className="btn-load-model" onClick={() => openConfigPanel({ name: card.name, path: card.modelPath, max_context_size: card.max_context_size, mmproj_path: card.mmproj_path })}>
                            <Settings size={14} /> {isLoaded ? 'Settings' : 'Configure'}
                          </button>
                          {isLoaded ? <button className="btn-unload-model" onClick={async () => {
                            await handleUnloadModel(card.loadedModelId);
                            setModelCards(current => current.map(item => item.id === card.id ? { ...item, loadedModelId: null } : item));
                          }} disabled={unloadingModelId === card.loadedModelId}><Power size={14} /> Eject</button>
                          : <button className="btn-load-model" onClick={async () => {
                            const configuration = { model_path: card.modelPath, gpu_layers: 99, context_size: Math.min(4096, card.max_context_size ?? 4096), mmproj_path: card.mmproj_path || detectedModels.find(m => m.path === card.modelPath)?.mmproj_path || undefined };
                            const modelId = await handleLoadModel(configuration);
                            if (modelId) setModelCards(current => current.map(item => item.id === card.id ? { ...item, loadedModelId: modelId } : item));
                          }} disabled={isLoadingModel}><Zap size={14} /> Load</button>}
                          {isLoaded && <button className="btn-load-model" onClick={() => startStudio(loadedEntry, card)} title="Start studio"><Play size={14} /></button>}
                        </div>
                      </div>
                    );
                  })}
                  {modelCards.length === 0 && <p style={{ color: 'var(--text-secondary)' }}>No model cards yet. Click "Add Model" to add one from the catalog.</p>}
                </div>
              </div>
            </div>
          )}

          {showModelPicker && (
            <div className="modal-backdrop" onClick={() => setShowModelPicker(false)}>
              <div className="modal-container" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                  <h3>Add a model</h3>
                  <button className="config-sidebar-close" onClick={() => setShowModelPicker(false)} title="Close">
                    <X size={16} />
                  </button>
                </div>
                <div className="modal-body">
                  <input
                    type="text"
                    className="control-input"
                    placeholder="Search models..."
                    value={modelPickerSearch}
                    onChange={e => setModelPickerSearch(e.target.value)}
                    autoFocus
                  />
                  <div className="modal-list">
                    {detectedModels
                      .filter(model => model.name.toLowerCase().includes(modelPickerSearch.toLowerCase()))
                      .map(model => (
                        <button
                          key={model.path}
                          className="modal-list-item"
                          onClick={() => {
                            setModelCards(current => [...current, {
                              id: crypto.randomUUID(),
                              modelPath: model.path,
                              name: model.name,
                              modality: model.modality,
                              size_bytes: model.size_bytes,
                              max_context_size: model.max_context_size ?? null,
                              image_input_available: model.image_input_available === true,
                              mmproj_path: model.mmproj_path || null,
                            }]);
                            setShowModelPicker(false);
                            setModelPickerSearch('');
                          }}
                        >
                          <strong>{model.name}</strong>
                          <span className="modal-list-item-meta">{model.modality?.toUpperCase()} · {(model.size_bytes / 1e9).toFixed(2)} GB</span>
                        </button>
                      ))}
                    {detectedModels.length === 0 && <p style={{ color: 'var(--text-secondary)' }}>No GGUF models found in AIATM's models folder.</p>}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'settings' && (
            <div className="tab-panel">
              <h2 style={{ fontSize: '1.4rem', marginBottom: '0.3rem' }}>General Settings</h2>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
                Preferences for this local AIATM desktop app. Changes are saved on this device.
              </p>
              <div className="card" style={{ maxWidth: '680px' }}>
                <div className="slider-header" style={{ padding: '0.8rem 0', borderBottom: '1px solid var(--border-color)' }}>
                  <div><strong>Auto-select newly loaded models</strong><div className="slider-hint">Use the latest loaded model for chat automatically.</div></div>
                  <input type="checkbox" checked={appSettings.autoSelectNewest} onChange={e => setAppSettings(current => ({ ...current, autoSelectNewest: e.target.checked }))} />
                </div>
                <div style={{ padding: '1rem 0', borderBottom: '1px solid var(--border-color)' }}>
                  <div className="slider-header"><div><strong>Model status refresh</strong><div className="slider-hint">How often AIATM checks loaded models.</div></div><span className="badge-value">{appSettings.refreshSeconds}s</span></div>
                  <input className="control-slider" type="range" min="1" max="10" step="1" value={appSettings.refreshSeconds} onChange={e => setAppSettings(current => ({ ...current, refreshSeconds: Number(e.target.value) }))} />
                </div>
                <div className="slider-header" style={{ paddingTop: '1rem' }}>
                  <div><strong>Show thought process panels</strong><div className="slider-hint">Display model reasoning when it is supplied in a response.</div></div>
                  <input type="checkbox" checked={appSettings.showThoughtProcess} onChange={e => setAppSettings(current => ({ ...current, showThoughtProcess: e.target.checked }))} />
                </div>
              </div>
            </div>
          )}

          {/* 2. Image Studio */}
          {activeTab === 'image' && (
            <div className="tab-panel">
              <h2 style={{ fontSize: '1.4rem', marginBottom: '0.3rem' }}>Stable Diffusion Image Studio</h2>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
                Local text-to-image synthesis using stable-diffusion.cpp
              </p>

              <div className="grid-2">
                <div className="card">
                  <div className="form-group">
                    <label>Prompt</label>
                    <input
                      type="text"
                      value={imgPrompt}
                      onChange={e => setImgPrompt(e.target.value)}
                    />
                  </div>

                  <div className="form-group">
                    <label>Model Path</label>
                    <input type="text" value="models/sd.safetensors" readOnly />
                  </div>

                  <button className="btn-primary" onClick={handleGenerateImage} disabled={isGeneratingImg}>
                    <Sparkles size={16} /> {isGeneratingImg ? 'Rendering...' : 'Generate Image'}
                  </button>
                </div>

                <div className="card">
                  <label>Generated Canvas Output</label>
                  <div className="preview-box">
                    {imgSrc ? (
                      <img src={imgSrc} alt="SD Output" />
                    ) : (
                      <>
                        <Image size={40} />
                        <span>Rendered output image will display here</span>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* 3. Voice Studio */}
          {activeTab === 'audio' && (
            <div className="tab-panel">
              <h2 style={{ fontSize: '1.4rem', marginBottom: '0.3rem' }}>Voice & Audio Studio</h2>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
                Kokoro Text-to-Speech & Whisper Speech-to-Text ASR
              </p>

              <div className="grid-2">
                <div className="card">
                  <h3>Text-to-Speech Synthesizer (Kokoro)</h3>
                  <br />
                  <div className="form-group">
                    <label>Input Text</label>
                    <input
                      type="text"
                      value={ttsInput}
                      onChange={e => setTtsInput(e.target.value)}
                    />
                  </div>

                  <button className="btn-primary" onClick={handleSynthesizeSpeech} disabled={isGeneratingTts}>
                    <Play size={16} /> {isGeneratingTts ? 'Synthesizing...' : 'Synthesize Speech'}
                  </button>

                  {audioSrc && <audio controls autoPlay src={audioSrc} style={{ width: '100%', marginTop: '1rem' }} />}
                </div>

                <div className="card">
                  <h3>Speech-to-Text ASR (Whisper)</h3>
                  <br />
                  <div className="preview-box" style={{ height: '180px' }}>
                    <FileAudio size={40} />
                    <span>Drop audio file for transcription</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* 4. Video Studio */}
          {activeTab === 'video' && (
            <div className="tab-panel">
              <h2 style={{ fontSize: '1.4rem', marginBottom: '0.3rem' }}>Wan Video Runner Studio</h2>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
                Text-to-Video generation engine
              </p>

              <div className="card" style={{ maxWidth: '650px' }}>
                <div className="form-group">
                  <label>Video Prompt</label>
                  <input type="text" defaultValue="A serene waterfall in a mystical forest, cinematic motion" />
                </div>
                <button className="btn-primary">
                  <Video size={16} /> Render Video
                </button>
                <div className="preview-box" style={{ marginTop: '1rem' }}>
                  <Play size={45} />
                  <span>Video player output preview</span>
                </div>
              </div>
            </div>
          )}

        </main>
      </div>

      {/* Configure Model Sidebar (left overlay) */}
      {configTarget && (
        <>
          <div className="config-sidebar-backdrop" onClick={() => setConfigTarget(null)} />
          <aside className="config-sidebar">
            <div className="config-sidebar-header">
              <button className="config-sidebar-close" onClick={() => setConfigTarget(null)} title="Close">
                <X size={16} />
              </button>
              <div className="config-sidebar-title">
                <Settings size={16} /> {configTarget.name || 'Model'} settings
              </div>
            </div>
            <div className="config-sidebar-body">
              <div className="sidebar-section">
                <label className="section-label">GGUF model path</label>
                <input className="control-input" value={configTarget?.model_path || ''} onChange={e => setConfigTarget(current => ({ ...current, model_path: e.target.value }))} />
              </div>
              <div className="sidebar-section">
                <label className="section-label">Vision projector (mmproj)</label>
                <div className="mmproj-row">
                  <input className="control-input" value={configTarget?.mmproj_path || ''} onChange={e => setConfigTarget(current => ({ ...current, mmproj_path: e.target.value }))} />
                  <button className="btn-browse" onClick={() => mmprojInputRef.current?.click()}>Browse</button>
                  <input
                    ref={mmprojInputRef}
                    type="file"
                    accept=".gguf"
                    style={{ display: 'none' }}
                    onChange={e => {
                      const file = e.target.files[0];
                      if (file) setConfigTarget(current => ({ ...current, mmproj_path: file.path || '' }));
                      e.target.value = '';
                    }}
                  />
                </div>
                <div className="slider-hint">Auto-detected from model directory. Leave empty if not a vision model.</div>
              </div>
              <div className="sidebar-section">
                <div className="slider-header"><label className="section-label">GPU layers</label><span className="badge-value">{configTarget?.gpu_layers ?? 99}</span></div>
                <input className="control-slider" type="range" min="0" max="99" value={configTarget?.gpu_layers ?? 99} onChange={e => setConfigTarget(current => ({ ...current, gpu_layers: Number(e.target.value) }))} />
              </div>
              <div className="sidebar-section">
                <div className="slider-header"><label className="section-label">Context window</label><span className="badge-value">{(configTarget?.context_size ?? 4096).toLocaleString()} / {configTarget?.max_context_size ? configTarget.max_context_size.toLocaleString() : '?'}</span></div>
                {configTarget?.max_context_size ? <>
                  <input className="control-slider" type="range" min="512" max={configTarget.max_context_size} step="512" value={Math.min(configTarget.context_size ?? 4096, configTarget.max_context_size)} onChange={e => setConfigTarget(current => ({ ...current, context_size: Number(e.target.value) }))} />
                  <div className="slider-hint">Maximum read from this GGUF model’s metadata.</div>
                </> : <div className="slider-hint">Maximum context is not present in this model’s metadata.</div>}
              </div>
              <div className="sidebar-section">
                <div className="slider-header"><label className="section-label">Temperature</label><span className="badge-value">{configTarget?.temperature ?? 0.7}</span></div>
                <input className="control-slider" type="range" min="0" max="1.5" step="0.05" value={configTarget?.temperature ?? 0.7} onChange={e => setConfigTarget(current => ({ ...current, temperature: Number(e.target.value) }))} />
              </div>
              <div className="sidebar-section">
                <div className="slider-header"><label className="section-label">Top-P</label><span className="badge-value">{configTarget?.top_p ?? 0.9}</span></div>
                <input className="control-slider" type="range" min="0.1" max="1" step="0.05" value={configTarget?.top_p ?? 0.9} onChange={e => setConfigTarget(current => ({ ...current, top_p: Number(e.target.value) }))} />
              </div>
              <div className="vram-preview-card">
                <div className="preview-title"><Cpu size={13} /> Estimated VRAM</div>
                <strong>{((modelFitPreview?.total_required_vram_bytes || 0) / 1e9).toFixed(2)} GB</strong>
              </div>
              <button className="btn-load-model config-sidebar-load" onClick={() => handleLoadModel({ model_path: configTarget?.model_path, gpu_layers: configTarget?.gpu_layers ?? 99, context_size: configTarget?.context_size ?? 4096, mmproj_path: configTarget?.mmproj_path || undefined })} disabled={isLoadingModel}>
                <Zap size={15} /> {isLoadingModel ? 'Loading...' : 'Load configured model'}
              </button>
              <div className="config-sidebar-section-title">Loaded models ({loadedModels.length})</div>
              {loadedModels.length === 0 ? <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>No models are loaded.</p> : loadedModels.map(model => (
                <div key={model.model_id} className="config-sidebar-loaded-row">
                  <strong>{model.model_path?.split('\\').pop() || model.model_id}</strong>
                  <div className="config-sidebar-loaded-meta">
                    {model.gpu_layers ?? 0} GPU layers · {model.context_size ?? 0} context
                  </div>
                  <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
                    <button className="btn-load-model" onClick={() => setSelectedModelId(model.model_id)} disabled={selectedModelId === model.model_id}>
                      {selectedModelId === model.model_id ? 'Selected for chat' : 'Use for chat'}
                    </button>
                    <button className="btn-unload-model" onClick={() => handleUnloadModel(model.model_id)} disabled={unloadingModelId === model.model_id}>
                      <Power size={14} /> Eject
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </aside>
        </>
      )}
    </div>
  );
}
