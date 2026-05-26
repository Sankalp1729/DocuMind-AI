import { useState, useEffect, useRef, useCallback } from "react";
import {
  MessageSquare,
  FileText,
  BarChart3,
  Settings,
  ChevronLeft,
  Share2,
  Paperclip,
  Mic,
  Send,
  Check,
  Circle,
  Clock,
  Sparkles,
  Zap,
  Target,
  TrendingUp,
  Activity,
  FileCheck,
  AlertCircle,
  Plus,
  X,
  ChevronRight,
  Cpu,
  Layers,
  Filter,
  ArrowUpRight,
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";

type Tab = "citations" | "trace" | "analytics";
type AIState = "idle" | "thinking" | "streaming" | "complete";
type ProcessingStage = "embedding" | "retrieval" | "reranking" | "generating";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  streaming?: boolean;
  alignmentScore?: number;
}

interface Citation {
  id: number;
  document: string;
  page: number;
  snippet: string;
  similarity: number;
  rerankerScore: number;
  docColor: string;
}

interface Document {
  id: string;
  name: string;
  color: string;
  pages: number;
  active: boolean;
}

interface StageState {
  stage: ProcessingStage;
  status: "pending" | "active" | "complete";
  latency?: number;
}

const documents: Document[] = [
  { id: "1", name: "Q4 Financial.pdf", color: "#5B7FFF", pages: 48, active: true },
  { id: "2", name: "Roadmap 2025.pdf", color: "#a78bfa", pages: 32, active: true },
  { id: "3", name: "Eng Spec v3.pdf", color: "#34d399", pages: 67, active: true },
  { id: "4", name: "Legal Comp.pdf", color: "#fbbf24", pages: 24, active: false },
];

const recentChats = [
  { id: "1", title: "Revenue Q4 analysis", time: "2h ago" },
  { id: "2", title: "Compliance checklist", time: "Yesterday" },
  { id: "3", title: "Roadmap priorities H1", time: "2d ago" },
];

const mockCitations: Citation[] = [
  {
    id: 1,
    document: "Q4 Financial.pdf",
    page: 12,
    snippet:
      "Revenue increased by 34% YoY, reaching $42.3M in Q4 2024, driven primarily by enterprise segment growth and successful product launches in the EMEA market.",
    similarity: 0.92,
    rerankerScore: 0.90,
    docColor: "#5B7FFF",
  },
  {
    id: 2,
    document: "Roadmap 2025.pdf",
    page: 5,
    snippet:
      "Strategic initiatives for H1 include expanding AI capabilities, enhancing platform security, and launching multi-region support across APAC and LATAM.",
    similarity: 0.87,
    rerankerScore: 0.75,
    docColor: "#a78bfa",
  },
  {
    id: 3,
    document: "Eng Spec v3.pdf",
    page: 23,
    snippet:
      "The vector database architecture supports hybrid search with BM25 and dense embeddings, enabling semantic retrieval at scale with sub-200ms p95 latency.",
    similarity: 0.84,
    rerankerScore: 0.82,
    docColor: "#34d399",
  },
];

const traceSteps = [
  { label: "Query embedding", latency: 45, status: "complete" as const, detail: "text-embedding-3-large" },
  { label: "Vector search", latency: 120, status: "complete" as const, detail: "HNSW k=24, cosine" },
  { label: "BM25 search", latency: 85, status: "complete" as const, detail: "sparse retrieval, BM25+" },
  { label: "Score fusion", latency: 12, status: "complete" as const, detail: "RRF α=0.6" },
  { label: "Reranker", latency: 230, status: "complete" as const, detail: "cross-encoder top-12" },
  { label: "Context assembly", latency: 18, status: "complete" as const, detail: "3,847 tokens" },
  { label: "LLM generation", latency: 1840, status: "complete" as const, detail: "gpt-4o, T=0.3" },
];

const FULL_RESPONSE =
  "Based on the uploaded documents, Q4 revenue increased by 34% YoY to $42.3M [①], driven primarily by enterprise segment growth across EMEA. This directly aligns with the 2025 roadmap [②], which prioritizes expanding AI capabilities and launching multi-region support. The technical foundation already in place [③] — hybrid search with sub-200ms p95 latency — provides the infrastructure needed to support these strategic initiatives at scale.";

const STAGES: StageState[] = [
  { stage: "embedding", status: "pending" },
  { stage: "retrieval", status: "pending" },
  { stage: "reranking", status: "pending" },
  { stage: "generating", status: "pending" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("citations");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [aiState, setAIState] = useState<AIState>("complete");
  const [stages, setStages] = useState<StageState[]>(STAGES.map((s) => ({ ...s, status: "complete" as const })));
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "user",
      content: "What were the key revenue drivers in Q4 and how do they align with our 2025 roadmap?",
    },
    {
      id: "2",
      role: "assistant",
      content: FULL_RESPONSE,
      citations: mockCitations,
      alignmentScore: 0.88,
    },
  ]);
  const [streamingText, setStreamingText] = useState("");
  const [stageTimes, setStageTimes] = useState<Record<string, number>>({
    embedding: 45,
    retrieval: 205,
    reranking: 247,
    generating: 2087,
  });
  const chatRef = useRef<HTMLDivElement>(null);
  const streamTimers = useRef<ReturnType<typeof setTimeout>[]>([]);

  const clearTimers = () => {
    streamTimers.current.forEach(clearTimeout);
    streamTimers.current = [];
  };

  const runStreamingDemo = useCallback((userMsg: string) => {
    clearTimers();

    const newUserMsgId = Date.now().toString();
    const newAIMsgId = (Date.now() + 1).toString();

    setMessages((prev) => [
      ...prev,
      { id: newUserMsgId, role: "user", content: userMsg },
    ]);
    setStreamingText("");
    setAIState("thinking");
    setStages(STAGES.map((s) => ({ ...s, status: "pending" })));
    setActiveTab("trace");

    const schedule = (fn: () => void, delay: number) => {
      const t = setTimeout(fn, delay);
      streamTimers.current.push(t);
    };

    // Stage 1: Embedding
    schedule(() => {
      setStages((s) =>
        s.map((st) => (st.stage === "embedding" ? { ...st, status: "active" } : st))
      );
    }, 300);

    schedule(() => {
      setStageTimes((t) => ({ ...t, embedding: 38 + Math.floor(Math.random() * 20) }));
      setStages((s) =>
        s.map((st) => (st.stage === "embedding" ? { ...st, status: "complete" } : st))
      );
    }, 700);

    // Stage 2: Retrieval
    schedule(() => {
      setStages((s) =>
        s.map((st) => (st.stage === "retrieval" ? { ...st, status: "active" } : st))
      );
    }, 800);

    schedule(() => {
      setStageTimes((t) => ({ ...t, retrieval: 180 + Math.floor(Math.random() * 60) }));
      setStages((s) =>
        s.map((st) => (st.stage === "retrieval" ? { ...st, status: "complete" } : st))
      );
    }, 1400);

    // Stage 3: Reranking
    schedule(() => {
      setStages((s) =>
        s.map((st) => (st.stage === "reranking" ? { ...st, status: "active" } : st))
      );
    }, 1500);

    schedule(() => {
      setStageTimes((t) => ({ ...t, reranking: 200 + Math.floor(Math.random() * 80) }));
      setStages((s) =>
        s.map((st) => (st.stage === "reranking" ? { ...st, status: "complete" } : st))
      );
    }, 2000);

    // Stage 4: Generating — start streaming
    schedule(() => {
      setStages((s) =>
        s.map((st) => (st.stage === "generating" ? { ...st, status: "active" } : st))
      );
      setAIState("streaming");
      setMessages((prev) => [
        ...prev,
        {
          id: newAIMsgId,
          role: "assistant",
          content: "",
          streaming: true,
        },
      ]);
    }, 2100);

    // Token streaming
    const tokens = FULL_RESPONSE.split(" ");
    tokens.forEach((token, i) => {
      schedule(() => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === newAIMsgId
              ? { ...m, content: tokens.slice(0, i + 1).join(" ") }
              : m
          )
        );
      }, 2100 + i * 38);
    });

    const endTime = 2100 + tokens.length * 38;

    schedule(() => {
      setStageTimes((t) => ({ ...t, generating: endTime - 2100 }));
      setStages((s) =>
        s.map((st) => (st.stage === "generating" ? { ...st, status: "complete" } : st))
      );
      setAIState("complete");
      setMessages((prev) =>
        prev.map((m) =>
          m.id === newAIMsgId
            ? { ...m, streaming: false, citations: mockCitations, alignmentScore: 0.88 }
            : m
        )
      );
      setActiveTab("citations");
    }, endTime + 200);
  }, []);

  const handleSend = () => {
    const text = inputValue.trim();
    if (!text || aiState === "thinking" || aiState === "streaming") return;
    setInputValue("");
    runStreamingDemo(text);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      handleSend();
    }
  };

  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div className="h-screen w-screen flex overflow-hidden bg-background dark select-none">
      {/* LEFT SIDEBAR */}
      <motion.aside
        initial={false}
        animate={{ width: sidebarCollapsed ? 0 : 220 }}
        transition={{ duration: 0.2, ease: "easeOut" }}
        className="bg-sidebar border-r border-sidebar-border flex-shrink-0 overflow-hidden relative z-10"
        style={{ willChange: "width" }}
      >
        <div className="w-[220px] h-full flex flex-col">
          {/* Logo */}
          <div className="px-4 py-4 border-b border-sidebar-border flex items-center justify-between flex-shrink-0">
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-[7px] bg-gradient-to-br from-[#5B7FFF] to-[#a78bfa] flex items-center justify-center">
                <Sparkles className="w-3.5 h-3.5 text-white" />
              </div>
              <span className="text-sm font-semibold tracking-tight text-sidebar-foreground font-['Syne']">
                DocuMind
              </span>
            </div>
            <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-[#5B7FFF]/15 text-[#5B7FFF] border border-[#5B7FFF]/25 tracking-wider">
              PRO
            </span>
          </div>

          {/* Nav */}
          <div className="px-3 py-3 border-b border-sidebar-border flex-shrink-0">
            <div className="text-[9px] font-semibold text-[#454d66] tracking-[0.1em] uppercase mb-2 px-2">
              Workspace
            </div>
            <nav className="space-y-0.5">
              {[
                { icon: <MessageSquare className="w-3.5 h-3.5" />, label: "AI Chat", active: true },
                { icon: <FileText className="w-3.5 h-3.5" />, label: "Documents", badge: "7" },
                { icon: <BarChart3 className="w-3.5 h-3.5" />, label: "Analytics" },
              ].map((item) => (
                <button
                  key={item.label}
                  className={`w-full flex items-center gap-2.5 px-2.5 py-1.5 rounded-lg text-xs transition-colors duration-150 ${
                    item.active
                      ? "bg-sidebar-accent text-sidebar-accent-foreground"
                      : "text-[#7c8299] hover:bg-sidebar-accent/60 hover:text-sidebar-foreground"
                  }`}
                >
                  <span className={item.active ? "text-[#5B7FFF]" : ""}>{item.icon}</span>
                  <span className="font-medium font-['Syne']">{item.label}</span>
                  {item.badge && (
                    <span className="ml-auto text-[9px] font-mono bg-[#1e2235] text-[#7c8299] px-1.5 py-0.5 rounded">
                      {item.badge}
                    </span>
                  )}
                </button>
              ))}
            </nav>
          </div>

          {/* Knowledge Base */}
          <div className="px-3 py-3 border-b border-sidebar-border flex-1 overflow-y-auto scrollbar-hide">
            <div className="flex items-center justify-between px-2 mb-2">
              <div className="text-[9px] font-semibold text-[#454d66] tracking-[0.1em] uppercase">
                Knowledge Base
              </div>
              <button className="w-4 h-4 rounded flex items-center justify-center text-[#454d66] hover:text-[#7c8299] hover:bg-sidebar-accent/50 transition-colors duration-150">
                <Plus className="w-3 h-3" />
              </button>
            </div>
            <div className="space-y-0.5">
              {documents.map((doc) => (
                <button
                  key={doc.id}
                  className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-[#7c8299] text-[11px] hover:bg-sidebar-accent/50 hover:text-sidebar-foreground transition-colors duration-150 group"
                >
                  <Circle
                    className="w-1.5 h-1.5 flex-shrink-0"
                    fill={doc.active ? doc.color : "#454d66"}
                    color={doc.active ? doc.color : "#454d66"}
                  />
                  <span className="truncate font-medium">{doc.name}</span>
                  <span className="ml-auto text-[9px] font-mono opacity-0 group-hover:opacity-60 transition-opacity duration-150">
                    {doc.pages}p
                  </span>
                </button>
              ))}
            </div>

            <div className="flex items-center justify-between px-2 mb-2 mt-4">
              <div className="text-[9px] font-semibold text-[#454d66] tracking-[0.1em] uppercase">
                Recent Chats
              </div>
            </div>
            <div className="space-y-0.5">
              {recentChats.map((chat, idx) => (
                <button
                  key={chat.id}
                  className={`w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-[11px] transition-colors duration-150 group ${
                    idx === 0
                      ? "bg-sidebar-accent/60 text-sidebar-foreground"
                      : "text-[#7c8299] hover:bg-sidebar-accent/50 hover:text-sidebar-foreground"
                  }`}
                >
                  <Clock className="w-3 h-3 flex-shrink-0" />
                  <span className="truncate font-medium">{chat.title}</span>
                  <span className="ml-auto text-[9px] font-mono opacity-50 group-hover:opacity-70 transition-opacity duration-150 flex-shrink-0">
                    {chat.time}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* User */}
          <div className="px-3 py-3 border-t border-sidebar-border space-y-1 flex-shrink-0">
            <button className="w-full flex items-center gap-2.5 px-2.5 py-1.5 rounded-lg text-[#7c8299] text-[11px] hover:bg-sidebar-accent/50 hover:text-sidebar-foreground transition-colors duration-150">
              <Settings className="w-3.5 h-3.5" />
              <span className="font-medium">Settings</span>
            </button>
            <div className="flex items-center gap-2.5 px-2.5 py-1.5">
              <div className="w-6 h-6 rounded-full bg-gradient-to-br from-[#34d399] to-[#5B7FFF] flex items-center justify-center text-[10px] font-bold text-white flex-shrink-0">
                AK
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-[11px] font-semibold text-sidebar-foreground">Aryan Kumar</div>
                <div className="text-[9px] text-[#454d66] font-mono">Pro · 12.4K tokens</div>
              </div>
            </div>
          </div>
        </div>
      </motion.aside>

      {/* CENTER PANEL */}
      <main className="flex-1 flex flex-col overflow-hidden min-w-0">
        {/* Top Bar */}
        <header className="h-12 border-b border-border flex items-center justify-between px-4 flex-shrink-0 bg-background/80 backdrop-blur-sm">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              aria-label="Toggle sidebar"
              className="w-7 h-7 rounded-lg hover:bg-secondary/60 flex items-center justify-center transition-colors duration-150"
            >
              <ChevronLeft
                className={`w-3.5 h-3.5 text-muted-foreground transition-transform duration-200 ${
                  sidebarCollapsed ? "rotate-180" : ""
                }`}
              />
            </button>
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold text-foreground font-['Syne']">
                Q4 Strategy Discussion
              </span>
              <AnimatePresence>
                {aiState === "streaming" && (
                  <motion.span
                    initial={{ opacity: 0, scale: 0.85 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.85 }}
                    transition={{ duration: 0.15 }}
                    className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-[#34d399]/10 border border-[#34d399]/20 text-[10px] font-mono font-semibold text-[#34d399]"
                  >
                    <motion.span
                      animate={{ opacity: [1, 0.3, 1] }}
                      transition={{ duration: 0.8, repeat: Infinity }}
                      className="w-1 h-1 rounded-full bg-[#34d399] block"
                    />
                    Streaming
                  </motion.span>
                )}
              </AnimatePresence>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1.5">
              <span className="px-2 py-1 rounded-md bg-secondary text-[10px] font-mono text-[#7c8299]">
                gpt-4o
              </span>
              <span className="px-2 py-1 rounded-md bg-[#a78bfa]/10 border border-[#a78bfa]/15 text-[10px] font-mono text-[#a78bfa]">
                Hybrid k=12
              </span>
            </div>
            <button
              aria-label="Share"
              className="w-7 h-7 rounded-lg hover:bg-secondary/60 flex items-center justify-center transition-colors duration-150"
            >
              <Share2 className="w-3.5 h-3.5 text-muted-foreground" />
            </button>
          </div>
        </header>

        {/* Stage Pipeline */}
        <div className="h-10 border-b border-border flex items-center px-5 gap-3 bg-[#111318] flex-shrink-0">
          {stages.map((s, i) => (
            <div key={s.stage} className="flex items-center gap-3">
              <StageIndicator
                stage={s.stage}
                label={
                  s.stage === "embedding"
                    ? "Embedding"
                    : s.stage === "retrieval"
                    ? "Retrieval"
                    : s.stage === "reranking"
                    ? "Reranking"
                    : "Generating"
                }
                status={s.status}
              />
              {i < stages.length - 1 && (
                <ChevronRight
                  className={`w-3 h-3 flex-shrink-0 transition-colors duration-150 ${
                    s.status === "complete" ? "text-[#34d399]/40" : "text-[#1e2235]"
                  }`}
                />
              )}
            </div>
          ))}

          <div className="ml-auto flex items-center gap-3">
            <AIStateIndicator state={aiState} />
          </div>
        </div>

        {/* Chat */}
        <div
          ref={chatRef}
          className="flex-1 overflow-y-auto px-6 py-5 scrollbar-hide"
          role="log"
          aria-live="polite"
        >
          <div className="max-w-2xl mx-auto space-y-5">
            {messages.map((message) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.15, ease: "easeOut" }}
                className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
              >
                {message.role === "user" ? (
                  <div className="max-w-[80%] px-4 py-2.5 rounded-xl bg-[#5B7FFF]/10 border border-[#5B7FFF]/15 text-sm text-foreground leading-relaxed">
                    {message.content}
                  </div>
                ) : (
                  <div className="max-w-[90%] space-y-2">
                    <div className="px-4 py-3.5 rounded-xl bg-card border border-border text-sm leading-relaxed text-foreground">
                      <CitationInlineText content={message.content} />
                      {message.streaming && (
                        <motion.span
                          animate={{ opacity: [1, 0, 1] }}
                          transition={{ duration: 0.8, repeat: Infinity }}
                          className="inline-block w-[2px] h-[14px] bg-[#5B7FFF] ml-0.5 align-middle"
                        />
                      )}
                    </div>
                    <AnimatePresence>
                      {message.alignmentScore && !message.streaming && (
                        <motion.div
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ duration: 0.3, delay: 0.1 }}
                          className="flex items-center gap-2 px-1"
                        >
                          <span className="text-[9px] font-mono text-[#454d66]">Alignment</span>
                          <div className="h-1 w-16 bg-secondary rounded-full overflow-hidden">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${message.alignmentScore * 100}%` }}
                              transition={{ duration: 0.3, ease: "easeOut" }}
                              className="h-full bg-[#34d399] rounded-full"
                            />
                          </div>
                          <span className="text-[9px] font-mono text-[#34d399]">
                            {message.alignmentScore.toFixed(2)}
                          </span>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        </div>

        {/* Input */}
        <div className="border-t border-border p-4 flex-shrink-0 bg-background">
          <div className="max-w-2xl mx-auto">
            <div
              className={`rounded-xl border bg-card p-3 transition-colors duration-150 ${
                aiState === "streaming" || aiState === "thinking"
                  ? "border-[#a78bfa]/20"
                  : "border-border focus-within:border-[#5B7FFF]/30"
              }`}
            >
              <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  aiState === "thinking"
                    ? "Processing your query..."
                    : aiState === "streaming"
                    ? "Generating response..."
                    : "Ask about your documents..."
                }
                disabled={aiState === "thinking" || aiState === "streaming"}
                className="w-full bg-transparent resize-none outline-none text-sm placeholder:text-[#454d66] min-h-[52px] disabled:opacity-40 transition-opacity duration-150"
                rows={2}
              />
              <div className="flex items-center justify-between mt-2 pt-2 border-t border-border/40">
                <div className="flex items-center gap-1.5">
                  <button
                    aria-label="Attach file"
                    className="w-7 h-7 rounded-lg hover:bg-secondary/60 flex items-center justify-center transition-colors duration-150"
                  >
                    <Paperclip className="w-3.5 h-3.5 text-muted-foreground" />
                  </button>
                  <button
                    aria-label="Voice input"
                    className="w-7 h-7 rounded-lg hover:bg-secondary/60 flex items-center justify-center transition-colors duration-150"
                  >
                    <Mic className="w-3.5 h-3.5 text-muted-foreground" />
                  </button>
                  <div className="w-px h-4 bg-border mx-1" />
                  <div className="flex items-center gap-1">
                    {documents.filter((d) => d.active).map((d) => (
                      <span key={d.id} className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: d.color }} />
                    ))}
                    <span className="text-[9px] font-mono text-[#454d66] ml-1">
                      {documents.filter((d) => d.active).length} docs
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-[9px] text-[#454d66] font-mono hidden sm:block">⌘↵ to send</span>
                  <button
                    onClick={handleSend}
                    disabled={!inputValue.trim() || aiState === "thinking" || aiState === "streaming"}
                    aria-label="Send message"
                    className="w-7 h-7 rounded-lg bg-[#5B7FFF] hover:bg-[#4a6ee8] flex items-center justify-center transition-colors duration-150 disabled:opacity-30 disabled:cursor-not-allowed"
                  >
                    <Send className="w-3.5 h-3.5 text-white" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* RIGHT PANEL */}
      <aside className="w-[280px] border-l border-border flex flex-col flex-shrink-0 bg-[#111318]">
        <div className="h-10 border-b border-border flex items-center px-2 gap-0.5 flex-shrink-0">
          <TabButton
            active={activeTab === "citations"}
            onClick={() => setActiveTab("citations")}
            icon={<FileCheck className="w-3 h-3" />}
            label="Citations"
            aria-label="Citations tab"
          />
          <TabButton
            active={activeTab === "trace"}
            onClick={() => setActiveTab("trace")}
            icon={<Activity className="w-3 h-3" />}
            label="Trace"
            aria-label="Trace tab"
          />
          <TabButton
            active={activeTab === "analytics"}
            onClick={() => setActiveTab("analytics")}
            icon={<TrendingUp className="w-3 h-3" />}
            label="Analytics"
            aria-label="Analytics tab"
          />
        </div>

        <div className="flex-1 overflow-y-auto p-3 scrollbar-hide">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.12 }}
            >
              {activeTab === "citations" && <CitationsTab citations={mockCitations} />}
              {activeTab === "trace" && <TraceTab steps={traceSteps} stageTimes={stageTimes} stages={stages} />}
              {activeTab === "analytics" && <AnalyticsTab />}
            </motion.div>
          </AnimatePresence>
        </div>
      </aside>
    </div>
  );
}

// ─── Inline text with citation pills ──────────────────────────────────────────

function CitationInlineText({ content }: { content: string }) {
  if (!content) return null;
  const parts = content.split(/(\[①\]|\[②\]|\[③\])/);
  const map: Record<string, number> = { "[①]": 1, "[②]": 2, "[③]": 3 };
  return (
    <>
      {parts.map((part, i) => {
        const num = map[part];
        if (num !== undefined) {
          return (
            <button
              key={i}
              aria-label={`Citation ${num}`}
              className="inline-flex items-center justify-center w-[18px] h-[18px] rounded bg-[#5B7FFF]/20 text-[#5B7FFF] text-[9px] font-mono font-semibold hover:bg-[#5B7FFF]/35 transition-colors duration-150 mx-0.5 align-middle"
            >
              {num}
            </button>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}

// ─── Stage indicator ──────────────────────────────────────────────────────────

function StageIndicator({
  stage,
  label,
  status,
}: {
  stage: string;
  label: string;
  status: "pending" | "active" | "complete";
}) {
  return (
    <div className="flex items-center gap-1.5">
      <AnimatePresence mode="wait">
        {status === "complete" ? (
          <motion.div
            key="check"
            initial={{ scale: 0.5, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.2, ease: "backOut" }}
            className="w-4 h-4 rounded-full bg-[#34d399]/15 border border-[#34d399]/25 flex items-center justify-center"
          >
            <Check className="w-2.5 h-2.5 text-[#34d399]" />
          </motion.div>
        ) : status === "active" ? (
          <motion.div
            key="active"
            initial={{ scale: 0.8 }}
            animate={{ scale: [1, 1.15, 1] }}
            transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
            className="w-4 h-4 rounded-full bg-[#5B7FFF]/15 border border-[#5B7FFF]/40 flex items-center justify-center"
          >
            <div className="w-1.5 h-1.5 rounded-full bg-[#5B7FFF]" />
          </motion.div>
        ) : (
          <motion.div
            key="pending"
            className="w-4 h-4 rounded-full bg-[#1e2235] border border-[#2a2f3d] flex items-center justify-center"
          >
            <div className="w-1.5 h-1.5 rounded-full bg-[#2a2f3d]" />
          </motion.div>
        )}
      </AnimatePresence>
      <span
        className={`text-[10px] font-medium font-['Syne'] transition-colors duration-150 ${
          status === "complete"
            ? "text-[#7c8299]"
            : status === "active"
            ? "text-foreground"
            : "text-[#454d66]"
        }`}
      >
        {label}
      </span>
    </div>
  );
}

// ─── AI state dot ─────────────────────────────────────────────────────────────

function AIStateIndicator({ state }: { state: AIState }) {
  if (state === "idle")
    return (
      <div className="flex items-center gap-1.5">
        <div className="w-1.5 h-1.5 rounded-full bg-[#5B7FFF]/50" />
        <span className="text-[9px] font-mono text-[#454d66]">Idle</span>
      </div>
    );
  if (state === "thinking")
    return (
      <div className="flex items-center gap-1.5">
        <motion.div
          animate={{ backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="w-12 h-1.5 rounded-full"
          style={{
            background: "linear-gradient(90deg, #5B7FFF, #a78bfa, #5B7FFF)",
            backgroundSize: "200% 100%",
          }}
        />
        <span className="text-[9px] font-mono text-[#7c8299]">Thinking</span>
      </div>
    );
  if (state === "streaming")
    return (
      <div className="flex items-center gap-1.5">
        <motion.div
          animate={{ opacity: [1, 0.3, 1] }}
          transition={{ duration: 0.8, repeat: Infinity }}
          className="w-1.5 h-1.5 rounded-full bg-[#34d399]"
        />
        <span className="text-[9px] font-mono text-[#34d399]">Streaming</span>
      </div>
    );
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-1.5 h-1.5 rounded-full bg-[#34d399]/40" />
      <span className="text-[9px] font-mono text-[#454d66]">Ready</span>
    </div>
  );
}

// ─── Tab button ───────────────────────────────────────────────────────────────

function TabButton({
  active,
  onClick,
  icon,
  label,
  "aria-label": ariaLabel,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
  "aria-label": string;
}) {
  return (
    <button
      onClick={onClick}
      aria-label={ariaLabel}
      className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] font-medium transition-colors duration-150 ${
        active
          ? "bg-secondary text-foreground"
          : "text-[#7c8299] hover:bg-secondary/50 hover:text-foreground"
      }`}
    >
      {icon}
      <span className="font-['Syne']">{label}</span>
    </button>
  );
}

// ─── Citations Tab ────────────────────────────────────────────────────────────

function CitationsTab({ citations }: { citations: Citation[] }) {
  return (
    <div className="space-y-2.5">
      {citations.map((citation, i) => (
        <motion.div
          key={citation.id}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.15, delay: i * 0.06 }}
          whileHover={{ y: -2 }}
          className="rounded-lg border border-[#1e2235] bg-[#13161f] p-3 space-y-2.5 hover:border-[#5B7FFF]/25 transition-colors duration-150 cursor-pointer group"
        >
          <div className="flex items-start gap-2">
            <div
              className="flex items-center justify-center w-5 h-5 rounded text-[9px] font-mono font-semibold flex-shrink-0"
              style={{ backgroundColor: `${citation.docColor}18`, color: citation.docColor }}
            >
              {citation.id}
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between gap-2">
                <div className="text-[11px] font-semibold font-['Syne'] text-foreground truncate">
                  {citation.document}
                </div>
                <ArrowUpRight className="w-3 h-3 text-[#454d66] opacity-0 group-hover:opacity-100 transition-opacity duration-150 flex-shrink-0" />
              </div>
              <div className="text-[9px] text-[#7c8299] font-mono">p. {citation.page}</div>
            </div>
          </div>

          <div
            className="pl-2.5 border-l-[2px] py-0.5"
            style={{ borderColor: `${citation.docColor}40` }}
          >
            <p className="text-[11px] leading-relaxed text-[#b8bdd4]">{citation.snippet}</p>
          </div>

          <div className="space-y-2">
            <ScoreBar label="Similarity" value={citation.similarity} from="#5B7FFF" to="#a78bfa" />
            <ScoreBar label="Reranker" value={citation.rerankerScore} from="#a78bfa" to="#34d399" />
          </div>
        </motion.div>
      ))}

      {/* Session Metrics */}
      <div className="rounded-lg border border-[#1e2235] bg-[#13161f] p-3">
        <h4 className="text-[10px] font-semibold font-['Syne'] text-foreground mb-2.5">
          Session Metrics
        </h4>
        <div className="grid grid-cols-2 gap-2">
          <MetricCard label="Alignment" value="0.88" color="#34d399" />
          <MetricCard label="Latency" value="235ms" color="#5B7FFF" />
          <MetricCard label="Chunks" value="18" color="#a78bfa" />
          <MetricCard label="Tokens" value="3.2K" color="#fbbf24" />
        </div>
      </div>

      {/* Reranker Breakdown */}
      <div className="rounded-lg border border-[#1e2235] bg-[#13161f] p-3 space-y-2.5">
        <h4 className="text-[10px] font-semibold font-['Syne'] text-foreground">
          Reranker Breakdown
        </h4>
        <div className="space-y-2">
          <ScoreBar label="Cross-encoder" value={0.9} from="#5B7FFF" to="#a78bfa" />
          <ScoreBar label="BM25 Hybrid" value={0.75} from="#a78bfa" to="#fbbf24" />
          <ScoreBar label="Dense Vector" value={0.82} from="#34d399" to="#5B7FFF" />
        </div>
      </div>

      {/* Tags */}
      <div className="flex flex-wrap gap-1.5">
        <Tag icon={<Check className="w-2.5 h-2.5" />} label="Grounded" color="#34d399" />
        <Tag icon={<Target className="w-2.5 h-2.5" />} label="Cross-doc" color="#5B7FFF" />
        <Tag icon={<Zap className="w-2.5 h-2.5" />} label="High confidence" color="#a78bfa" />
      </div>
    </div>
  );
}

// ─── Trace Tab ────────────────────────────────────────────────────────────────

function TraceTab({
  steps,
  stageTimes,
  stages,
}: {
  steps: { label: string; latency: number; status: string; detail: string }[];
  stageTimes: Record<string, number>;
  stages: StageState[];
}) {
  const total = Object.values(stageTimes).reduce((a, b) => a + b, 0);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-[10px] font-semibold font-['Syne'] text-foreground">
          Retrieval Timeline
        </h4>
        <span className="text-[9px] font-mono text-[#7c8299]">{total}ms total</span>
      </div>

      {/* Live stage status */}
      <div className="rounded-lg border border-[#1e2235] bg-[#13161f] p-2.5 space-y-1.5 mb-3">
        {stages.map((s) => {
          const stageKey = s.stage;
          const time = stageTimes[stageKey];
          return (
            <div key={s.stage} className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <div
                  className={`w-1 h-1 rounded-full flex-shrink-0 ${
                    s.status === "complete"
                      ? "bg-[#34d399]"
                      : s.status === "active"
                      ? "bg-[#5B7FFF]"
                      : "bg-[#2a2f3d]"
                  }`}
                />
                <span
                  className={`text-[10px] font-medium font-['Syne'] capitalize ${
                    s.status === "pending" ? "text-[#454d66]" : "text-[#b8bdd4]"
                  }`}
                >
                  {s.stage}
                </span>
              </div>
              <span className="text-[9px] font-mono text-[#7c8299]">
                {s.status === "complete" ? `${time}ms` : s.status === "active" ? "…" : "—"}
              </span>
            </div>
          );
        })}
      </div>

      {steps.map((step, idx) => (
        <div key={idx} className="relative">
          {idx < steps.length - 1 && (
            <div className="absolute left-[7px] top-9 w-px h-4 bg-[#1e2235]" />
          )}
          <div className="flex items-start gap-2.5 rounded-lg border border-[#1e2235] bg-[#13161f] p-2.5 hover:border-[#5B7FFF]/20 transition-colors duration-150">
            <div className="w-3.5 h-3.5 rounded-full bg-[#34d399]/10 border border-[#34d399]/20 flex items-center justify-center flex-shrink-0 mt-0.5">
              <div className="w-1 h-1 rounded-full bg-[#34d399]" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-medium font-['Syne'] text-foreground">
                  {step.label}
                </span>
                <span className="text-[9px] font-mono text-[#7c8299]">{step.latency}ms</span>
              </div>
              <div className="text-[9px] font-mono text-[#454d66] mt-0.5">{step.detail}</div>
            </div>
          </div>
        </div>
      ))}

      {/* Latency bar */}
      <div className="rounded-lg border border-[#1e2235] bg-[#13161f] p-2.5 mt-3">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[10px] font-semibold font-['Syne'] text-foreground">
            Pipeline Breakdown
          </span>
        </div>
        <div className="flex h-2 rounded-full overflow-hidden gap-px">
          {Object.entries(stageTimes).map(([key, val], i) => {
            const colors = ["#5B7FFF", "#a78bfa", "#fbbf24", "#34d399"];
            const pct = (val / total) * 100;
            return (
              <motion.div
                key={key}
                initial={{ width: 0 }}
                animate={{ width: `${pct}%` }}
                transition={{ duration: 0.3, ease: "easeOut", delay: i * 0.05 }}
                className="h-full rounded-sm"
                style={{ backgroundColor: colors[i] }}
                title={`${key}: ${val}ms`}
              />
            );
          })}
        </div>
        <div className="flex items-center gap-3 mt-1.5">
          {Object.entries(stageTimes).map(([key, _], i) => {
            const colors = ["#5B7FFF", "#a78bfa", "#fbbf24", "#34d399"];
            return (
              <div key={key} className="flex items-center gap-1">
                <div className="w-1.5 h-1.5 rounded-sm" style={{ backgroundColor: colors[i] }} />
                <span className="text-[8px] font-mono text-[#454d66] capitalize">{key}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ─── Analytics Tab ────────────────────────────────────────────────────────────

function AnalyticsTab() {
  return (
    <div className="space-y-2.5">
      <div className="rounded-lg border border-[#1e2235] bg-[#13161f] p-3">
        <h4 className="text-[10px] font-semibold font-['Syne'] text-foreground mb-3">
          Confidence Distribution
        </h4>
        <div className="space-y-2">
          <ConfidenceBar label="High (0.8+)" value={65} color="#34d399" delay={0} />
          <ConfidenceBar label="Med (0.6–0.8)" value={28} color="#fbbf24" delay={0.06} />
          <ConfidenceBar label="Low (<0.6)" value={7} color="#f87171" delay={0.12} />
        </div>
      </div>

      <div className="rounded-lg border border-[#1e2235] bg-[#13161f] p-3">
        <h4 className="text-[10px] font-semibold font-['Syne'] text-foreground mb-3">
          Doc Contribution
        </h4>
        <div className="space-y-2">
          {documents.map((doc, idx) => {
            const vals = [42, 28, 18, 12];
            return (
              <div key={doc.id} className="space-y-1">
                <div className="flex items-center justify-between text-[9px]">
                  <div className="flex items-center gap-1.5">
                    <Circle className="w-1.5 h-1.5" fill={doc.color} color={doc.color} />
                    <span className="text-[#7c8299] font-medium truncate max-w-[120px]">{doc.name}</span>
                  </div>
                  <span className="font-mono text-[#b8bdd4]">{vals[idx]}%</span>
                </div>
                <div className="h-1.5 bg-[#1e2235] rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${vals[idx]}%` }}
                    transition={{ duration: 0.3, ease: "easeOut", delay: idx * 0.05 }}
                    className="h-full rounded-full"
                    style={{ backgroundColor: doc.color }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="rounded-lg border border-[#1e2235] bg-[#13161f] p-3">
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-[10px] font-semibold font-['Syne'] text-foreground">Token Activity</h4>
          <span className="text-[9px] font-mono text-[#5B7FFF] font-semibold">~450 tok/s</span>
        </div>
        <Sparkline />
        <div className="text-[9px] font-mono text-[#454d66] mt-2">Last 40 inference calls</div>
      </div>

      <div className="rounded-lg border border-[#1e2235] bg-[#13161f] p-3">
        <h4 className="text-[10px] font-semibold font-['Syne'] text-foreground mb-2.5">
          Query Stats
        </h4>
        <div className="grid grid-cols-2 gap-2">
          <MetricCard label="Queries" value="24" color="#5B7FFF" />
          <MetricCard label="Avg latency" value="1.8s" color="#a78bfa" />
          <MetricCard label="Cache hits" value="71%" color="#34d399" />
          <MetricCard label="Error rate" value="0.4%" color="#f87171" />
        </div>
      </div>
    </div>
  );
}

// ─── Sparkline ────────────────────────────────────────────────────────────────

function Sparkline() {
  const bars = Array.from({ length: 40 }, (_, i) => {
    const base = Math.sin(i * 0.4) * 0.3 + 0.5;
    const noise = Math.random() * 0.4;
    return Math.min(1, Math.max(0.05, base + noise - 0.2));
  });

  return (
    <div className="h-12 flex items-end gap-px">
      {bars.map((h, i) => (
        <motion.div
          key={i}
          initial={{ height: 0 }}
          animate={{ height: `${h * 100}%` }}
          transition={{ duration: 0.4, delay: i * 0.01, ease: "easeOut" }}
          className="flex-1 rounded-sm"
          style={{
            background: `linear-gradient(to top, #5B7FFF, #a78bfa)`,
            opacity: 0.55 + h * 0.45,
          }}
        />
      ))}
    </div>
  );
}

// ─── Shared primitives ────────────────────────────────────────────────────────

function ScoreBar({
  label,
  value,
  from,
  to,
}: {
  label: string;
  value: number;
  from: string;
  to: string;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-[9px]">
        <span className="text-[#7c8299] font-medium font-['Syne']">{label}</span>
        <span className="font-mono text-[#b8bdd4]">{value.toFixed(2)}</span>
      </div>
      <div className="h-1 bg-[#1e2235] rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 0.3, ease: "easeOut" }}
          className="h-full rounded-full"
          style={{ background: `linear-gradient(to right, ${from}, ${to})` }}
        />
      </div>
    </div>
  );
}

function ConfidenceBar({
  label,
  value,
  color,
  delay,
}: {
  label: string;
  value: number;
  color: string;
  delay: number;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-[9px]">
        <span className="text-[#7c8299] font-medium font-['Syne']">{label}</span>
        <span className="font-mono text-[#b8bdd4]">{value}%</span>
      </div>
      <div className="h-1 bg-[#1e2235] rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value}%` }}
          transition={{ duration: 0.3, ease: "easeOut", delay }}
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
        />
      </div>
    </div>
  );
}

function MetricCard({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="rounded-lg border border-[#1e2235] bg-[#191c24] p-2">
      <div className="text-[8px] text-[#454d66] font-medium font-['Syne'] mb-0.5">{label}</div>
      <div className="text-sm font-semibold font-mono" style={{ color }}>
        {value}
      </div>
    </div>
  );
}

function Tag({ icon, label, color }: { icon: React.ReactNode; label: string; color: string }) {
  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[9px] font-medium font-['Syne'] border"
      style={{
        backgroundColor: `${color}12`,
        borderColor: `${color}28`,
        color,
      }}
    >
      {icon}
      {label}
    </span>
  );
}
