import { useState, useEffect, useRef } from "react";

const GATEWAY_URL = "http://localhost:8000";

const fmt = {
  usd:     v => (v < 0.0001 && v > 0) ? "<$0.0001" : `$${Number(v).toFixed(6)}`,
  usdMed:  v => `$${Number(v).toFixed(4)}`,
  pct:     v => `${Number(v).toFixed(1)}%`,
  ms:      v => `${Math.round(v)}ms`,
  num:     v => Number(v).toLocaleString(),
  short:   v => v > 1000 ? `${(v/1000).toFixed(1)}k` : String(v),
};

const TIER_STYLE = {
  cheap:    { bg: "#EEF9F1", fg: "#1A7F3C" },
  mid:      { bg: "#FEF9EC", fg: "#92600A" },
  powerful: { bg: "#FEF0EF", fg: "#B91C1C" },
};

const CB_STYLE = {
  closed:    { dot: "#22C55E", bg: "#F0FDF4", border: "#BBF7D0", text: "#15803D" },
  open:      { dot: "#EF4444", bg: "#FEF2F2", border: "#FECACA", text: "#B91C1C" },
  half_open: { dot: "#F59E0B", bg: "#FFFBEB", border: "#FDE68A", text: "#92600A" },
};

const BASELINE = {
  gpt4o:     { prompt: 0.0025,  completion: 0.01  },
  claude:    { prompt: 0.003,   completion: 0.015 },
  geminiPro: { prompt: 0.00125, completion: 0.005 },
};

function calcBaseCost(tokens, model) {
  if (!tokens) return 0;
  return (tokens.prompt / 1000) * model.prompt +
         (tokens.completion / 1000) * model.completion;
}

function TierPill({ tier }) {
  const s = TIER_STYLE[tier];
  if (!s) return null;
  return (
    <span style={{ background: s.bg, color: s.fg }}
      className="text-[10px] font-bold tracking-widest px-2 py-0.5 rounded uppercase">
      {tier}
    </span>
  );
}

function Dot({ color, pulse }) {
  return (
    <span style={{ background: color }}
      className={`inline-block w-2 h-2 rounded-full shrink-0 ${pulse ? "animate-pulse" : ""}`} />
  );
}

function SectionLabel({ children, live }) {
  return (
    <div className="flex items-center justify-between mb-3">
      <span className="text-[10px] font-bold tracking-[0.12em] text-gray-400 uppercase">{children}</span>
      {live && (
        <div className="flex items-center gap-1">
          <Dot color="#22C55E" pulse />
          <span className="text-[10px] text-gray-400">live</span>
        </div>
      )}
    </div>
  );
}

function Card({ children, className = "" }) {
  return (
    <div className={`bg-white border border-gray-200 rounded-xl p-4 ${className}`}>
      {children}
    </div>
  );
}

function Row({ label, value, accent, mono }) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-gray-100 last:border-0">
      <span className="text-xs text-gray-400">{label}</span>
      <span className={`text-xs font-semibold tabular-nums ${mono ? "font-mono" : ""} ${accent || "text-gray-800"}`}>
        {value}
      </span>
    </div>
  );
}

function StatBox({ label, value, accent }) {
  return (
    <div className="bg-gray-50 rounded-lg p-2.5 text-center">
      <div className={`text-base font-bold tabular-nums leading-tight ${accent || "text-gray-800"}`}>{value}</div>
      <div className="text-[10px] text-gray-400 mt-0.5 leading-tight">{label}</div>
    </div>
  );
}

function CacheBar({ redis, qdrant, miss, total }) {
  const r = total ? (redis / total) * 100 : 0;
  const q = total ? (qdrant / total) * 100 : 0;
  const m = total ? (miss / total) * 100 : 0;
  return (
    <div>
      <div className="flex h-2 rounded-full overflow-hidden bg-gray-100 mb-2">
        <div style={{ width: `${r}%`, background: "#3B82F6" }} className="transition-all duration-700" />
        <div style={{ width: `${q}%`, background: "#8B5CF6" }} className="transition-all duration-700" />
        <div style={{ width: `${m}%`, background: "#E5E7EB" }} className="transition-all duration-700" />
      </div>
      <div className="grid grid-cols-3 text-center gap-1">
        <div><div className="text-sm font-bold text-blue-600 tabular-nums">{redis}</div><div className="text-[10px] text-gray-400">Redis</div></div>
        <div><div className="text-sm font-bold text-purple-600 tabular-nums">{qdrant}</div><div className="text-[10px] text-gray-400">Qdrant</div></div>
        <div><div className="text-sm font-bold text-gray-400 tabular-nums">{miss}</div><div className="text-[10px] text-gray-400">Miss</div></div>
      </div>
    </div>
  );
}

function TierBar({ cheap, mid, powerful, total }) {
  if (!total) return null;
  return (
    <div>
      <div className="flex h-2 rounded-full overflow-hidden bg-gray-100 mb-1.5">
        <div style={{ width: `${(cheap/total)*100}%`, background: "#22C55E" }} className="transition-all duration-700" />
        <div style={{ width: `${(mid/total)*100}%`, background: "#F59E0B" }} className="transition-all duration-700" />
        <div style={{ width: `${(powerful/total)*100}%`, background: "#EF4444" }} className="transition-all duration-700" />
      </div>
      <div className="grid grid-cols-3 text-center gap-1">
        <div className="text-[10px] font-semibold text-green-600">{cheap} cheap</div>
        <div className="text-[10px] font-semibold text-yellow-600">{mid} mid</div>
        <div className="text-[10px] font-semibold text-red-600">{powerful} powerful</div>
      </div>
    </div>
  );
}

function ProviderCard({ cb }) {
  const s = CB_STYLE[cb.state] || CB_STYLE.closed;
  return (
    <div style={{ background: s.bg, borderColor: s.border }} className="border rounded-lg p-3">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Dot color={s.dot} pulse={cb.state === "closed"} />
          <span className="text-sm font-semibold text-gray-800 capitalize">{cb.provider}</span>
        </div>
        <span className="text-[10px] font-bold tracking-wider px-2 py-0.5 rounded border bg-white uppercase"
          style={{ color: s.text, borderColor: s.border }}>
          {cb.state.replace("_", " ")}
        </span>
      </div>
      <div className="grid grid-cols-4 gap-1 text-center">
        {[["Req", cb.total_requests], ["OK", cb.success_count], ["Fail", cb.failure_count], ["Err", fmt.pct(cb.error_rate * 100)]].map(([l, v]) => (
          <div key={l}>
            <div className="text-xs font-bold text-gray-800 tabular-nums">{v}</div>
            <div className="text-[9px] text-gray-400">{l}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function Bubble({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div className={`max-w-[78%] flex flex-col gap-1.5 ${isUser ? "items-end" : "items-start"}`}>
        <span className="text-[10px] font-bold tracking-widest text-gray-400 uppercase px-1">
          {isUser ? "YOU" : "GATEWAY"}
        </span>
        <div className={`px-4 py-3 rounded-xl text-sm leading-relaxed whitespace-pre-wrap
          ${isUser ? "bg-gray-900 text-white" : "bg-white border border-gray-200 text-gray-800 shadow-sm"}`}>
          {msg.content}
        </div>
        {!isUser && msg.meta && (
          <div className="flex flex-wrap gap-1.5 px-1">
            {msg.meta.cache_hit ? (
              <>
                <span className="text-[11px] font-medium bg-green-50 text-green-700 border border-green-200 px-2 py-0.5 rounded">
                  + {msg.meta.cache_layer === "redis" ? "Redis" : "Qdrant"} hit
                  {msg.meta.similarity < 1 ? ` · ${(msg.meta.similarity * 100).toFixed(0)}% match` : " · exact"}
                </span>
                <span className="text-[11px] font-medium bg-gray-50 text-gray-500 border border-gray-200 px-2 py-0.5 rounded tabular-nums">
                  Free · {fmt.ms(msg.meta.latency?.total_ms ?? 0)}
                </span>
              </>
            ) : (
              <>
                <span className="text-[11px] font-medium bg-blue-50 text-blue-700 border border-blue-200 px-2 py-0.5 rounded">
                  + {msg.meta.provider_used} · {msg.meta.model?.split("/").pop()}
                </span>
                {msg.meta.routing && <TierPill tier={msg.meta.routing.tier} />}
                {msg.meta.routing && (
                  <span className="text-[11px] font-medium bg-gray-50 text-gray-500 border border-gray-200 px-2 py-0.5 rounded">
                    Score {msg.meta.routing.complexity_score}/10
                  </span>
                )}
                <span className="text-[11px] font-medium bg-gray-50 text-gray-600 border border-gray-200 px-2 py-0.5 rounded tabular-nums">
                  {fmt.usd(msg.meta.cost?.estimated_usd ?? 0)}
                </span>
                <span className="text-[11px] font-medium bg-gray-50 text-gray-500 border border-gray-200 px-2 py-0.5 rounded tabular-nums">
                  {fmt.ms(msg.meta.latency?.total_ms ?? 0)}
                </span>
                {msg.meta.tokens && (
                  <span className="text-[11px] font-medium bg-gray-50 text-gray-400 border border-gray-200 px-2 py-0.5 rounded tabular-nums">
                    {msg.meta.tokens.prompt}p + {msg.meta.tokens.completion}c tok
                  </span>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function LivePanel({ stats, health, session, lastRequest }) {
  return (
    <div className="flex flex-col gap-3">
      <Card>
        <SectionLabel>Session</SectionLabel>
        <div className="grid grid-cols-2 gap-2 mb-3">
          <StatBox label="Requests"     value={fmt.num(stats?.total_lookups ?? 0)} />
          <StatBox label="Hit rate"     value={fmt.pct(stats?.hit_rate ?? 0)} accent="text-green-600" />
          <StatBox label="Session cost" value={fmt.usdMed(session.cost)} />
          <StatBox label="Tokens used"  value={fmt.short(session.tokens)} />
        </div>
        <CacheBar redis={stats?.redis_hits ?? 0} qdrant={stats?.qdrant_hits ?? 0} miss={stats?.misses ?? 0} total={stats?.total_lookups ?? 0} />
      </Card>

      <Card>
        <SectionLabel>Savings vs alternatives</SectionLabel>
        <Row label="Your cost"        value={fmt.usdMed(session.cost)} />
        <Row label="vs GPT-4o"        value={`+${fmt.usdMed(session.savedVsGpt4o)}`}      accent="text-green-600" />
        <Row label="vs Claude Sonnet" value={`+${fmt.usdMed(session.savedVsClaude)}`}     accent="text-green-600" />
        <Row label="vs Gemini Pro"    value={`+${fmt.usdMed(session.savedVsGeminiPro)}`}  accent="text-green-600" />
        <Row label="Free cache calls" value={`${session.cacheHits} requests`}             accent="text-blue-600" />
        <Row label="Tokens saved"     value={fmt.num(session.tokensSaved)}                accent="text-blue-600" />
      </Card>

      <Card>
        <SectionLabel>Routing</SectionLabel>
        <div className="space-y-0 mb-3">
          <Row label="Cheap tier"    value={session.tierCount.cheap}    accent="text-green-600" />
          <Row label="Mid tier"      value={session.tierCount.mid}      accent="text-yellow-600" />
          <Row label="Powerful tier" value={session.tierCount.powerful} accent="text-red-600" />
          <Row label="Cache served"  value={session.cacheHits}          accent="text-blue-600" />
        </div>
        <TierBar cheap={session.tierCount.cheap} mid={session.tierCount.mid} powerful={session.tierCount.powerful} total={session.totalNonCached} />
      </Card>

      {Object.keys(session.modelCount).length > 0 && (
        <Card>
          <SectionLabel>Models used</SectionLabel>
          {Object.entries(session.modelCount).sort((a,b) => b[1]-a[1]).map(([model, count]) => (
            <div key={model} className="flex items-center justify-between py-1.5 border-b border-gray-100 last:border-0">
              <span className="text-[11px] font-mono text-gray-600 truncate max-w-[160px]">{model}</span>
              <span className="text-xs font-bold text-gray-800 tabular-nums">{count}x</span>
            </div>
          ))}
        </Card>
      )}

      {lastRequest && (
        <Card>
          <SectionLabel>Last request</SectionLabel>
          <Row label="Provider"    value={lastRequest.provider_used} />
          <Row label="Model"       value={lastRequest.model?.split("/").pop() ?? "-"} mono />
          <Row label="Cache"
            value={lastRequest.cache_hit
              ? `Hit · ${lastRequest.cache_layer} · ${lastRequest.similarity < 1 ? (lastRequest.similarity*100).toFixed(0)+"%" : "exact"}`
              : "Miss"}
            accent={lastRequest.cache_hit ? "text-green-600" : "text-gray-800"} />
          {lastRequest.routing && !lastRequest.cache_hit && (
            <>
              <Row label="Complexity" value={`${lastRequest.routing.complexity_score}/10`} />
              <Row label="Tier"       value={lastRequest.routing.tier?.toUpperCase()} />
              <Row label="Classified" value={lastRequest.routing.classified_by} />
            </>
          )}
          <Row label="Prompt tokens"      value={lastRequest.cache_hit ? "0" : fmt.num(lastRequest.tokens?.prompt ?? 0)} />
          <Row label="Completion tokens"  value={lastRequest.cache_hit ? "0" : fmt.num(lastRequest.tokens?.completion ?? 0)} />
          <Row label="Cost"
            value={lastRequest.cache_hit ? "Free" : fmt.usd(lastRequest.cost?.estimated_usd ?? 0)}
            accent={lastRequest.cache_hit ? "text-green-600" : "text-gray-800"} />
          {!lastRequest.cache_hit && (
            <Row label="Saved vs GPT-4o" value={fmt.usd(lastRequest.cost?.savings_vs_gpt4o_usd ?? 0)} accent="text-green-600" />
          )}
          <Row label="Total latency"   value={fmt.ms(lastRequest.latency?.total_ms ?? 0)} />
          <Row label="Cache lookup"    value={fmt.ms(lastRequest.latency?.cache_lookup_ms ?? 0)} accent="text-green-600" />
          <Row label="Provider call"   value={fmt.ms(lastRequest.latency?.provider_ms ?? 0)} />
          {lastRequest.routing && !lastRequest.cache_hit && (
            <div className="mt-2 text-[10px] text-gray-400 bg-gray-50 rounded-lg p-2 leading-relaxed font-mono">
              {lastRequest.routing.reasoning}
            </div>
          )}
        </Card>
      )}

      <Card>
        <SectionLabel live>Provider health</SectionLabel>
        <div className="flex flex-col gap-2">
          {health?.circuit_breakers?.map(cb => <ProviderCard key={cb.provider} cb={cb} />)}
        </div>
        <p className="text-[10px] text-gray-400 mt-2 text-center leading-relaxed">
          Circuit opens after 5 failures. Recovers after 30s half-open probe.
        </p>
      </Card>
    </div>
  );
}

function HistoricalPanel({ analytics }) {
  if (!analytics?.overall) {
    return (
      <div className="flex items-center justify-center h-32">
        <p className="text-xs text-gray-400">Send messages to populate historical data</p>
      </div>
    );
  }

  const o  = analytics.overall;
  const lp = analytics.latency_percentiles ?? {};
  const speedup = Math.round(o.avg_provider_latency_ms / Math.max(o.avg_cache_latency_ms, 1));

  return (
    <div className="flex flex-col gap-3">
      <Card>
        <SectionLabel>All time — Postgres</SectionLabel>
        <div className="grid grid-cols-2 gap-2 mb-3">
          <StatBox label="Total requests" value={fmt.num(o.total_requests)} />
          <StatBox label="Hit rate"       value={fmt.pct(o.hit_rate_pct)}   accent="text-green-600" />
          <StatBox label="Total cost"     value={fmt.usdMed(o.total_cost_usd)} />
          <StatBox label="Total saved"    value={fmt.usdMed(o.total_savings_usd)} accent="text-green-600" />
          <StatBox label="Total tokens"   value={fmt.short(o.total_tokens)} />
          <StatBox label="Avg complexity" value={o.avg_complexity} />
        </div>
        <div className="bg-gray-50 rounded-lg p-3">
          <div className="text-[10px] font-bold tracking-wider text-gray-400 uppercase mb-2">Latency comparison</div>
          <Row label="Cache avg"     value={fmt.ms(o.avg_cache_latency_ms)}    accent="text-green-600" />
          <Row label="Provider avg"  value={fmt.ms(o.avg_provider_latency_ms)} />
          <Row label="Cache speedup" value={`${speedup}x faster`}              accent="text-blue-600" />
        </div>
      </Card>

      <Card>
        <SectionLabel>Latency percentiles</SectionLabel>
        <Row label="Provider p50" value={fmt.ms(lp.provider_p50 ?? 0)} />
        <Row label="Provider p95" value={fmt.ms(lp.provider_p95 ?? 0)} accent="text-yellow-600" />
        <Row label="Provider p99" value={fmt.ms(lp.provider_p99 ?? 0)} accent="text-red-600" />
        <Row label="Cache p50"    value={fmt.ms(lp.cache_p50 ?? 0)}    accent="text-green-600" />
      </Card>

      <Card>
        <SectionLabel>By provider</SectionLabel>
        {analytics.by_provider?.map(p => (
          <div key={p.provider} className="flex items-center justify-between py-1.5 border-b border-gray-100 last:border-0 gap-2">
            <span className="text-xs font-semibold text-gray-700 capitalize w-14 shrink-0">{p.provider}</span>
            <span className="text-[11px] text-gray-400 tabular-nums w-10 text-right">{p.requests}req</span>
            <span className="text-[11px] text-purple-600 tabular-nums w-16 text-right">{fmt.usdMed(p.cost)}</span>
            <span className="text-[11px] text-gray-400 tabular-nums w-14 text-right">{fmt.ms(p.avg_latency)}</span>
          </div>
        ))}
      </Card>

      <Card>
        <SectionLabel>By tier</SectionLabel>
        <div className="space-y-0 mb-3">
          {analytics.by_tier?.map(t => (
            <div key={t.tier} className="flex items-center justify-between py-1.5 border-b border-gray-100 last:border-0 gap-2">
              <TierPill tier={t.tier} />
              <span className="text-[11px] text-gray-400 tabular-nums">{t.requests} req</span>
              <span className="text-[11px] text-purple-600 tabular-nums">{fmt.usdMed(t.cost)}</span>
              <span className="text-[11px] text-gray-400 tabular-nums">{fmt.num(t.tokens)} tok</span>
            </div>
          ))}
        </div>
      </Card>

      <Card>
        <SectionLabel>By model</SectionLabel>
        {analytics.by_model?.map(m => (
          <div key={m.model} className="flex items-center justify-between py-1.5 border-b border-gray-100 last:border-0 gap-2">
            <span className="text-[11px] font-mono text-gray-600 truncate max-w-[120px]">{m.model.split("/").pop()}</span>
            <span className="text-[11px] text-gray-400 tabular-nums">{m.requests}x</span>
            <span className="text-[11px] text-purple-600 tabular-nums">{fmt.usdMed(m.cost)}</span>
            <span className="text-[11px] text-gray-400 tabular-nums">{fmt.ms(m.avg_latency)}</span>
          </div>
        ))}
      </Card>

      {analytics.cache_layers?.length > 0 && (
        <Card>
          <SectionLabel>Cache layer performance</SectionLabel>
          {analytics.cache_layers.map(l => (
            <div key={l.layer} className="flex items-center justify-between py-1.5 border-b border-gray-100 last:border-0">
              <span className="text-xs font-semibold text-gray-700 capitalize w-16">{l.layer}</span>
              <span className="text-[11px] text-gray-400 tabular-nums">{l.count} hits</span>
              <span className="text-[11px] text-green-600 tabular-nums">{fmt.ms(l.avg_latency)} avg</span>
            </div>
          ))}
        </Card>
      )}

      <Card>
        <SectionLabel>Recent requests</SectionLabel>
        {analytics.recent_requests?.slice(0, 15).map(r => (
          <div key={r.id} className="flex items-center gap-2 py-1.5 border-b border-gray-100 last:border-0">
            <Dot color={r.cache_hit ? "#22C55E" : "#D1D5DB"} />
            <span className="text-[10px] font-mono text-gray-400 w-10 shrink-0">{r.created_at.slice(11,16)}</span>
            <span className="text-[11px] text-gray-600 capitalize w-10 shrink-0 truncate">{r.provider}</span>
            {r.tier ? <TierPill tier={r.tier} /> : <span className="text-[10px] text-blue-500 w-14">cache</span>}
            <span className="text-[11px] text-gray-400 tabular-nums ml-auto shrink-0">{fmt.ms(r.total_latency_ms)}</span>
            <span className="text-[11px] text-purple-600 tabular-nums shrink-0">{fmt.usd(r.cost_usd)}</span>
          </div>
        ))}
      </Card>
    </div>
  );
}

const SUGGESTED = [
  "what is machine learning",
  "compare microservices vs monolith in detail",
  "implement binary search in Python",
  "what is recursion",
  "how are you",
];

export default function App() {
  const [messages, setMessages]     = useState([{ role: "assistant", content: "Gateway ready.\n\nSend a message to observe cost routing, semantic caching, and circuit breaker behaviour. Try the same question twice to see cache hit metrics.", meta: null }]);
  const [input, setInput]           = useState("");
  const [loading, setLoading]       = useState(false);
  const [activeTab, setActiveTab]   = useState("Live");
  const [stats, setStats]           = useState(null);
  const [health, setHealth]         = useState(null);
  const [analytics, setAnalytics]   = useState(null);
  const [lastRequest, setLastRequest] = useState(null);
  const [session, setSession]       = useState({
    cost: 0, tokens: 0, cacheHits: 0, tokensSaved: 0,
    savedVsGpt4o: 0, savedVsClaude: 0, savedVsGeminiPro: 0,
    tierCount: { cheap: 0, mid: 0, powerful: 0 },
    modelCount: {}, totalNonCached: 0,
  });
  const bottomRef = useRef(null);

  useEffect(() => {
    const poll = async () => {
      try {
        const [s, h, a] = await Promise.all([
          fetch(`${GATEWAY_URL}/cache/stats`).then(r => r.json()),
          fetch(`${GATEWAY_URL}/health`).then(r => r.json()),
          fetch(`${GATEWAY_URL}/analytics`).then(r => r.json()),
        ]);
        setStats(s); setHealth(h); setAnalytics(a);
      } catch (_) {}
    };
    poll();
    const id = setInterval(poll, 2000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  const updateSession = (data) => {
    setSession(prev => {
      const next = { ...prev, tierCount: { ...prev.tierCount }, modelCount: { ...prev.modelCount } };
      if (data.cache_hit) { next.cacheHits += 1; next.tokensSaved += 200; return next; }
      const cost = data.cost?.estimated_usd ?? 0;
      const tokens = data.tokens ?? { prompt: 0, completion: 0 };
      const tier = data.routing?.tier ?? "cheap";
      const shortModel = (data.model ?? "unknown").split("/").pop();
      next.cost += cost;
      next.tokens += tokens.prompt + tokens.completion;
      next.totalNonCached += 1;
      next.savedVsGpt4o    += Math.max(0, calcBaseCost(tokens, BASELINE.gpt4o)     - cost);
      next.savedVsClaude   += Math.max(0, calcBaseCost(tokens, BASELINE.claude)    - cost);
      next.savedVsGeminiPro += Math.max(0, calcBaseCost(tokens, BASELINE.geminiPro) - cost);
      if (tier in next.tierCount) next.tierCount[tier] += 1;
      next.modelCount[shortModel] = (next.modelCount[shortModel] ?? 0) + 1;
      return next;
    });
  };

  const sendMessage = async (text) => {
    const msg = (text || input).trim();
    if (!msg || loading) return;
    setInput(""); setLoading(true);
    setMessages(prev => [...prev, { role: "user", content: msg, meta: null }]);
    try {
      const res = await fetch(`${GATEWAY_URL}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: [{ role: "user", content: msg }], preferred_provider: "gemini" }),
      });
      const data = await res.json();
      updateSession(data);
      setLastRequest(data);
      setMessages(prev => [...prev, { role: "assistant", content: data.content || data.detail || "Error.", meta: data }]);
    } catch (_) {
      setMessages(prev => [...prev, { role: "assistant", content: "Gateway unreachable. Ensure the server is running on port 8000 and CORS is configured.", meta: null }]);
    } finally { setLoading(false); }
  };

  const onKey = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } };

  return (
    <div className="h-screen bg-[#F7F7F8] flex flex-col overflow-hidden" style={{ fontFamily: "'Inter', system-ui, sans-serif" }}>
      <header className="bg-white border-b border-gray-200 px-6 py-3 flex items-center gap-3 shrink-0">
        <Dot color="#22C55E" pulse />
        <span className="font-bold text-gray-900 tracking-tight">LLM Gateway</span>
        <div className="h-4 w-px bg-gray-200" />
        <span className="text-[10px] font-semibold tracking-wider text-gray-400 uppercase hidden sm:block">
          Semantic Cache · Cost Router · Circuit Breaker
        </span>
        <div className="ml-auto flex items-center gap-1.5">
          {health?.available_providers?.map(p => (
            <span key={p} className="text-[10px] font-bold tracking-wider px-2 py-0.5 rounded bg-green-50 text-green-700 border border-green-200 uppercase">{p}</span>
          ))}
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
          <div className="flex-1 overflow-y-auto px-6 py-5">
            {messages.map((msg, i) => <Bubble key={i} msg={msg} />)}
            {loading && (
              <div className="flex justify-start mb-3">
                <div className="flex flex-col gap-1.5 items-start">
                  <span className="text-[10px] font-bold tracking-widest text-gray-400 uppercase px-1">GATEWAY</span>
                  <div className="bg-white border border-gray-200 rounded-xl px-4 py-3 shadow-sm">
                    <div className="flex gap-1">
                      {[0,150,300].map(d => <div key={d} className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: `${d}ms` }} />)}
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="px-6 pb-2 flex gap-2 flex-wrap">
            {SUGGESTED.map(q => (
              <button key={q} onClick={() => sendMessage(q)} disabled={loading}
                className="text-[11px] font-medium bg-white border border-gray-200 text-gray-500 px-3 py-1.5 rounded-lg hover:border-gray-400 hover:text-gray-800 transition-colors disabled:opacity-40">
                {q}
              </button>
            ))}
          </div>

          <div className="border-t border-gray-200 bg-white px-6 py-4 shrink-0">
            <div className="flex gap-3">
              <textarea value={input} onChange={e => setInput(e.target.value)} onKeyDown={onKey}
                placeholder="Send a message... (Enter to send, Shift+Enter for newline)" rows={2}
                className="flex-1 resize-none border border-gray-200 rounded-xl px-4 py-3 text-sm text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-gray-900 focus:border-gray-900 transition-colors" />
              <button onClick={() => sendMessage()} disabled={loading || !input.trim()}
                className="bg-gray-900 text-white px-6 rounded-xl text-sm font-semibold hover:bg-gray-700 disabled:opacity-30 disabled:cursor-not-allowed transition-colors">
                Send
              </button>
            </div>
            <p className="text-[10px] text-gray-400 mt-2 text-center tracking-wide">
              FastAPI · Redis · Qdrant · sentence-transformers · Gemini · Groq / Llama 3
            </p>
          </div>
        </div>

        <div className="w-px bg-gray-200 shrink-0" />

        <div className="w-[320px] shrink-0 bg-[#F7F7F8] flex flex-col overflow-hidden">
          <div className="flex border-b border-gray-200 bg-white shrink-0">
            {["Live", "Historical"].map(tab => (
              <button key={tab} onClick={() => setActiveTab(tab)}
                className={`flex-1 text-xs font-bold py-3 transition-colors tracking-wide
                  ${activeTab === tab ? "text-gray-900 border-b-2 border-gray-900" : "text-gray-400 hover:text-gray-600"}`}>
                {tab.toUpperCase()}
              </button>
            ))}
          </div>
          <div className="flex-1 overflow-y-auto p-3">
            {activeTab === "Live"
              ? <LivePanel stats={stats} health={health} session={session} lastRequest={lastRequest} />
              : <HistoricalPanel analytics={analytics} />}
          </div>
        </div>
      </div>
    </div>
  );
}