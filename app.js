// ZENITH Interactive Landing Page Logic

document.addEventListener('DOMContentLoaded', () => {
  initMobileNav();
  initSearchSimulator();
  initArchitectureVisualizer();
  initCLITerminal();
  initCopyButtons();
});

function initMobileNav() {
  const navToggle = document.getElementById('nav-toggle');
  const navLinks = document.querySelector('.nav-links');
  if (!navToggle || !navLinks) return;

  navToggle.addEventListener('click', () => {
    const isExpanded = navToggle.getAttribute('aria-expanded') === 'true';
    navToggle.setAttribute('aria-expanded', !isExpanded);
    navLinks.classList.toggle('active');
  });

  navLinks.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => {
      navLinks.classList.remove('active');
      navToggle.setAttribute('aria-expanded', 'false');
    });
  });
}

// 1. Interactive Search Simulator
const MOCK_DOCUMENTS = [
  {
    id: 1,
    title: "internal/storage/lsm.go",
    path: "~/ZENITH/internal/storage/lsm.go",
    snippet: "func (l *LSMTree) FlushMemTable() error { ... SkipList flush to SSTable with sparse index and Bloom filter generation }",
    bm25: 8.92,
    vector: 0.94,
    fuzzyDist: 0,
    keywords: ["lsm", "memtable", "sstable", "bloom", "flush", "storage", "skip-list"]
  },
  {
    id: 2,
    title: "docs/kubernetes_memory_debugging.md",
    path: "~/Documents/docs/kubernetes_memory_debugging.md",
    snippet: "# Troubleshooting Kubernetes Pod OOMKilled Errors\nWhen debugging memory pressure in Go microservices under K8s cgroups v2...",
    bm25: 9.45,
    vector: 0.88,
    fuzzyDist: 0,
    keywords: ["kubernetes", "memory", "debugging", "k8s", "pod", "oomkilled", "deployement"]
  },
  {
    id: 3,
    title: "internal/ranking/rrf.go",
    path: "~/ZENITH/internal/ranking/rrf.go",
    snippet: "func CalculateRRF(lexicalRank, vectorRank int, k float64) float64 { return 1.0 / (k + float64(lexicalRank)) + 1.0 / (k + float64(vectorRank)) }",
    bm25: 7.80,
    vector: 0.96,
    fuzzyDist: 0,
    keywords: ["rrf", "rank", "fusion", "vector", "bm25", "hybrid", "scoring", "cosine"]
  },
  {
    id: 4,
    title: "nerve/main.py",
    path: "~/ZENITH/nerve/main.py",
    snippet: "@app.post('/embed')\ndef generate_embeddings(req: TextRequest):\n    return embedder.encode(req.texts, convert_to_numpy=True).tolist()",
    bm25: 6.15,
    vector: 0.91,
    fuzzyDist: 0,
    keywords: ["nerve", "embed", "embeddings", "python", "fastapi", "sentence-transformers", "vector"]
  },
  {
    id: 5,
    title: "internal/analysis/bktree.go",
    path: "~/ZENITH/internal/analysis/bktree.go",
    snippet: "type BKTree struct { root *BKNode }\nfunc (t *BKTree) Search(query string, maxDist int) []FuzzyMatch { ... Levenshtein O(log n) pruning }",
    bm25: 8.10,
    vector: 0.79,
    fuzzyDist: 1,
    keywords: ["bktree", "fuzzy", "levenshtein", "distance", "typo", "matching", "pruning"]
  }
];

function initSearchSimulator() {
  const input = document.getElementById('sim-input');
  const resultsContainer = document.getElementById('sim-results');
  const chips = document.querySelectorAll('.chip');
  const modeBtns = document.querySelectorAll('.mode-btn');

  if (!input || !resultsContainer) return;

  let currentEmbedderMode = 'nerve';

  function renderResults(query) {
    if (!query.trim()) {
      query = "kubernetes memory debugging";
    }

    const qLower = query.toLowerCase();
    const queryTokens = qLower.split(/\s+/).filter(t => t.length > 0);

    // Calculate dynamic matching scores based on query
    const docScores = MOCK_DOCUMENTS.map(doc => {
      let lexicalMatch = 0;
      let vectorMatch = doc.vector;
      let isFuzzy = false;

      // Simple keyword matching simulation
      doc.keywords.forEach(kw => {
        if (qLower.includes(kw)) lexicalMatch += 2.5;
        // Check fuzzy match for individual query tokens
        queryTokens.forEach(token => {
          if (token.length < 3) return;
          if (!kw.includes(token)) {
            const maxDist = token.length <= 4 ? 1 : 2;
            if (levenshteinDistance(token, kw) <= maxDist) {
              lexicalMatch += 1.8;
              isFuzzy = true;
            }
          }
        });
      });

      // Embedder mode adjustments
      if (currentEmbedderMode === 'deterministic') {
        vectorMatch = Math.max(0.4, vectorMatch - 0.2); // Hash embeddings lower precision
      } else if (currentEmbedderMode === 'ollama') {
        vectorMatch = Math.min(0.98, vectorMatch + 0.02);
      }

      return { doc, lexicalMatch, vectorMatch, isFuzzy };
    });

    // Derive lexical and vector ranks across full document set
    const sortedByLexical = [...docScores].sort((a, b) => b.lexicalMatch - a.lexicalMatch);
    const lexicalRankMap = new Map();
    sortedByLexical.forEach((item, idx) => lexicalRankMap.set(item.doc.id, idx + 1));

    const sortedByVector = [...docScores].sort((a, b) => b.vectorMatch - a.vectorMatch);
    const vectorRankMap = new Map();
    sortedByVector.forEach((item, idx) => vectorRankMap.set(item.doc.id, idx + 1));

    const scoredDocs = docScores.map(({ doc, lexicalMatch, vectorMatch, isFuzzy }) => {
      const lexRank = lexicalRankMap.get(doc.id);
      const vecRank = vectorRankMap.get(doc.id);

      const rrfVal = (lexicalMatch > 0 ? 1 / (60 + lexRank) : 0) + (1 / (60 + vecRank));
      const rrfScore = rrfVal.toFixed(4);

      const bm25Score = lexicalMatch > 0 ? (doc.bm25 + lexicalMatch).toFixed(2) : "0.00";
      const vecScore = (vectorMatch * 100).toFixed(1) + "%";

      return {
        ...doc,
        bm25Score,
        vecScore,
        rrfScore,
        isFuzzy,
        relevance: parseFloat(rrfScore)
      };
    });

    // Sort by RRF score descending
    scoredDocs.sort((a, b) => b.relevance - a.relevance);

    resultsContainer.innerHTML = scoredDocs.map(doc => `
      <div class="result-card">
        <div class="result-main">
          <div class="result-title">
            📄 ${doc.title}
            ${doc.isFuzzy ? '<span class="logo-badge" style="background:rgba(255,183,3,0.1);color:#ffb703;border-color:rgba(255,183,3,0.3)">BK-Tree Fuzzy Match</span>' : ''}
          </div>
          <div class="result-path">${doc.path}</div>
          <div class="result-snippet">${highlightQuery(doc.snippet, qLower)}</div>
        </div>
        <div class="scores-breakdown">
          <div class="score-badge score-bm25">
            <span class="score-label">BM25 Lexical</span>
            <span class="score-val">${doc.bm25Score}</span>
          </div>
          <div class="score-badge score-vector">
            <span class="score-label">Vector Cosine</span>
            <span class="score-val">${doc.vecScore}</span>
          </div>
          <div class="score-badge score-rrf">
            <span class="score-label">RRF Score</span>
            <span class="score-val">${doc.rrfScore}</span>
          </div>
        </div>
      </div>
    `).join('');
  }

  input.addEventListener('input', (e) => renderResults(e.target.value));

  chips.forEach(chip => {
    chip.addEventListener('click', () => {
      chips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      const q = chip.getAttribute('data-query');
      input.value = q;
      renderResults(q);
    });
  });

  modeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      modeBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentEmbedderMode = btn.getAttribute('data-mode');
      renderResults(input.value);
    });
  });

  renderResults(input.value);
}

function highlightQuery(text, query) {
  if (!query) return text;
  const words = query.split(/\s+/).filter(w => w.length > 2);
  let highlighted = text;
  words.forEach(word => {
    const escapedWord = word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const regex = new RegExp(`(${escapedWord})`, 'gi');
    highlighted = highlighted.replace(regex, '<mark>$1</mark>');
  });
  return highlighted;
}

function levenshteinDistance(a, b) {
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;
  const matrix = [];
  for (let i = 0; i <= b.length; i++) matrix[i] = [i];
  for (let j = 0; j <= a.length; j++) matrix[0][j] = j;
  for (let i = 1; i <= b.length; i++) {
    for (let j = 1; j <= a.length; j++) {
      if (b.charAt(i - 1) === a.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1,
          matrix[i][j - 1] + 1,
          matrix[i - 1][j] + 1
        );
      }
    }
  }
  return matrix[b.length][a.length];
}

// 2. Architecture Explorer
const ARCH_DATA = {
  wal: {
    title: "Write-Ahead Log (WAL)",
    desc: "Every write operation is first appended to a crash-safe WAL file with CRC32 verification frames before updating the in-memory MemTable.",
    specs: [
      { key: "Format", val: "CRC-framed binary log" },
      { key: "Durability", val: "fsync on commit window" },
      { key: "Recovery Time", val: "< 15ms for 100k ops" }
    ]
  },
  memtable: {
    title: "MemTable (Skip-List)",
    desc: "In-memory sorted key-value skip list providing O(log n) insertions and lookups. Automatically flushed to SSTable when threshold (64MB) is reached.",
    specs: [
      { key: "Structure", val: "Lock-free Skip-List" },
      { key: "Flush Limit", val: "64 MB default" },
      { key: "Concurrency", val: "sync.RWMutex protected" }
    ]
  },
  sstable: {
    title: "SSTables (Sorted String Tables)",
    desc: "Immutable block-structured files stored on disk. Contains document postings, TF-IDF weights, and 384-dim float16 vector embeddings.",
    specs: [
      { key: "Storage", val: "Block-compressed binary" },
      { key: "Immutability", val: "Write-once read-many" },
      { key: "Vector Format", val: "Float16 quantized" }
    ]
  },
  bloom: {
    title: "Per-SSTable Bloom Filter",
    desc: "Probabilistic bit array generated for each SSTable. Allows O(1) bypass of disk reads for terms that do not exist in the segment.",
    specs: [
      { key: "False Positive Rate", val: "1%" },
      { key: "Lookup Speed", val: "< 50 nanoseconds" },
      { key: "Memory Footprint", val: "~1.2 bytes per key" }
    ]
  },
  bktree: {
    title: "BK-Tree Fuzzy Matcher",
    desc: "Metric tree discrete metric structure over Levenshtein edit distance. Evaluates typos in O(log n) by leveraging the triangle inequality.",
    specs: [
      { key: "Algorithm", val: "Levenshtein Distance" },
      { key: "Max Distance", val: "2 edits (configurable)" },
      { key: "Dictionary Size", val: "100k+ terms" }
    ]
  },
  nerve: {
    title: "Nerve Embedder Cascade",
    desc: "Embedded Python sidecar auto-bootstrapping `all-MiniLM-L6-v2`. Auto-cascades to local Ollama or deterministic hash fallback if Python is absent.",
    specs: [
      { key: "Default Model", val: "all-MiniLM-L6-v2 (384-d)" },
      { key: "Fallbacks", val: "Ollama nomic-embed-text → Hash" },
      { key: "Bootstrap Time", val: "< 200ms on warm start" }
    ]
  },
  rrf: {
    title: "Reciprocal Rank Fusion (RRF)",
    desc: "Combines lexical inverted index rankings (BM25) and neural vector cosine similarity rankings into a single unified hybrid score list.",
    specs: [
      { key: "Constant (k)", val: "60.0" },
      { key: "Tie-breaker", val: "BM25 Score" },
      { key: "Output", val: "Unified top-K ranking" }
    ]
  }
};

function initArchitectureVisualizer() {
  const nodes = document.querySelectorAll('.arch-node');
  const titleEl = document.getElementById('arch-detail-title');
  const descEl = document.getElementById('arch-detail-desc');
  const specsEl = document.getElementById('arch-detail-specs');

  if (!nodes.length || !titleEl || !descEl || !specsEl) return;

  function updateDetail(key) {
    const data = ARCH_DATA[key] || ARCH_DATA['wal'];
    titleEl.textContent = data.title;
    descEl.textContent = data.desc;

    specsEl.innerHTML = data.specs.map(s => `
      <div class="spec-row">
        <span class="spec-key">${s.key}</span>
        <span class="spec-val">${s.val}</span>
      </div>
    `).join('');
  }

  nodes.forEach(node => {
    node.addEventListener('click', () => {
      nodes.forEach(n => n.classList.remove('active'));
      node.classList.add('active');
      const key = node.getAttribute('data-node');
      updateDetail(key);
    });
  });

  updateDetail('wal');
}

// 3. CLI Terminal Showcase
const CLI_COMMANDS = {
  index: `<span class="t-prompt">$</span> <span class="t-cmd">zenith index ~/Documents</span>
<span class="t-dim">[17:42:01]</span> <span class="t-info">INF</span> Scanning directory: <span class="t-cmd">~/Documents</span>
<span class="t-dim">[17:42:01]</span> <span class="t-info">INF</span> Embedded sidecar 'nerve' status: <span class="t-success">ACTIVE</span> (http://127.0.0.1:8000)
<span class="t-dim">[17:42:02]</span> <span class="t-info">INF</span> Extracted 428 documents (.md, .go, .html, .txt)
<span class="t-dim">[17:42:04]</span> <span class="t-info">INF</span> Generated 384-d vector embeddings via nerve (all-MiniLM-L6-v2)
<span class="t-dim">[17:42:05]</span> <span class="t-info">INF</span> SSTable flush complete: memtable_size=14.2MB sstable_id=00004
<span class="t-dim">[17:42:05]</span> <span class="t-success">SUCCESS</span> Indexed 428 files in 3.84s (111.4 docs/sec)`,

  search: `<span class="t-prompt">$</span> <span class="t-cmd">zenith search "kubernetes memory debugging"</span>
<span class="t-dim">Rank  RRF Score  BM25   Vector   File</span>
────────────────────────────────────────────────────────────────────────────────
<span class="t-success">  1   0.0328     12.45   0.941   ~/Documents/docs/kubernetes_memory_debugging.md</span>
<span class="t-info">  2   0.0314     10.12   0.887   ~/Documents/k8s/pod_oom_handler.go</span>
<span class="t-dim">  3   0.0298      8.50   0.820   ~/Documents/notes/golang_cgroup_limits.txt</span>

<span class="t-dim">Query executed in 4.2ms (LSM Bloom bypass: 8/10 SSTables, BK-tree fuzzy: 0 edits)</span>`,

  watch: `<span class="t-prompt">$</span> <span class="t-cmd">zenith watch add ~/Documents</span>
<span class="t-success">✓</span> Added '~/Documents' to persistent watchlist (~/.zenith/watchlist.json)

<span class="t-prompt">$</span> <span class="t-cmd">zenith watch install</span>
<span class="t-success">✓</span> Registered Task Scheduler auto-start task 'ZenithWatch' on boot (login trigger)

<span class="t-prompt">$</span> <span class="t-cmd">zenith watch start</span>
<span class="t-info">INF</span> Watching 3 paths: [~/Documents, ~/Projects, ~/Desktop]
<span class="t-dim">[17:45:12]</span> <span class="t-info">EVENT</span> WRITE "docs/notes.md" -> incremental re-index (12ms)`,

  log: `<span class="t-prompt">$</span> <span class="t-cmd">zenith log -n 10 --type SEARCH</span>
<span class="t-dim">2026-07-26T17:30:11Z</span> | <span class="t-info">SEARCH</span> | query="goroutine channel leak" results=8 duration=3.8ms embedder=nerve
<span class="t-dim">2026-07-26T17:34:42Z</span> | <span class="t-info">SEARCH</span> | query="kubernets deployement" results=12 duration=4.5ms embedder=nerve (fuzzy=true)
<span class="t-dim">2026-07-26T17:40:02Z</span> | <span class="t-info">SEARCH</span> | query="lsm tree skip-list" results=5 duration=2.1ms embedder=deterministic`,

  serve: `<span class="t-prompt">$</span> <span class="t-cmd">zenith serve --port 8080</span>
<span class="t-info">INF</span> Starting ZENITH gRPC Search Engine Server on :8080
<span class="t-info">INF</span> Protobuf service: zenithproto.SearchService
<span class="t-success">READY</span> Listening for incoming gRPC connections...`
};

function initCLITerminal() {
  const tabs = document.querySelectorAll('.cli-tab');
  const body = document.getElementById('terminal-body');

  if (!tabs.length || !body) return;

  function selectTab(selectedTab) {
    const cmd = selectedTab.getAttribute('data-cmd');
    tabs.forEach(t => {
      const isActive = t === selectedTab;
      t.classList.toggle('active', isActive);
      t.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });

    const content = Object.prototype.hasOwnProperty.call(CLI_COMMANDS, cmd)
      ? CLI_COMMANDS[cmd]
      : CLI_COMMANDS['index'];

    body.innerHTML = content;
  }

  tabs.forEach(tab => {
    tab.addEventListener('click', () => selectTab(tab));
  });

  const initialTab = document.querySelector('.cli-tab.active') || tabs[0];
  if (initialTab) {
    selectTab(initialTab);
  }
}

// 4. Copy Buttons
function initCopyButtons() {
  const copyBtn = document.getElementById('copy-cmd-btn');
  if (!copyBtn) return;

  const originalText = copyBtn.innerHTML;
  let copyTimeout = null;

  const showFeedback = (msg, bg, color) => {
    if (copyTimeout) {
      clearTimeout(copyTimeout);
    }
    copyBtn.innerHTML = msg;
    copyBtn.style.background = bg;
    copyBtn.style.color = color;

    copyTimeout = setTimeout(() => {
      copyBtn.innerHTML = originalText;
      copyBtn.style.background = "";
      copyBtn.style.color = "";
      copyTimeout = null;
    }, 2000);
  };

  copyBtn.addEventListener('click', () => {
    const textToCopy = "go install github.com/shramanb113/ZENITH/cmd/zenith@latest";

    if (!navigator.clipboard || !navigator.clipboard.writeText) {
      showFeedback('❌ Failed!', '#ef4444', '#fff');
      return;
    }

    navigator.clipboard.writeText(textToCopy).then(() => {
      showFeedback('✓ Copied!', '#22c55e', '#04100b');
    }).catch(() => {
      showFeedback('❌ Failed!', '#ef4444', '#fff');
    });
  });
}
